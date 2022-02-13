import re

import torch
import torch.nn.functional as F

from models.programs import AbstractProgram
from utils import copy_dict, apply, EPS


class SymbolicProgram(AbstractProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.kwargs = {}
        self.registered = None, []

    def evaluate(self, box_registry, **kwargs):
        p = super(SymbolicProgram, self)._transform(box_registry, **kwargs)
        p.kwargs = copy_dict(kwargs)
        p.registered = self.registered
        return p

    @property
    def object_collections(self):
        return self.kwargs["features"]

    @object_collections.setter
    def object_collections(self, other):
        self.kwargs["features"] = other
        for k in dir(self):
            if re.search("child", k):
                getattr(self, k).object_collections = other

    @property
    def relation_collections(self):
        return self.kwargs["relations"]

    @relation_collections.setter
    def relation_collections(self, other):
        self.kwargs["relations"] = other
        for k in dir(self):
            if re.search("child", k):
                getattr(self, k).relation_collections = other

    def apply(self, f):
        p = type(self)(*(apply(arg, f) for arg in self.arguments))
        p.kwargs = apply(self.kwargs, f)
        p.registered = self.registered
        return p

    def __call__(self, executor):
        raise NotImplementedError

    @property
    def right_most(self):
        if isinstance(self.arguments[-1], SymbolicProgram):
            return self.arguments[-1].right_most
        else:
            return self.arguments[-1]

    def __len__(self):
        length = 0
        for a in self.arguments:
            if isinstance(a, SymbolicProgram):
                length = max(length, len(a))
        return length + 1

    def register_token(self, concept_id):
        for i, arg in enumerate(self.arguments):
            # noinspection PyUnresolvedReferences
            if isinstance(arg, SymbolicProgram):
                arg.register_token(concept_id)
            else:
                arg = torch.tensor(arg)
                if (arg == concept_id).any():
                    self.registered = i, (arg == concept_id).nonzero(as_tuple=False).tolist()

        return self

    def evaluate_token(self, queried_embedding):
        arguments = []
        for i, arg in enumerate(self.arguments):
            if torch.is_tensor(arg) and i == self.registered[0]:
                for t in self.registered[1]:
                    arg[tuple(t)] = queried_embedding
            elif isinstance(arg, SymbolicProgram):
                arg = arg.evaluate_token(queried_embedding)
            arguments.append(arg)
        program = type(self)(*arguments)
        program.kwargs = copy_dict(self.kwargs)
        program.registered = self.registered
        return program


class Scene(SymbolicProgram):
    BIG_NUMBER = 10

    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self, executor):
        logit = torch.ones((1, 1), device=self.object_collections.device) * self.BIG_NUMBER
        return {"end": logit}


class Unique(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args

    def __call__(self, executor):
        child = self.child(executor)
        logit = child["end"]
        if executor.training:
            prob = F.softmax(logit, dim=-1)
        else:
            prob = F.one_hot(logit.max(-1).indices, logit.shape[-1])
        return {**child, "end": prob}


class Filter(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, self.concept_collections = args

    def __call__(self, executor):
        child = self.child(executor)
        mask = executor.entailment(self.object_collections.unsqueeze(-3),
            self.concept_collections.unsqueeze(-2))
        filter_logit = torch.min(child["end"], mask)
        query_object = mask[..., 0].max(-1).indices
        return {**child, "end": filter_logit, "queried_embedding": self.concept_collections,
            "feature": self.object_collections, "query_object": query_object}


class Relate(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, self.direction_collections = args

    def __call__(self, executor):
        child = self.child(executor)
        mask = executor.entailment(self.relation_collections.unsqueeze(0),
            self.direction_collections.unsqueeze(1).unsqueeze(1))
        new_prob = (child["end"].unsqueeze(1) * torch.sigmoid(mask)).sum(-1).clamp(EPS, 1 - EPS)
        new_logit = torch.log(new_prob) - torch.log(1 - new_prob)
        return {**child, "end": new_logit}


class RelateAttributeEqual(Relate):
    pass


class Intersect(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        logit = torch.min(left_child["end"], right_child["end"])
        return {**left_child, **right_child, "end": logit}


class Union(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        logit = torch.max(left_child["end"], right_child["end"])
        return {**left_child, **right_child, "end": logit}


class Count(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args

    def __call__(self, executor):
        child = self.child(executor)
        if executor.training:
            count = torch.sigmoid(child["end"]).sum(-1)
        else:
            count = (child["end"] >= 0).sum(-1)

        return {**child, "end": count}


class CountGreater(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        if executor.training:
            left_count = torch.sigmoid(left_child["end"]).sum(-1)
            right_count = torch.sigmoid(right_child["end"]).sum(-1)
            logit = 4 * (left_count - right_count - .5)
        else:
            left_count = (left_child["end"] >= 0).sum(-1)
            right_count = (right_child["end"] >= 0).sum(-1)
            logit = -10 + 20 * (left_count > right_count).float()

        return {**left_child, **right_child, "end": logit}


class CountLess(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        if executor.training:
            left_count = torch.sigmoid(left_child["end"]).sum(-1)
            right_count = torch.sigmoid(right_child["end"]).sum(-1)
            logit = 4 * (-left_count + right_count - .5)
        else:
            left_count = (left_child["end"] >= 0).sum(-1)
            right_count = (right_child["end"] >= 0).sum(-1)
            logit = -10 + 20 * (left_count < right_count).float()

        return {**left_child, **right_child, "end": logit}


class CountEqual(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        if executor.training:
            left_count = torch.sigmoid(left_child["end"]).sum(-1)
            right_count = torch.sigmoid(right_child["end"]).sum(-1)
            logit = 8 * (.5 - (left_count - right_count).abs())
        else:
            left_count = (left_child["end"] >= 0).sum(-1)
            right_count = (right_child["end"] >= 0).sum(-1)
            logit = -10 + 20 * (left_count == right_count).float()

        return {**left_child, **right_child, "end": logit}


class Query(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, self.concept_collections = args

    def __call__(self, executor):
        child = self.child(executor)
        mask = executor.entailment(self.object_collections.unsqueeze(0).unsqueeze(2),
            self.concept_collections.unsqueeze(1))
        mask = torch.sigmoid(mask) / (torch.sigmoid(mask).sum(-1, keepdim=True) +EPS)
        new_prob = (child["end"].unsqueeze(2) * mask).sum(1).clamp(EPS, 1 - EPS)
        out = {**child, "end": new_prob, "queried_embedding": self.concept_collections[0],
            "feature": self.object_collections}
        if self.registered[0] is not None:
            out["query_object"] = mask[0, :, self.registered[1][0][1]].max(-1).indices
        return out


class QueryAttributeEqual(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child, self.attribute_collections = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        mask = executor.entailment(self.relation_collections.unsqueeze(0),
            self.attribute_collections.unsqueeze(1).unsqueeze(1))
        new_prob = (left_child["end"].unsqueeze(1) * torch.sigmoid(mask) * right_child["end"].unsqueeze(2)).sum(
            (1, 2)).clamp(EPS, 1 - EPS)
        new_logit = torch.log(new_prob) - torch.log(1 - new_prob)
        return {**left_child, **right_child, "end": new_logit}


class Exist(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args

    def __call__(self, executor):
        child = self.child(executor)
        max_logit, query_object = child["end"].max(-1)
        return {**child, "end": max_logit}
