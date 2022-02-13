from collections import defaultdict

import torch

from .fewshot import Fewshot
from .symbolic import SymbolicProgram


class Composite(SymbolicProgram):

    def __init__(self, *arg):
        super().__init__(*arg)
        self.gt_embeddings, self.relations, self.concept_index = arg

    def evaluate(self, box_registry, **kwargs):
        k = {}
        for group in ["train_sample", "val_sample"]:
            k[group] = [q.evaluate(box_registry, features=f, relations=r) for q, f, r in
                zip(kwargs[group].pop("program"), kwargs[group].pop("features"),
                    kwargs[group].pop("relations"))]
        p = super(Composite, self).evaluate(box_registry, **kwargs)
        for group in ["train_sample", "val_sample"]:
            p.kwargs[group]["program"] = k[group]
        p.relations = torch.tensor(self.relations).to(box_registry.device)
        p.concept_index = box_registry[torch.tensor([self.concept_index]).to(box_registry.device)]
        return p

    def to_fewshot(self, train_features):
        fewshot = Fewshot(self.gt_embeddings, self.relations, train_features, self)
        return fewshot

    @property
    def hypernym_embeddings(self):
        return self.gt_embeddings[list(i for i, r in
        enumerate(self.relations if not torch.is_tensor(self.relations) else self.relations.tolist()) if
        r == 0)]

    @property
    def samekind_embeddings(self):
        return self.gt_embeddings[list(i for i, r in
        enumerate(self.relations if not torch.is_tensor(self.relations) else self.relations.tolist()) if
        r == 2)]

    @property
    def is_fewshot(self):
        return not any(i == -1 for i in self.train_image_index)

    @property
    def is_attached(self):
        return len(self.relations) > 0

    def __getattr__(self, item):
        if item.startswith("train_"):
            return self.kwargs["train_sample"][item[len("train_"):]]
        elif item.startswith("val_"):
            return self.kwargs["val_sample"][item[len("val_"):]]
        elif item.startswith('metaconcept_') or item.startswith('composite_'):
            return self.kwargs['task'][item]
        else:
            return self.__getattribute__(item)

    def __setattr__(self, item, value):
        if item.startswith("train_"):
            self.kwargs["train_sample"][item[len("train_"):]] = value
        elif item.startswith("val_"):
            self.kwargs["val_sample"][item[len("val_"):]] = value
        elif item.startswith('metaconcept_') or item.startswith('composite_'):
            self.kwargs['task'][item] = value
        else:
            super().__setattr__(item, value)

    @staticmethod
    def sequence2text(tensor, concepts):
        if tensor.ndim == 1:
            return [f"{concepts[tensor[-1]]} from {len(tensor) - 1:01d} masks"]
        else:
            return ['']

    def __mod__(self, dataset):
        return [f"{dataset.named_entries_[self.concept_index]} from {len(self.relations):01d} supports."]

    def evaluate_logits(self, executor, **kwargs):
        queried_embedding = kwargs["queried_embedding"]
        train_ends, train_query_objects = [], []
        for p in self.train_program:
            q = p.evaluate_token(queried_embedding)
            out = q(executor)
            e, o = out["end"].max(-1)
            train_ends.append(e)
            train_query_objects.append(o.squeeze(0))
        train_query_objects = torch.stack(train_query_objects)

        val_ends, val_query_objects = [], []
        for p in self.val_program:
            q = p.evaluate_token(queried_embedding)
            out = q(executor)
            val_ends.append(out["end"])
            val_query_objects.append(out["query_object"])
        val_query_objects = torch.stack(val_query_objects)

        output = defaultdict(dict, **kwargs)
        output["train_sample"] = {"end": train_ends, "query_object": train_query_objects}
        output["val_sample"] = {"end": val_ends, "query_object": val_query_objects}
        return output

    def __call__(self, executor):
        outputs = executor(self)
        outputs = executor.compute_logits(self, **outputs)
        return outputs
