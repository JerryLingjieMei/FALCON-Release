from abc import abstractmethod
from collections import abc

import torch

from utils import apply


class AbstractProgram:
    PROGRAMS_REGISTRY = {}

    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def __init__(self, *args):
        self.arguments = args

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.PROGRAMS_REGISTRY[cls.__name__] = cls

    def _transform(self, box_registry, **kwargs):
        new_args = []
        for arg in self.arguments:
            if isinstance(arg, AbstractProgram):
                new_arg = arg.evaluate(box_registry, **kwargs)
            elif isinstance(arg, list):
                if len(arg) == 0:
                    new_arg = torch.zeros(0, box_registry.size).to(box_registry.device)
                elif isinstance(arg[0], AbstractProgram):
                    # train_program, val_program
                    new_arg = [a.evaluate(box_registry, **kwargs) for a in arg]
                else:
                    new_arg = box_registry(torch.tensor(arg).to(box_registry.device))
            elif isinstance(arg, int):
                # concept_id, kind_id
                new_arg = arg
            else:
                raise NotImplementedError(f"Unknown program {arg}")
            new_args.append(new_arg)
        program = type(self)(*new_args)
        return program

    def evaluate(self, box_registry, **kwargs):
        return self._transform(box_registry, **kwargs)

    def __call__(self, executor):
        raise NotImplementedError

    def __str__(self):
        return "(" + ', '.join([self.name, *(str(arg) for arg in self.arguments)]) + ")"

    @staticmethod
    def sequence2text(tensor, concepts):
        if isinstance(tensor[0], abc.Sequence):
            return [str([concepts[_] for _ in t]) for t in tensor]
        else:
            return [concepts[_] for _ in tensor]

    def __mod__(self, dataset):
        texts = []
        for arg in self.arguments:
            if torch.is_tensor(arg):
                text = self.sequence2text(arg, dataset.named_entries_)
            elif isinstance(arg, list):
                text = self.sequence2text(arg, dataset.named_entries_)
            elif isinstance(arg, AbstractProgram):
                text = arg % dataset
            elif isinstance(arg, int):
                text = [dataset.named_entries_[arg]]
            else:
                raise NotImplementedError(f"Unexpected type {type(arg)}.")
            texts.append(text)
        if len(texts) > 0:
            maximal_lines = max(len(_) for _ in texts)
            for i, t in enumerate(texts):
                texts[i] = t * (maximal_lines // len(t))
            return [f"({', '.join([self.name] + list(_))})" for _ in zip(*texts)]
        else:
            return [f'({self.name}, )']

    def apply(self, f):
        return type(self)(*(apply(arg, f) for arg in self.arguments))

    def clone(self):
        return apply(self, lambda _: _)

    @property
    def device(self):
        for arg in self.arguments:
            if torch.is_tensor(arg):
                return arg.device
        else:
            raise AttributeError(f"{str(self)} is not on any device.")
