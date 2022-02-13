from itertools import chain

from .abstract_program import AbstractProgram
from .composite import Composite
from .fewshot import Fewshot
from .symbolic import *


def build_program(args):
    if isinstance(args, list) or isinstance(args, tuple):
        if len(args) == 0:
            return args
        elif isinstance(args[0], str):
            return AbstractProgram.PROGRAMS_REGISTRY[args[0]](*(build_program(arg) for arg in args[1:]))
        else:
            return args
    elif isinstance(args, str) or isinstance(args, int):
        return args
    elif isinstance(args, AbstractProgram):
        return args
    else:
        raise NotImplementedError


def to_batch(programs):
    results = []
    assert all(len(p) == len(programs[0]) for p in programs)
    for args in zip(*programs):
        if isinstance(args[0], str):
            results.append(args[0])
        elif isinstance(args[0], list):
            results.append(list(chain(*args)))
        else:
            results.append(to_batch(args))
    return tuple(results)
