import inspect
from functools import wraps

import torch.autograd.profiler as profiler

def record_wrapper(func, name):
    @wraps(func)
    def new_func(*args, **kwargs):
        with profiler.record_function(name):
            out = func(*args, **kwargs)
        return out

    return new_func


def record_model(model, cfg):
    for modules in model.modules():
        name = modules.__class__.__name__
        modules.forward = record_wrapper(modules.forward, name)


def record_dataset(dataset):
    for attr, value in inspect.getmembers(dataset, predicate=inspect.ismethod):
        setattr(dataset, attr, record_wrapper(value, attr))
