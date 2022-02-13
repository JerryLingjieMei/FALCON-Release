import torch
from torch import nn
# noinspection PyProtectedMember
from torch.nn.parallel._functions import Scatter, Gather

from .misc import is_debug
from .tensor import to_device

def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        # Equally divide lists
        if isinstance(obj, list) and len(obj) > 0:
            return [[to_device(obj[_], device) for _ in chunk] for device, chunk in
                zip(target_gpus, torch.arange(len(obj)).chunk(len(target_gpus)))]
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        # noinspection PyTypeChecker
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            for o in outputs:
                if not torch.is_tensor(o):
                    import IPython
                    IPython.embed()
            return Gather.apply(target_device, dim, *outputs)
        elif isinstance(out, list):
            gathered = []
            for o in outputs:
                gathered.extend(to_device(o, target_device))
            if torch.is_tensor(gathered[0]):
                if all(g.shape == gathered[0].shape for g in gathered) and all(
                        g.dtype == gathered[0].dtype for g in gathered):
                    return torch.stack(gathered)
                else:
                    return gathered
            else:
                return gathered
        elif isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            # noinspection PyArgumentList
            return type(out)(((k, gather_map([d[k] for d in outputs])) for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


def data_parallel(model, gpu_ids):
    if len(gpu_ids) > 1 and not is_debug():
        return DataParallel(model, gpu_ids)
    else:
        return model


class DataParallel(nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return getattr(self._modules["module"], name)

