from collections import abc

import torch
import torch.nn.functional as F
from torch import nn


def apply(x, f):
    if torch.is_tensor(x):
        return f(x)
    elif isinstance(x, list) or isinstance(x, tuple):
        # noinspection PyArgumentList
        return type(x)(apply(_, f) for _ in x)
    elif isinstance(x, dict):
        return {k: apply(v, f) for k, v in x.items()}
    elif getattr(x, "apply", None) is not None:
        return x.apply(f)
    else:
        return x


def to_device(x, device):
    return apply(x, lambda _: _.to(device))


def to_cuda(x):
    return apply(x, lambda _: _.to("cuda"))


def to_cpu(x):
    return apply(x, lambda _: _.to("cpu"))


def to_cpu_detach(x):
    return apply(x, lambda _: _.to("cpu").detach())


def to_serializable(x):
    return apply(x, lambda _: _.tolist() if torch.is_tensor(_) else _)


def nonzero(x, offset=0) -> list:
    return (x.nonzero(as_tuple=False).squeeze(1) + offset).tolist()


def bind(x):
    return torch.cat(x.unbind(-1), -1)


def unbind(x, n):
    return torch.stack(x.chunk(n, -1), -1)


def gather_loss(outputs):
    loss_dict = {}
    for k, v in outputs.items():
        if "loss" in k or "reg" in k:
            loss_dict[k] = v.mean()
    return loss_dict


def is_nan(xs):
    if isinstance(xs, list):
        return any(is_nan(_) for _ in xs)
    elif torch.is_tensor(xs):
        return torch.isnan(xs).any()
    else:
        return False


def is_inf(xs):
    if isinstance(xs, list):
        return any(is_inf(_) for _ in xs)
    elif torch.is_tensor(xs):
        return torch.isinf(xs).any()
    else:
        return False


def collate_fn(batch, no_collate_keys=()):
    if torch.is_tensor(batch):
        return batch
    elif isinstance(batch, abc.Mapping):
        return {k: v if k in no_collate_keys else collate_fn(v, no_collate_keys) for k, v in batch.items()}
    elif torch.is_tensor(batch[0]):
        if all(b.shape == batch[0].shape and b.dtype == batch[0].dtype for b in batch):
            return torch.stack(batch)
        else:
            return batch
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], abc.Mapping):
        return {
            key: collate_fn([d[key] for d in batch], no_collate_keys) if key not in no_collate_keys else [d[key]
                for d in batch] for key in batch[0]}
    else:
        return batch


def pad_topk(logit, k):
    topk = min(len(logit), k)
    return F.pad(torch.topk(logit, min(len(logit), topk)).indices, [0, k - topk], "constant", -1)


def log_normalize(tensor, dim=-1):
    return tensor - torch.logsumexp(tensor, dim)


def apply_grid(x):
    u, v = torch.meshgrid([torch.arange(x.shape[-2]), torch.arange(x.shape[-1])])
    return torch.cat([x, u.expand(x.shape[0], *([-1] * (len(x.shape) - 1))).to(x.device),
        v.expand(x.shape[0], *([-1] * (len(x.shape) - 1))).to(x.device)], -3)


EPS = 1e-4
INF = 1e4


def nan_hook(self, inputs, outputs, key=""):
    if isinstance(outputs, dict):
        for k, v in outputs.items():
            nan_hook(self, inputs, v, f"{key}.{k}")
        return
    has_nan = is_nan(outputs)
    if has_nan:
        raise ValueError(f"Found Nan in entry \'{key}\' of class {self.__class__.__name__}.")
    has_inf = is_inf(outputs)
    if has_inf:
        raise ValueError(f"Found Inf in entry \'{key}\' of class {self.__class__.__name__}.")


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def compose_image(image, mask):
    return torch.cat([image.unsqueeze(0) * mask.unsqueeze(1), image.unsqueeze(0).expand(len(mask), -1, -1, -1)],
        1)


def mask2bbox(mask):
    return mask.sum(-1, keepdims=True).bool() & mask.bool().sum(-2, keepdims=True).bool()


def create_dummy():
    return nn.Parameter(torch.zeros(1).squeeze())


def invert(x):
    assert x.ndim == 1
    y = torch.zeros_like(x)
    y[x] = torch.arange(len(x))
    return y
