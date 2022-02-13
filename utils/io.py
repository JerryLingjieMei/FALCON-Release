import json
import logging
import os
import pickle
import re
from functools import wraps

import h5py
import torch
import yaml
from PIL import Image

from .misc import skip_cache


def read_image(file_name, mode="RGB"):
    return Image.open(file_name).convert(mode)


def save_image(var, file_name):
    return var.save(file_name, "JPEG")


def join(*args):
    args = [arg for arg in args if arg is not None]
    return os.path.join(*args)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def load(filename):
    """Read json and yaml file"""
    filename = re.sub("http://vision38.csail.mit.edu", "", filename)
    with open(filename, "r") as f:
        if filename.endswith(".json"):
            x = json.load(f)
        elif filename.endswith(".yaml"):
            x = yaml.full_load(f)
        elif filename.endswith(".pth"):
            x = torch.load(filename, map_location="cpu")
        elif filename.endswith(".pkl"):
            with open(filename, "rb") as g:
                x = pickle.load(g)
        elif filename.endswith('.h5'):
            return h5py.File(filename, 'r')
        else:
            raise FileNotFoundError(f"{f} not found.")
    return x


def dump(x, filename):
    """Write json and yaml file"""
    assert filename is not None
    with open(filename, "w") as f:
        try:
            if filename.endswith(".json"):
                json.dump(x, f, indent=4)
            elif filename.endswith(".yaml"):
                yaml.safe_dump(x, f, indent=4)
            elif filename.endswith(".pth"):
                torch.save(x, filename)
            else:
                raise FileNotFoundError()
        except FileNotFoundError as e:
            print(f"{filename} invalid.")
            raise e
        except TypeError as e:
            os.remove(filename)
            raise e


def file_cached(*prefix):
    def decorator(fn):
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            assert len(args) == 0 and len(kwargs) == 0
            filename = join(self.augmented_root, self.get_augmented_name(fn.__qualname__.split('.')[0]), *prefix)
            filename = f"{os.path.splitext(filename)[0]}.json"
            if os.path.exists(filename) and not skip_cache():
                return load(filename)
            logger = logging.getLogger("falcon_logger")
            logger.info(f"Running {fn.__qualname__}.")
            x = fn(self)
            mkdir(os.path.split(filename)[0])
            dump(x, filename)
            logger.info(f"Saved to {join(self.get_augmented_name(fn.__qualname__.split('.')[0]), *prefix)}.")
            return x

        return wrapped_fn

    return decorator


def cast(*types):
    def decorator(fn):
        @wraps(fn)
        def wrapped_func(self):
            out = fn(self)
            if len(types) > 1:
                assert len(types) == len(out)
                outs = [o if t is None else t(o) for t, o in zip(types, out)]
            else:
                outs = types[0](out)
            return outs

        return wrapped_func

    return decorator


def symlink_recursive(src, dest):
    for name in os.listdir(src):
        src_file = join(src, name)
        dest_file = join(dest, name)
        try:
            os.symlink(src_file, dest_file)
        except:
            pass
