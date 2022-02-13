import functools
import random
import re
import time
import weakref

import torch


class Singleton(type):
    _instances_with_args_kwargs = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances_with_args_kwargs:
            cls._instances_with_args_kwargs[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances_with_args_kwargs[cls]


class Timer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(time.time() - self.start_time)


def map_method(self, regex):
    method_map = {}
    for k in dir(self):
        match = re.fullmatch(regex, k)
        if match is not None:
            method_map[match.group(1)] = getattr(self, k)
    return method_map


def underscores(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def camel_case(name):
    return ''.join(x.capitalize() or '_' for x in name.split('_'))


def copy_dict(d):
    return d if not isinstance(d, dict) else {k: copy_dict(v) for k, v in d.items()}


def lru_cache(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            for i, arg in enumerate(args):
                assert not torch.is_tensor(arg), f"The {i}-th element of {func.__name__} is a tensor."
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


class IdentityDict(dict):
    def __missing__(self, key):
        return key


def map_wrap(f):
    @functools.wraps(f)
    def new_f(self, *args):
        return list(zip(*map(f, [self]*len(args[0]), *args)))

    return new_f
