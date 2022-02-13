import random
from itertools import accumulate

import torch


def sample_with_ratio(n, split_ratio, seed):
    random.seed(seed)
    perm_indices = list(range(n))
    random.shuffle(perm_indices)
    cum_lengths = [int(_ * n) for _ in accumulate(split_ratio)]
    cum_lengths.insert(0, 0)
    split_specs = [0] * n
    for i in range(len(split_ratio)):
        for _ in perm_indices[cum_lengths[i]: cum_lengths[i + 1]]:
            split_specs[_] = i
    return torch.tensor(split_specs)


def hierarchy_detach(hierarchy, n_detach, seed):
    random.seed(seed)
    parents = {}

    def _mark_parent(name, node):
        for k, v in node.items():
            parents[k] = name, node
            _mark_parent(k, v)

    try:
        _mark_parent(-1, hierarchy)
    finally:
        _mark_parent = None

    leaves = set()

    def _mark_initial_leaves(n):
        for k, v in n.items():
            if len(v) == 0:
                leaves.add(k)
            else:
                _mark_initial_leaves(v)

    _mark_initial_leaves(hierarchy)

    detached = []
    while len(detached) < n_detach:
        to_detach = random.choice(list(leaves))
        while True:
            detached.append(to_detach)
            if to_detach in leaves:
                leaves.remove(to_detach)
            parent_name, parent_node = parents[to_detach]
            parent_node.pop(to_detach)
            if len(parent_node) == 0:
                to_detach = parent_name
            else:
                break

    return detached
