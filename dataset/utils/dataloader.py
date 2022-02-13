import copy

import torch
from torch.utils.data.dataloader import DataLoader as DataLoader_
from tqdm import tqdm

from utils import is_debug, collate_fn

def deepcopy(x):
    if isinstance(x,dict):
        return {k:deepcopy(v) for k,v in x.items()}
    elif isinstance(x, list):
        if torch.is_tensor(x[0]):
            return [xx.clone().detach() for xx in x]
    elif torch.is_tensor(x):
            return x.clone().detach()
    return x

def cycle(data_loader):
    while True:
        size = len(data_loader.dataset)
        for x in data_loader:
            if size == len(data_loader.dataset):
                # yield deepcopy(x)
                yield x
            else:
                break

def tqdm_cycle(data_loader):
    for x in tqdm(data_loader):
        yield x

class DataLoader(DataLoader_):
    DATASET_LOGGED = set()

    def __init__(self, dataset, cfg):
        if is_debug():
            num_workers = 0
            batch_size = 1
        else:
            num_workers = cfg.DATALOADER.NUM_WORKERS
            batch_size = cfg.SOLVER.BATCH_SIZE
        batch_sampler = dataset.get_batch_sampler(batch_size)
        if batch_sampler is not None:
            kwargs = {"batch_sampler": batch_sampler}
        else:
            kwargs = {"batch_size": batch_size, "sampler": dataset.sampler}
        super().__init__(dataset, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
        if str(dataset) not in self.DATASET_LOGGED:
            dataset.log_info()
            self.DATASET_LOGGED.add(str(dataset))
