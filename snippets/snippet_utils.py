import torch

from dataset.utils.dataloader import DataLoader
from models import build_model
from tools.dataset_catalog import DatasetCatalog
from utils import Checkpointer, start_up, data_parallel


def cfg2train_loader(cfg, args):
    train_set = DatasetCatalog(cfg).get(cfg.DATASETS.TRAIN, args)
    train_loader = DataLoader(train_set, cfg)
    return train_loader


def cfg2val_loader(cfg, args):
    val_set = DatasetCatalog(cfg).get(cfg.DATASETS.VAL, args)
    val_loader = DataLoader(val_set, cfg)
    return val_loader


def cfg2test_loader(cfg, args):
    test_set = DatasetCatalog(cfg).get(cfg.DATASETS.TEST, args)
    test_loader = DataLoader(test_set, cfg)
    return test_loader


def cfg2second_test_loader(cfg, args):
    test_set = DatasetCatalog(cfg).get(cfg.DATASETS.TEST, args, as_tuple=True)[1][1]
    test_loader = DataLoader(test_set, cfg)
    return test_loader


def cfg2model(cfg, args):
    model = build_model(cfg).to("cuda")
    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    model = data_parallel(model, gpu_ids)
    start_up()
    checkpointer = Checkpointer(cfg, model, None, None)
    checkpointer.load(args.iteration)
    model.eval()
    return model


def cfg2iteration(cfg, args):
    return Checkpointer(cfg, None, None, None).iteration
