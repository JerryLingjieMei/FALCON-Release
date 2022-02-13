import logging
import os

import torch.autograd.profiler as profiler
from torchviz import make_dot

from dataset.utils import DataLoader
from experiments import cfg_from_args
from models import build_model
from solver import make_scheduler, Optimizer
from tools.dataset_catalog import DatasetCatalog
from utils import Checkpointer, mkdir, join,setup_logger,start_up, ArgumentParser,record_model, record_dataset, to_cuda, gather_loss


def profile(cfg, args):
    train_set = DatasetCatalog(cfg).get(cfg.DATASETS.TRAIN, args)
    train_loader = DataLoader(train_set, cfg)

    model = build_model(cfg).to("cuda")

    logger = logging.getLogger("falcon_logger")
    logger.info("Start profiling")
    output_dir = cfg.OUTPUT_DIR

    optimizer = Optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    weight = cfg.WEIGHT
    checkpointer = Checkpointer(cfg, model, optimizer, scheduler)
    start_iteration = checkpointer.load(args.iteration)
    model.train()

    inputs = next(iter(train_loader))
    inputs = to_cuda(inputs)
    outputs = model(inputs)
    with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        record_dataset(train_loader.dataset)
        if args.dataset_only:
            for i, inputs in enumerate(train_loader):
                if i == 4: break

        else:
            record_model(model, cfg)
            inputs = to_cuda(next(iter(train_loader)))
            outputs = model(inputs)

    loss_dict = gather_loss(outputs)
    total_loss = sum(loss_dict.values())

    logger.warning(prof)
    logger.warning(prof.key_averages())
    mkdir("profiles")
    prof.export_chrome_trace(join("profiles", f"{os.path.basename(cfg.OUTPUT_DIR)}_trace.json"))
    dot = make_dot(total_loss, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render(join("profiles", f"{os.path.basename(cfg.OUTPUT_DIR)}_dot"))


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()

    cfg = cfg_from_args(args)
    output_dir = mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("falcon_logger", os.path.join(output_dir, "train_log.txt"))
    start_up()

    logger.info(f"Running with args:\n{args}")
    logger.info(f"Running with config:\n{cfg}")
    profile(cfg, args)


if __name__ == "__main__":
    main()
