import datetime
import logging
import os
import time

import torch

from dataset.utils import DataLoader, tqdm_cycle
from experiments import cfg_from_args
from models import build_model
from tools.dataset_catalog import DatasetCatalog
from utils import Checkpointer, mkdir
from utils import setup_logger, SummaryWriter, Metric
from utils import start_up, ArgumentParser, data_parallel
from utils import to_cuda
from visualization import build_visualizer


def test(cfg, args):
    logger = logging.getLogger("falcon_logger")
    logger.info("Setting up dependencies.")

    logger.info("Setting up models.")
    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    model = build_model(cfg).to("cuda")
    model = data_parallel(model, gpu_ids)

    logger.info("Setting up utilities.")
    output_dir = cfg.OUTPUT_DIR

    checkpointer = Checkpointer(cfg, model, None, None)
    iteration = checkpointer.load(args.iteration)
    summary_writer = SummaryWriter(output_dir)

    model.eval()
    for dataset_name, test_set in DatasetCatalog(cfg).get(cfg.DATASETS.TEST, args, as_tuple=True):
        start_testing_time = time.time()
        last_batch_time = time.time()
        test_loader = DataLoader(test_set, cfg)
        test_metrics = Metric(delimiter="  ", summary_writer=summary_writer)
        visualizer = build_visualizer(cfg.VISUALIZATION, test_set, summary_writer)
        logger.info(f"Start testing on {test_set} with mode {test_set.mode}.")

        with torch.no_grad():
            model.eval()
            evaluated = test_set.init_evaluate(args.mode)
            for i, inputs in enumerate(tqdm_cycle(test_loader)):
                data_time = time.time() - last_batch_time
                inputs = to_cuda(inputs)
                outputs = model(inputs)
                model.callback(inputs, outputs)
                test_set.callback(i)
                test_set.batch_evaluate(inputs, outputs, evaluated)

                batch_time = time.time() - last_batch_time
                last_batch_time = time.time()
                test_metrics.update(batch_time=batch_time, data_time=data_time)

                if i % 5 == 0:
                    visualizer.visualize(inputs, outputs, model, iteration + i)

            metrics = test_set.evaluate_metric(evaluated)
            visualizer.visualize(evaluated, model, iteration)
            test_set.save(output_dir, evaluated, iteration, metrics)
            test_metrics.update(**metrics)
            test_metrics.log_summary(test_set.tag, iteration)
            checkpointer.save(9999999, False)
            logger.warning(test_metrics.delimiter.join([f"iter: {iteration}", f"{test_metrics}"]))

        total_training_time = time.time() - start_testing_time
        logger.info(f"Total testing time: {datetime.timedelta(seconds=total_training_time)}")


def main():
    parser = ArgumentParser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)
    output_dir = mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("falcon_logger", os.path.join(output_dir, "test_log.txt"))
    start_up()

    logger.info(f"Running with args:\n{args}")
    logger.info(f"Running with config:\n{cfg}")
    test(cfg, args)


if __name__ == "__main__":
    main()
