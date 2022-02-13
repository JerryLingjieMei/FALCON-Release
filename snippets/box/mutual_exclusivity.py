import os

import torch

from dataset.utils import tqdm_cycle
from experiments import cfg_from_args
from snippets.snippet_utils import cfg2test_loader, cfg2model, cfg2iteration
from utils import join, dump, setup_logger
from utils import ArgumentParser, nonzero, to_cuda

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    args.iteration = 9999999
    cfg = cfg_from_args(args)

    model = cfg2model(cfg, args)
    test_loader = cfg2test_loader(cfg, args)
    test_set = test_loader.dataset
    iteration = cfg2iteration(cfg, args)
    output_dir = cfg.OUTPUT_DIR
    entailment = model.program_executor.learner.entailment

    train_indices = nonzero(test_set.concept_split_specs == 0)
    test_indices = nonzero(test_set.concept_split_specs == 6)
    train_embeddings = model.box_registry[train_indices]
    test_embeddings = model.box_registry[test_indices]
    all_embeddings = torch.cat([train_embeddings, test_embeddings])

    scores = []
    for inputs in tqdm_cycle(test_loader):
        inputs = to_cuda(inputs)
        outputs = model(inputs)
        features = torch.stack(outputs["val_sample"]["features"][0][:15]).squeeze(1)
        logits = entailment(features.unsqueeze(1), all_embeddings.unsqueeze(0)) * 4
        probs = torch.sigmoid(logits) / torch.sigmoid(logits).sum(-1, keepdim=True)
        s = probs[:, len(train_indices):].sum(-1).mean(0)
        scores.append(s)

    scores = torch.stack(scores).mean()
    logger = setup_logger("falcon_logger", os.path.join(output_dir, "snippet_log.txt"))
    logger.warning(f"mutual exclusivity: {scores}")
    dump({"mutual_exclusivity": scores}, join(output_dir, "inference", cfg.DATASETS.TEST.split("&")[0],
        f"mutual_exclusivity_{iteration:07d}.pth"))
