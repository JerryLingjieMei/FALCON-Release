from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from experiments import cfg_from_args
from snippets.snippet_utils import cfg2model, cfg2test_loader, cfg2iteration
from utils import load, join, SummaryWriter
from utils import ArgumentParser

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    cfg = cfg_from_args(args)

    test_loader = cfg2test_loader(cfg, args)
    model = cfg2model(cfg, args)
    iteration = cfg2iteration(cfg, args)
    summary_writer = SummaryWriter(cfg.OUTPUT_DIR)

    filename = join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST.split("&")[0], f"meta_{iteration:07d}.pth")
    raw_results = load(filename)
    measure = model.program_executor.entailment
    widths_by_split = defaultdict(list)
    for u in sorted(torch.unique(raw_results["image_class"])):
        specific_boundary = measure.specific_boundary(raw_results["feature"][raw_results["image_class"] == u])
        width = specific_boundary[measure.mid_channels:].mean()
        widths_by_split[test_loader.dataset.concept_split_specs[u].item()].append(width)
    widths_by_split = {k: torch.Tensor(v).mean() for k, v in widths_by_split.items()}

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    split = widths_by_split.keys()
    width = widths_by_split.values()
    ax.bar(split, width)
    for s, w in widths_by_split.items():
        ax.text(s - .35, w + .002, "{:.03f}".format(w))
    plt.title("Average width of specific boundary formed by all image within the same class\n"
              "Sorted by split, <=0 for train concept images, >0 for test concept images")
    summary_writer.add_figure("visualize/feature_width", fig)
