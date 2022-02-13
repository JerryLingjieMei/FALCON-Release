from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from snippets.snippet_utils import cfg2model, cfg2logger, cfg2test_loader
from utils import load, join, ArgumentParser

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    config_file = args.config_file

    cfg, logger, summary_writer = cfg2logger(config_file, args)
    test_loader = cfg2test_loader(config_file, args)
    model = cfg2model(config_file, args)

    filename = join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, "concept_feature.pth")
    raw_results = load(filename)
    measure = model.program_executor.entailment
    similarity_by_split = defaultdict(list)
    for u in sorted(torch.unique(raw_results["image_class"])):
        features = raw_results["feature"][raw_results["image_class"] == u]
        specific_boundary = measure.specific_boundary(features)[:measure.mid_channels]
        mean_similarity = F.cosine_similarity(features[:, :measure.mid_channels], specific_boundary.unsqueeze(0)).mean()
        similarity_by_split[test_loader.dataset.concept_split_specs[u].item()].append(mean_similarity)
    similarity_by_split = {k: torch.Tensor(v).mean() for k, v in similarity_by_split.items()}

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    split = similarity_by_split.keys()
    width = similarity_by_split.values()
    ax.bar(split, width)
    for s, w in similarity_by_split.items():
        ax.text(s - .35, w + .002, "{:.03f}".format(w))
    plt.title("Average similarity with specific boundary formed by all image within the same class\n"
              "Sorted by split, <=0 for train concept images, >0 for test concept images")
    summary_writer.add_figure("visualize/box2cone_view", fig)
