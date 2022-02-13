from experiments import cfg_from_args
from snippets.snippet_utils import cfg2test_loader, cfg2model
from utils import ArgumentParser, SummaryWriter
from utils import nonzero
from visualization import build_visualizer

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    cfg = cfg_from_args(args)

    summary_writer = SummaryWriter(cfg.OUTPUT_DIR)
    model = cfg2model(cfg, args)
    test_loader = cfg2test_loader(cfg, args)
    test_set = test_loader.dataset
    indices = nonzero(test_set.concept2splits == 0)
    embeddings = model.box_registry[indices]
    logits = model.program_executor.learner.entailment(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
    results = {"logit": logits, "embedding": embeddings, "label": indices}

    visualizer = build_visualizer(["concept_embedding", "hypernym_logit"], test_set, summary_writer)
    visualizer.visualize(results, model, args.iteration)
