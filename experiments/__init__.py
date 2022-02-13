import os

from experiments.defaults import C
from utils import join


def cfg_from_args(args):
    cfg = C.clone()
    cfg.merge_from_file(args.config_file)
    for meta in cfg.TEMPLATE:
        cfg.merge_from_file(join("experiments", "template", f"{meta}.yaml"))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = join("output", os.path.splitext(os.path.basename(args.config_file))[0])
    seed = os.path.splitext(args.config_file)[0].split('_')[-1]
    if seed.isnumeric():
        cfg.CATALOG.SPLIT_SEED = int(seed)
    cfg.freeze()
    return cfg
