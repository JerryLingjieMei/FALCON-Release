import argparse
import logging
import socket
import warnings
from datetime import datetime

import resource
import torch
import torch.multiprocessing

FLAGS = {}


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str, )
        self.add_argument("--iteration", type=int)
        self.add_argument("--mode", type=str, default=None, choices=[None, "feature", "predict", "logit"])
        self.add_argument("--debug", action="store_true")
        self.add_argument("--skip-cache", action="store_true")
        self.add_argument("opts", help="Modify config options using the command-line", default=None,
            nargs=argparse.REMAINDER, )

    def parse_args(self, args=None, namespace=None):
        args = super(ArgumentParser, self).parse_args(args, namespace)
        if args.debug:  FLAGS["DEBUG"] = True
        if args.skip_cache: FLAGS['SKIP_CACHE'] = True
        return args


def is_debug():
    return "DEBUG" in FLAGS


def skip_cache():
    return "SKIP_CACHE" in FLAGS


def get_trial():
    if "DATETIME" not in FLAGS:
        FLAGS["DATETIME"] = datetime.now()
    return FLAGS["DATETIME"].strftime("%Y%m%d-%H%M%S")


def start_up():
    logger = logging.getLogger("falcon_logger")
    logger.info(f"Running on machine {socket.gethostname()}.")
    torch.autograd.set_detect_anomaly(True)
    warnings.filterwarnings("ignore",
        "This overload of nonzero is deprecated:\n\tnonzero(Tensor input, *, Tensor out)\nConsider using one "
        "of the following signatures instead:\n\tnonzero(Tensor input, *, bool as_tuple) (Triggered "
        "internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:766.)")
    warnings.filterwarnings("ignore",
        "Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze "
        "and return a vector.", UserWarning)

    _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard_limit))
    torch.multiprocessing.set_sharing_strategy(
        'file_system')  # faulthandler.enable()  # sys.excepthook = ultratb.FormattedTB(mode='Plain',
    # color_scheme='Linux', call_pdb=True)


def check_entries(true_entries, given_entries):
    message = f"{true_entries} entries are in the model, while {given_entries} are in the dataset."
    assert true_entries == given_entries, message


def num2word(i):
    assert i < 10
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return numbers[i]
