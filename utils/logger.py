import logging
import os
import re
import shutil
import sys
# noinspection PyProtectedMember
import traceback
from datetime import datetime

import torch
# noinspection PyProtectedMember
from torch.utils.tensorboard._utils import figure_to_image
from torchvision.transforms import functional as TF

from .io import join, mkdir, save_image, dump
from .misc import is_debug, get_trial

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    green = "\x1b[32;21m"
    cyan = "\x1b[36;21m"
    bold_red = "\x1b[31;21m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(name)s %(levelname)s: %(message)s"

    FORMATS = {logging.DEBUG: grey + format + reset, logging.INFO: format,
        logging.WARNING: green + format + reset, logging.ERROR: cyan + format + reset,
        logging.CRITICAL: bold_red + format + reset}

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = CustomFormatter()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    if log_file and not is_debug():
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class SummaryWriter(torch.utils.tensorboard.SummaryWriter):
    _minimal_calls = 5
    datetime_format = "%Y%m%d-%H%M%S"

    def __init__(self, log_dir=None, comment='', **kwargs):
        self.trial = get_trial()
        super().__init__(mkdir(join(log_dir, "summary", self.trial)), comment, **kwargs)
        self.calls = 0
        self.flush()
        self.start_up(log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super().add_scalar(tag, scalar_value, global_step, walltime)
        self.flush()

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
        self.flush()

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        super().add_image(tag, img_tensor, global_step, walltime, dataformats='CHW')
        output_folder = mkdir(join(self.log_dir, "images", tag))
        save_image(TF.to_pil_image(img_tensor, "RGB"), join(output_folder, f"image_{global_step:07d}.jpeg"))
        self.flush()

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default',
            metadata_header=None):
        super().add_embedding(mat, metadata, label_img, global_step, tag, metadata_header)
        output_folder = mkdir(join(self.log_dir, "embeddings", tag))
        dump(mat, join(output_folder, f"embedding_{global_step:07d}.pth"))
        dump(metadata, join(output_folder, f"metadata_{global_step:07d}.json"))
        # remove duplicate embeddings
        projection_filename = join(self.log_dir, "projector_config.pbtxt")
        regex = r"embeddings \{[^\{]+\}\n"
        with open(projection_filename, "r+") as f:
            logs = f.read()
            groups = re.findall(regex, logs)
            groups = list(set(groups))
            f.seek(0)
            f.write(''.join(groups))
            f.truncate()
        self.flush()

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        super().add_image(tag, figure_to_image(figure, close), global_step, walltime)
        output_dir = mkdir(join(self.log_dir, "figures"))
        figure.savefig(join(output_dir, f"{tag.replace('/', '-')}_{global_step:07d}.png"))
        self.flush()

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        text_string = text_string.replace("\n", "\n\r")
        super().add_text(tag, text_string, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        logger = logging.getLogger('falcon_logger')
        if len(values) > 0:
            super().add_histogram(tag, values, global_step, bins, walltime, max_bins)
            self.flush(self._minimal_calls)
            # logger.warning(f'Histogram with {tag} drawn.')
        else:
            logger.warning(f"Empty histogram with tag {tag} and step {global_step}.")
            traceback.print_exc()

    def flush(self, n=1):
        self.calls += n
        with open(join(self.log_dir, "calls"), "w") as f:
            f.write(str(self.calls))
        super().flush()

    def start_up(self, log_dir):
        trials = os.listdir(join(log_dir, "summary"))
        trials = [t for t in trials if re.fullmatch("[\d]+-[\d]+", t) is not None]

        trials.remove(self.trial)
        for trial in trials:
            file = join(log_dir, "summary", trial, "calls")
            to_remove = False
            if os.path.exists(file):
                with open(file, "r") as f:
                    try:
                        calls = int(f.read().strip())
                        if calls < self._minimal_calls:
                            to_remove = True
                    except:
                        pass
            elif datetime.strptime(trial, self.datetime_format) > datetime.strptime("20210620-000000",
                    self.datetime_format):
                to_remove = True
            if to_remove:
                logger = logging.getLogger("falcon_logger")
                logger.info(f"Removing summary in {trial}.")
                shutil.rmtree(join(log_dir, "summary", trial), ignore_errors=True)
