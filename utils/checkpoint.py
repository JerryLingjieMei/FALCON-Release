import logging
import os
import re

import torch
import torch.nn.functional as F

from .io import mkdir, join, dump
from .misc import get_trial, is_debug

LOAD_MODEL_DECORATOR = [("^(?<!module\.)", "module.", lambda self: hasattr(self.model, "module")),
    ("^module\.", "", lambda self: not hasattr(self.model, "module")),
    ("program_executor.network", "program_executor.learner.network", lambda self: True),
    ("program_executor.entailment", "program_executor.learner.entailment", lambda self: True),
    ("box_registry.planes.weight", "box_registry.cones.weight", lambda self: 'cone' in self.output_dir)]

SAVE_MODEL_DECORATOR = [("^module\.", "", lambda self: True)]


class Checkpointer:
    def __init__(self, cfg, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = cfg.OUTPUT_DIR
        self.weight_file = cfg.WEIGHT.FILE.format(cfg.CATALOG.SPLIT_SEED)
        self.weight_regex = cfg.WEIGHT.REGEX
        self.max_iter = cfg.SOLVER.MAX_ITER

    def save(self, iteration, to_tag=True):
        logger = logging.getLogger("falcon_logger")
        if iteration == 0:
            return
        save_file = self._checkpoint_file(iteration)
        data = {"models": self._decorate_save_names(self.model.state_dict())}
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        logger.warning(f"Saving checkpoint to {save_file}")
        dump(data, save_file)
        if to_tag:
            self._tag_last_checkpoint(iteration)
        if iteration % 10000 == 0:
            save_file = self._old_checkpoint_file(iteration)
            data = {"models": self._decorate_save_names(self.model.state_dict())}
            if self.optimizer is not None:
                data["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()
            logger.info(f"Saving old checkpoint to {save_file}")
            dump(data, save_file)

    def load(self, iteration):
        logger = logging.getLogger("falcon_logger")
        if is_debug():
            filename, iteration = None, 0
        elif iteration is None:
            filename, iteration = self._last_checkpoint()
        elif iteration > 0:
            filename = self._checkpoint_file(iteration)
            assert os.path.exists(filename);
            f"Checkpoint {filename} must exist."
        else:
            filename, iteration = None, 0

        if iteration == 0:
            logger.info("No checkpoint found. Initializing models from scratch")
            self.load_pretrained()
        else:
            logger.info(f"Loading checkpoint from {filename}")
            checkpoint = self._load_file(filename)
            model_state_dict = self._decorate_params(self._decorate_load_names(checkpoint.pop("models")))
            self.model.load_state_dict(model_state_dict)
            if "optimizer" in checkpoint and self.optimizer:
                logger.info(f"Loading optimizer from {filename}")
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                logger.info(f"Loading scheduler from {filename}")
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        return iteration

    def load_pretrained(self):
        logger = logging.getLogger("falcon_logger")
        if self.weight_file == "":
            return
        assert os.path.exists(self.weight_file), f"Pretrained weights {self.weight_file} must exist."
        logger.info(f"Loading pretrained weights from {self.weight_file}")
        loaded = self._load_file(self.weight_file)
        filtered_loaded = self._decorate_params(self._decorate_load_names(
            {k: param for k, param in loaded["models"].items() if re.search(self.weight_regex, k) is not None}))
        state_dict = self.model.state_dict()
        state_dict.update(filtered_loaded)
        try:
            self.model.load_state_dict(state_dict)
        except ValueError as e:
            logger.info(e)
            if not is_debug():
                if not input("Type 2 if allowed\n"):
                    raise e
        except RuntimeError as e:
            logger.info(e)
            if not is_debug():
                if not input("Type 2 if allowed\n"):
                    raise e

    def _decorate_load_names(self, state_dict):
        for pattern, replacement, condition in LOAD_MODEL_DECORATOR:
            if condition(self):
                state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
        return state_dict

    def _decorate_save_names(self, state_dict):
        for pattern, replacement, condition in SAVE_MODEL_DECORATOR:
            if condition(self):
                state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
        return state_dict

    def _checkpoint_file(self, iteration):
        mkdir(join(self.output_dir, "checkpoints"))
        return join(self.output_dir, "checkpoints", f"model_{iteration:07d}.pth")

    def _old_checkpoint_file(self, iteration):
        trial = get_trial()
        mkdir(join(self.output_dir, "old_checkpoints", trial))
        return join(self.output_dir, "old_checkpoints", trial, f"model_{iteration:07d}.pth")

    def _last_checkpoint(self):
        save_file = os.path.join(self.output_dir, "last_checkpoint")
        if not os.path.exists(save_file):
            return None, 0
        with open(save_file, "r") as f:
            last_saved = f.read().strip()
        return last_saved, int(last_saved[-11:-4])

    def _tag_last_checkpoint(self, iteration):
        if iteration <= self.max_iter:
            with open(os.path.join(self.output_dir, "last_checkpoint"), "w") as f:
                f.write(self._checkpoint_file(iteration))

    def _load_file(self, f):
        loaded = torch.load(f, map_location="cpu")
        if "models" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def _decorate_params(self, state_dict):
        component_state_dict = self.model.state_dict()
        keys = set(component_state_dict.keys()).intersection(state_dict.keys())
        for k in keys:
            if torch.is_tensor(component_state_dict[k]):
                model_shape = component_state_dict[k].shape
                state_shape = state_dict[k].shape
                diff = [i for i in range(len(model_shape)) if model_shape[i] != state_shape[i]]
                if len(diff) == 1:
                    i = diff[0]
                    if model_shape[i] < state_shape[i]:
                        state_dict[k] = state_dict[k].narrow(i, 0, model_shape[i])
                    else:
                        pad = [0] * len(state_shape) * 2
                        pad[-2 * i - 1] = model_shape[i] - state_shape[i]
                        state_dict[k] = F.pad(state_dict[k], pad)
                    logger = logging.getLogger("falcon_logger")
                    logger.info(f"Key {k} in checkpoint fixed.")
        return state_dict

    @property
    def iteration(self):
        return self._last_checkpoint()[-1]
