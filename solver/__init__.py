from solver.scheduler import ReduceLROnPlateau, MultiStepLR
from .optimizer import Optimizer


def make_scheduler(cfg, optimizer):
    if len(cfg.SOLVER.STEP) == 0:
        return ReduceLROnPlateau(cfg, optimizer)
    else:
        return MultiStepLR(cfg, optimizer)
