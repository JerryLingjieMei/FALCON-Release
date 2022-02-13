import torch
import bisect

# noinspection PyUnresolvedReferences,PyTypeChecker
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, cfg, optimizer):
        super().__init__(optimizer, mode='max', factor=cfg.SOLVER.GAMMA, patience=cfg.SOLVER.PATIENCE)
        self.warmup_factor = cfg.SOLVER.WARMUP_FACTOR
        self.warmup_iter = cfg.SOLVER.WARMUP_ITER
        self.last_iter = 0
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        for param_group, lr in zip(optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def step_iteration(self):
        self.last_iter += 1
        if self.last_iter <= self.warmup_iter:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

    def get_lr(self):
        alpha = float(self.last_iter) / self.warmup_iter
        warmup_factor = self.warmup_factor ** (1 - alpha)
        return [warmup_factor * lr for lr in self.base_lrs]

    def step_on_metrics(self, metrics):
        self.step(metrics)


# noinspection PyUnresolvedReferences,PyTypeChecker
class MultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    removed_keys = ["milestones", "gamma"]

    def __init__(self, cfg, optimizer):
        self.warmup_factor = cfg.SOLVER.WARMUP_FACTOR
        self.warmup_iter = cfg.SOLVER.WARMUP_ITER
        super().__init__(optimizer, cfg.SOLVER.STEP, cfg.SOLVER.GAMMA)

    def step_iteration(self):
        self.step()

    def step_on_metrics(self, metrics):
        pass

    def state_dict(self):
        return {key: value for key, value in super().state_dict().items() if key not in self.removed_keys}

    def get_lr(self):
        if self.last_epoch <= self.warmup_iter:
            alpha = float(self.last_epoch) / self.warmup_iter
            warmup_factor = self.warmup_factor ** (1 - alpha)
        else:
            warmup_factor = 1
        milestones = list(sorted(self.milestones.elements()))
        return [warmup_factor * base_lr * self.gamma ** bisect.bisect_right(milestones, self.last_epoch) for base_lr in
            self.base_lrs]
