import re
import torch


class Optimizer(torch.optim.Adam):
    def __init__(self, cfg, model):
        params = []
        solver_cfg = cfg.SOLVER
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            this_lr = solver_cfg.BASE_LR
            for regex, factor in solver_cfg.LR_SPECS:
                if re.search(regex,key) is not None:
                    this_lr *=factor
            params += [{"params": [value], "lr":this_lr}]
        super().__init__(params)
        for i, param_group in enumerate(self.param_groups):
            if param_group["lr"] == solver_cfg.BASE_LR:
                default_param_group_id = i
                break
        else:
            default_param_group_id = 0
        self.default_param_group_id = default_param_group_id

    @property
    def lr(self):
        return self.param_groups[self.default_param_group_id]["lr"]
