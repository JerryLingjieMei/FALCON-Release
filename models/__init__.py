from utils import is_debug, nan_hook
from .fewshot.model import FewshotModel
from .parser.model import ParserModel
from .pretrain.model import PretrainModel

_META_ARCHITECTURES = {"pretrain": PretrainModel, "fewshot": FewshotModel, 'parser': ParserModel}


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    model = meta_arch(cfg.MODEL)
    if is_debug():
        for module in model.modules():
            if "torch" not in module.__module__:
                module.register_forward_hook(nan_hook)
    return model
