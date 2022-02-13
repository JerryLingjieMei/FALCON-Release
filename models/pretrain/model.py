from collections import defaultdict

from torch import nn

from models.nn import build_box_registry, BatchedFeatureExtractor
from models.pretrain.nn import PretrainLoss
from models.pretrain.program_executor import PretrainProgramExecutor
from models.pretrain.reg import PretrainReg
from utils import check_entries, collate_fn, compose_image


class PretrainModel(nn.Module):
    _NO_COLLATE_KEYS = ["end", "queried_embedding", "query_object"]

    def __init__(self, cfg):
        super().__init__()
        self.box_registry = build_box_registry(cfg)
        self.feature_extractor = BatchedFeatureExtractor(cfg)

        self.program_executor = PretrainProgramExecutor(cfg)
        self.loss = PretrainLoss(cfg)

    @property
    def concept_entries(self):
        return len(self.box_registry)

    @classmethod
    def build_loss(cls, cfg):
        meta_arch = cls._LOSS_METHODS[cfg.LOSS.NAME]
        return meta_arch(cfg)

    def forward(self, inputs):
        check_entries(self.concept_entries, inputs["info"]["concept_entries"][0])
        features, relations = self.feature_extractor(self.get_stacked_images(inputs))

        program = inputs["program"]
        outputs = defaultdict(list)
        for p, feature, relation in zip(program, features, relations):
            q = p.evaluate(self.box_registry, features=feature, relations=relation)
            output = self.program_executor(q)
            for k, v in output.items():
                outputs[k].append(v)
            outputs["program"].append(q)
        outputs = collate_fn(outputs, ("end",))
        self.program_executor.clean_output(outputs, inputs)

        losses = {}
        if self.training:
            losses.update(self.loss(outputs, inputs))
        return {**outputs, **losses}

    def get_stacked_images(self, inputs):
        if "pretrained" in inputs:
            return inputs["pretrained"]
        elif "mask" in inputs:
            return list(compose_image(image, mask) for image, mask in zip(inputs["image"], inputs["mask"]))
        else:
            return list(img.unsqueeze(0) for img in inputs["image"])

    @property
    def rep(self):
        return self.feature_extractor.rep

    def callback(self, inputs, outputs):
        if self.training:
            self.box_registry.clamp_dimensions()
