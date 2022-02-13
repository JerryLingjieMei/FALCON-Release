from collections import defaultdict

import torch
from torch import nn

from models.fewshot.nn import FewshotLoss
from models.fewshot.program_executor import FewshotProgramExecutor
from models.nn import build_box_registry, CachedFeatureExtractor
from utils import check_entries, map_wrap
from utils import collate_fn, freeze, compose_image


class FewshotModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.box_registry = build_box_registry(cfg)
        freeze(self.box_registry)

        self.feature_extractor = CachedFeatureExtractor(cfg)
        self.program_executor = FewshotProgramExecutor(cfg)
        self.loss = FewshotLoss(cfg)

    @property
    def concept_entries(self):
        return len(self.box_registry)

    @map_wrap
    @map_wrap
    def _forward_pretrained(self, index, pf):
        return self.feature_extractor.get(index, pf)

    @map_wrap
    @map_wrap
    def _forward_masked(self, index, image, mask):
        return self.feature_extractor.get(index, compose_image(image, mask))

    @map_wrap
    @map_wrap
    def _forward_unmasked(self, index, image):
        return self.feature_extractor.get(index, image.unsqueeze(0))

    def forward_feature_extractor(self, samples):
        if "pretrained" in samples:
            features, relations = self._forward_pretrained(*map(samples.get, ["image_index", "pretrained"]))
        elif "mask" in samples:
            features, relations = self._forward_masked(*map(samples.get, ["image_index", "image", "mask"]))
        else:
            features, relations = self._forward_unmasked(*map(samples.get, ["image_index", "image"]))
        return {"features": features, "relations": relations}

    def make_kwargs(self, inputs, train_samples, val_samples, i):
        kwargs = defaultdict(dict, device=self.box_registry.device)
        for group, samples in zip(["train_sample", "val_sample"], [train_samples, val_samples]):
            for k, v in inputs[group].items():
                kwargs[group][k] = v[i]
            kwargs[group]["features"] = samples["features"][i]
            kwargs[group]["relations"] = samples["relations"][i]
        for k, v in inputs['task'].items():
            kwargs['task'][k] = v[i]
        return kwargs

    def forward(self, inputs):
        check_entries(self.concept_entries, inputs["info"]["concept_entries"][0])
        train_samples = self.forward_feature_extractor(inputs["train_sample"])
        val_samples = self.forward_feature_extractor(inputs["val_sample"])

        outputs = defaultdict(list)
        for i, p in enumerate(inputs["program"]):
            kwargs = self.make_kwargs(inputs, train_samples, val_samples, i)
            q = p.evaluate(self.box_registry, **kwargs)
            o = self.program_executor(q)
            for k, v in o.items():
                outputs[k].append(v)
            outputs["program"].append(q)
        outputs = collate_fn(outputs)

        losses = {}
        if self.training:
            losses = self.loss(outputs, inputs)

        outputs["train_sample"].update(train_samples)
        outputs["val_sample"].update(val_samples)
        return {**outputs, **losses}

    def callback(self, inputs, outputs):
        if not self.training and inputs["info"]["split"][0] != "train" and not inputs['info']['use_text'][0]:
            # Fill the resulting model with fewshot results
            with torch.no_grad():
                for p, e in zip(inputs["program"], outputs["queried_embedding"]):
                    self.box_registry[p.concept_index] = e
        # Fill in the cache for extracted features
        if not self.feature_extractor.has_cache:
            for group in ["train_sample", "val_sample"]:
                for features, relations, indices in zip(outputs[group]["features"], outputs[group]["relations"],
                        inputs[group]["image_index"]):
                    for feature, relation, index in zip(features, relations, indices):
                        self.feature_extractor.set(index, (feature, relation))

    @property
    def rep(self):
        return self.feature_extractor.rep
