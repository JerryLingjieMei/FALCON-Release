from yacs.config import CfgNode as CN

from dataset import build_dataset
from dataset.utils.weighted_dataset import WeightedConcatDataset
from utils import join


class DatasetCatalog:
    DATASET_ROOT = "/data/vision/billf/scratch/jerrymei/datasets/"
    CLEVR_ROOT = join(DATASET_ROOT, "CLEVR_v1.0")
    CUB_ROOT = join(DATASET_ROOT, "CUB-200-2011")
    GQA_ROOT = join(DATASET_ROOT, "GQA")

    NAME2ROOT = {"clevr": CLEVR_ROOT, "cub": CUB_ROOT, 'gqa': GQA_ROOT}
    for root in NAME2ROOT.values():
        assert not root.endswith("/"), "No trailing slashes. "

    def __init__(self, cfg):
        catalog_cfg = cfg.CATALOG
        self.cfg = catalog_cfg

    def get(self, name, args, as_tuple=False):
        if as_tuple:
            names = name.split("&")
            return tuple((n, self.get(n, args)) for n in names)
        if "&" in name:
            names = name.split("&")
            return self.get(names[0], args)
        if "+" in name:
            datasets_with_ratio = name.split("+")
            datasets, ratio = [], []
            for _ in datasets_with_ratio:
                dataset, ration = _.split("*")
                datasets.append(self.get(dataset, args))
                ratio.append(float(ration))
            return WeightedConcatDataset(datasets, ratio)
        parts = name.split('_')
        root = self.NAME2ROOT[parts[0]]
        split_index, split = None, None
        for split in ["train", "val", "test", "all"]:
            if split in parts:
                split_index = parts.index(split)
                break
        dataset_name = '_'.join(parts[:split_index])
        return build_dataset(
            CN(dict(NAME=dataset_name, SPLIT=split, ROOT=root, OPTS=parts[split_index + 1:], **self.cfg)), args)
