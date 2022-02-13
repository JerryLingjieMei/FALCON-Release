import glob
import os
import re
from collections import defaultdict
from itertools import chain

from tqdm import tqdm

from dataset.clevr.clevr_fewshot_dataset import ClevrFewshotDataset, ClevrFewshotBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, load, join, mkdir, dump, symlink_recursive, num2word


class ClevrFewshotDebiasDataset(ClevrFewshotDataset):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.concept2splits[self.question_concepts]
        self.indices_split = self.select_split(self.split_specs)

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class ClevrDetachedDebiasDataset(ClevrFewshotDebiasDataset):
    pass


class ClevrFewshotDebiasBuilderDataset(ClevrFewshotBuilderDataset):
    N_SAMPLES = 500

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self._build_mac()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    @file_cached('questions')
    def _build_questions(self):
        refexps = self._build_refexps()
        fewshot_questions = self._build_fewshot_questions()
        questions = self._build_composite_questions(refexps, fewshot_questions)
        return questions

    def _build_refexps(self):
        refexps = []
        for r in tqdm(load(join(self.augmented_root, self._refexp_file))["refexps"]):
            object_id = r["program"][-1]["_output"][0]
            objects = self.obj2concepts[r["image_index"]]
            used_concepts = list(chain(*(p["value_inputs"] for p in r["program"])))
            for c in set(objects[object_id]).difference(used_concepts):
                has_cs = set.intersection(*(set(o) for o in objects if c in o))
                has_not_cs = set(self.concepts).difference(
                    set.union(*(set(o) for o in objects if c not in o), set()))
                if len(has_cs.intersection(has_not_cs)) > 1:
                    refexps.append({**r, 'concept_index': c})
        return refexps

    @file_cached('mac')
    def _build_mac(self):
        super()._build_mac()

