from itertools import chain

import torch
from tqdm import tqdm

from dataset.clevr.clevr_dataset import ClevrDataset, ClevrBuilderDataset
from dataset.meta_dataset import MetaDataset, MetaBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, join, load


class ClevrFewshotDataset(ClevrDataset, MetaDataset):
    @property
    def question_concepts(self):
        return torch.tensor([q['concept_index'] for q in self.questions])

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.concept2splits[self.question_concepts]
        self.indices_split = self.select_split(self.split_specs)

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} "
            f"should already "
            f"exist.")


class ClevrDetachedDataset(ClevrFewshotDataset):
    pass


class ClevrFewshotBuilderDataset(MetaBuilderDataset, ClevrBuilderDataset):
    N_SAMPLES = 500

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self._build_mac()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def decorate_refexp(self, text, concept):
        text = f"{text} is a {self.named_entries_[concept]}"
        if concept not in self.concept_maps_["shape"]:
            text = f"{text} object"
        return f"{text}."

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
            concepts = self.obj2concepts[r["image_index"]][object_id]
            used_concepts = [self.entry2idx_[c] for c in chain(*(p["value_inputs"] for p in r["program"]))]
            for c in set(concepts).difference(used_concepts):
                refexps.append({**r, 'concept_index': c})
        return refexps

    def _build_fewshot_questions(self):
        fewshot_questions = []
        for q in tqdm(load(join(self.augmented_root, self._fewshot_file))["questions"]):
            used_concepts = [self.entry2idx_[c] for c in
                chain.from_iterable(p["value_inputs"] for p in q["program"] if p['type'] != 'relate')]
            answer = q["answer"]
            if isinstance(answer, str):
                used_concepts.extend([i for i in self.itself2samekinds[self.entry2idx_[answer]] if
                    self.concept_split_specs[i] == 0])
            used_concepts = list(set(used_concepts))
            if self.concept_split_specs[used_concepts].sum() - self.concept_split_specs[
                used_concepts].max() == 0:
                for c in used_concepts:
                    fewshot_questions.append({**q, 'concept_index': c})
        return fewshot_questions

    @file_cached('mac')
    def _build_mac(self):
        super()._build_mac()

    @property
    def valid_shots(self):
        return ['fewshot', 'detached']
