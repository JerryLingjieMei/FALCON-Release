import random
from collections import Counter
from itertools import chain
from multiprocessing import Pool

import torch
from tqdm import tqdm
# noinspection PyUnresolvedReferences
from tqdm.contrib.itertools import product as tqdm_product

from dataset.gqa.gqa_dataset import GqaDataset, GqaBuilderDataset
from dataset.meta_dataset import MetaDataset, MetaBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, join, nonzero


class GqaMetaDataset(GqaDataset, MetaDataset):

    @property
    def question_concepts(self):
        return torch.tensor([q['concept_index'] for q in self.questions])

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.concept2splits[self.question_concepts]
        self.indices_split = self.select_split(self.split_specs)


class GqaFewshotDataset(GqaMetaDataset):
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


class GqaDetachedDataset(GqaFewshotDataset):
    pass


class GqaFewshotBuilderDataset(MetaBuilderDataset, GqaBuilderDataset):
    MAC_ROOT = "/data/vision/billf/scratch/jerrymei/BoxEmbedding/vendor/mac-network-gqa"
    N_PER_TEMPLATE = 1
    N_SAMPLES = 100

    @property
    def concept_sets(self):
        return set(nonzero(self.concept_split_specs >= 0))

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.concept_frequencies = self._build_concept_frequencies()
        self.questions = self._build_questions()
        self._build_mac()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for this_object in self.obj2concepts:
            concept_frequencies[list(set(chain.from_iterable(this_object)))] += 1
        return concept_frequencies

    def decorate_refexp(self, text, concept):
        return text

    @file_cached('questions')
    def _build_questions(self):
        fewshot_questions = self._build_fewshot_questions()
        refexps = self._build_refexps()
        questions = self._build_composite_questions(refexps, fewshot_questions)
        return questions

    def _build_statement(self, image_index):
        candidates = []
        objects = self.obj2concepts[image_index]
        valid_objects = [[oo for oo in o if self.concept_split_specs[oo] <= 0] for o in objects]
        for o in valid_objects:
            candidates.extend((x,) for x in o)
            candidates.extend((x, y) for x in o for y in o if x != y)
        candidates = [k for k, v in Counter(candidates).items() if v == 1]
        refexps = []
        for f in candidates:
            o = [o for o in objects if all(ff in o for ff in f)][0]
            for itself in set(o).difference(f).intersection(self.concept_sets):
                program = self.filter_statement_program(itself, f)
                refexp = self.filter_statement(itself, f)
                refexps.append(
                    {'refexp': refexp, 'program': program, 'image_index': image_index, 'concept_index': itself,
                        "concept_contained": [*f, itself]})
        return refexps

    def _build_refexps(self):
        with Pool(16) as p:
            results = list(tqdm(p.imap(self._build_statement, list(range(len(self.obj2concepts))), 1000),
                total=len(self.obj2concepts)))
        refexps = list(chain.from_iterable(results))
        return refexps

    def _build_filter_question(self, image_index):
        questions = []
        objects = self.obj2concepts[image_index]
        valid_objects = [[oo for oo in o if self.concept_split_specs[oo] <= 0] for o in objects]
        concepts = list(set(chain.from_iterable(objects)).intersection(self.concept_sets))
        if len(concepts) == 0:
            return questions
        itself = random.choices(concepts, 1 / self.concept_frequencies[concepts], k=1)[0]
        tf, ff = [], []
        for vo, o in zip(valid_objects, objects):
            group = tf if itself in o else ff
            group.extend((x,) for x in o if x != itself)
            group.extend((x, y) for x in o for y in o if len({x, y, itself}) == 3)
        tf, ff = set(k for k, v in Counter(tf).items() if v == 1).difference(ff), set(
            k for k, v in Counter(ff).items() if v == 1).difference(tf)
        if len(tf) == 0 or len(ff) == 0: return []
        for candidates, answer in zip([tf, ff], [True, False]):
            f = random.choice(list(candidates))
            question = self.filter_question(itself, f)
            program = self.filter_question_program(itself, f)
            c = random.choice([f, itself])
            questions.append(
                {"question": question, "program": program, "answer": answer, "image_index": image_index,
                    "concept_index": c, "concept_contained": [*f, itself]})
        return questions

    def _build_exist_questions(self, image_index):
        questions = []
        concepts = list(set(chain.from_iterable(self.obj2concepts[image_index])))
        if len(concepts) == 0: return []
        true = random.choices(concepts, 1 / self.concept_frequencies[concepts], k=1)[0]
        false = random.choice(list(self.concept_sets.difference(concepts)))
        for concept, a in zip([true, false], [True, False]):
            question = self.exist_question(concept)
            program = self.exist_question_program(concept)
            questions.append({"question": question, 'program': program, "answer": a, "image_index": image_index,
                'concept_index': concept, "concept_contained": [concept]})
        return questions

    def _build_fewshot_questions(self):
        with Pool(16) as p:
            filtered_results = list(
                tqdm(p.imap(self._build_filter_question, list(range(len(self.obj2concepts))), 1000),
                    total=len(self.obj2concepts)))
        questions = list(chain.from_iterable(filtered_results))
        return questions

    @file_cached('mac')
    def _build_mac(self):
        super()._build_mac()
