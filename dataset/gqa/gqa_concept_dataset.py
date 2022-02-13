import random
from itertools import chain
from multiprocessing import Pool

import torch
from tqdm import tqdm

from dataset.gqa.gqa_dataset import GqaDataset, GqaBuilderDataset
from dataset.pretrain_dataset import PretrainDataset, PretrainBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, nonzero, join


class GqaConceptDataset(GqaDataset, PretrainDataset):

    @property
    def question_images(self):
        return torch.tensor([q['image_index'] for q in self.questions])

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.image_split_specs[self.question_images]
        self.indices_split = self.select_split(self.split_specs)

    def _build_questions(self):
        raise NotImplementedError


class GqaConceptSupportDataset(GqaConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class GqaConceptFullDataset(GqaConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class GqaPretrainBuilderDataset(PretrainBuilderDataset, GqaBuilderDataset):
    N_SAMPLES = 5

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.concept_frequencies = self._build_concept_frequencies()
        self.questions = self._build_questions()
        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for this_object in self.obj2concepts:
            concept_frequencies[list(set(chain.from_iterable(this_object)))] += 1
        return concept_frequencies

    @property
    def concept_sets(self):
        raise NotImplementedError

    def _build_exist_questions(self, image_index):
        questions = []
        objects = [[oo for oo in o if self.concept_split_specs[oo] <= 0] for o in
            self.obj2concepts[image_index]] if self.is_support else self.obj2concepts[image_index]
        candidates = list(set(chain.from_iterable(objects)))
        if len(candidates) == 0: return []
        positive_concepts = random.choices(candidates, 1 / self.concept_frequencies[candidates],
            k=self.N_SAMPLES)
        negative_concepts = random.choices(list(set(self.concept_sets).difference(candidates)),
            k=self.N_SAMPLES)
        answers = [True] * len(positive_concepts) + [False] * len(negative_concepts)
        for concept, a in zip(positive_concepts + negative_concepts, answers):
            question = self.exist_question(concept)
            question_encoded, question_length = self.encode_text(question)
            question_program = self.exist_question_program(concept)
            question_target, _ = self.encode_program(question_program)
            questions.append({"question": question, "answer": a, "image_index": image_index,
                "question_target": question_target, "question_encoded": question_encoded,
                'question_length': question_length})
        return questions

    def _build_filter_questions(self, image_index):
        questions, candidates, concepts = [], [], []
        objects = [[oo for oo in o if self.concept_split_specs[oo] <= 0] for o in
            self.obj2concepts[image_index]] if self.is_support else self.obj2concepts[image_index]
        for o in objects:
            for itself in o:
                f = random.sample([c for c in o if c != itself], min(len(o) - 1, 2))
                examples = [o_ for o_ in objects if all(f in o_ for f in f)]
                if len(f) > 0 and len(examples) == 1:
                    candidates.append((f, examples[0]))
                    concepts.append(itself)
        if len(candidates) == 0: return []
        for i in random.choices(list(range(len(candidates))), 1 / self.concept_frequencies[concepts],
                k=self.N_SAMPLES):
            f, example = candidates[i]
            itself = concepts[i]
            negative = random.choice(list(set(self.concept_sets).difference(example)))
            for candidate, a in zip([itself, negative], [True, False]):
                question = self.filter_question(candidate, f)
                question_encoded, question_length = self.encode_text(question)
                question_program = self.filter_question_program(candidate, f)
                question_target, _ = self.encode_program(question_program)
                questions.append({"question": question, "answer": a, "image_index": image_index,
                    "question_target": question_target, "question_encoded": question_encoded,
                    'question_length': question_length})
        return questions


class GqaConceptSupportBuilderDataset(GqaPretrainBuilderDataset):

    @property
    def concept_sets(self):
        return nonzero(self.concept_split_specs <= 0)

    @file_cached("questions")
    def _build_questions(self):
        selected = random.sample(list(range(len(self.obj2concepts))), k=len(self.obj2concepts) // 10)
        with Pool(16) as p:
            exist_results = list(tqdm(p.imap(self._build_exist_questions, selected, 1000), total=len(selected)))
        with Pool(16) as p:
            filtered_results = list(tqdm(p.imap(self._build_filter_questions, selected, 1000), total=len(selected)))
        questions = list(chain.from_iterable(exist_results)) + list(chain.from_iterable(filtered_results))
        return questions


class GqaConceptFullBuilderDataset(GqaPretrainBuilderDataset):
    @property
    def concept_sets(self):
        return self.concepts

    @file_cached("questions")
    def _build_questions(self):
        selected = random.sample(list(range(len(self.obj2concepts))), k=len(self.obj2concepts) // 10)
        with Pool(16) as p:
            exist_results = list(
                tqdm(p.imap(self._build_exist_questions, selected, 1000), total=len(self.obj2concepts)))
        with Pool(16) as p:
            filtered_results = list(
                tqdm(p.imap(self._build_filter_questions, selected, 1000), total=len(self.obj2concepts)))
        questions = list(chain.from_iterable(exist_results)) + list(chain.from_iterable(filtered_results))
        return questions
