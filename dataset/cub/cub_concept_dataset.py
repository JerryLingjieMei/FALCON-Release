import copy
import random

import torch
from tqdm import tqdm

from dataset.cub.cub_dataset import CubDataset, CubBuilderDataset
from dataset.pretrain_dataset import PretrainBuilderDataset, PretrainDataset
from dataset.utils import FixedCropTransform, RandomCropTransform, sample_with_ratio
from models.programs import build_program
from utils import file_cached, join, nonzero


class CubConceptDataset(CubDataset, PretrainDataset):

    @property
    def transform_fn(self):
        return RandomCropTransform if self.split == "train" else FixedCropTransform

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


class CubConceptSupportDataset(CubConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class CubConceptFullDataset(CubConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class CubPretrainBuilderDataset(PretrainBuilderDataset, CubBuilderDataset):
    N_SAMPLES = 100

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.concept_frequencies = self._build_concept_frequencies()
        self.questions = self._build_questions()
        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def _build_questions(self):
        raise NotImplementedError

    @property
    def concept_sets(self):
        raise NotImplementedError

    def _build_image_questions(self, objects, image_index):
        questions = []
        candidates = objects[0]
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


class CubConceptSupportBuilderDataset(CubPretrainBuilderDataset):

    @property
    def concept_sets(self):
        return nonzero(self.concept_split_specs <= 0)

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for hypo in nonzero((self.concept_split_specs <= 0) & (self.concept2kinds == 0)):
            concept_frequencies[self.hypo2hyper[hypo] + [hypo]] += 1
        return concept_frequencies

    @file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2classes)):
            if self.concept_split_specs[cls] > 0: continue
            questions.extend(self._build_image_questions(self.obj2concepts[image_index], image_index))
        return questions


class CubConceptFullBuilderDataset(CubPretrainBuilderDataset):

    @property
    def concept_sets(self):
        return self.concepts

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for hypo in nonzero(self.concept2kinds == 0):
            concept_frequencies[self.hypo2hyper[hypo] + [hypo]] += 1
        return concept_frequencies

    @file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2classes)):
            questions.extend(self._build_image_questions(self.obj2concepts[image_index], image_index))
        return questions
