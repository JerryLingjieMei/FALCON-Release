import os
import random
from itertools import chain, cycle

import math
import torch
from tqdm import tqdm

from dataset.cub.cub_dataset import CubDataset, CubBuilderDataset
from dataset.meta_dataset import MetaDataset, MetaBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, join, nonzero, mkdir, dump


class CubFewshotDataset(CubDataset, MetaDataset):

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

    def get_batch_sampler(self, batch_size):
        if self.split == "train":
            return None
        elif self.split == "val":
            return OnePerConceptBatchSampler(self, batch_size)
        elif self.split == "test":
            return ManyPerConceptBatchSampler(self, batch_size)

    def save_meta_handler(self, output_dir, evaluated, iteration, metrics):
        super().save_meta_handler(output_dir, evaluated, iteration, metrics)
        if self.split == "test" and not self.use_text:
            head, tail = os.path.split(output_dir)
            parts = tail.split("_")
            output_dir = mkdir(join(head, '_'.join(parts[:-1] + ['shallow'] + parts[-1:])))
            filename_prefix = evaluated["mode"]
            metrics = {**metrics, 'principal': metrics['accuracy_006']}
            dump(metrics, join(output_dir, f"{filename_prefix}_{iteration:07d}.json"))


class CubDetachedDataset(CubFewshotDataset):
    pass


class CubZeroshotDataset(CubFewshotDataset):
    pass


class OnePerConceptBatchSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        unique_indices = [0] + dataset.question_concepts[dataset.indices_split].unique_consecutive(
            return_counts=True)[-1].cumsum(-1)[:-1].tolist()
        split_specs = dataset.concept_split_specs[
            dataset.question_concepts[list(dataset.indices_split[i] for i in unique_indices)]]
        self.segments = [[unique_indices[n] for n in nonzero(split_specs == s)] for s in
            split_specs.unique().sort().values.tolist()]

    def __iter__(self):
        for segment in self.segments:
            for i in range(0, len(segment), self.batch_size):
                yield segment[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(segment) / self.batch_size) for segment in self.segments)


class ManyPerConceptBatchSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        unique_indices = torch.arange(len(dataset)).tolist()
        split_specs = dataset.concept_split_specs[
            dataset.question_concepts[list(dataset.indices_split[i] for i in unique_indices)]]
        self.segments = [[unique_indices[n] for n in nonzero(split_specs == s)] for s in
            split_specs.unique().sort().values.tolist()]

    def __iter__(self):
        for segment in self.segments:
            for i in range(0, len(segment), self.batch_size):
                yield segment[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(segment) / self.batch_size) for segment in self.segments)


class CubFewshotBuilderDataset(CubBuilderDataset, MetaBuilderDataset):
    N_SAMPLES = 5

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self._build_mac()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    @file_cached('questions')
    def _build_questions(self):
        questions = []
        for itself in tqdm(self.concepts[:-1]):
            hyper = min(self.hypo2hyper[itself])
            samekinds = [s for s in self.itself2samekinds[itself] if
                self.concept_split_specs[s] < max(self.concept_split_specs[itself], 1)]
            valid_classes = set(nonzero(self.concept2splits[:len(self.classes)] == self.concept2splits[itself]))
            true_classes = list(sorted(valid_classes.intersection([itself, *self.hyper2hypo[itself]])))
            true_candidates = list(chain.from_iterable(self.class2images[c] for c in true_classes))
            false_classes = list(sorted(valid_classes.difference([itself, *self.hyper2hypo[itself]])))
            false_candidates = list(chain.from_iterable(self.class2images[c] for c in false_classes))
            for i in range(self.N_SAMPLES):
                this_samekinds = self.dropout(samekinds, self.concept_split_specs[itself] <= 0)
                supports = [hyper, *this_samekinds]
                relations = [0] + [2] * len(this_samekinds)
                encoded_metaconcept = self.encode(self.metaconcept_text(supports, relations, itself),
                    self.metaconcept_program(supports, relations), 'metaconcept')
                encoded_statement = self.encode(self.exist_statement(itself),
                    self.exist_statement_program(itself), 'statement')
                encoded_question = self.encode(self.exist_question(itself), self.exist_question_program(itself),
                    'question')
                true_image_index = random.choices(true_candidates, k=self.query_k // 2 + self.shot_k)
                train_image_index, true_image_index = true_image_index[:self.shot_k], true_image_index[
                self.shot_k:]
                false_image_index = random.choices(false_candidates, k=self.query_k // 2)
                val_image_index = true_image_index + false_image_index
                answers = [True] * len(true_image_index) + [False] * len(false_image_index)
                for j, (ti, vi, answer) in enumerate(zip(cycle(train_image_index), val_image_index, answers)):
                    questions.append(
                        {**encoded_statement, **encoded_metaconcept, **encoded_question, 'answer': answer,
                            'concept_index': itself, 'train_image_index': ti, 'image_index': vi,
                            'family': (itself, i, j)})
        return questions

    @file_cached('mac')
    def _build_mac(self):
        super()._build_mac()

    def mac_split(self, concept_index):
        if self.concept_split_specs[concept_index] <= 0:
            return 'train'
        elif self.concept_split_specs[concept_index] == 1:
            return 'val'
        elif self.concept_split_specs[concept_index] == 6:
            return 'test'
        else:
            return None
