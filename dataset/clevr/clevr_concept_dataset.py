import bisect

import torch

from dataset.clevr.clevr_dataset import ClevrDataset, ClevrBuilderDataset
from dataset.pretrain_dataset import PretrainDataset, PretrainBuilderDataset
from dataset.utils import sample_with_ratio
from utils import load, file_cached, join


class ClevrConceptDataset(ClevrDataset, PretrainDataset):
    _curriculum_strategy = [(8, 3, 3), (24, 3, 4), (48, 3, 5), (60, 4, 6), (72, 4, 8), (84, 5, 11),  (1000, 1000, 1000)]

    @property
    def _curriculum_epoch(self):
        return [e * 500 for e, s, l in self._curriculum_strategy]

    def callback(self, iteration):
        self.iteration = iteration
        if self.split != "train":
            return
        i = bisect.bisect_left(self._curriculum_epoch, iteration)
        e, s, l = self._curriculum_strategy[i]
        self.split_specs = self.image_split_specs[self.question_images] + 100 * (self.question_lengths > l)
        self.indices_split = self.select_split(self.split_specs)

    @property
    def question_images(self):
        return torch.tensor([q['image_index'] for q in self.questions])

    @property
    def question_lengths(self):
        return torch.tensor([q['length'] for q in self.questions])

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.image_split_specs[self.question_images]
        self.indices_split = self.select_split(self.split_specs)

    def _build_questions(self):
        raise NotImplementedError


class ClevrConceptSupportDataset(ClevrConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class ClevrConceptFullDataset(ClevrConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class ClevrPretrainBuilderDataset(PretrainBuilderDataset, ClevrBuilderDataset):
    N_SAMPLES = 20

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def _build_questions_from_file(self, data):
        questions = []
        for question in data:
            question_encoded, question_length = self.encode_text(question['question'])
            question_program = question['program']
            question_target, _ = self.encode_program(question_program)
            questions.append({"question": question['question'], "answer": question['answer'],
                "image_index": question['image_index'], "question_target": question_target,
                "question_encoded": question_encoded, 'question_length': question_length})
        return questions


class ClevrConceptSupportBuilderDataset(ClevrPretrainBuilderDataset):
    @file_cached("questions")
    def _build_questions(self):
        return self._build_questions_from_file(
            load(join(self.augmented_root, self._pretrain_file))['questions'])


class ClevrConceptFullBuilderDataset(ClevrPretrainBuilderDataset):
    @file_cached("questions")
    def _build_questions(self):
        return self._build_questions_from_file(load(join(self.augmented_root, self._fewshot_file))['questions'])
