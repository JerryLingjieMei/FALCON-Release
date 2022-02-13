import logging
import os
import random
import re
from collections import defaultdict
from statistics import mean

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.utils import draw_bounding_boxes

from dataset.utils import WordVocab, ProgramVocab
from utils import join, mkdir, underscores, camel_case, num2word
from utils import to_cpu_detach, nonzero


class Dataset(torch.utils.data.Dataset):
    DATASET_REGISTRY = {}
    image_size = (224, 224)
    split2spec = {"train": 0, "val": 1, "test": 2}
    period_map = {"train": 1, "val": 1, "test": 1}
    concept_maps_ = {}
    max_question_length = 50

    @property
    def is_builder(self):
        return 'Builder' in self.__class__.__name__

    @property
    def is_support(self):
        return 'Support' in self.__class__.__name__

    @property
    def is_debias(self):
        return 'Debias' in self.__class__.__name__

    @staticmethod
    def get_name(name):
        return underscores(name[:-len("Dataset")])

    def get_augmented_name(self, name):
        name = underscores(name[:-len("Dataset")])
        if self.shot_k > 1: name = name.replace('few', num2word(self.shot_k))
        return name

    @property
    def base_dataset(self):
        return underscores(self.__class__.__name__).split('_')[0]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = Dataset.get_name(cls.__name__)
        cls.DATASET_REGISTRY[name] = cls
        cls.name = name

    def __str__(self):
        return f"{self.name}_{self.split}"

    @property
    def kinds(self):
        return torch.arange(len(self.kinds_))

    @property
    def tag(self):
        if self.split in ["train", "val"]:
            return self.split
        else:
            return f"{self}"

    @property
    def sampler(self):
        if self.split in ["train", "val"]:
            sampler = RandomSampler(self)
        else:
            sampler = SequentialSampler(self)
        return sampler

    def get_batch_sampler(self, batch_size):
        return None

    def select_split(self, split_spec):
        if self.split == "all":
            return list(range(len(split_spec)))
        else:
            return nonzero(split_spec == self.split2spec[self.split])

    @property
    def principal_metric(self):
        return "accuracy"

    def __init__(self, cfg, args):
        self.root = cfg.ROOT
        self.has_mask = cfg.HAS_MASK
        self.shot_k = cfg.SHOT_K
        self.query_k = cfg.QUERY_K
        self.dropout_rate = cfg.DROPOUT_RATE
        self.use_text = cfg.USE_TEXT
        self.opts = cfg.OPTS

        self.args = args
        self.split = cfg.SPLIT
        self.split_ratio = cfg.SPLIT_RATIO
        self.indices_split = []
        self.split_specs = torch.LongTensor([])
        self.kinds_ = []
        self.concepts_ = []
        self.concepts = []
        self.concept2kinds = torch.LongTensor([])
        self.named_entries_ = []
        self.names = []
        self.synonyms = {}
        self.entry2kinds = torch.LongTensor([])
        self.entry2idx_ = {}

        self.image_split_specs = torch.LongTensor([])
        self.concept_split_specs = torch.LongTensor([])
        self.itself2samekinds = {}

        self.transform = self.transform_fn(self.image_size)
        self.word_vocab = None
        self.program_vocab = None
        self.split_seed = cfg.SPLIT_SEED
        self.iteration = 0
        self.mode = "default"

    @property
    def augmented_root(self):
        root_head, root_tail = os.path.split(self.root)
        assert os.access(root_head, os.W_OK)
        return mkdir(join(root_head, '.augmented', root_tail, str(self.split_seed)))

    @property
    def is_alt(self):
        return 'alt' in self.opts

    @property
    def is_shallow(self):
        return 'shallow' in self.opts

    def log_info(self):
        logger = logging.getLogger("falcon_logger")
        logger.warning(f"{self} dataset uses {len(self.indices_split)} out of "
                       f"{len(self.split_specs)} samples.")
        logger.info(f"{self} has {len(self.kinds)} kinds.")
        logger.info(f"{self} has a word vocabulary size of {len(self.word_vocab)}.")
        if self.program_vocab is not None:
            logger.info(f"{self} has a program vocabulary size of {len(self.program_vocab)}.")

    @property
    def info(self):
        return {"concept_entries": len(self.named_entries_), "kind_entries": len(self.kinds_),
            "split": self.split, 'use_text': self.use_text}

    @property
    def empty_image(self):
        return torch.zeros((6 if self.has_mask else 3, *self.image_size))

    @property
    def transform_fn(self):
        raise NotImplementedError

    def get_stacked_scenes(self, image_index):
        raise NotImplementedError

    def get_image(self, image_index):
        raise NotImplementedError

    def get_mask(self, image_index):
        raise NotImplementedError

    def get_annotated_image(self, image_index, mask_index=None):
        masks, image = self.transform(self.get_mask(image_index), self.get_image(image_index))
        if mask_index is not None:
            masks = masks[mask_index:mask_index + 1]
        y_sum = masks.sum(-2).bool()
        y_min_value, y_min = y_sum.max(-1)
        y_max = y_sum.shape[-1] - 1 - y_sum.flip(-1).max(-1).indices
        x_sum = masks.sum(-1).bool()
        x_min_value, x_min = x_sum.max(-1)
        x_max = x_sum.shape[-1] - 1 - x_sum.flip(-1).max(-1).indices
        bboxes = torch.stack([y_min, x_min, y_max, x_max], -1)
        bboxes = bboxes[y_min_value & x_min_value]
        annotated_image = draw_bounding_boxes((image * 255).to(torch.uint8), bboxes, width=1) / 255
        return TF.resize(annotated_image, self.image_size)

    def __len__(self):
        return len(self.indices_split)

    def init_evaluate(self, mode):
        if mode is None: mode = self.mode
        return defaultdict(list, mode=mode)

    def batch_evaluate(self, inputs, outputs, evaluated):
        inputs = to_cpu_detach(inputs)
        outputs = to_cpu_detach(outputs)
        handler = getattr(self, f"batch_{evaluated['mode']}_handler", self.batch_default_handler)
        for k, v in handler(inputs, outputs).items():
            evaluated[k].extend(v)

    def evaluate_metric(self, evaluated):
        handler = getattr(self, f"metric_{evaluated['mode']}_handler", self.metric_default_handler)
        metrics = handler(evaluated)
        if self.principal_metric in metrics: metrics['principal'] = metrics[self.principal_metric]
        return  metrics

    def save(self, output_dir, evaluated, checkpointer, metrics):
        output_dir = mkdir(join(output_dir, "inference", str(self)))
        handler = getattr(self, f"save_{evaluated['mode']}_handler", self.save_default_handler)
        handler(output_dir, evaluated, checkpointer, metrics)

    def batch_default_handler(self, inputs, outputs):
        return {}

    def metric_default_handler(self, evaluated):
        return {}

    def save_default_handler(self, output_dir, evaluated, iteration, metrics):
        pass

    def callback(self, iteration):
        pass

    def _build_word_tokens(self):
        raise NotImplementedError

    def _build_word_vocab(self):
        word_tokens = self._build_word_tokens()
        vocab = WordVocab()
        vocab.words.update(word_tokens)
        vocab.freeze()
        return vocab

    def _build_program_tokens(self):
        raise NotImplementedError

    def _build_program_vocab(self):
        program_tokens = self._build_program_tokens()
        vocab = ProgramVocab()
        vocab.words.update(program_tokens)
        vocab.freeze()
        return vocab

    @staticmethod
    def answer2text(target):
        assert target is not None
        if isinstance(target, bool):
            return "Yes" if target else "No"
        else:
            return str(target)

    def answer2category(self, target):
        if self.use_text:
            return "token"
        elif isinstance(target, str):
            return "choice"
        elif isinstance(target, bool):
            return "boolean"
        else:
            return "count"

    @staticmethod
    def answer2target(answer):
        return torch.tensor(answer).float()

    @staticmethod
    def clevr2nscl(clevr_program):
        nscl_program = list()
        mapping = dict()

        for block_id, block in enumerate(clevr_program):
            op = block['type'] if 'type' in block else block['function']
            current = None
            if op == 'scene':
                current = dict(op='scene')
            elif op.startswith('filter'):
                concept = block['value_inputs'][0]
                last = nscl_program[mapping[block['inputs'][0]]]
                if last['op'] == 'filter':
                    last['concept'].append(concept)
                else:
                    current = dict(op='filter', concept=[concept])
            elif op.startswith('relate'):
                concept = block['value_inputs'][0]
                current = dict(op='relate', relational_concept=[concept])
            elif op.startswith('same'):
                current = dict(op='relate_attribute_equal', attribute=(op.split('_')[1]))
            elif op in ('intersect', 'union'):
                current = dict(op=op)
            elif op == 'unique':
                pass  # We will ignore the unique operations.
            else:
                if op.startswith('query'):
                    if block_id == len(clevr_program) - 1:
                        current = dict(op='query', attribute=(op.split('_')[1]))
                elif op.startswith('equal') and op != 'equal_integer':
                    current = dict(op='query_attribute_equal', attribute=(op.split('_')[1]))
                elif op == 'exist':
                    current = dict(op='exist')
                elif op == 'count':
                    if block_id == len(clevr_program) - 1:
                        current = dict(op='count')
                elif op == 'equal_integer':
                    current = dict(op='count_equal')
                elif op == 'less_than':
                    current = dict(op='count_less')
                elif op == 'greater_than':
                    current = dict(op='count_greater')
                else:
                    raise ValueError('Unknown CLEVR operation: {}.'.format(op))

            if current is None:
                assert len(block['inputs']) == 1
                mapping[block_id] = mapping[block['inputs'][0]]
            else:
                current['inputs'] = list(map(mapping.get, block['inputs']))
                nscl_program.append(current)
                mapping[block_id] = len(nscl_program) - 1

        return nscl_program

    def nscl2program(self, nscl_program):
        programs = []
        for np in nscl_program:
            op = np["op"]
            if op == "filter":
                p = programs[np["inputs"][0]]
                for c in np['concept']:
                    p = ('Filter', p, [self.entry2idx_[c]])
            elif op == "query":
                p = ('Query', ('Unique', programs[np["inputs"][0]]),
                [[self.entry2idx_[_] for _ in self.concept_maps_[np["attribute"]]]])
            else:
                p = [camel_case(op)]
                for i in np['inputs']:
                    if 'relational_concept' in np or 'attribute' in np:
                        p.append(('Unique', programs[i]))
                    else:
                        p.append((programs[i]))
                if 'relational_concept' in np:
                    p.append([self.entry2idx_[np['relational_concept'][0]]])
                if 'attribute' in np:
                    p.append([self.entry2idx_[np['attribute']]])
                p = tuple(p)
            programs.append(p)
        return programs[-1]

    def metaconcept_text(self, supports, relations, concept_index):
        raise NotImplementedError

    def encode_text(self, question):
        encoded = self.word_vocab(question)
        length = len(encoded)
        assert length < self.max_question_length
        encoded.extend([self.word_vocab.pad] * (self.max_question_length - length))
        return encoded, length

    def encode_unknown(self, text, concept_id):
        for word in self.synonyms.get(self.names[concept_id], [self.names[concept_id]]):
            text = re.subn(f"(?<!\w){word}s?(\\b)", "unk\\1", text)[0]
        encoded = self.word_vocab(text)
        length = len(encoded)
        assert length < self.max_question_length
        encoded.extend([self.word_vocab.pad] * (self.max_question_length - length))
        return encoded, length


class BuilderDataset(Dataset):
    MAC_ROOT = "/data/vision/billf/scratch/jerrymei/BoxEmbedding/vendor/mac-network-original"
    texts = ["statement"]
    num_inputs = {}
    num_value_inputs = {}
    max_program_length = 50
    max_samekinds = 8
    metaconcepts_ = ['hypernym', 'hyponym', 'samekind']

    @property
    def info(self):
        return {**super(BuilderDataset, self).info, 'word_entries': len(self.word_vocab),
            'program_entries': len(self.program_vocab), 'start_id': self.program_vocab.start,
            'end_id': self.program_vocab.end, 'max_length': self.max_program_length}

    @property
    def principal_metric(self):
        return 'accuracy'

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = []
        self.program_vocab = None
        self.mode = "build" if self.split != 'all' else 'inference'

    def _build_program_tokens(self):
        raise NotImplementedError

    def encode_program(self, program):
        encoded = [[self.program_vocab.start, self.program_vocab.start]]

        def _build_subtree(node):
            t = node['type']
            arg = self.program_vocab.unk if self.num_value_inputs[t] == 0 else self.program_vocab[
                node['value_inputs'][0]]
            entry = [self.program_vocab[t], arg]

            encoded.append(entry)
            for i in node['inputs']:
                _build_subtree(program[i])

        _build_subtree(program[-1])
        encoded.append([self.program_vocab.end, self.program_vocab.end])
        length = len(encoded)

        encoded.extend([[self.program_vocab.pad, self.program_vocab.pad]] * (self.max_program_length - length))
        assert length < self.max_program_length
        return encoded, length

    def encode_metaconcept(self, program):
        encoded = [[self.program_vocab.start, self.program_vocab.start]]
        for p in program:
            encoded.append([self.program_vocab[p['type']], self.program_vocab[p['value_inputs'][0]]])
        encoded.append([self.program_vocab.end, self.program_vocab.end])
        length = len(encoded)

        assert length < self.max_program_length
        encoded.extend([[self.program_vocab.pad, self.program_vocab.pad]] * (self.max_program_length - length))
        return encoded, length

    def decode_program(self, predicted):
        def canonical(operation):
            return operation.replace('_material', '').replace('_size', '').replace('_color', '').replace(
                '_shape', '')

        es = nonzero(predicted[..., 0] == self.program_vocab.end)
        end_index = es[0] if len(es) > 0 else predicted.shape[0]
        program_encoded = predicted[1:end_index].tolist()
        family = tuple(canonical(self.program_vocab.index2word[t]) for t, a in program_encoded)
        program = []

        def expand_node():
            if len(program_encoded) == 0:
                program.append({'type': 'scene', 'value_inputs': [], 'inputs': []})
                return len(program) - 1
            t, a = [self.program_vocab.index2word[_] for _ in program_encoded.pop(0)]
            value_inputs = [a] if a in self.named_entries_ else [] if self.num_value_inputs[t] == 0 else [
                random.choice(self.named_entries_)]
            p = {'type': t, 'value_inputs': value_inputs}
            ii = [expand_node() for _ in range(self.num_inputs[t])]
            p['inputs'] = ii
            program.append(p)
            return len(program) - 1

        expand_node()
        return program, family

    def decode_metaconcept(self, predicted):
        end_index = nonzero(predicted[..., 0] == self.program_vocab.end)[0]
        program_encoded = predicted[1:end_index].tolist()
        family = tuple(self.program_vocab.index2word[t] for t, a in program_encoded)
        program = [
            {'type': self.program_vocab.index2word[t], 'value_inputs': [self.program_vocab.index2word[a]],
                'inputs': []} for t, a in program_encoded if a not in self.program_vocab.special_tokens]
        return program, family

    def encode(self, text, program, name):
        encoded, length = self.encode_text(text)
        target, _ = self.encode_program(program) if name != 'metaconcept' else self.encode_metaconcept(program)
        return {f'{name}_encoded': encoded, f'{name}_length': length, f'{name}_target': target, f'{name}': text,
            f"{name}_program": program}

    def batch_build_handler(self, inputs, outputs):
        accuracies = {}
        for t in ["statement", "metaconcept", "question"]:
            # noinspection PyTypeChecker
            accuracies[f"{t}_accuracy"] = torch.all(
                torch.all(inputs[f"{t}_target"] == outputs[f"{t}_predicted"], -1), -1).tolist()
        return accuracies

    def metric_build_handler(self, evaluated):
        metrics = {}
        for k, v in evaluated.items():
            if k != "mode":
                metrics[k] = torch.tensor(v).float().mean().item()
        metrics['accuracy'] = mean(metrics.values())
        return metrics

    def exist_question_program(self, candidate):
        return [{'type': 'scene', 'inputs': [], 'value_inputs': []},
            {'type': 'filter', 'inputs': [0], 'value_inputs': [self.named_entries_[candidate]]},
            {'type': 'exist', 'inputs': [1], 'value_inputs': []}]

    def exist_statement_program(self, candidate):
        return [{'type': 'scene', 'inputs': [], 'value_inputs': []}]

    def __getitem__(self, index):
        question_index = self.indices_split[index]
        question = {k: torch.tensor(v) if re.search('encoded|target', k) else v for k, v in
            self.questions[question_index].items()}
        return {**question, 'question_index': question_index, 'index': index, 'info': self.info}

    def filter_question_program(self, candidate, filters):
        return [{'type': 'scene', 'inputs': [], 'value_inputs': []},
            *({'type': 'filter', 'inputs': [i], 'value_inputs': [self.named_entries_[f]]} for i, f in
                enumerate(filters)),
            {'type': 'filter', 'inputs': [len(filters)], 'value_inputs': [self.named_entries_[candidate]]},
            {'type': 'exist', 'inputs': [len(filters) + 1], 'value_inputs': []}]

    def filter_statement_program(self, candidate, filters):
        return [{'type': 'scene', 'inputs': [], 'value_inputs': []},
            *({'type': 'filter', 'inputs': [i], 'value_inputs': [self.named_entries_[f]]} for i, f in
                enumerate(filters))]
