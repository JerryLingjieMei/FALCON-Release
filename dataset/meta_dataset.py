import csv
import glob
import os
import random
import re
from collections import defaultdict
from itertools import groupby, cycle

import torch
from tqdm import tqdm

from dataset.dataset import Dataset, BuilderDataset
from models.programs import build_program
from utils import collate_fn, num2word, dump, join, mkdir, symlink_recursive


class MetaDataset(Dataset):

    @property
    def is_attached(self):
        return 'Detached' not in self.__class__.__name__

    @property
    def is_fewshot(self):
        return 'Zeroshot' not in self.__class__.__name__

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = []
        self.mode = "meta" if self.split != 'all' else 'inference'

    @property
    def principal_metric(self):
        return "accuracy_999" if self.split == "test" else "accuracy_001"

    def tokenize_statement(self, text, concept):
        category = "boolean"
        target = torch.tensor(1)
        tokenized, length = self.encode_unknown(text, concept)
        tokenized = torch.tensor(tokenized)
        return {"target": target, "category": category, "tokenized": tokenized, "length": length}

    def tokenize_question(self, text, answer, concept):
        category = self.answer2category(answer)
        target = self.answer2target(answer)
        tokenized, length = self.encode_unknown(text, concept)
        tokenized = torch.tensor(tokenized)
        answer_text = self.answer2text(answer)
        answer_tokenized = torch.tensor(self.word_vocab[answer_text])
        return {"category": category, "target": target, "text": text, "answer_text": answer_text,
            "tokenized": tokenized, "length": length, "answer_tokenized": answer_tokenized}

    def encode_composite(self, train_text, metaconcept_text, val_text, concept_id):
        for word in self.synonyms.get(self.names[concept_id], [self.names[concept_id]]):
            train_text = re.subn(f"(?<!\w){word}s?(\\b)", "unk\\1", train_text)[0]
            metaconcept_text = re.subn(f"(?<!\w){word}s?(\\b)", "unk\\1", metaconcept_text)[0]
            val_text = re.subn(f"(?<!\w){word}s?(\\b)", "unk\\1", val_text)[0]
        train_encoded = self.word_vocab(train_text)[:-1]
        metaconcept_encoded = self.word_vocab(metaconcept_text)[1:-1]
        val_encoded = self.word_vocab(val_text)[1:]
        if not self.is_attached:
            metaconcept_encoded = []
        if not self.is_fewshot:
            train_encoded = []
        encoded = [*train_encoded, self.word_vocab.pad, *metaconcept_encoded, self.word_vocab.pad, *val_encoded]
        segment = [1] * (len(train_encoded) + 1) + [2] * (len(metaconcept_encoded) + 1) + [3] * len(val_encoded)
        length = len(encoded)
        assert length < self.max_question_length * 3
        encoded.extend([self.word_vocab.pad] * (self.max_question_length * 3 - length))
        segment.extend([0] * (self.max_question_length * 3 - length))
        return encoded, length, segment

    def batch_meta_handler(self, inputs, outputs):
        accuracies = []
        concept_index = inputs["concept_index"]
        queried_embedding = outputs["queried_embedding"]
        for i, categories in enumerate(inputs["val_sample"]["category"]):
            val_end = outputs["val_sample"]["end"][i]
            val_target = inputs["val_sample"]["target"][i]
            val_token = inputs["val_sample"]["answer_tokenized"][i]
            piece_accuracy = []
            for j, category in enumerate(categories):
                if category == "boolean":
                    pa = (~torch.logical_xor(val_end[j] > 0, val_target[j])).float()
                elif category == "count":
                    pa = (val_end[j] == val_target[j]).float()
                elif category == "choice":
                    pa = (val_end[j].max(-1).indices == val_target[j]).float()
                elif category == "token":
                    pa = (val_end[j].max(-1).indices == val_token[j]).float()
                else:
                    raise NotImplementedError
                piece_accuracy.append(pa)
            accuracies.append(torch.stack(piece_accuracy).mean())
        out = {"accuracy": accuracies, "concept_index": concept_index, "queried_embedding": queried_embedding}
        return out

    def metric_meta_handler(self, evaluated):
        accuracy = torch.stack(evaluated["accuracy"])
        metrics = {}
        index = torch.tensor(evaluated["concept_index"])
        spec = self.concept_split_specs[index]
        kind = self.concept2kinds[index]
        for unique in torch.unique(spec).tolist():
            metrics[f"accuracy_{unique:03d}"] = accuracy[spec == unique].mean().item()
        metrics["accuracy_999"] = accuracy.mean().item()
        for unique in torch.unique(kind).tolist():
            metrics[f"accuracy_{self.kinds_[unique]}"] = accuracy[kind == unique].mean(0).item()
        return metrics

    def save_meta_handler(self, output_dir, evaluated, iteration, metrics):
        filename_prefix = evaluated["mode"]
        metric_titles = ["group", "accuracy"]
        metric_width = [len(m) for m in metric_titles]
        with open(join(output_dir, f"{filename_prefix}_{iteration:07d}.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=metric_titles, dialect="excel")
            writer.writeheader()
            for s in torch.unique(self.concept_split_specs, sorted=True).tolist() + [999]:
                row = {"group": f"{s:>{metric_width[0]:}}"}
                for title, width in zip(metric_titles, metric_width):
                    m = metrics.get(f'{title}_{s:03d}', None)
                    if m is not None:
                        row[title] = f"{m :{width}.3f}"
                if len(row) > 1:
                    writer.writerow(row)
            for s in self.kinds_:
                row = {"group": f"{s:>{width:}}"}
                for title, width in zip(metric_titles, metric_width):
                    m = metrics.get(f'{title}_{s}', None)
                    if m is not None:
                        row[title] = f"{m :{width}.3f}"
                if len(row) > 1:
                    writer.writerow(row)
        dump(metrics, join(output_dir, f"{filename_prefix}_{iteration:07d}.json"))
        queried_embedding = torch.stack(evaluated["queried_embedding"])
        dump(queried_embedding, join(output_dir, f"{filename_prefix}_{iteration:07d}.pth"))

    def __getitem__(self, index):
        question_index = self.indices_split[index]
        question = self.questions[question_index]
        concept_index = question['concept_index']

        train_samples = question['train_sample']
        train_stacked_scenes = collate_fn([self.get_stacked_scenes(i) for i in train_samples['image_index']])
        train_tokenized_statements = collate_fn(
            [self.tokenize_statement(t, concept_index) for t in train_samples['text']])
        train_program = [build_program(p) for p in train_samples['program']]
        train_image_indices = train_samples['image_index'] if self.is_fewshot else [-1] * len(
            train_samples['image_index'])
        train_samples = {**train_samples, **train_stacked_scenes, **train_tokenized_statements,
            'program': train_program, 'image_index': train_image_indices}

        val_samples = question['val_sample']
        val_stacked_scenes = collate_fn([self.get_stacked_scenes(i) for i in val_samples['image_index']])
        val_tokenized_question = collate_fn([self.tokenize_question(t, a, concept_index) for t, a in
            zip(val_samples['text'], val_samples['answer'])])
        val_program = [build_program(p).register_token(concept_index) for p in val_samples['program']]
        val_samples = {**val_samples, **val_stacked_scenes, **val_tokenized_question, 'program': val_program}

        metaconcept_tokenized, metaconcept_length = self.encode_unknown(train_samples['metaconcept_text'],
            concept_index)
        metaconcept_tokenized = torch.tensor(metaconcept_tokenized)
        composite_tokenized, composite_length, composite_segment = list(zip(*(
            self.encode_composite(train_samples['text'][0], train_samples['metaconcept_text'], val_text,
                concept_index) for val_text in val_samples["text"])))
        task = {'metaconcept_tokenized': metaconcept_tokenized, 'metaconcept_length': metaconcept_length,
            'composite_tokenized': torch.tensor(composite_tokenized), 'composite_length': composite_length,
            'composite_segment': torch.tensor(composite_segment)}

        if self.is_attached:
            program = build_program(("Composite", question['supports'], question['relations'], concept_index))
        else:
            program = build_program(('Composite', [], [], concept_index))

        return {**question, 'train_sample': train_samples, 'val_sample': val_samples, 'task': task,
            'program': program, 'index': index, 'question_index': question_index, 'info': self.info}


class MetaBuilderDataset(BuilderDataset):
    split_seed2rate = {7: .25, 8: .5, 9: .75}

    def dropout(self, samekinds, dropout=False):
        if len(samekinds) == 0:
            return samekinds
        # if dropout:
        #     samekinds = list(filter(lambda _: random.random() > self.dropout_rate, samekinds))
        # else:
        samekinds = samekinds[:int(len(samekinds) * self.split_seed2rate.get(self.split_seed, 1))]
        random.shuffle(samekinds)
        if len(samekinds) == 0:
            return self.dropout(samekinds, dropout)
        else:
            return samekinds[:self.max_samekinds]

    def decorate_refexp(self, text, concept):
        raise NotImplementedError

    def metaconcept_program(self, supports, relations):
        return [{'type': self.metaconcepts_[r], 'value_inputs': [self.named_entries_[s]], 'inputs': []} for r, s
            in zip(relations, supports)]

    def _build_composite_questions(self, refexps, fewshot_questions):
        questions = []
        for itself, samekinds in tqdm(self.itself2samekinds.items()):
            samekinds = [s for s in samekinds if self.concept_split_specs[s] <= 0]
            if len(samekinds) == 0:
                continue
            statement_candidates = random.choices([r for r in refexps if r['concept_index'] == itself],
                k=self.N_SAMPLES * self.shot_k)
            for i in range(self.N_SAMPLES):
                supports = self.dropout(samekinds, self.concept_split_specs[itself] <= 0)
                relations = [2] * len(supports)
                encoded_metaconcept = self.encode(self.metaconcept_text(supports, relations, itself),
                    self.metaconcept_program(supports, relations), 'metaconcept')

                statement = statement_candidates[i * self.shot_k:(i + 1) * self.shot_k]
                valid_questions = [q for q in fewshot_questions if q['concept_index'] == itself]
                if self.base_dataset == "clevr":
                    qs = random.choices(valid_questions, k=self.query_k)
                else:
                    valid_questions = [q for q in valid_questions if any(
                        len(set(s['concept_contained']).intersection(q['concept_contained'])) == 1 for s in
                            statement)]
                    true_qs = [q for q in valid_questions if q['answer']]
                    false_qs = [q for q in valid_questions if not q['answer']]
                    qs = []
                    if len(true_qs) > 0: qs += random.choices(true_qs, k=self.query_k // 2)
                    if len(false_qs) > 0: qs += random.choices(false_qs, k=self.query_k // 2)
                for j, (s, q) in enumerate(zip(cycle(statement), qs)):
                    encoded_statement = self.encode(self.decorate_refexp(s['refexp'], itself), s['program'],
                        'statement')
                    encoded_question = self.encode(q['question'], q['program'], 'question')
                    questions.append(
                        {**encoded_statement, **encoded_metaconcept, **encoded_question, 'answer': q['answer'],
                            'concept_index': itself, 'train_image_index': s['image_index'],
                            'image_index': q['image_index'], 'family': (itself, i, j)})
        return questions

    def batch_inference_handler(self, inputs, outputs):
        questions = []
        for i, (sp, mp, qp) in enumerate(zip(outputs['statement_predicted'], outputs['metaconcept_predicted'],
                outputs['question_predicted'])):
            statement = self.nscl2program(self.clevr2nscl(self.decode_program(sp)[0]))
            metaconcept = self.decode_metaconcept(mp)[0]
            question = self.nscl2program(self.clevr2nscl(self.decode_program(qp)[0]))
            out = {k: inputs[k][i] for k in
                ['statement', 'metaconcept', 'question', 'answer', 'image_index', 'train_image_index',
                    'concept_index', 'question_index', 'family']}
            questions.append({**out, 'statement_program': statement, 'metaconcept_program': metaconcept,
                'question_program': question})
        return {"questions": questions}

    def save_inference_handler(self, output_dir, evaluated, iteration, metrics):
        questions = []
        for family, qs in groupby(evaluated['questions'], lambda q: (q['family'][:2])):
            qs = sorted(list(qs), key=lambda q: q['family'][2])
            train_texts = [q['statement'] for q in qs[:self.shot_k]]
            train_programs = [q['statement_program'] for q in qs[:self.shot_k]]
            train_answers = [True] * self.shot_k
            train_image_indices = [q['image_index'] for q in qs[:self.shot_k]]
            metaconcept_text = qs[0]['metaconcept']
            train_samples = {'text': train_texts, 'program': train_programs, 'answer': train_answers,
                'image_index': train_image_indices, 'metaconcept_text': metaconcept_text}

            val_text = [q['question'] for q in qs]
            val_program = [q['question_program'] for q in qs]
            val_answer = [q['answer'] for q in qs]
            val_image_indices = [q['image_index'] for q in qs]
            val_samples = {'text': val_text, 'program': val_program, 'answer': val_answer,
                'image_index': val_image_indices}

            supports = [self.entry2idx_[p['value_inputs'][0]] for p in qs[0]['metaconcept_program']]
            relations = [self.metaconcepts_.index(p['type']) for p in qs[0]['metaconcept_program']]

            text = ' '.join(train_texts + [qs[0]['metaconcept']] + val_text)
            concept_index = qs[0]['concept_index']
            q = {'text': text, 'train_sample': train_samples, 'val_sample': val_samples, 'supports': supports,
                'relations': relations, 'concept_index': concept_index}
            questions.append(q)
        dest_dataset = self.get_augmented_name(self.__class__.__qualname__).replace('_builder', '')
        dump(questions, join(output_dir, f"{evaluated['mode']}_{iteration:07d}.json"))
        dump(questions, join(mkdir(join(self.augmented_root, dest_dataset)), 'questions.json'))
        self._build_mac()

    def mac_split(self, concept_index):
        if self.concept_split_specs[concept_index] <= 0:
            return 'train'
        elif self.concept_split_specs[concept_index] == 1:
            return 'val'
        elif self.concept_split_specs[concept_index] == 2:
            return 'test'
        else:
            return None

    @property
    def valid_shots(self):
        return ['fewshot', 'zeroshot', 'detached']

    def _build_mac(self):
        others = defaultdict(list)
        for q in self.questions:
            others[tuple(q['family'][:2])].append(q)
        others = {k: list(sorted(v, key=lambda q: q['family'][2]))[:self.shot_k] for k, v in others.items()}

        for dataset in self.valid_shots:
            questions = {'train': [], 'val': [], 'test': []}
            for question in self.questions:
                concept_index = question['concept_index']
                split = self.mac_split(concept_index)
                if split is None: continue
                collections = questions[split]

                other = others[tuple(question['family'][:2])]
                s = list(o['statement'] for o in other)
                m = question['metaconcept']
                q = question['question']
                if dataset == "fewshot":
                    text = ' '.join([*s, m, q])
                elif dataset == "zeroshot":
                    text = ' '.join([m, q])
                else:
                    text = ' '.join([*s, q])

                for s in self.synonyms.get(self.names[concept_index], [self.names[concept_index]]):
                    text = re.subn(f"(?<!\w){s}s?(\\b)", "unk\\1", text)[0]
                    text = re.subn(f"(?<!\w){s.capitalize()}s?(\\b)", "unk\\1", text)[0]
                answer = str(question['answer'])
                if answer == self.names[concept_index]:
                    answer = 'unk'

                image_index = list(o['train_image_index'] for o in other) if dataset != 'zeroshot' else []
                image_index.append(question['image_index'])
                if self.base_dataset == 'gqa': image_index = [int(self.image_filenames[i].split('.')[0]) for i
                    in image_index]
                collections.append({'image_index': image_index, 'question': text, 'answer': answer})

            name = f'{self.base_dataset}_{dataset}'
            if self.shot_k > 1: name = name.replace('few', num2word(self.shot_k))
            if self.is_debias: name = f"{name}_debias"
            name = f"{name}_{self.split_seed}"
            folder = mkdir(join(self.MAC_ROOT, name))
            for split, collections in questions.items():
                if not self.is_debias or split == 'test':
                    dump({'questions': collections},
                        join(folder, f'{self.base_dataset}_{split}_questions.json'))
                else:
                    try:
                        os.symlink(
                            join(folder.replace('debias_', ''), f'{self.base_dataset}_{split}_questions.json'),
                            join(folder, f'{self.base_dataset}_{split}_questions.json'))
                    except:
                        pass
                symlink_recursive(join(self.MAC_ROOT, self.base_dataset), folder)
            # Remove generated file
            for g in glob.glob(join(folder, 'gen*')):
                os.remove(g)
