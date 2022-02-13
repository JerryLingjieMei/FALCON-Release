from itertools import groupby

import torch

from dataset.dataset import Dataset, BuilderDataset
from models.programs import to_batch, build_program
from utils import join, dump, mkdir, to_serializable, nonzero


class PretrainDataset(Dataset):

    @property
    def principal_metric(self):
        return 'accuracy'

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = []
        self.mode = "pretrain" if self.split != 'all' else 'inference'

    def tokenize_support(self, texts, answers):
        category = self.answer2category(answers[0])
        answer_texts = [self.answer2text(t) for t in answers]
        targets = torch.stack([self.answer2target(a) for a in answers])
        question_tokenized, question_length = list(zip(*(self.encode_text(q) for q in texts)))
        question_tokenized = torch.tensor(question_tokenized)
        answer_tokenized = torch.tensor([self.word_vocab[a] for a in answer_texts])
        return {"category": category, "target": targets, "tokenized": question_tokenized,
            "length": question_length, "answer_text": answer_texts, "answer_tokenized": answer_tokenized}

    def batch_pretrain_handler(self, inputs, outputs):
        accuracies = []
        for i, category in enumerate(inputs["category"]):
            end = outputs["end"][i]
            target = inputs["target"][i]
            if category == "boolean":
                accuracy = (~torch.logical_xor(end > 0, target)).float().mean()
            elif category == "count":
                accuracy = (end == target).float().mean()
            else:
                accuracy = (end.max(-1).indices == target).float().mean()
            accuracies.append(accuracy)
        return {"accuracy": accuracies}

    def batch_feature_handler(self, inputs, outputs):
        embeddings = outputs["feature"]
        labels = [p.right_most[t.bool()] for p, t in zip(inputs["program"], inputs["target"])]
        return {"feature": embeddings, "labels": labels}

    def batch_inference_handler(self, inputs, outputs):
        positives, trues = [], []
        for i, p in enumerate(inputs["program"]):
            candidates = p.right_most
            positives.append(candidates[nonzero(outputs["logit"][i] > 0)])
            trues.append(candidates[nonzero(inputs["target"][i] > 0)])
        return {"positive": positives, "true": trues}

    def metric_pretrain_handler(self, evaluated):
        accuracy = torch.stack(evaluated["accuracy"]).mean()
        return {"accuracy": accuracy}

    def save_pretrain_handler(self, output_dir, evaluated, iteration, metrics):
        dump(to_serializable(metrics), join(output_dir, f"{evaluated['mode']}_{iteration:07d}.json"))

    def save_feature_handler(self, output_dir, evaluated, iteration, metrics):
        features = torch.stack(evaluated["feature"])
        dump(features, join(output_dir, f"{evaluated['mode']}_{iteration:07d}.pth"))

    def save_confusion_handler(self, output_dir, evaluated, iteration, metrics):
        evaluated = to_serializable(evaluated)
        dump(evaluated, join(output_dir, f"{evaluated['mode']}_{iteration:07d}.json"))
        dump(metrics, join(output_dir, f"{evaluated['mode']}_{iteration:07d}.pth"))

    def __getitem__(self, index):
        question_index = self.indices_split[index]
        question = self.questions[question_index]
        program = build_program(question['program'])
        stacked_scenes = self.get_stacked_scenes(question['image_index'])
        tokenized_support = self.tokenize_support(question['text'], question['answer'])
        return {**tokenized_support, **stacked_scenes, **question, 'program': program, 'index': index,
            'question_index': question_index, 'info': self.info}


class PretrainBuilderDataset(BuilderDataset):

    def batch_inference_handler(self, inputs, outputs):
        questions = []
        for i, p in enumerate(outputs['question_predicted']):
            program, family = self.decode_program(p)
            length = len(program)
            program = self.nscl2program(self.clevr2nscl(program))
            question = {k: inputs[k][i] for k in ['question', 'answer', 'image_index']}
            questions.append({**question, 'program': program, 'family': family, 'length': length})
        return {"questions": questions}

    def save_inference_handler(self, output_dir, evaluated, iteration, metrics):
        questions = []
        for (image_index, family), qs in groupby(evaluated['questions'],
                lambda q: (q['image_index'], q['family'])):
            qs = list(qs)
            question = [q['question'] for q in qs]
            program = to_batch([q['program'] for q in qs])
            length = qs[0]['length']
            answer = [q['answer'] for q in qs]
            q = {'text': question, 'program': program, 'answer': answer, 'image_index': image_index,
                'length': length}
            questions.append(q)
        dest_dataset = self.get_augmented_name(self.__class__.__qualname__).replace('_builder', '')
        dump(questions, join(output_dir, f"{evaluated['mode']}_{iteration:07d}.json"))
        dump(questions, join(mkdir(join(self.augmented_root, dest_dataset)), 'questions.json'))
