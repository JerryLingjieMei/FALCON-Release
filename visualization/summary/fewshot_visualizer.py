import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from math import ceil
from torchvision.utils import make_grid

from utils import text2img, to_cpu_detach
from utils.images import to_numbered_list
from visualization.visualizer import Visualizer


class FewshotVisualizer(Visualizer):
    per_batch = True
    _DOWN_SCALE = .75
    _N_SAMPLES = 4
    _TOP_K = 4
    _COLUMNS = 2

    def text(self, target, end, category):
        if category == "boolean":
            return f"{end.item():< 8.3f} {target.item() * 10 - 5:< 8.3f}", 'green' if (end > 0) == (
                    target > .5) else 'red'
        elif category == "count":
            return f"{end.item():< 8.3f} {target.item():< 8.3f}", 'green' if abs(end - target) < .5 else 'red'
        elif category == "choice":
            return f"{end[0].max().item():< 8.3f} {end[0][target].item():< 8.3f}", 'green' if end.max(
                -1).indices == target else 'red'
        elif category == "token":
            index2word = self.dataset.word_vocab.index2word
            return f"{index2word[end.max(0).indices.item()]} {index2word[target.item()]}", 'green' if end.max(
                -1).indices == target else 'red'
        else:
            raise NotImplementedError

    def visualize(self, inputs, outputs, model, iteration, **kwargs):
        summaries = []
        inputs = to_cpu_detach(inputs)
        outputs = to_cpu_detach(outputs)
        for i in range(min(self._N_SAMPLES, len(inputs["program"]))):
            in_program = inputs["program"][i]
            out_program = outputs["program"][i]
            summary = []
            for group in ["train_sample", "val_sample"]:
                h, w = self.dataset.empty_image.shape[-2:]
                this_inputs = inputs[group]

                if self.dataset.has_mask:
                    queried_objects = outputs[group]["query_object"][i]
                    images = torch.stack([self.dataset.get_annotated_image(i, q) for i, q in
                        zip(this_inputs["image_index"][i], queried_objects)])
                else:
                    images = this_inputs["image"][i]

                n = len(images)
                if group == "val_sample":
                    images = images[torch.cat([torch.arange(5), torch.arange(5) + n // 2])]
                img_grid = make_grid(images, ceil(len(images) / 5), 0)
                pad = (h * 5 - img_grid.shape[-2]) // 2
                summary.append(F.pad(img_grid, [0, 0, pad, pad], "constant", 0))

                target = this_inputs["target"][i]
                end = outputs[group]["end"][i]
                category = this_inputs["category"][i]

                for chunk in torch.arange(len(end)).chunk(self._COLUMNS):
                    texts, colors = [], []
                    if self.dataset.use_text:
                        if group == "val_sample":
                            token = this_inputs["answer_tokenized"][i]
                            texts, colors = list(
                                zip(*(self.text(token[j], end[j], category[j]) for j in chunk.tolist())))
                    elif out_program.is_fewshot or group == "val_sample":
                        texts, colors = list(
                            zip(*(self.text(target[j], end[j], category[j]) for j in chunk.tolist())))
                    summary.append(TF.to_tensor(text2img(texts, (h * 5, w), colors)))

            summary = torch.cat(summary, dim=2)
            program_img = TF.to_tensor(text2img(in_program % self.dataset, (h // 2, summary.shape[-1])))
            text = self.partition(inputs["text"][i], self.MAX_TEXT_LINES)
            question_img = TF.to_tensor(text2img(text, (h, summary.shape[-1])))
            answer = self.partition(out_program.val_answer_text, self.MAX_ANSWER_LINES)
            answer_img = TF.to_tensor(text2img(answer, (h // 2, summary.shape[-1])))
            train_image_indices = [self.dataset.image_filenames[t] for t in out_program.train_image_index]
            val_image_indices = [self.dataset.image_filenames[t] for t in out_program.val_image_index]
            info_img = TF.to_tensor(text2img(
                [f"train_indices:", *to_numbered_list(out_program.train_image_index, 1),
                    *to_numbered_list(train_image_indices, 1), "val_indices",
                    *to_numbered_list(out_program.val_image_index, 2), *to_numbered_list(val_image_indices, 2),
                    f"index: {inputs['index'][i]}", f"question index: {inputs['question_index'][i]}"],
                (h, summary.shape[-1])))

            summaries.append(torch.cat([summary, program_img, question_img, answer_img, info_img], dim=1))

        self.summary_writer.add_image(f"fewshot/{self.dataset.tag}", make_grid(summaries, nrow=1), iteration)

        # distribution of embeddings
        concepts = outputs["queried_embedding"]
        if model.feature_extractor.rep == "box":
            concept_locs, concept_scales = concepts.chunk(2, -1)
            self.summary_writer.add_histogram(f"concept/{self.dataset.tag}/location", concept_locs, iteration)
            self.summary_writer.add_histogram(f"concept/{self.dataset.tag}/scale", concept_scales, iteration)
        else:
            self.summary_writer.add_histogram(f"concept/{self.dataset.tag}/size", concepts, iteration)

        # distribution of logits
        true_logits, false_logits = [], []
        for i, _ in enumerate(inputs["index"]):
            target = inputs["val_sample"]["target"][i]
            end = outputs["val_sample"]["end"][i]
            category = inputs["val_sample"]["category"][i]
            if category == "boolean":
                true_logits.append(end[target.bool()])
                false_logits.append(end[~target.bool()])
        if len(true_logits) > 0:            self.summary_writer.add_histogram(
            f"fewshot/{self.dataset.tag}/true", torch.cat(true_logits), iteration)
        if len(false_logits) > 0:            self.summary_writer.add_histogram(
            f"fewshot/{self.dataset.tag}/false", torch.cat(false_logits), iteration)
