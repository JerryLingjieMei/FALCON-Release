import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from utils import text2img, to_cpu_detach
from visualization.visualizer import Visualizer


class PretrainVisualizer(Visualizer):
    per_batch = True
    _DOWN_SCALE = .75
    _N_SAMPLES = 16
    _TOP_K = 9

    def boolean_window(self, program, target, end):
        names = program % self.dataset
        lines, colors = [], []
        if len(end) > self._TOP_K + 1:
            top_ks = torch.topk(end, min(self._TOP_K, len(end)))
            true_index = torch.max(target, 0).indices.item()
            indices = [*top_ks.indices, true_index]
        else:
            indices = range(len(end))
        for i in indices:
            line = f"{end[i]:< 8.3f} {names[i]:<12}"
            color = 'green' if (end[i] > 0) == (target[i] > .5) else 'red'
            lines.append(line)
            colors.append(color)
        return lines, colors

    def count_window(self, program, target, end):
        names = program % self.dataset
        lines, colors = [], []
        if len(end) > self._TOP_K + 1:
            top_ks = torch.topk(end, min(self._TOP_K, len(end)))
            true_index = torch.max(target, 0).indices.item()
            indices = [*top_ks.indices, true_index]
        else:
            indices = range(len(end))
        for i in indices:
            line = f"{end[i]:< 8.3f} {names[i]:<12}"
            color = 'green' if (end[i] - target[i]).abs() < .5 else 'red'
            lines.append(line)
            colors.append(color)
        return lines, colors

    def choice_window(self, program, target, end):
        names = program % self.dataset
        lines, colors = [], []
        for i in range(min(self._TOP_K, len(end))):
            line = f"{end[i][target[i]]:< 8.3f} {names[i]:<12}"
            color = 'green' if end[i].max(0).indices == target[i] else 'red'
            lines.append(line)
            colors.append(color)
        return lines, colors

    def visualize(self, inputs, outputs, model, iteration, **kwargs):
        summaries = []
        inputs = to_cpu_detach(inputs)
        outputs = to_cpu_detach(outputs)
        # summary image
        n_samples = min(self._N_SAMPLES, len(inputs["image_index"]))
        for i in range(n_samples):
            h, w = self.dataset.empty_image.shape[-2:]
            image_index = inputs["image_index"][i]
            images = self.dataset.get_annotated_image(image_index)
            program = inputs["program"][i]
            target = inputs["target"][i]
            category = inputs["category"][i]
            end = outputs["end"][i]
            text, colors = getattr(self, f"{category}_window")(program, target, end)
            text_img = TF.to_tensor(text2img(text, (h, w * 4), colors))
            summary = torch.cat([images, text_img], dim=2)

            question = self.partition(inputs["text"][i], self.MAX_TEXT_LINES)
            question_img = TF.to_tensor(text2img(question, (h, summary.shape[-1])))
            answer = self.partition(inputs["answer_text"][i], self.MAX_ANSWER_LINES)
            answer_img = TF.to_tensor(text2img(answer, (h // 4, summary.shape[-1])))
            info_img = TF.to_tensor(text2img([f"image_index: {image_index}", f"index:{inputs['index'][i]}",
                f"question index:{inputs['question_index'][i]}",
                f"filename: {self.dataset.image_filenames[image_index]}"], (h // 4, summary.shape[-1])))
            summary = torch.cat([summary, question_img, answer_img, info_img], dim=1)
            summaries.append(summary)

        grids = make_grid(summaries, nrow=1)
        self.summary_writer.add_image(f"pretrain/{self.dataset.tag}", grids, iteration)

        # distribution of embeddings
        feature = outputs["feature"]
        # relation = outputs["relation"]
        concept = outputs["queried_embedding"]
        if model.feature_extractor.rep == "box":
            feature_location, feature_scale = feature.chunk(2, -1)
            self.summary_writer.add_histogram(f"feature/{self.dataset.tag}/location", feature_location,
                iteration)
            concept_location, concept_scale = concept.chunk(2, -1)
            self.summary_writer.add_histogram(f"concept/{self.dataset.tag}/location", concept_location,
                iteration)
            self.summary_writer.add_histogram(f"concept/{self.dataset.tag}/scale", concept_scale, iteration)
        else:
            self.summary_writer.add_histogram(f"feature/{self.dataset.tag}/size", feature, iteration)
            # self.summary_writer.add_histogram(f"relation/{self.dataset.tag}/size", relation, iteration)
            self.summary_writer.add_histogram(f"concept/{self.dataset.tag}/size", concept, iteration)

        # distribution of logits
        true_logits, false_logits = [], []
        true_counts, false_counts = [], []
        choices = []
        for i, _ in enumerate(inputs["index"]):
            target = inputs["target"][i]
            category = inputs["category"][i]
            end = outputs["end"][i]
            if category == "boolean":
                true_logits.append(end[target.bool()])
                false_logits.append(end[~target.bool()])
            elif category == "count":
                true_counts.append(end[target.bool()])
                false_counts.append(end[~target.bool()])
            else:
                choices.append(end[target.max(0).indices])
        if sum(len(t) for t in true_logits) > 0: self.summary_writer.add_histogram(
            f"pretrain/boolean/{self.dataset.tag}/true", torch.cat(true_logits), iteration)
        if sum(len(f) for f in false_logits) > 0: self.summary_writer.add_histogram(
            f"pretrain/boolean/{self.dataset.tag}/false", torch.cat(false_logits), iteration)

        if sum(len(t) for t in true_counts) > 0:  self.summary_writer.add_histogram(
            f"pretrain/count/{self.dataset.tag}/true", torch.cat(true_counts), iteration)
        if sum(len(f) for f in false_counts) > 0: self.summary_writer.add_histogram(
            f"pretrain/count/{self.dataset.tag}/false", torch.cat(false_counts), iteration)

        if sum(len(t) for t in choices) > 0:  self.summary_writer.add_histogram(
            f"pretrain/choice/{self.dataset.tag}", torch.cat(choices), iteration)
