import matplotlib.pyplot as plt
import torch

from visualization.visualizer import Visualizer


class FeatureWidthVisualizer(Visualizer):
    per_batch = False

    def visualize(self, results, model, iteration, **kwargs):
        if results is None or model is None:
            return
        image_classes = torch.stack(results["image_class"])
        features = torch.stack(results["feature"])
        specific_boundaries = []
        unique_classes = torch.unique(image_classes).sort().values
        for u in unique_classes:
            specific_boundary = model.entailment.specific_boundary(features[image_classes == u])
            specific_boundaries.append(specific_boundary)
        specific_boundaries = torch.stack(specific_boundaries)
        dim = model.entailment.mid_channels

        widths = specific_boundaries[:, dim:].mean(1)
        widths_by_split = []
        unique_splits = torch.unique(self.dataset.concept_split_specs[unique_classes]).sort().values
        for split in unique_splits:
            widths_by_split.append(
                widths[self.dataset.concept_split_specs[unique_classes] == split].mean().item())

        distance_from_neighbour = (specific_boundaries.unsqueeze(0) - specific_boundaries.unsqueeze(1))[...,
        :dim].abs().mean(-1).topk(2, largest=False).values[:, 1]
        distance_by_split = []
        for split in unique_splits:
            distance_by_split.append(distance_from_neighbour[
                self.dataset.concept_split_specs[unique_classes] == split].mean().item())

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.bar(unique_splits, widths_by_split)
        for s, w in zip(unique_splits, widths_by_split):
            ax.text(s - .35, w + .002, f"{w:.03f}")
        self.summary_writer.add_figure(f"feature_by_class/{self.dataset.tag}/width", fig, iteration)

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.bar(unique_splits, distance_by_split)
        for s, w in zip(unique_splits, distance_by_split):
            ax.text(s - .35, w + .002, f"{w:.03f}")
        self.summary_writer.add_figure(f"feature_by_class/{self.dataset.tag}/distance", fig, iteration)
