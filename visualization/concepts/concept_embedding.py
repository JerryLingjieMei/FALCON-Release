from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from visualization.visualizer import Visualizer


class ConceptEmbeddingVisualizer(Visualizer):
    per_batch = False

    def visualize(self, results, model, iteration, **kwargs):
        embeddings = results["embedding"]
        labels = results["label"]

        # dimension_stddev
        if model.rep == "box":
            dim = embeddings.shape[-1] // 2
            anchors = embeddings[:, :dim]
        else:
            anchors = embeddings
        std = torch.std(anchors, 0)
        self.summary_writer.add_histogram(f"weight_stddev_by_dimension/{self.dataset.tag}/ordered", std,
            iteration)
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.scatter(list(range(len(std))), std.sort().values.tolist())
        ax.set_ylabel("Stddev")
        self.summary_writer.add_figure(f"weight_stddev_by_dimension/{self.dataset.tag}/sorted", fig, iteration)

        # weight by split
        split2weight = defaultdict(list)
        for e, l in zip(embeddings, labels):
            if model.rep == "box":
                dim = e.shape[-1] // 2
                size = e[dim:].mean()
            else:
                size = e.norm()
            split2weight[self.dataset.concept_split_specs[l].item()].append(size)
        split2weight = {k: torch.stack(v) for k, v in split2weight.items()}

        for split, weight in sorted(split2weight.items()):
            self.summary_writer.add_histogram(f"weight_by_split/{self.dataset.tag}", weight, split)

        # weight by dimension
        if model.rep == "box":
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            dim = embeddings.shape[-1] // 2
            offset = torch.arange(dim)
            xs = (embeddings[:, offset] - embeddings[:, dim + offset]).flatten().tolist()
            ys = (embeddings[:, offset] + embeddings[:, dim + offset]).flatten().tolist()
            ax.hist2d(xs, ys, bins=40, range=[[-.5, .5], [-.5, .5]])
            ax.plot([0, 1], [0, 1], transform=ax.transAxes)
            ax.set_xlabel("Box begin")
            ax.set_ylabel("Box end")
            self.summary_writer.add_figure(f"weight_by_dimension/{self.dataset.tag}/global", fig, iteration)

            for offset in range(5):
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 8)
                xs = (embeddings[:, offset] - embeddings[:, dim + offset]).flatten().tolist()
                ys = (embeddings[:, offset] + embeddings[:, dim + offset]).flatten().tolist()
                ax.hist2d(xs, ys, bins=40, range=[[-.5, .5], [-.5, .5]])
                ax.plot([0, 1], [0, 1], transform=ax.transAxes)
                ax.set_xlabel("Box begin")
                ax.set_ylabel("Box end")
                self.summary_writer.add_figure(f"weight_by_dimension/{self.dataset.tag}/local", fig,
                    iteration + offset)
        else:
            self.summary_writer.add_histogram(f"weight_by_dimension/{self.dataset.tag}/global", embeddings,
                iteration)
            for offset in range(5):
                self.summary_writer.add_histogram(f"weight_by_dimension/{self.dataset.tag}/local",
                    embeddings[:, offset], iteration + offset)
