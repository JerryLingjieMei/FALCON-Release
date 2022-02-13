from copy import copy

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import XKCD_COLORS
from matplotlib.patches import Circle
from sklearn.decomposition import PCA

from utils import to_cpu_detach
from visualization.polygon.utils import regular_polygon_vertices, circle_projection
from visualization.visualizer import Visualizer


class DriftBoxVisualizer(Visualizer):
    per_batch = True

    # noinspection PyTypeChecker
    def visualize(self, inputs, outputs, model, iteration, **kwargs):
        if self.dataset.use_text:
            return
        entries = []
        inputs = to_cpu_detach(inputs)
        outputs = to_cpu_detach(outputs)

        program = outputs["program"][0]
        train_query_object = outputs["train_sample"]["query_object"][0]
        val_query_object = outputs["val_sample"]["query_object"][0]
        for he in program.hypernym_embeddings:
            entries.append((he, "hypernym"))
        for se in program.samekind_embeddings:
            entries.append((se, "samekind"))

        train_features = torch.stack(
            [p.object_collections[o] for p, o in zip(program.train_program, train_query_object)])
        val_features = torch.stack(
            [p.object_collections[o] for p, o in zip(program.val_program, val_query_object)])


        if "specific_boundary" in outputs:
            scaled_specific_boundary = outputs["specific_boundary"][0]
            if model.rep == "box":
                dim = val_features.shape[-1] // 2
                scaled_specific_boundary[dim:] = scaled_specific_boundary[dim:].clamp(min=0.02)
            entries.append((outputs["specific_boundary"][0], f"{self.dataset.shot_k}-shot_before"))

        entries.append((outputs["queried_embedding"][0], f"{self.dataset.shot_k}-shot_after"))

        metadata = [m for e, m in entries]
        embeddings = torch.stack([e for e, m in entries])
        concept_name = self.dataset.named_entries_[inputs["concept_index"][0]]

        pca_global = self.pca_global(model)
        fig_global = self.box_plot(model, pca_global, embeddings, metadata, train_features, val_features)
        plt.title(f"{concept_name}: Before and After")
        self.summary_writer.add_figure(f"drift_box/{self.dataset.tag}/global", fig_global, iteration)

        pca_local = self.pca_local(model, val_features)
        fig_local = self.box_plot(model, pca_local, embeddings, metadata, train_features,val_features)
        plt.title(f"{concept_name}: Before and After")
        self.summary_writer.add_figure(f"drift_box/{self.dataset.tag}/local", fig_local, iteration)

    def pca_global(self, model):
        with torch.no_grad():
            indices = (self.dataset.concept_split_specs <= 0).nonzero(as_tuple=False).squeeze(1)
            embeddings = to_cpu_detach(model.box_registry[indices])
        pca_2d = PCA(n_components=2)
        anchors = embeddings[:, :embeddings.shape[-1] // 2] if model.rep == "box" else embeddings
        pca_2d.fit(anchors)
        return pca_2d

    def pca_local(self, model, feature):
        pca_2d = PCA(n_components=2)
        anchors = feature[:, :feature.shape[-1] // 2] if model.rep == "box" else feature
        pca_2d.fit(anchors)
        return pca_2d

    def box_plot(self, model, pca, embeddings, metadata, train_features, val_features):
        # figure visualization
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)

        patches = []
        xkcd_colors = list(XKCD_COLORS.values())
        colors = {m: xkcd_colors.pop() for m in metadata}
        # concept_boxes
        dim = embeddings.shape[-1] // 2
        if model.rep == "box":
            center = embeddings[:, :dim]
            offset = embeddings[:, dim:]
            polygon_vertices = regular_polygon_vertices()
            for concept_index in range(len(metadata)):
                color = colors[metadata[concept_index]]
                point, radius = circle_projection(pca, center[concept_index], offset[concept_index],
                    polygon_vertices)
                circle = Circle(point, radius=radius, fill=None, color=color)
                ax.add_patch(circle)
                patches.append(circle)
        else:
            vertices = pca.transform(embeddings)
            for concept_index in range(len(metadata)):
                color = colors[metadata[concept_index]]
                point = vertices[concept_index]
                other_point = [point[0] - point[1], point[1] + point[0]]
                # noinspection PyUnresolvedReferences
                line = plt.axline(point, other_point, color=color)
                patches.append(line)

        labels = copy(metadata)
        train_2d = torch.Tensor(
            pca.transform(train_features[:, :dim] if model.rep == "box" else train_features))
        train_dots = ax.scatter(train_2d[:, 0], train_2d[:, 1], c=xkcd_colors.pop(), alpha=self.ALPHA,
            marker=".", s=100)
        patches.append(train_dots)
        labels.append("trains")

        val_2d = torch.Tensor(pca.transform(val_features[:, :dim] if model.rep == "box" else val_features))
        val_dots = ax.scatter(val_2d[:, 0], val_2d[:, 1], c=xkcd_colors.pop(), alpha=self.ALPHA, marker="+",
            s=100)
        patches.append(val_dots)
        labels.append("vals")

        fig.legend(patches, labels)
        plt.tight_layout()
        return fig
