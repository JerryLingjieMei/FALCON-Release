import torch
from matplotlib import pyplot as plt

from utils import to_cpu_detach
from visualization.visualizer import Visualizer


class HypernymMatrixVisualizer(Visualizer):

    per_batch = False
    def visualize(self, results, model, iteration, **kwargs):
        labels = to_cpu_detach(results["label"])
        logits = to_cpu_detach(results["logit"])
        embeddings = results["embedding"]
        hyper_indices = torch.tensor([labels.index(min(self.dataset.hypo2hyper[_])) for _ in labels[:-1]])

        self.hypernym_matrix(labels, logits, iteration)
        # noinspection PyTypeChecker
        self.hypernym_logits(hyper_indices, labels, logits, iteration)
        self.hypernym_dimension(model, hyper_indices, embeddings, iteration)
        self.hypernym_occupancy(model, hyper_indices, embeddings, iteration)

    def hypernym_dimension(self, model, hyper_indices, embeddings, iteration):
        # hypernym dimension
        if model.rep == "box":
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            dim = embeddings.shape[-1] // 2
            offset = torch.arange(dim)
            xs = (embeddings[:-1, offset] - embeddings[:-1, dim + offset]).flatten().tolist()
            hyper_embeddings = embeddings[hyper_indices]
            ys = (hyper_embeddings[:, offset] - hyper_embeddings[:, dim + offset]).flatten().tolist()
            ax.hist2d(xs, ys, bins=40)

            ax.set_aspect('equal', 'box')
            ax.set_xlabel("Hyponym")
            ax.set_ylabel("Hypernym")
            ax.set_title("Hypernym and hyponym start in dimension all.")
            fig.tight_layout()
            self.summary_writer.add_figure(f"hypernym_dimension/{self.dataset.tag}/start", fig, iteration)

            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)

            dim = embeddings.shape[-1] // 2
            offset = torch.arange(dim)
            xs = (embeddings[:-1, offset] + embeddings[:-1, dim + offset]).flatten().tolist()
            hyper_embeddings = embeddings[hyper_indices]
            ys = (hyper_embeddings[:, offset] + hyper_embeddings[:, dim + offset]).flatten().tolist()
            ax.hist2d(xs, ys, bins=40)

            ax.set_aspect('equal', 'box')
            ax.set_xlabel("Hyponym")
            ax.set_ylabel("Hypernym")
            ax.set_title("Hypernym and hyponym end in all dimension.s")
            fig.tight_layout()
            self.summary_writer.add_figure(f"hypernym_dimension/{self.dataset.tag}/end", fig, iteration)
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            xs = embeddings[:-1].flatten().tolist()
            hyper_embeddings = embeddings[hyper_indices]
            ys = hyper_embeddings.flatten().tolist()
            ax.hist2d(xs, ys, bins=40)

            ax.set_aspect('equal', 'box')
            ax.set_xlabel("Hyponym")
            ax.set_ylabel("Hypernym")
            ax.set_title("Hypernym and hyponym in dimension all.")
            fig.tight_layout()
            self.summary_writer.add_figure(f"hypernym_dimension/{self.dataset.tag}/all", fig, iteration)

    def hypernym_logits(self, hyper_indices, labels, logits, iteration):
        # immediate hypernym logits
        hyper_logits = logits[:, :-1].gather(0, hyper_indices.unsqueeze(0))
        self.summary_writer.add_histogram(f"hypernym_logits/{self.dataset.tag}/immediate", hyper_logits,
            iteration)
        # non-hypernym logits
        non_hyper_indices, non_hyper_hypernyms = [], []
        for ii, i in enumerate(labels[:-1]):
            non_hypers = list(range(len(labels)))
            for _ in self.dataset.hypo2hyper[i] + [i]:
                non_hypers.remove(labels.index(_))
            non_hyper_hypernyms.extend(non_hypers)
            non_hyper_indices.extend([ii] * len(non_hypers))
        non_hyper_logits = logits[non_hyper_hypernyms, non_hyper_indices]
        self.summary_writer.add_histogram(f"hypernym_logits/{self.dataset.tag}/non-hypernym", non_hyper_logits,
            iteration)

    def hypernym_matrix(self, labels, logits, iteration):
        # Hypernym matrix
        fig, ax = plt.subplots()
        fig.set_size_inches(100, 100)
        plt.imshow(logits)
        ax.set_xticks(torch.arange(len(labels)))
        ax.set_yticks(torch.arange(len(labels)))
        ax.set_xticklabels([self.dataset.named_entries_[_] for _ in labels])
        ax.set_yticklabels([self.dataset.named_entries_[_] for _ in labels])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if logits[i, j] > -5:
                    ax.text(j, i, f"{logits[i, j].item():.01f}", ha="center", va="center", color="w",
                        fontsize=6)
        self.summary_writer.add_figure(f"hypernym_matrix/{self.dataset.tag}", fig, iteration)

    def hypernym_occupancy(self, model, hyper_indices, embeddings, iteration):
        hypo_embeddings = embeddings[:-1]
        hyper_embeddings = embeddings[hyper_indices]
        intersection = model.program_executor.entailment.entailment.intersection(hypo_embeddings, hyper_embeddings)
        dim = embeddings.shape[-1] // 2
        occupancy = intersection[:, dim:] / hyper_embeddings[:, dim:]
        self.summary_writer.add_histogram(f"hypernym_occupancy/{self.dataset.tag}/all", occupancy, iteration)
        start_occupancy = (intersection[:, :dim] - intersection[:, dim:] - hyper_embeddings[:,
        :dim] + hyper_embeddings[:, dim:]) / hyper_embeddings[:, dim:] / 2
        self.summary_writer.add_histogram(f"hypernym_occupancy/{self.dataset.tag}/start", start_occupancy)
        end_occupancy = (intersection[:, :dim] + intersection[:, dim:] - hyper_embeddings[:,
        :dim] - hyper_embeddings[:, dim:]) / hyper_embeddings[:, dim:] / 2
        self.summary_writer.add_histogram(f"hypernym_occupancy/{self.dataset.tag}/end", end_occupancy,
            iteration)
