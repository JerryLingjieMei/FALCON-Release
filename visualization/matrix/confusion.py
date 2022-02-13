import matplotlib.pyplot as plt
import torch

from utils import nonzero
from visualization.visualizer import Visualizer


class ConfusionVisualizer(Visualizer):
    per_batch = False

    def visualize(self, results, model, iteration, **kwargs):
        confusion = torch.zeros(len(self.dataset.named_entries_), len(self.dataset.named_entries_)).long()
        for p, l in zip(results["positive"], results["true"]):
            row = torch.zeros_like(confusion[0:1]).index_fill_(1, p, 1)
            col = torch.zeros_like(confusion[:, 0:1]).index_fill(0, l, 1)
            confusion += row * col

        unique_labels = set()
        for l in results["true"]:
            unique_labels.update(l.tolist())
        unique_labels = list(sorted(unique_labels))
        unique_kinds = self.dataset.entry2kinds[unique_labels].unique()
        for kind in unique_kinds:
            kind_name = self.dataset.kinds_[kind]
            entries = set(nonzero(self.dataset.entry2kinds == kind))
            entries = entries.intersection(unique_labels)
            entries = list(sorted(entries))
            sub_confusion = confusion[entries][:, entries]

            fig, ax = plt.subplots()
            fig.set_size_inches(.5 * len(entries) + 5, .5 * len(entries) + 5)
            plt.imshow(sub_confusion)
            ax.set_xticks(torch.arange(len(entries)))
            ax.set_yticks(torch.arange(len(entries)))
            ax.set_xticklabels([self.dataset.named_entries_[_] for _ in entries])
            ax.set_yticklabels([self.dataset.named_entries_[_] for _ in entries])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(len(entries)):
                for j in range(len(entries)):
                    ax.text(j, i, sub_confusion[i, j].item(), ha="center", va="center", color="r", fontsize=20)

            plt.tight_layout()
            self.summary_writer.add_figure(f"{self.dataset.tag}/confusion/{kind_name}", fig, iteration)

        self.dataset.save(self.summary_writer.log_dir, {"mode": "confusion"}, iteration, confusion)
