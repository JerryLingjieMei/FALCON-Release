import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from utils import to_cpu_detach
from visualization.visualizer import Visualizer


class HypernymLogitVisualizer(Visualizer):
    per_batch = False

    def visualize(self, results, model, iteration, **kwargs):
        logits = to_cpu_detach(results["logit"])
        names = ['train', 'test', 'all']
        fs = [lambda c: True,
            lambda c: self.dataset.concept_split_specs[c] <= 1 or self.dataset.concept_split_specs[
                c] == getattr(self.dataset, '_hierarchy_n', 1) + 1,
            lambda c: self.dataset.concept2splits[c] == 0]
        for name, f in zip(names, fs):
            self.hypernym_logits(logits, name, f, iteration)

    def hypernym_logits(self, logits, name, f, iteration):
        hypernym_pairs = []
        for hyper, hypos in self.dataset.hyper2hypo.items():
            for hypo in hypos:
                if f(hyper) and f(hypo):
                    hypernym_pairs.append((hypo, hyper))
        hypos, hypers = zip(*hypernym_pairs)
        hypernym_logits = logits[hypos, hypers]
        self.summary_writer.add_histogram(f"{self.dataset.tag}/hypernym_logits", hypernym_logits, iteration)

        all_pairs = [(i, j) for i in self.dataset.concepts for j in self.dataset.concepts if
            i != j and f(i) and f(j)]
        non_hypernym_pairs = list(set(all_pairs).difference(hypernym_pairs))
        hypos, hypers = zip(*non_hypernym_pairs)
        non_hypernym_logits = logits[hypos, hypers]
        self.summary_writer.add_histogram(f"{self.dataset.tag}/non-hypernym_logits", non_hypernym_logits,
            iteration)

        all_labels = [1] * len(hypernym_logits) + [0] * len(non_hypernym_logits)
        all_logits = torch.cat([hypernym_logits, non_hypernym_logits])
        self.plot_roc(all_labels, all_logits, name, iteration)

    def plot_roc(self, labels, logits, name, iteration):
        fpr, tpr, _ = roc_curve(labels, logits)
        auc = roc_auc_score(labels, logits)
        self.summary_writer.add_scalar(f"{self.dataset.tag}/auc-{name}", auc)

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (area = {auc:0.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC of hypernym prediction")
        plt.legend(loc="lower right")
        self.summary_writer.add_figure(f"{self.dataset.tag}/roc_curve-{name}", fig, iteration)
