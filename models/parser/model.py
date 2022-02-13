import torch
import torch.nn as nn
import torch.nn.functional as F

from models.parser.nn import DoubleSeq2seq
from utils import check_entries, invert, EPS, INF


class ParserModel(nn.Module):
    """Model interface for seq2seq parser"""

    def __init__(self, cfg):
        super().__init__()
        self.seq2seq = nn.ModuleDict({t: DoubleSeq2seq(cfg) for t in ["statement", "metaconcept", "question"]})

    @property
    def word_entries(self):
        return self.seq2seq['question'].word_entries

    @property
    def program_entries(self):
        return self.seq2seq['question'].program_entries

    def loss(self, output, target):
        return F.nll_loss(output[:, :-1].contiguous().view(-1, output.shape[-1]).clamp(min=-INF, max=-EPS),
            target[:, 1:].contiguous().view(-1))

    def forward(self, inputs):
        check_entries(self.seq2seq['question'].word_entries, inputs['info']['word_entries'][0])
        check_entries(self.seq2seq['question'].program_entries, inputs['info']['program_entries'][0])
        outputs = {}
        for t in ["statement", "metaconcept", "question"]:
            seq2seq = self.seq2seq[t]
            if f"{t}_encoded" in inputs:
                lengths_sorted, index_sorted = torch.tensor(inputs[f"{t}_length"]).sort(-1, descending=True)
                index_inverted = invert(index_sorted)
                info = {k: v[0] for k, v in inputs['info'].items()}
                if self.training:
                    encoded = inputs[f"{t}_encoded"][index_sorted]
                    target = inputs[f"{t}_target"][index_sorted]
                    end = seq2seq(encoded, target, lengths_sorted, info=info)
                    outputs[f"{t}_end"] = end[index_inverted]
                    outputs[f"{t}_predicted"] = end[index_inverted].max(-1).indices
                    outputs[f"{t}_loss"] = self.loss(end, target)
                else:
                    encoded = inputs[f"{t}_encoded"][index_sorted]
                    end = seq2seq.sample_output(encoded, lengths_sorted, info=info)
                    outputs[f"{t}_predicted"] = end[index_inverted]
        return outputs

    def callback(self, inputs, outputs):
        pass
