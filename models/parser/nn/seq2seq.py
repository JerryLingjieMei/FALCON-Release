import torch
import torch.nn as nn

from models.parser.nn.decoder import Decoder
from models.parser.nn.encoder import Encoder


class Seq2seq(nn.Module):
    """Seq2seq model module
    To do: add docstring to methods
    """

    def __init__(self, cfg):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @property
    def word_entries(self):
        return len(self.encoder.embedding.weight)

    @property
    def program_entries(self):
        return len(self.decoder.out_linear.weight)

    def forward(self, x, y, input_lengths=None, info=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return decoder_outputs

    def sample_output(self, x, input_lengths=None, info=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        output_symbols, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden, info)
        return torch.stack(output_symbols).transpose(0, 1)


class DoubleSeq2seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder_type = Decoder(cfg)
        self.decoder_argument = Decoder(cfg)

    @property
    def word_entries(self):
        return len(self.encoder.embedding.weight)

    @property
    def program_entries(self):
        return len(self.decoder_type.out_linear.weight)

    def forward(self, x, y, input_lengths=None, info=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        decoder_outputs_type, _ = self.decoder_type(y[..., 0], encoder_outputs, encoder_hidden)
        decoder_outputs_argument, _ = self.decoder_argument(y[..., 1], encoder_outputs, encoder_hidden)
        return torch.stack([decoder_outputs_type.contiguous(), decoder_outputs_argument.contiguous()], -2)

    def sample_output(self, x, input_lengths=None, info=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        output_types, _ = self.decoder_type.forward_sample(encoder_outputs, encoder_hidden, info=info)
        output_arguments, _ = self.decoder_argument.forward_sample(encoder_outputs, encoder_hidden, info=info)
        return torch.stack(
            [torch.stack(output_types).transpose(0, 1), torch.stack(output_arguments).transpose(0, 1)], -1)
