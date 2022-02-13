import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import EPS, INF
from .attention import Attention


class Decoder(nn.Module):
    """Decoder RNN module
    To do: add docstring to methods
    """

    def __init__(self, cfg):
        super().__init__()
        language_cfg = cfg.LANGUAGE
        self.hidden_channels = language_cfg.HIDDEN_CHANNELS
        word_channels = language_cfg.WORD_CHANNELS
        self.bidirectional = language_cfg.BIDIRECTIONAL
        if self.bidirectional:
            self.hidden_channels *= 2
        use_attention = language_cfg.USE_ATTENTION
        self.program_entries = language_cfg.PROGRAM_ENTRIES
        self.embedding = nn.Embedding(self.program_entries, word_channels)
        if language_cfg.RNN_CELL == "lstm":
            rnn_cell = nn.LSTM
        elif language_cfg.RNN_CELL == "gru":
            rnn_cell = nn.GRU
        else:
            raise NotImplementedError(f"Unsupported rnn cell {language_cfg.RNN_CELL}.")
        self.rnn = rnn_cell(word_channels, self.hidden_channels, language_cfg.N_LAYERS, batch_first=True)
        self.out_linear = nn.Linear(self.hidden_channels, self.program_entries)
        if use_attention:
            self.attention = Attention(self.hidden_channels)
        else:
            self.attention = None

    def forward_step(self, input_var, hidden, encoder_outputs):
        embedded = self.embedding(input_var)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.attention is not None:
            output, attn = self.attention(output, encoder_outputs)

        output = self.out_linear(output.contiguous().view(-1, self.hidden_channels))
        predicted_softmax = F.log_softmax(output.view(input_var.size(0), input_var.size(1), -1), 2)
        return predicted_softmax, hidden, attn

    def forward(self, y, encoder_outputs, encoder_hidden):
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs, decoder_hidden, attn = self.forward_step(y, decoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_sample(self, encoder_outputs, encoder_hidden, reinforce_sample=False, info=None):
        start_id, end_id, max_length = map(info.get, ["start_id", "end_id", "max_length"])
        device = encoder_hidden[0].device
        if isinstance(encoder_hidden, tuple):
            batch_size = encoder_hidden[0].size(1)
        else:
            batch_size = encoder_hidden.size(1)
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_input = torch.LongTensor(batch_size, 1).fill_(start_id).to(device)

        l = -torch.ones(batch_size, self.program_entries).to(device) * INF
        l[:, start_id] = - EPS
        output_logprobs = [l]
        output_symbols = [torch.tensor([start_id] * batch_size).to(device)]
        output_lengths = np.array([max_length] * batch_size)

        def decode(i, output):
            output_logprobs.append(output.squeeze())
            symbols = output.topk(1)[1].view(batch_size, -1)
            output_symbols.append(symbols.squeeze())

            eos_batches = symbols.data.eq(end_id)
            if eos_batches.ndim > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((output_lengths > i) & eos_batches) != 0
                output_lengths[update_idx] = len(output_symbols)

            return symbols

        for i in range(max_length - 1):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                encoder_outputs)
            # noinspection PyTypeChecker
            decoder_input = decode(i, decoder_output)

        return output_symbols, output_logprobs

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
