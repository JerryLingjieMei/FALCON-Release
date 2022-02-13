import torch.nn as nn


class Encoder(nn.Module):
    """Encoder RNN module"""

    def __init__(self, cfg):
        super().__init__()
        language_cfg = cfg.LANGUAGE
        self.embedding = nn.Embedding(language_cfg.WORD_ENTRIES, language_cfg.WORD_CHANNELS)
        if language_cfg.RNN_CELL == "lstm":
            rnn_cell = nn.LSTM
        elif language_cfg.RNN_CELL == "gru":
            rnn_cell = nn.GRU
        else:
            raise NotImplementedError(f"Unsupported rnn cell {language_cfg.RNN_CELL}.")
        self.rnn = rnn_cell(language_cfg.WORD_CHANNELS, language_cfg.HIDDEN_CHANNELS, language_cfg.N_LAYERS,
            batch_first=True, bidirectional=language_cfg.BIDIRECTIONAL)

    def forward(self, input_var, input_lengths=None):
        """
        To do: add input, output dimensions to docstring
        """
        embedded = self.embedding(input_var)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
