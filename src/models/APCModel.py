__author__ = "Lisa van Staden"

from torch import nn
import torch.nn.functional as F


class APCEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0, residual=False):
        super().__init__()
        self.rnn_layers = nn.ModuleList(
            [nn.GRU(input_dim, hidden_dim)] + [nn.GRU(hidden_dim, hidden_dim)] * (num_layers - 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x):
        for i, rnn in enumerate(self.rnn_layers):
            h, _ = rnn(x)
            if i + 1 < len(self.rnn_layers):
                h = self.dropout(h)
            if x.size()[-1] == h.size()[-1] and self.residual:
                x = h + x
            else:
                x = h
        return h


class APCPredictor(nn.Module):

    def __init__(self, hidden_dim, input_dim, num_pred_steps):
        super(APCPredictor, self).__init__()
        self.W = nn.Linear(hidden_dim, input_dim)
        self.num_pred_steps = num_pred_steps

    def forward(self, x, h):
        x_n = x[:, self.num_pred_steps:]
        # h = h.transpose(1, 2)
        if self.num_pred_steps == 0:
            y = self.W(h)
        else:
            y = self.W(h)[:, :-self.num_pred_steps]
        # y = y.transpose(1, 2)
        l1_loss = F.l1_loss(x_n, y, reduction='sum')
        return l1_loss
