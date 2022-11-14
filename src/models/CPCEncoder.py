__author__ = "Lisa van Staden"

import math

import torch
from torch import nn
from torch.nn import functional as F


class CPCEncoder(nn.Module):

    def __init__(self, input_dim, enc_dim, z_dim, c_dim):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.long_type = torch.long if self.device == 'cuda' else torch.LongTensor

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, enc_dim, bias=False),
            nn.LayerNorm(enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, enc_dim, bias=False),
            nn.LayerNorm(enc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(enc_dim, enc_dim, bias=False),
            nn.LayerNorm(enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, enc_dim, bias=False),
            nn.LayerNorm(enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, z_dim)
        )

        self.sum_rnn = nn.LSTM(input_size=z_dim, hidden_size=c_dim, batch_first=True)

    def forward(self, x):
        z = self.encoder(x)
        c, _ = self.sum_rnn(z)

        return z, c


class CPCPredictor(nn.Module):

    def __init__(self, z_dim, c_dim, num_pred_steps):
        super().__init__()

        self.num_pred_steps = num_pred_steps
        self.W = nn.ModuleList([
            nn.Linear(in_features=c_dim, out_features=z_dim) for _ in range(num_pred_steps)
        ])

    def forward(self, z, c, num_negatives, num_utts=12, num_speakers=12):
        pred_c = c[:, :-self.num_pred_steps, :]
        sample_len = z.size()[1] - self.num_pred_steps

        [batch_size, _, z_dim] = z.size()

        losses = []
        accs = []

        for k in range(self.num_pred_steps):
            z_k = z[:, k + 1: sample_len + k + 1, :]

            Wk_c = self.W[k](pred_c)
            # print('Wkc', Wk_c.size())
            Wk_c = Wk_c.view(12, 12, -1, z_dim)
            # print('Wkc', Wk_c.size())
            batch_index = torch.randint(
                0, num_utts,
                size=(
                    num_utts,
                    num_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, num_utts, num_negatives, 1
            )

            seq_index = torch.randint(
                0, sample_len,
                size=(
                    num_speakers,
                    num_utts,
                    num_negatives,
                    sample_len
                ),
                device=z.device
            )
            seq_index += torch.arange(sample_len, device=z.device)
            seq_index = torch.remainder(seq_index, sample_len)

            speaker_index = torch.arange(num_speakers, device=z.device)
            speaker_index = speaker_index.view(-1, 1, 1, 1)

            z_negatives = z_k.reshape(num_speakers, num_utts, sample_len, -1)[speaker_index, batch_index, seq_index, :]

            z_all = torch.cat((z_k.reshape(12, 12, 1, -1, z_dim), z_negatives), dim=2)
            f = torch.sum(z_all * Wk_c.unsqueeze(2) / math.sqrt(z_dim), dim=-1)
            f = f.view(batch_size, num_negatives + 1, -1)
            labels = torch.zeros(batch_size, sample_len, dtype=int).to(z.device)
            loss = F.cross_entropy(f, labels)
            acc = torch.mean((f.argmax(dim=1) == labels).float())
            losses.append(loss)
            accs.append(acc)

        loss = torch.stack(losses).mean()

        return loss, accs
