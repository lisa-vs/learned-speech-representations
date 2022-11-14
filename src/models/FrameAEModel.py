__author__ = "Lisa van Staden"

from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_dim, enc_dim, z_dim, num_layers=6):
        super().__init__()
        self.encode_layers = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else enc_dim, enc_dim) for i in range(num_layers)]
        )
        self.relu = nn.ReLU(inplace=True)
        self.feature_layer = nn.Linear(enc_dim, z_dim)

    def forward(self, frame_x):
        for layer in self.encode_layers:
            frame_x = layer(frame_x)
            frame_x = self.relu(frame_x)
        z = self.feature_layer(frame_x)

        return z


class Decoder(nn.Module):

    def __init__(self, z_dim, dec_dim, input_dim, num_layers=6):
        super().__init__()
        self.decode_layers = nn.ModuleList(
            [nn.Linear(z_dim if i == 0 else dec_dim, dec_dim if i < num_layers - 1 else input_dim) for i in range(num_layers)]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, frame_y):
        for layer in self.decode_layers:
            frame_y = layer(frame_y)
            frame_y = self.relu(frame_y)

        return frame_y
