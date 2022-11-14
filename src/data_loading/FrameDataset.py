__author__ = "Lisa van Staden"

import torch
from torch.utils import data
import numpy as np


class FrameDataset(data.Dataset):

    def __init__(self, x_frames, y_frames=None, reverse=False):

        self.x_frames = torch.as_tensor(x_frames, dtype=torch.float32)
        print(self.x_frames.size())
        self.include_y_frames = y_frames is not None
        self.reverse = reverse and self.include_y_frames
        if self.include_y_frames:
            self.y_frames = torch.as_tensor(y_frames, dtype=torch.float32)

    def __len__(self):

        if self.reverse:
            return len(self.x_frames) * 2
        else:
            return len(self.x_frames)

    def __getitem__(self, index):

        num_x_items = len(self.x_frames)
        if self.reverse and index >= num_x_items:
            x = self.y_frames[index - num_x_items]
            y = self.x_frames[index - num_x_items]
        elif self.include_y_frames:
            x = self.x_frames[index]
            y = self.y_frames[index]
        else:
            x = self.x_frames[index]
            y = x

        return [x, y]

