from scipy import signal
from scipy.spatial.distance import pdist

from training import samediff

__author__ = "Lisa van Staden"

import torch
import numpy as np


def custom_mse(x, y, mask):
    return torch.mean(torch.sum(torch.mean(torch.pow(x - y, 2), -1), -1) / torch.sum(mask, 1))


def custom_mae(x, y, mask):
    return torch.mean(torch.sum(torch.mean(torch.abs(x - y), -1), -1) / torch.sum(mask, 1))


def calculate_ap_and_prb(z, labels):
    distances = pdist(z, metric="cosine")
    matches = samediff.generate_matches_array(labels)
    ap, prb = samediff.average_precision(
        distances[matches == True], distances[matches == False])
    return ap, prb


def downsample(in_features, labels, n_samples=10, d_frame=256, flatten_order="C"):
    downsampled_features = []
    for f in in_features:
        y = f[:, :d_frame].T
        downsampled_features.append(signal.resample(y, n_samples, axis=1).flatten(flatten_order))

    return downsampled_features


def print_string(values, k=1):
    m = np.mean(values)
    s = np.std(values)
    return f'$ {round(m * k, 2)} \pm {round(s * k, 2)} $'
