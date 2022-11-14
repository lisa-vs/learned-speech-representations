__author__ = "Lisa van Staden"

import os
import json
import sys

import numpy as np
from torch.utils import data

from data_loading.FrameDataset import FrameDataset


class FrameDataLoader(data.DataLoader):

    def __init__(self, dataset_type, batch_size, language="english", pairs=False, reverse_pairs=False):

        if os.getcwd() == "/home":
            root = "/home"
        else:
            root = "../.."
        with open(root + "/config/data_paths.json") as paths_file:
            path_dict = json.load(paths_file)
            if dataset_type == "training":
                x_frames = np.load(root + path_dict["{0}_word_1_train_data".format(language)])
                y_frames = None if not pairs else np.load(root + path_dict["{0}_word_2_train_data".format(language)])
            elif dataset_type == "validation":
                x_frames = np.load(root + path_dict["{0}_word_1_train_data".format(language)])
                y_frames = None if not pairs else np.load(root + path_dict["{0}_word_2_train_data".format(language)])
            elif dataset_type == "test":
                x_frames = np.load(root + path_dict["{0}_word_1_train_data".format(language)])
                y_frames = None if not pairs else np.load(root + path_dict["{0}_word_2_train_data".format(language)])
            else:
                sys.exit("Invalid dataset type given.")

        dataset = FrameDataset(x_frames=x_frames, y_frames=y_frames, reverse=reverse_pairs)

        super().__init__(dataset=dataset, shuffle=dataset_type == "training", batch_size=batch_size)
