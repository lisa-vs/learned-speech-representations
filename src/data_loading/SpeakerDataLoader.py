import os
import random
import glob

__author__ = "Lisa van Staden"

from torch.utils import data
import json
import numpy as np
import sys

from data_loading.BucketSampler import BucketSampler
from data_loading.SpeakerDataset import SpeakerDataset
from data_loading.SpeakerBatch import SpeakerBatch


class SpeakerDataLoader(data.DataLoader):
    dataset: SpeakerDataset

    def __init__(self, dataset_type, batch_size=1, pairs=False, num_buckets=3, language="english", for_analysis=False,
                 max_seq_len=100, dframe=39):
        """
        :param dataset_type: validation or training
        :param batch_size: Set batch_size=0 for no mini-batching
        """

        if os.getcwd() == "/home":
            root = "/home"
        else:
            root = "../.."
        npz = None
        with open(root + "/config/data_paths.json") as paths_file:
            path_dict = json.load(paths_file)
            if dataset_type == "training":
                npz = np.load(root + path_dict["{0}_train_data".format(language)])
            elif dataset_type == "validation" and not language.startswith("xitsonga"):
                npz = np.load(root + path_dict["{0}_validation_data".format(language)])
            elif dataset_type == "test":
                npz = np.load(root + path_dict["{0}_test_data".format(language)])

            else:
                sys.exit("Invalid dataset type given.")

        if npz is not None:
            self.dataset = SpeakerDataset(npz, language=language, pairs=pairs,
                                          max_seq_len=max_seq_len, d_frame=dframe)

            if batch_size == 0:
                batch_size = len(self.dataset)

            if (dataset_type != "training" or for_analysis) and language != 'english_full':
                super(SpeakerDataLoader, self).__init__(self.dataset, shuffle=False,
                                                        collate_fn=(lambda sp: SpeakerBatch(sp)),
                                                        batch_size=batch_size, drop_last=True)
            else:
                self.sampler = BucketSampler(self.dataset, num_buckets)
                super(SpeakerDataLoader, self).__init__(self.dataset,
                                                        sampler=self.sampler,
                                                        collate_fn=(lambda sp: SpeakerBatch(sp)), batch_size=batch_size,
                                                        drop_last=True)
