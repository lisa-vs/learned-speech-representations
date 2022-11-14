__author__ = "Lisa van Staden"

import random

from torch.utils import data

from data_loading.SpeakerDataset import SpeakerDataset


class BucketSampler(data.Sampler):

    def __init__(self, data_source: SpeakerDataset, num_buckets=4, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_buckets = num_buckets
        self.flat_buckets = []
        self.buckets = [[] for i in range(self.num_buckets)]
        self.shuffle_buckets = shuffle

        if isinstance(self, BucketSampler):
            self.flat_buckets = list(range(len(self.data_source)))
            self._allocate_buckets()

    def __iter__(self):
        if self.shuffle_buckets:
            self.shuffle()
        return iter(sum(self.buckets, []))

    def __len__(self):
        return len(self.data_source)

    def shuffle(self):
        for i in range(self.num_buckets):
            random.shuffle(self.buckets[i])

    def _allocate_buckets(self):
        seq_lengths = [(i, self.data_source[i].get("X_length")) for i in self.flat_buckets]
        lengths = list(seq_len for (i, seq_len) in seq_lengths)
        lengths.sort()
        bucket_means = []
        if len(lengths) < self.num_buckets:
            print("Error: Too many buckets")
        for i in range(self.num_buckets):
            bucket_means.append(lengths[(i + 1) * int(len(lengths) / (self.num_buckets + 1))])

        for (i, seq_len) in seq_lengths:
            dists = [abs(seq_len - bm) for bm in bucket_means]
            index = dists.index(min(dists))
            self.buckets[index].append(i)
