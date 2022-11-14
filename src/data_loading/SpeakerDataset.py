__author__ = "Lisa van Staden"

from torch.utils import data
import torch


class SpeakerDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, npz, language="english", d_frame=26, max_seq_len=100, pairs=False):

        print("Creating Speaker Dataset")

        self.language = language
        self.pairs = pairs
        self.d_frame = d_frame
        self.max_seq_len = max_seq_len
        self.utt_keys = list(set([key.split(".")[0] for key in sorted(npz)]))
        self.labels = [utt_key.split("_")[0] for utt_key in self.utt_keys]
        self.data = {}
        for key in self.utt_keys:
            if "_cpc_feats" in self.language or "capc_feats" in self.language:
                self.data[key] = torch.as_tensor(npz[f"{key}.c"][:self.max_seq_len, :])
            elif "aligned" in self.language:
                self.data[f"{key}.X"] = torch.as_tensor(npz[f"{key}.X"][:self.max_seq_len, :d_frame],
                                                        dtype=torch.float32)
                self.data[f"{key}.Y"] = torch.as_tensor(npz[f"{key}.Y"][:self.max_seq_len, :d_frame],
                                                        dtype=torch.float32)
            else:
                self.data[key] = torch.as_tensor(npz[key][:self.max_seq_len, :d_frame], dtype=torch.float32)

    def __len__(self):
        return len(self.utt_keys)

    def __getitem__(self, index):
        sample = {"index": index}
        if "aligned" in self.language:
            utt_key1 = self.utt_keys[index]
            sample["Y"] = self.data[f"{utt_key1}.Y"]
            sample["Y_length"] = len(sample["Y"])
            sample["X"] = self.data[f"{utt_key1}.X"]
            sample["X_length"] = len(sample["X"])
            sample["word"] = utt_key1.split("_")[0]
            sample["utt_key"] = utt_key1
        else:
            utt_key1 = self.utt_keys[index]

        if "aligned" not in self.language:
            sample["utt_key"] = utt_key1

            sample["X"] = self.data[utt_key1]
            sample["X_length"] = len(sample["X"])
            sample["word"] = utt_key1.split("_")[0]

            if self.language == "english" or self.language == "english_cpc_feats" \
                    or self.language == "english_cae_feats" or self.language == "english_capc_feats":
                sample["speaker_X"] = utt_key1.split("_")[1][:3]
            elif self.language == "xitsonga_cpc":
                sample["speaker_X"] = utt_key1.split("-")[2]
            elif self.language.startswith("xitsonga"):
                sample["speaker_X"] = utt_key1.split("_")[1].split("-")[2]
            elif self.language == "hausa":
                sample["speaker_X"] = utt_key1.split("_")[1].lower()
            elif self.language == "english_full":
                sample["speaker_X"] = utt_key1.split("_")[0][:3]

        return sample
        # Select sample
