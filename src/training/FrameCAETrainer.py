__author__ = "Lisa van Staden"

import json
import numpy as np
import torch

from data_loading.FrameDataLoader import FrameDataLoader
from data_loading.SpeakerDataLoader import SpeakerDataLoader
from training.FrameAETrainer import FrameAETrainer
from torch.nn import functional as F


class FrameCAETrainer(FrameAETrainer):

    def __init__(self, config, checkpoint_path="cae", language="english", config_key="frame_cae", reverse=True):
        super().__init__(config, checkpoint_path=checkpoint_path, language=language, config_key=config_key)
        self.reverse = reverse

    def load_pretrained_ae(self, ae_path):

        checkpoint = torch.load(ae_path)

        self.encoder.load_state_dict(checkpoint['enc_model_state_dict'])
        self.decoder.load_state_dict(checkpoint['dec_model_state_dict'])

    def _train_step(self):

        if self.train_data_loader is None:
            batch_size = self.config[self.config_key]["batch_size"]
            self.train_data_loader = FrameDataLoader(dataset_type="training", batch_size=batch_size,
                                                     language=self.language, pairs=True, reverse_pairs=self.reverse)

        self.encoder.train()
        self.decoder.train()

        epoch_loss = 0
        num_batches = 0

        for batch in self.train_data_loader:
            self.optimizer.zero_grad()
            [x, y] = batch
            x = x.to(self.device)

            z = self.encoder(x).to(self.device)
            y_pred = self.decoder(z)

            loss = F.mse_loss(y.to(self.device), y_pred)
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        return epoch_loss / num_batches

    def save_features(self, data_loader, save_path):
        self.encoder.eval()
        feature_dict = {}

        z_dim = self.config[self.config_key]["z_dim"]

        for batch in data_loader:
            word_x = batch.X.to(self.device)
            word_x = word_x.reshape((-1, word_x.size()[-1]))
            word_z = torch.empty(size=(word_x.size()[0], z_dim))

            with torch.no_grad():
                for i in range(len(word_x)):
                    word_z[i] = self.encoder(word_x[i])

            key = batch.utt_key[0]
            feature_dict[key] = word_z.cpu().numpy()

        np.savez_compressed(save_path, **feature_dict)


if __name__ == '__main__':
    with open('/home/config/herman_model_config.json') as config_file:
        config = json.load(config_file)

        for i in range(1, 4):

            trainer = FrameAETrainer(config, "/home/saved_models/frame_ae", language="english")
            trainer.build()
            trainer.train()
            trainer.save_model(f"frame_ae_{i}")

            trainer = FrameCAETrainer(config, "/home/saved_models/frame_cae", language="english")
            trainer.build()
            trainer.load_pretrained_ae(f"/home/saved_models/frame_ae/frame_ae_{i}.ckpt")
            trainer.train()
            trainer.save_model(f"frame_cae_{i}")

            feature_dl = SpeakerDataLoader("training", batch_size=1, max_seq_len=100, language="xitsonga", dframe=39)
            trainer.save_features(feature_dl, f'cae_feats_train_x_from_e_{i}')

            feature_dl = SpeakerDataLoader("test", batch_size=1, max_seq_len=100, language="xitsonga", dframe=39)
            trainer.save_features(feature_dl, f'cae_feats_test_x_from_e_{i}')




