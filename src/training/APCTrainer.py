__author__ = "Lisa van Staden"

import json
import random

import numpy as np

import torch
from torch import optim
from torch import nn

from data_loading.SpeakerDataLoader import SpeakerDataLoader
from models.APCModel import APCEncoder, APCPredictor
from training import helpers
from training.CPCTrainer import CPCTrainer


class APCTrainer(CPCTrainer):

    def __init__(self, config, checkpoint_path, language='english', config_key='apc'):
        super().__init__(config, checkpoint_path, language, config_key)
        self.config_key = config_key
        self.include_aux_loss = self.config[self.config_key]["include_aux_loss"]
        self.encoder_aux = None
        self.predictor_aux = None

    def build(self):
        input_dim = self.config[self.config_key]["input_dim"]
        hidden_dim = self.config[self.config_key]["hidden_dim"]
        num_pred_steps = self.config[self.config_key]["num_pred_steps"]
        learning_rate = self.config[self.config_key]["learning_rate"]
        num_layers = self.config[self.config_key]["num_layers"]

        self.encoder = APCEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
        self.predictor = APCPredictor(hidden_dim=hidden_dim, input_dim=input_dim,
                                      num_pred_steps=num_pred_steps).to(self.device)

        if self.include_aux_loss:
            num_pred_steps_aux = self.config[self.config_key]["num_pred_steps_aux"]
            self.encoder_aux = APCEncoder(input_dim=input_dim, hidden_dim=hidden_dim,
                                          num_layers=num_layers).to(self.device)
            self.predictor_aux = APCPredictor(hidden_dim=hidden_dim, input_dim=input_dim,
                                              num_pred_steps=num_pred_steps_aux).to(self.device)

        model_params = list(self.encoder.parameters()) + list(self.predictor.parameters())

        if self.include_aux_loss:
            model_params += list(self.encoder_aux.parameters()) + list(self.predictor_aux.parameters())
        self.optimizer = optim.Adam(model_params, lr=learning_rate)

    def train(self, validate=True, loss_limit=-1.0, save_best_ap=False, evaluate_after=-1):

        batch_size = self.config[self.config_key]["batch_size"] or 100
        num_buckets = self.config[self.config_key].get("num_buckets")

        checkpoint_interval = self.config[self.config_key]["checkpoint_interval"]
        num_epochs = self.config[self.config_key]["num_epochs"]
        starting_epoch = self.epoch

        if self.train_data_loader is None:

            if "aligned" in self.language:
                self.train_data_loader = SpeakerDataLoader(dataset_type="training",
                                                           language=self.language,
                                                           batch_size=batch_size,
                                                           num_buckets=num_buckets, max_seq_len=100,
                                                           speaker_sampler=False,
                                                           dframe=13)
            else:
                self.train_data_loader = SpeakerDataLoader(dataset_type="training",
                                                           language=self.language,
                                                           batch_size=batch_size,
                                                           num_buckets=num_buckets, max_seq_len=100,
                                                           speaker_sampler=True,
                                                           num_utterances=9, include_speaker_ids=True, dframe=13)

        if self.dev_data_loader is None:
            self.dev_data_loader = SpeakerDataLoader("validation", batch_size=1, max_seq_len=100, dframe=13)

        for epoch in range(starting_epoch, num_epochs):
            self.encoder.train()
            self.predictor.train()
            losses = []
            for batch in self.train_data_loader:
                losses.append(self._train_step(batch))

            if (epoch + 1) % checkpoint_interval == 0:
                print(f'Epoch: {epoch}')
                print(f'Train Loss: {torch.stack(losses).mean()}')

            if (epoch +  1) % checkpoint_interval == 0:
                self.epoch = epoch
                self.save_checkpoint()
                self.evaluate_downsampled_features()

    def _train_step(self, batch):
        if self.include_aux_loss:
            aux_seq_len = self.config[self.config_key]["aux_seq_len"]
            num_aux_steps_back = self.config[self.config_key]["num_aux_steps_back"]
            num_aux_anchors = self.config[self.config_key]["num_aux_anchors"]
            aux_loss_weight = self.config[self.config_key]["aux_loss_weight"]
        self.optimizer.zero_grad()

        x = batch.X.to(self.device)
        h = self.encoder(x)
        loss = self.predictor(x, h)

        if self.include_aux_loss:
            x_idxs = [i for i in range(num_aux_steps_back, x.size()[1])]
            anchors = random.sample(x_idxs, num_aux_anchors)  # choose M anchor indices from x
            anchor_cum_loss = 0
            for a in anchors:
                x_anchor = x[:, a - num_aux_steps_back: a - num_aux_steps_back + aux_seq_len, :]
                h_anchor = self.encoder_aux(x_anchor)
                anchor_cum_loss += self.predictor_aux(x_anchor, h_anchor)
            loss += aux_loss_weight * anchor_cum_loss / num_aux_anchors
        loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1)
        if self.include_aux_loss:
            nn.utils.clip_grad_norm_(self.encoder_aux.parameters(), 1)
            nn.utils.clip_grad_norm_(self.predictor_aux.parameters(), 1)
        self.optimizer.step()
        return loss

    def evaluate_downsampled_features(self, verbose=True):
        cs = []
        labels = []
        for batch in self.dev_data_loader:
            X = batch.X.to(self.device)

            with torch.no_grad():
                c = self.encoder(X)
                cs.append(c.squeeze().cpu().numpy())
                labels.append(batch.labels[0])
        downsampled = helpers.downsample(cs, labels)
        ap, prb = helpers.calculate_ap_and_prb(downsampled, labels)

        if verbose:
            print("Downsampled AP:", ap)
        return ap

    def save_features(self, data_loader, save_path):
        print("saving features")
        self.encoder.eval()
        feature_dict = {}

        for batch in data_loader:
            X = batch.X.to(self.device)
            with torch.no_grad():
                c = self.encoder(X)

                for i in range(len(batch.utt_key)):
                    feature_dict[f"{batch.utt_key[i]}.c"] = c[i].cpu().numpy()

        np.savez_compressed(save_path, **feature_dict)


def main():
    with open('/home/config/config.json') as config_file:
        config = json.load(config_file)

        for i in range(1, 4):
            trainer = APCTrainer(config, "/home/saved_models/apc", config_key='apc', language="english_full",
                                 include_aux_loss=True)
            trainer.build()
            #trainer.train()
            #trainer.evaluate_downsampled_features()
            # trainer.save_model(f"apc_{i}")
            feature_dl = SpeakerDataLoader("training", batch_size=1, max_seq_len=100, dframe=13, language="xitsonga")
            # trainer.save_features(feature_dl, f'apc_feats_train_e_to_x_{i}')
            feature_dl = SpeakerDataLoader("test", batch_size=1, max_seq_len=100, dframe=13, language="xitsonga")
            # trainer.save_features(feature_dl, f'apc_feats_test_e_to_x_{i}')


if __name__ == '__main__':
    main()
