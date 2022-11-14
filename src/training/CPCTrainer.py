__author__ = "Lisa van Staden"

import json
import os
import numpy as np

import torch
from torch import optim
from torch import nn

from data_loading.SpeakerDataLoader import SpeakerDataLoader
from models.CPCEncoder import CPCEncoder, CPCPredictor
from training import helpers


class CPCTrainer:

    def __init__(self, config, checkpoint_path, language='english', config_key='cpc'):

        self.encoder = None
        self.predictor = None
        self.optimizer = None
        self.epoch = 0
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.language = language
        self.config_key = config_key

        self.train_data_loader = None
        self.dev_data_loader = None

    def load_checkpoint(self, epoch=-1, only_model=False):

        if epoch == -1:  # load latest checkpoint
            checkpoints = os.listdir(self.checkpoint_path)
            epochs = [int(c.split('_')[1].split('.')[0]) for c in checkpoints]
            epoch = max(epochs)

        checkpoint = torch.load(f'{self.checkpoint_path}/ckpt_{epoch}.pt')

        self.encoder.load_state_dict(checkpoint['enc_model_state_dict'])
        self.predictor.load_state_dict(checkpoint['pred_model_state_dict'])

        if not only_model:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

    def build(self):

        input_dim = self.config[self.config_key]["input_dim"]
        enc_dim = self.config[self.config_key]["enc_dim"]
        z_dim = self.config[self.config_key]["z_dim"]
        c_dim = self.config[self.config_key]["c_dim"]
        num_pred_steps = self.config[self.config_key]["num_pred_steps"]
        learning_rate = self.config[self.config_key]["learning_rate"]

        self.encoder = CPCEncoder(input_dim=input_dim, enc_dim=enc_dim, z_dim=z_dim, c_dim=c_dim).to(self.device)
        self.predictor = CPCPredictor(z_dim=z_dim, c_dim=c_dim, num_pred_steps=num_pred_steps).to(self.device)

        model_params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        self.optimizer = optim.Adam(model_params, lr=learning_rate)

    def train(self, validate=True, loss_limit=-1.0, save_best_ap=True):

        batch_size = self.config[self.config_key]["batch_size"] or 100
        num_buckets = self.config[self.config_key].get("num_buckets")

        checkpoint_interval = self.config[self.config_key]["checkpoint_interval"]
        num_epochs = self.config[self.config_key]["num_epochs"]
        num_negatives = self.config[self.config_key]["num_negatives"]
        milestones = self.config[self.config_key]["lr_milestones"]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones)
        starting_epoch = self.epoch

        self.train_data_loader = SpeakerDataLoader(dataset_type="training",
                                                   language=self.language,
                                                   batch_size=batch_size,
                                                   num_buckets=num_buckets, max_seq_len=100,
                                                   speaker_sampler=True,
                                                   num_utterances=12, include_speaker_ids=True, dframe=13)
        print("Using language:", self.language)
        if validate:
            self.dev_data_loader = SpeakerDataLoader("validation", batch_size=1, max_seq_len=100, dframe=13)

        best_ap = 0
        best_epoch = 0
        for epoch in range(starting_epoch, num_epochs):
            self.encoder.train()
            self.predictor.train()
            losses = []
            accs = []
            for batch in self.train_data_loader:
                self.optimizer.zero_grad()

                X = batch.X.to(self.device)
                z, c = self.encoder(X)
                loss, acc = self.predictor(z, c, num_negatives)
                losses.append(torch.tensor(loss))
                accs.append(torch.tensor(acc))

                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
                nn.utils.clip_grad_norm_(self.predictor.parameters(), 5)
                self.optimizer.step()
                scheduler.step()
            # dev_loss, dev_acc = self.evaluate()
            train_loss = torch.stack(losses).mean()

            if epoch % 200 == 0:
                print(f'Epoch: {epoch}')
                print(f'Train Loss: {train_loss}')
                print(f'Train Accuracy: {torch.stack(accs).mean(dim=0)}')

            if validate and (epoch % checkpoint_interval == 0) and epoch > 7000:
                self.epoch = epoch
                self.save_checkpoint()
                ap = self.evaluate_downsampled_features()

                if ap > best_ap:
                    best_ap = ap
                    best_epoch = epoch

            if train_loss < loss_limit:
                print("Loss limit reached.")
                self.epoch = epoch
                self.save_checkpoint()
                self.load_checkpoint(epoch)
                break

        if save_best_ap:
            self.load_checkpoint(best_epoch)

    def evaluate_downsampled_features(self, verbose=True):

        if self.dev_data_loader is None:
            self.dev_data_loader = SpeakerDataLoader("validation", batch_size=1, max_seq_len=100, dframe=13)
        cs = []
        labels = []
        for batch in self.dev_data_loader:
            X = batch.X.to(self.device)

            with torch.no_grad():
                _, c = self.encoder(X)
                cs.append(c.squeeze().cpu().numpy())
                labels.append(batch.labels[0])
        downsampled = helpers.downsample(cs, labels)
        ap, prb = helpers.calculate_ap_and_prb(downsampled, labels)

        if verbose:
            print("Downsampled AP:", ap)
        return ap

    def save_checkpoint(self):
        torch.save({'epoch': self.epoch + 1,
                    'enc_model_state_dict': self.encoder.state_dict(),
                    'pred_model_state_dict': self.predictor.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, f"{self.checkpoint_path}/ckpt_{self.epoch}.pt")

    def save_model(self, file_name):
        torch.save({'epoch': self.epoch + 1,
                    'enc_model_state_dict': self.encoder.state_dict(),
                    'pred_model_state_dict': self.predictor.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, f"{self.checkpoint_path}/{file_name}.ckpt")

    def load_model(self, file_name, only_model=True):
        checkpoint = torch.load(file_name)

        self.encoder.load_state_dict(checkpoint['enc_model_state_dict'])
        self.predictor.load_state_dict(checkpoint['pred_model_state_dict'])

        if not only_model:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

    def save_features(self, data_loader, save_path):
        print("saving features")
        self.encoder.eval()
        feature_dict = {}

        for batch in data_loader:
            X = batch.X.to(self.device)
            # print(X.size())
            with torch.no_grad():
                z, c = self.encoder(X)

                for i in range(len(batch.utt_key)):
                    feature_dict[f"{batch.utt_key[i]}.z"] = z[i].cpu().numpy()

                    feature_dict[f"{batch.utt_key[i]}.c"] = c[i].cpu().numpy()
                    # print(feature_dict[f"{batch.utt_key[i]}.c"].shape)

        np.savez_compressed(save_path, **feature_dict)


def main():
    with open('/home/config/config.json') as config_file:
        config = json.load(config_file)

        for i in range(1, 4):
            trainer = CPCTrainer(config, "/home/saved_models/cpc", config_key='cpc', language="english_full")
            trainer.build()
            # trainer.load_model(f"/home/saved_models/cpc/cpc_e_{i}.ckpt", only_model=False)
            trainer.train(save_best_ap=True, loss_limit=-1, validate=True)
            trainer.save_model(f"cpc_e_{i}")
            trainer.evaluate_downsampled_features()

            feature_dl = SpeakerDataLoader("training", batch_size=1, max_seq_len=100, language="english", dframe=13)
            trainer.save_features(feature_dl, f"/home/features/cpc_features/buckeye/cpc_feats_train_{i}.npz")

            feature_dl = SpeakerDataLoader("validation", batch_size=1, max_seq_len=100, language="english", dframe=13)
            trainer.save_features(feature_dl, f"/home/features/cpc_features/buckeye/cpc_feats_val_{i}.npz")

            feature_dl = SpeakerDataLoader("test", batch_size=1, max_seq_len=100, language="english", dframe=13)
            trainer.save_features(feature_dl, f"/home/features/cpc_features/buckeye/cpc_feats_test_{i}.npz")


if __name__ == '__main__':
    main()
