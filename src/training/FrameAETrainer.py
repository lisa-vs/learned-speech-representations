import json
import os

from data_loading.SpeakerDataLoader import SpeakerDataLoader
from training import helpers

__author__ = "Lisa van Staden"

import sys
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.FrameAEModel import Encoder
from models.FrameAEModel import Decoder
from data_loading.FrameDataLoader import FrameDataLoader


class FrameAETrainer:
    """ Class responsible for building and training an Auto Encoder model given the model attributes"""

    def __init__(self, config, checkpoint_path="ae.pt", language="english", config_key="frame_ae"):

        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.language = language

        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.epoch = 0

        self.train_data_loader = None
        self.test_data_loader = None
        self.dev_data_loader = None

        self.config_key = config_key

    def build(self):
        """ Builds the model with given attributes or default values"""

        # attributes
        input_dim = self.config[self.config_key]["input_dim"]
        enc_dim = self.config[self.config_key]["enc_dim"]
        num_layers = self.config[self.config_key].get("num_layers")
        z_dim = self.config[self.config_key]["z_dim"]

        # model
        self.encoder = Encoder(input_dim=input_dim, enc_dim=enc_dim, z_dim=z_dim, num_layers=num_layers).to(self.device)
        self.decoder = Decoder(z_dim=z_dim, dec_dim=enc_dim, input_dim=input_dim, num_layers=num_layers).to(self.device)

        learning_rate = self.config[self.config_key]["learning_rate"]

        model_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params=model_params, lr=learning_rate)

    def save_checkpoint(self):
        torch.save({'enc_model_state_dict': self.encoder.state_dict(),
                    'dec_model_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.epoch
                    }, f"{self.checkpoint_path}/{self.epoch}.ckpt")

    def save_model(self, file_name):
        torch.save({'enc_model_state_dict': self.encoder.state_dict(),
                    'dec_model_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.epoch
                    }, f"{self.checkpoint_path}/{file_name}.ckpt")

    def load_checkpoint(self, only_model=False, epoch=-1):

        if epoch == -1:  # load latest checkpoint
            checkpoints = os.listdir(self.checkpoint_path)
            epochs = [int(c.split('_')[1].split('.')[0]) for c in checkpoints]
            epoch = max(epochs)

        checkpoint = torch.load(f'{self.checkpoint_path}/{epoch}.ckpt')

        self.encoder.load_state_dict(checkpoint['enc_model_state_dict'])
        self.decoder.load_state_dict(checkpoint['dec_model_state_dict'])

        if not only_model:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

    def train(self, num_epochs=-1, verbose=True):

        num_epochs = self.config[self.config_key]["num_epochs"] if num_epochs == -1 else num_epochs
        evaluate_epochs = self.config[self.config_key]["evaluate_epochs"]
        checkpoint_interval = self.config[self.config_key]["checkpoint_interval"]

        for epoch in range(num_epochs):
            epoch_loss = self._train_step()
            if epoch >= evaluate_epochs and (epoch + 1) % checkpoint_interval == 0:
                if self.language == "english":
                    self.evaluate_downsampled_features()
                self.epoch = epoch
                self.save_checkpoint()
            if verbose:
                print(f'Epoch {epoch}: \t Loss: {epoch_loss}')

    def _train_step(self):

        if self.train_data_loader is None:
            batch_size = self.config[self.config_key]["batch_size"]
            self.train_data_loader = FrameDataLoader(dataset_type="training", batch_size=batch_size,
                                                     language=self.language, pairs=False)

        self.encoder.train()
        self.decoder.train()

        epoch_loss = 0
        num_batches = 0

        for batch in self.train_data_loader:
            self.optimizer.zero_grad()
            [x, _] = batch
            x = x.to(self.device)

            z = self.encoder(x).to(self.device)
            x_pred = self.decoder(z)

            loss = F.mse_loss(x, x_pred)
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        return epoch_loss / num_batches

    def evaluate_downsampled_features(self, verbose=True):
        self.encoder.eval()
        self.decoder.eval()

        z_dim = self.config[self.config_key]["z_dim"]
        word_zs = []
        labels = []

        if self.dev_data_loader is None:
            input_dim = self.config[self.config_key]["input_dim"]
            self.dev_data_loader = SpeakerDataLoader(dataset_type="validation", batch_size=1, language=self.language,
                                                     dframe=input_dim)

        for batch in self.dev_data_loader:
            word_x = batch.X.to(self.device)
            word_x = word_x.reshape((-1, word_x.size()[-1]))
            label = batch.labels[0]

            with torch.no_grad():
                word_z = torch.empty(size=(word_x.size()[0], z_dim))
                for i in range(len(word_x)):
                    word_z[i] = self.encoder(word_x[i])
                word_zs.append(word_z)
                labels.append(label)

        downsampled = helpers.downsample(word_zs, labels, d_frame=z_dim)
        ap, prb = helpers.calculate_ap_and_prb(downsampled, labels)

        if verbose:
            print("Downsampled AP:", ap)
        return ap


if __name__ == '__main__':
    with open('/home/config/herman_model_config.json') as config_file:
        config = json.load(config_file)

        trainer = FrameAETrainer(config, "/home/saved_models/frame_ae_x", language="xitsonga")
        trainer.build()
        trainer.train()
        results = trainer.evaluate_downsampled_features()

        # print(results)
