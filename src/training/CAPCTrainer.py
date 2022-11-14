__author__ = "Lisa van Staden"

import json
from torch import nn

from data_loading.SpeakerDataLoader import SpeakerDataLoader
from training.APCTrainer import APCTrainer


class CAPCTrainer(APCTrainer):

    def __init__(self, config, checkpoint_path, language='english', config_key='capc'):
        super().__init__(config, checkpoint_path, language, config_key)

    def _train_step(self, batch):
        self.optimizer.zero_grad()
        x = batch.X.to(self.device)
        y = batch.Y.to(self.device)
        h = self.encoder(x)
        loss = self.predictor(y, h)
        loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1)
        self.optimizer.step()
        return loss

def main():
    with open('/home/config/config.json') as config_file:
        config = json.load(config_file)

        for i in range(1, 4):
            #trainer = APCTrainer(config, "/home/saved_models/apc_pairs", config_key='apc', language="english_aligned")
            #trainer.build()
            #trainer.train()
            trainer = CAPCTrainer(config, "/home/saved_models/apc_pairs", config_key='capc', language="english_aligned")
            trainer.build()
            #trainer.train()
            #trainer.load_checkpoint(epoch=49, only_model=False)
            trainer.train(validate=False)
            #trainer.evaluate_downsampled_features()
            feature_dl = SpeakerDataLoader("training", batch_size=1, max_seq_len=100, dframe=13, language="xitsonga")
            trainer.save_features(feature_dl, f'/home/features/capc_features/xitsonga/cross_capc_feats_train_{i}')
            #feature_dl = SpeakerDataLoader("validation", batch_size=1, max_seq_len=100, dframe=13, language="xitsonga")
            #trainer.save_features(feature_dl, f'/home/features/capc_features/xitsonga/capc_feats_val_{i}')
            feature_dl = SpeakerDataLoader("test", batch_size=1, max_seq_len=100, dframe=13, language="xitsonga")
            trainer.save_features(feature_dl, f'/home/features/capc_features/xitsonga/cross_capc_feats_test_{i}')

if __name__ == '__main__':
    main()
