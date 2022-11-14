import argparse
import json
import sys

import torch

from data_loading.SpeakerDataLoader import SpeakerDataLoader
from training import APCTrainer, CAPCTrainer, CPCTrainer, FrameAETrainer, FrameCAETrainer


class Arg(object):
    pass


def main(argv: Arg):
    config = json.load(argv.config_file)

    if argv.representation_type == 'apc':
        trainer = APCTrainer.APCTrainer(
            config,
            checkpoint_path=argv.checkpoint_path,
            language=argv.language
        )
    elif argv.representation_type == 'capc':
        trainer = CAPCTrainer.CAPCTrainer(
            config,
            checkpoint_path=argv.checkpoint_path,
            language=argv.language
        )
    elif argv.representation_type == 'cpc':
        trainer = CPCTrainer.CPCTrainer(
            config,
            checkpoint_path=argv.checkpoint_path,
            language=argv.language
        )
    elif argv.representation_type == 'cpc':
        trainer = CPCTrainer.CPCTrainer(
            config,
            checkpoint_path=argv.checkpoint_path,
            language=argv.language
        )
    elif argv.representation_type == 'frame_cae':
        trainer = FrameCAETrainer.FrameCAETrainer(
            config,
            checkpoint_path=argv.checkpoint_path,
            language=argv.language
        )

    trainer.build()

    if argv.load_from_epoch:
        trainer.load_checkpoint(argv.load_from_epoch)

    if argv.action in ['all', 'train']:
        trainer.train()

    if argv.action in ['all', 'evaluate']:
        trainer.evaluate_downsampled_features()

    if argv.action in ['all', 'extract']:
        extract_all = not (argv.extract_training and argv.extract_validation and argv.extract_test)
        language = argv.extract_languag or argv.language
        if extract_all or argv.extract_training:
            data_loader = SpeakerDataLoader('training', batch_size=1, max_seq_len=100, dframe=13, language=language)
            trainer.save_features(save_path=argv.save_path + '_training', data_loader=data_loader)
        if extract_all or argv.extract_validation:
            data_loader = SpeakerDataLoader('validation', batch_size=1, max_seq_len=100, dframe=13, language=language)
            trainer.save_features(save_path=argv.save_path + '_validation', data_loader=data_loader)
        if extract_all or argv.extract_test:
            data_loader = SpeakerDataLoader('test', batch_size=1, max_seq_len=100, dframe=13, language=language)
            trainer.save_features(save_path=argv.save_path + '_test', data_loader=data_loader)


def check_args():
    argv = Arg()
    arg_parser = argparse.ArgumentParser(argument_default=None)

    # rep type apc / capc / cpc / ae / cae /
    arg_parser.add_argument('representation_type', choices=['apc', 'capc', 'cpc', 'frame_cae'])
    arg_parser.add_argument('action', choices=['all', 'train', 'evaluate', 'extract'])
    arg_parser.add_argument('config_file', type=argparse.FileType('r'))
    arg_parser.add_argument('checkpoint_path', type=str)
    arg_parser.add_argument('--load_from_epoch', default=0, type=int)
    arg_parser.add_argument('--language', default='english_full', choices=['english_full', 'english', 'xitsonga'])
    extract_mode = 'all' in sys.argv or 'extract' in sys.argv
    arg_parser.add_argument('--save_path', required=extract_mode, type=str)
    arg_parser.add_argument('--extract-language', choices=['english_full', 'english', 'xitsonga'])
    arg_parser.add_argument('--extract-training', action='store_true')
    arg_parser.add_argument('--extract-validation', action='store_true')
    arg_parser.add_argument('--extract-test', action='store_true')
    arg_parser.parse_args(namespace=argv)
    return argv


if __name__ == '__main__':
    arg = check_args()
    main(arg)
