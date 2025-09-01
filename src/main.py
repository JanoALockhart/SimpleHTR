import argparse
from path import Path
import tensorflow as tf
import random
import numpy as np

from dataloader_iam import IAMDataLoader, JPSDSmallTestSet
from model import Model, DecoderType
from settings import Settings

def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


def main():
    """Main function."""
    # Disable eager mode
    tf.compat.v1.disable_eager_execution()

    # Seting seed
    random.seed(Settings.SEED)
    np.random.seed(Settings.SEED)
    tf.random.set_seed(Settings.SEED)

    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    datasets_loader = IAMDataLoader(
        args.data_dir, 
        args.batch_size, 
        args.line_mode, 
        args.fast, 
        train_split=0.95, 
        validation_split=0.04,
        data_augmentation=False
    )

    train_set, validation_set, test_set = datasets_loader.get_configured_datasets()

    # train the model
    if args.mode == 'train':
        model = Model(datasets_loader.get_alphabet(), decoder_type)
        model.train(train_set, validation_set, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        model = Model(datasets_loader.get_alphabet(), decoder_type, must_restore=True)
        model.validate(test_set)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(datasets_loader.get_alphabet(), decoder_type, must_restore=True, dump=args.dump)
        model.infer(args.img_file)

if __name__ == '__main__':
    main()
