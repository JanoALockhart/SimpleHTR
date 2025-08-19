import argparse
from path import Path
import tensorflow as tf
import random
import numpy as np

from dataset import Dataset, BaseDataset
from image_loader import BaseImageLoader, LMDBImageLoader
from preprocessor import Preprocessor
from dataloader_iam import IAMDataLoader
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

    datasets_loader = IAMDataLoader(args.data_dir, train_split=0.95, validation_split=0.04)
    
    image_loader = BaseImageLoader()
    if args.fast:
        image_loader = LMDBImageLoader(args.data_dir)
    
    train_set, validation_set, test_set = datasets_loader.get_datasets()

    train_preprocessor = Preprocessor(data_augmentation=True, line_mode=args.line_mode)
    train_set.set_image_loader(image_loader).map(train_preprocessor).batch(args.batch_size, drop_remainder=True).shuffle()

    validation_preprocessor = Preprocessor(line_mode=args.line_mode)
    validation_set.set_image_loader(image_loader).map(validation_preprocessor).batch(args.batch_size)

    # train the model
    if args.mode == 'train':
        model = Model(datasets_loader.get_alphabet(), decoder_type)
        model.train(train_set, validation_set, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        model = Model(datasets_loader.get_alphabet(), decoder_type, must_restore=True)
        model.validate(validation_set)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(datasets_loader.get_alphabet(), decoder_type, must_restore=True, dump=args.dump)
        model.infer(args.img_file)

if __name__ == '__main__':
    main()
