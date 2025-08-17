import argparse
from typing import List
from path import Path
import tensorflow as tf

from settings import Settings
from dataloader_iam import DataLoaderIAM
from model import Model, DecoderType

def char_list_from_file() -> List[str]:
    with open(Settings.CHAR_LIST_FILE_PATH) as f:
        return list(f.read())

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

    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train the model
    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        train_set, validation_set, test_set = loader.get_datasets()
        model = Model(loader.char_list, decoder_type)
        model.train(train_set, validation_set, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        _, validation_set, _ = loader.get_datasets()
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        model.validate(validation_set, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        model.infer(args.img_file)


if __name__ == '__main__':
    main()
