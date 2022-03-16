from argparse import ArgumentParser
from email.policy import default

INPUT_LANG = ["spa"]
OUTPUT_LANG = ["kr"]


def create_train_parser():
    parser = ArgumentParser(
        description="CLI to train a neural network for machine translation")
    parser.add_argument("--input-lang", type=str, default="spa",
                        choices=INPUT_LANG, help="input language to translate from")
    parser.add_argument("--output-lang", type=str, default="kr",
                        choices=OUTPUT_LANG, help="output language to translate to")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of epochs")
    parser.add_argument("--sentences-per-epoch", type=int,
                        default=1000, help="sentences used in training per epoch")
    parser.add_argument("--gpu", action="store_true",
                        dest="gpu", help="specify whether use gpu or not")
    parser.add_argument("--samples", type=int, default=10,
                        help="number of samples")
    parser.add_argument("--split-size", type=int, default=0.8,
                        help="train split size ratio")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="hidden size of the encoder and decoder model")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--save-model", action="store_true", dest="save_model",
                        help="specify whether to save the model after training or not")
    parser.set_defaults(gpu=False, save_model=False)
    return parser


def create_test_parser():
    parser = ArgumentParser(
        description="CLI to train a neural network for machine translation")
    parser.add_argument("--input-lang", type=str, default="spa",
                        choices=INPUT_LANG, help="input language to translate from")
    parser.add_argument("--output-lang", type=str, default="kr",
                        choices=OUTPUT_LANG, help="output language to translate to")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="hidden size of the encoder and decoder model")
    parser.add_argument("--gpu", action="store_true",
                        dest="gpu", help="specify whether use gpu or not")
    parser.set_defaults(gpu=False)

    return parser
