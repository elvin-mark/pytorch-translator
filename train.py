import torch
from utils import evaluate_one_sentence, train
from lang import get_rosetta
from models import EncoderRNN, DecoderRNN
from cli_parser import create_train_parser
import random
import os
import pickle

if not os.path.exists("trained_models"):
    os.mkdir("trained_models")


args = create_train_parser().parse_args()
if args.gpu and torch.cuda.is_available():
    print("Training with GPU!")
    dev = torch.device("cuda")
else:
    print("Training with CPU. It can be a little bit slow")
    dev = torch.device("cpu")

print("Loading sentences ...")
rosetta = get_rosetta(args.input_lang, args.output_lang)
raw_data = rosetta.sentencepairs2tensors()

N = len(raw_data)
train_size = int(N * args.split_size)
test_size = N - train_size

train_ds, test_ds = torch.utils.data.random_split(
    raw_data, [train_size, test_size])

print("Creating the model")
encoder = EncoderRNN(rosetta.num_words_vocab1, args.hidden_size)
decoder = DecoderRNN(args.hidden_size, rosetta.num_words_vocab2)

encoder = encoder.to(dev)
decoder = decoder.to(dev)

encoder_optim = torch.optim.SGD(encoder.parameters(), lr=args.lr)
decoder_optim = torch.optim.SGD(decoder.parameters(), lr=args.lr)

crit = torch.nn.NLLLoss()

print("Training ...")
train(train_ds, encoder, decoder,
      encoder_optim, decoder_optim, crit, dev, epochs=args.epochs)

print("Generating some samples")
for _ in range(args.samples):
    x, y = random.choice(test_ds)
    sx = rosetta.tensor2sentence(x.squeeze())
    sy = rosetta.tensor2sentence(y.squeeze(), lang=2)
    yhat = evaluate_one_sentence(x, encoder, decoder, dev)
    syhat = rosetta.tensor2sentence(yhat, lang=2)
    print("original sentence:    ", sx)
    print("translation :         ", syhat)
    print("original translation: ", sy)
    print("=" * 50)

print("Saving model")
if args.save_model:
    model_dir = f"trained_models/{args.input_lang}-{args.output_lang}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(encoder, os.path.join(model_dir, "encoder.ckpt"))
    torch.save(decoder, os.path.join(model_dir, "decoder.ckpt"))
    rosetta_dir = os.path.join(model_dir, "rosetta")
    if not os.path.exists(rosetta_dir):
        os.mkdir(rosetta_dir)
    rosetta.save(rosetta_dir)
