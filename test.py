import torch
from tokenizer import get_tokenizer
from utils import evaluate_one_sentence
from lang import Rosetta
from models import EncoderRNN, DecoderRNN
from cli_parser import create_test_parser
import gradio as gr
import os

args = create_test_parser().parse_args()

if args.gpu and torch.cuda.is_available():
    print("Testing with GPU!")
    dev = torch.device("cuda")
else:
    print("Testing with CPU. It can be a little bit slow")
    dev = torch.device("cpu")

model_dir = f"trained_models/{args.input_lang}-{args.output_lang}"
rosetta_dir = os.path.join(model_dir, "rosetta")

print("Loading Rosetta ...")
tokenizer1 = get_tokenizer(args.input_lang)
tokenizer2 = get_tokenizer(args.output_lang)
rosetta = Rosetta([], tokenizer1, tokenizer2)
rosetta.load(rosetta_dir)

print("Creating the model")
encoder = torch.load(os.path.join(model_dir, "encoder.ckpt"), map_location=dev)
decoder = torch.load(os.path.join(model_dir, "decoder.ckpt"), map_location=dev)

encoder = encoder.to(dev)
decoder = decoder.to(dev)


def translate(s):
    x = rosetta.sentence2tensor(s).to(dev)
    y = evaluate_one_sentence(x, encoder, decoder, dev)
    sout = rosetta.tensor2sentence(y, lang=2)
    return sout


print("Testing ...")
iface = gr.Interface(fn=translate, inputs=["text"], outputs=["text"]).launch()
