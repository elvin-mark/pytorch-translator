from base64 import decode
import torch
import random
import tqdm
import os

LANG_ROOT_PATH = "./sentence-pairs"
MAX_LENGTH = 20
sos_idx = 0
eos_idx = 1


def get_sentencepairs(lang1, lang2):
    raw_data = open(os.path.join(
        LANG_ROOT_PATH, f"{lang1}-{lang2}.tsv"), "r").read().strip().split("\n")

    def extract_sentences(x): return (x[1], x[3])
    raw_data = [extract_sentences(elem.split("\t")) for elem in raw_data]
    return raw_data


def train_one_sentence(x, y, encoder, decoder, encoder_optim, decoder_optim, crit, dev, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5):
    encoder.train()
    decoder.train()
    x = x.to(dev)
    y = y.to(dev)
    h = encoder.init_hidden().to(dev)

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    lx = len(x)
    ly = len(y)

    eo = torch.zeros(max_length, encoder.hidden_size).to(dev)
    loss = 0

    for i in range(lx):
        tmp_out, h = encoder(x[i], h)
        eo[i] = tmp_out[0, 0]

    di = torch.tensor([[sos_idx]]).to(dev)
    dh = h

    if random.random() < teacher_forcing_ratio:
        for i in range(ly):
            do, dh = decoder(di, dh)
            loss += crit(do, y[i])
            di = y[i]
    else:
        for i in range(ly):
            do, dh = decoder(di.long(), dh)
            di = do.topk(1)[1].squeeze().detach()
            loss += crit(do, y[i])
            if di.item() == 1:
                break
    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / len(y)


def train(train_ds, encoder, decoder, encoder_optim, decoder_optim, crit, dev, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5, epochs=5, iter_per_epoch=1000):
    for epoch in tqdm.tqdm(range(epochs)):
        tot_loss = 0
        for j in range(iter_per_epoch):
            x, y = random.choice(train_ds)
            loss = train_one_sentence(x, y, encoder, decoder, encoder_optim, decoder_optim,
                                      crit, dev, max_length=max_length, teacher_forcing_ratio=teacher_forcing_ratio)
            tot_loss += loss
        print(f"epoch: {epoch}, loss: {tot_loss / iter_per_epoch}")


def evaluate_one_sentence(x, encoder, decoder, dev, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        x = x.to(dev)
        lx = len(x)
        h = encoder.init_hidden().to(dev)
        eo = torch.zeros(max_length, encoder.hidden_size).to(dev)

        for i in range(lx):
            o, h = encoder(x[i], h)
            eo[i] = o[0, 0]

        di = torch.tensor([[sos_idx]]).to(dev)

        decoder_output = []
        for i in range(max_length):
            o, h = decoder(di, h)
            di = o.topk(1)[1].squeeze().detach()
            if di.item() == 1:
                break
            else:
                decoder_output.append(di.item())

    return decoder_output
