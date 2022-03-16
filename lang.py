import torch
from tokenizer import get_tokenizer
from utils import get_sentencepairs
import pickle
import os

sos = "<SOS>"
eos = "<EOS>"
sos_idx = 0
eos_idx = 1
MAX_LENGTH = 20


class LangVocab:
    def __init__(self, raw):
        self.raw = raw
        self.i2w = [sos, eos] + list(raw)
        self.w2i = {k: v for v, k in enumerate(self.i2w)}

    def get_word(self, idx):
        assert idx < len(self.i2w) and idx >= 0, "index out of bound"
        return self.i2w[idx]

    def get_index(self, w):
        assert w in self.w2i, "word not in dictionary"
        return self.w2i[w]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.i2w, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.i2w = pickle.load(f)
            self.w2i = {k: v for v, k in enumerate(self.i2w)}


class Rosetta:
    def __init__(self, sentence_pairs, lang_tokenizer1, lang_tokenizer2, max_length=MAX_LENGTH):
        self.max_length = max_length
        self.sentence_pairs = sentence_pairs
        self.lang_tokenizer1 = lang_tokenizer1
        self.lang_tokenizer2 = lang_tokenizer2
        bag1 = set()
        bag2 = set()
        for s1, s2 in sentence_pairs:
            for w in self.lang_tokenizer1(s1):
                bag1.add(w)
            for w in self.lang_tokenizer2(s2):
                bag2.add(w)
        self.lang_vocab1 = LangVocab(bag1)
        self.lang_vocab2 = LangVocab(bag2)
        self.num_words_vocab1 = len(self.lang_vocab1.i2w)
        self.num_words_vocab2 = len(self.lang_vocab2.i2w)

    def sentence2tensor(self, s, lang=1):
        if lang == 1:
            lang_vocab = self.lang_vocab1
            tokenizer = self.lang_tokenizer1
        elif lang == 2:
            lang_vocab = self.lang_vocab2
            tokenizer = self.lang_tokenizer2
        else:
            assert False, "Wrong option for lang"

        return torch.tensor([lang_vocab.get_index(w) for w in tokenizer(s)] + [eos_idx]).view(-1, 1)

    def sentencepairs2tensors(self):
        sp = []
        for s1, s2 in self.sentence_pairs:
            t1, t2 = self.sentence2tensor(
                s1, lang=1), self.sentence2tensor(s2, lang=2)
            if len(t1) > self.max_length or len(t2) > self.max_length:
                continue
            sp.append((t1, t2))
        return sp

    def tensor2sentence(self, x, lang=1):
        if lang == 1:
            lang_vocab = self.lang_vocab1
            tokenizer = self.lang_tokenizer1
        elif lang == 2:
            lang_vocab = self.lang_vocab2
            tokenizer = self.lang_tokenizer2
        else:
            assert False, "Wrong option for language"

        words = [lang_vocab.get_word(i) for i in x]
        return tokenizer.detokenize(words)

    def save(self, path):
        path1 = os.path.join(path, "lang1.pkl")
        path2 = os.path.join(path, "lang2.pkl")
        self.lang_vocab1.save(path1)
        self.lang_vocab2.save(path2)

    def load(self, path):
        path1 = os.path.join(path, "lang1.pkl")
        path2 = os.path.join(path, "lang2.pkl")
        self.lang_vocab1.load(path1)
        self.lang_vocab2.load(path2)


def get_rosetta(lang1, lang2):
    sp = get_sentencepairs(lang1, lang2)
    tokenizer1 = get_tokenizer(lang1)
    tokenizer2 = get_tokenizer(lang2)
    return Rosetta(sp, tokenizer1, tokenizer2)
