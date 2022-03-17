DEFAULT_IGNORE_CHAR = [",", ".", "¡", "!", "¿", "?", "[", "]", "(", ")"]


class Tokenizer:
    def __init__(self, delim=" ", ignore_char=DEFAULT_IGNORE_CHAR, transform=lambda x: x.lower()):
        self.delim = delim
        self.ignore_char = ignore_char
        self.transform = transform

    def __call__(self, s):
        for c in self.ignore_char:
            s = s.replace(c, "")
        if len(self.delim) > 0:
            return [self.transform(elem) for elem in s.split(self.delim)]
        else:
            return list(s)

    def detokenize(self, words):
        if len(self.delim) > 0:
            return " ".join(words)
        elif len(self.delim) == 0:
            return "".join(words)
        return None


def get_tokenizer(lang):
    if lang == "spa":
        return Tokenizer()
    elif lang == "ind":
        return Tokenizer()
    elif lang == "kr":
        return Tokenizer(delim="")
    elif lang == "jap":
        return Tokenizer(delim="")
    elif lang == "cn":
        return Tokenizer(delim="")
    else:
        return None
