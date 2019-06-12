import MeCab


class MeCabTokenizer:
    def __init__(self, dic=None):
        if dic:
            self._mecab = MeCab.Tagger(f"-d {dic}")
        else:
            self._mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

    def tokenize(self, text, filter_token=None):
        tokens = []

        for result in self._mecab.parse(text).split("\n"):
            if result in ["EOS", ""]:
                continue

            word, columns = result.split("\t")
            columns = columns.split(",")
            token = columns[6]

            if not filter_token:
                tokens.append(token)
            elif filter_token(word, columns):
                tokens.append(token)

        return tokens
