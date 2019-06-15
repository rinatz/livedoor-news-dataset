from pathlib import Path

import MeCab


class MeCabTokenizer:
    def __init__(self, dic=None):
        if dic:
            if not Path(dic).exists():
                raise RuntimeError(f"MeCab dictionary {dic} is not found.")
        else:
            dic = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"

            if not Path(dic).exists():
                raise RuntimeError(
                    f"NEologd is not installed in {dic}. You have to install NEologd from https://github.com/neologd/mecab-ipadic-neologd ."
                )

        self._mecab = MeCab.Tagger(f"-d {dic}")

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
