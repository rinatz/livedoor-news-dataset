from pathlib import Path

import MeCab


class MeCabTokenizer:
    def __init__(self, dic=None):
        if dic:
            if not Path(dic).exists():
                raise RuntimeError(f"MeCab dictionary is not found: {dic}")
        else:
            dic = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"

            if not Path(dic).exists():
                raise RuntimeError(
                    "MeCabTokenizer needs to take a path to NEologd. "
                    f"If not, assumes to be installed in {dic}: "
                    "https://github.com/neologd/mecab-ipadic-neologd"
                )

        self._mecab = MeCab.Tagger(f"-d {dic}")

    def tokenize(self, text):
        tokens = []

        for node in self._mecab.parse(text).split("\n"):
            if node in ["EOS", ""]:
                continue

            _surface, feature = node.split("\t")
            parts = feature.split(",")
            pos, lemma = parts[0:4], parts[6]

            if pos[0] not in ["名詞", "動詞", "形容詞"]:
                continue

            if pos[0:2] == ["名詞", "数詞"]:
                continue

            tokens.append(lemma)

        return tokens
