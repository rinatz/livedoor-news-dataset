from pathlib import Path

import MeCab


MECAB_DIC_PATH = Path("/usr/local/lib/mecab/dic/mecab-ipadic-neologd")


class MeCabTokenizer:
    def __init__(self):
        if not MECAB_DIC_PATH.exists():
            raise RuntimeError(
                "MeCabTokenizer requires NEologd: "
                "https://github.com/neologd/mecab-ipadic-neologd"
            )

        self._mecab = MeCab.Tagger(f"-d {MECAB_DIC_PATH}")

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

            if pos[0:2] == ["名詞", "数"]:
                continue

            tokens.append(lemma)

        return " ".join(tokens)
