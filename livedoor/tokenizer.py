from pathlib import Path

import numpy as np
from rich import progress
from sklearn.model_selection import train_test_split
import tensorflow as tf
from vibrato import Vibrato
from zstandard import ZstdDecompressor

from livedoor.config import DATA_PATH, TOKENIZER_PATH, CATEGORIES


class VibratoTokenizer:
    DEFAULT_DICTIONARY = "ipadic-mecab-2_7_0/system.dic.zst"

    def __init__(self, dic=DEFAULT_DICTIONARY):
        unzstd = ZstdDecompressor()

        with open(dic, "rb") as zst:
            with unzstd.stream_reader(zst) as r:
                self._vibrato = Vibrato(r.read())

        self._tokenizer = tf.keras.preprocessing.text.Tokenizer()

    def tokenize(self, text):
        tokens = []

        for token in self._vibrato.tokenize(text):
            feature = token.feature().split(",")
            part_of_speech, lemma = feature[0:2], feature[6]

            if part_of_speech[0] not in ["名詞", "動詞", "形容詞"]:
                continue

            if part_of_speech[0:2] == ["名詞", "数"]:
                continue

            tokens.append(lemma)

        return " ".join(tokens)

    def fit_on_texts(self, texts):
        texts = [
            self.tokenize(text)
            for text in progress.track(texts, description="Fitting on texts...")
        ]
        self._tokenizer.fit_on_texts(texts)
        sequences = self._tokenizer.texts_to_sequences(texts)

        return np.array(sequences, dtype=object)

    def texts_to_matrix(self, texts):
        texts = [self.tokenize(text) for text in texts]
        return self._tokenizer.texts_to_matrix(texts, mode="tfidf")

    def sequences_to_matrix(self, sequences):
        return self._tokenizer.sequences_to_matrix(sequences, mode="tfidf")

    def save(self, path):
        with open(path, "w") as f:
            f.write(self._tokenizer.to_json())

    def load(self, path):
        with open(path) as f:
            self._tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())


def load_directory_data(directory):
    texts = []
    directory = Path(directory)
    txt_paths = filter(lambda x: x.name != "LICENSE.txt", directory.glob("**/*.txt"))

    for txt_path in txt_paths:
        with txt_path.open() as txt:
            _site_url = next(txt)
            _wrote_at = next(txt)

            texts.append(txt.read())

    return texts


def create_data():
    tar_path = tf.keras.utils.get_file(
        "ldcc-20140209.tar.gz",
        "https://www.rondhuit.com/download/ldcc-20140209.tar.gz",
        cache_subdir="datasets/livedoor",
        extract=True,
    )

    texts = []
    labels = []
    livedoor = Path(tar_path).parent

    for _index, category in CATEGORIES.iterrows():
        directory = livedoor / "text" / category.directory_name
        site_texts = load_directory_data(directory)
        texts += site_texts
        labels += [category.label] * len(site_texts)

    tokenizer = VibratoTokenizer()

    x = tokenizer.fit_on_texts(texts)
    y = np.array(labels, dtype=object)

    with DATA_PATH.open("wb") as npz:
        np.savez(npz, x=x, y=y)

    tokenizer.save(TOKENIZER_PATH)


def load_data(test_split=0.2):
    with np.load(DATA_PATH, allow_pickle=True) as npz:
        x = npz["x"]
        y = npz["y"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)

        return (x_train, y_train), (x_test, y_test)


def get_tokenizer():
    tokenizer = VibratoTokenizer()
    tokenizer.load(TOKENIZER_PATH)
    return tokenizer
