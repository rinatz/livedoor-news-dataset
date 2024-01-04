from pathlib import Path

from natto import MeCab
import numpy as np
from rich import progress
from sklearn.model_selection import train_test_split
import tensorflow as tf

from livedoor.config import DATA_PATH, TOKENIZER_PATH, CATEGORIES


class MeCabTokenizer:
    DEFAULT_DICTIONARY = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"

    def __init__(self, dic=DEFAULT_DICTIONARY):
        self._mecab = MeCab(f"-d {dic} -F%f[0],%f[1],%f[2],%f[3],%f[6]")
        self._tokenizer = tf.keras.preprocessing.text.Tokenizer()

    def tokenize(self, text):
        tokens = []

        for node in self._mecab.parse(text, as_nodes=True):
            if node.is_eos():
                continue

            feature = node.feature.split(",")
            part_of_speech, lemma = feature[0:4], feature[4]

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

    tokenizer = MeCabTokenizer()

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
    tokenizer = MeCabTokenizer()
    tokenizer.load(TOKENIZER_PATH)
    return tokenizer
