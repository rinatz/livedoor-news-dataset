from pathlib import Path

import numpy as np
import keras
from keras_preprocessing.text import Tokenizer, tokenizer_from_json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .japanese import MeCabTokenizer


def get_classifications():
    return [
        # site_name, description
        ("dokujo-tsushin", "独身女性"),
        ("it-life-hack", "IT"),
        ("kaden-channel", "家電"),
        ("livedoor-homme", "男性"),
        ("movie-enter", "映画"),
        ("peachy", "女性"),
        ("smax", "モバイル"),
        ("sports-watch", "スポーツ"),
        ("topic-news", "ニュース"),
    ]


def load_directory_data(directory):
    texts = []
    directory = Path(directory)
    site_name = directory.name
    file_paths = list(directory.glob("**/*.txt"))
    mecab = MeCabTokenizer()

    for file_path in tqdm(file_paths, desc=site_name, ncols=100):
        if file_path.name == "LICENSE.txt":
            continue

        with file_path.open() as txt:
            _site_url = next(txt)
            _wrote_at = next(txt)

            text = txt.read()
            tokens = mecab.tokenize(text)
            texts.append(" ".join(tokens))

    return texts


def save_data(num_words=None):
    tar_path = keras.utils.get_file(
        "ldcc-20140209.tar.gz",
        "https://www.rondhuit.com/download/ldcc-20140209.tar.gz",
        cache_subdir="datasets/livedoor_news",
        extract=True,
    )

    texts = []
    labels = []
    livedoor_news = Path(tar_path).parent

    for label, (site_name, _description) in enumerate(get_classifications()):
        site_texts = load_directory_data(livedoor_news / "text" / site_name)
        texts += site_texts
        labels += [label] * len(site_texts)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)

    with livedoor_news.joinpath("livedoor_news.npz").open("wb") as npz:
        x = tokenizer.texts_to_matrix(texts, mode="tfidf")
        y = np.array(labels)
        np.savez(npz, x=x, y=y)

    with livedoor_news.joinpath("livedoor_news_tokenizer.json").open("w") as json_file:
        json_file.write(tokenizer.to_json())


def load_data(path=None, num_words=None, test_split=0.2):
    path = path or "~/.keras/datasets/livedoor_news/livedoor_news.npz"
    path = Path(path).expanduser()

    if not path.exists() or num_words:
        save_data(num_words=num_words)

    with path.open("rb") as npz:
        data = np.load(npz)
        x = data["x"]
        y = data["y"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)

        return (x_train, y_train), (x_test, y_test)


def get_tokenizer(path=None):
    path = path or "~/.keras/datasets/livedoor_news/livedoor_news_tokenizer.json"
    path = Path(path).expanduser()

    if not path.exists():
        raise RuntimeError("load_data() must be invoked before taking tokenizer.")

    with path.open("r") as json_file:
        return tokenizer_from_json(json_file.read())
