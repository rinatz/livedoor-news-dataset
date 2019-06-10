from pathlib import Path
import pickle

import MeCab
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def parse_japanese(japanese):
    mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

    words = []

    for result in mecab.parse(japanese).split("\n"):
        if result in ["EOS", ""]:
            continue

        columns = result.split("\t")[1].split(",")
        category, category_detail1, word = columns[0], columns[1], columns[6]

        if category not in ["名詞", "動詞", "形容詞"]:
            continue

        if [category, category_detail1] == ["名詞", "数"]:
            continue

        words.append(word)

    return " ".join(words)


def get_classifications():
    return [
        # site_name, description
        ("dokujo-tsushin", "女性"),
        ("it-life-hack", "IT"),
        ("kaden-channel", "家電"),
        ("livedoor-homme", "男性"),
        ("movie-enter", "映画"),
        ("peachy", "グルメ"),
        ("smax", "モバイル"),
        ("sports-watch", "スポーツ"),
        ("topic-news", "ニュース"),
    ]


def load_directory_data(directory):
    texts = []

    for file_path in Path(directory).glob("**/*.txt"):
        if file_path.name == "LICENSE.txt":
            continue

        with file_path.open() as txt:
            texts.append(parse_japanese(txt.read()))

    return texts


def save_data(num_words=None):
    tar_path = tf.keras.utils.get_file(
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

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)

    with livedoor_news.joinpath("livedoor_news.npz").open("wb") as npz:
        np.savez(
            npz, x=tokenizer.texts_to_matrix(texts, mode="tfidf"), y=np.array(labels)
        )

    with livedoor_news.joinpath("livedoor_news_tokenizer.pickle").open("wb") as pkl:
        pickle.dump(tokenizer, pkl)


def load_data(num_words=None, test_split=0.2):
    path = Path("~/.keras/datasets/livedoor_news/livedoor_news.npz").expanduser()

    if not path.exists() or num_words:
        save_data(num_words=num_words)

    with path.open("rb") as npz:
        data = np.load(npz)
        x = data["x"]
        y = data["y"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)

        return (x_train, y_train), (x_test, y_test)


def get_tokenizer():
    path = Path("~/.keras/datasets/livedoor_news/livedoor_news_tokenizer.pickle").expanduser()

    if not path.exists():
        raise ValueError("save_data() must be called before taking tokenizer.")

    with path.open("rb") as pkl:
        return pickle.load(pkl)
