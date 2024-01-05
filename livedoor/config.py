from pathlib import Path

import pandas as pd

DATA_HOME = Path("~/.keras/datasets/livedoor").expanduser()
DATA_PATH = DATA_HOME / "data.npz"
TOKENIZER_PATH = DATA_HOME / "tokenizer.json"
MODEL_PATH = DATA_HOME / "model.h5"

CATEGORIES = pd.DataFrame(
    [
        [0, "dokujo-tsushin", "独身女性"],
        [1, "it-life-hack", "IT"],
        [2, "kaden-channel", "家電"],
        [3, "livedoor-homme", "男性"],
        [4, "movie-enter", "映画"],
        [5, "peachy", "女性"],
        [6, "smax", "モバイル"],
        [7, "sports-watch", "スポーツ"],
        [8, "topic-news", "ニュース"],
    ],
    columns=["label", "directory_name", "genre"],
)
