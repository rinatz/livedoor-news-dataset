from pathlib import Path

import pandas as pd

DATA_PATH = Path("~/.keras/datasets/livedoor/livedoor.npz").expanduser()
TOKENIZER_PATH = Path("~/.keras/datasets/livedoor/tokenizer.json").expanduser()
MODEL_PATH = Path("~/.keras/datasets/livedoor/model.h5").expanduser()

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
    columns=["label", "directory_name", "site_name"],
)
