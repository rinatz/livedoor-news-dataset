"""Create a model

Usage: train.py [options]

Options:
    -w MODEL --with=MODEL   Machine learning model [default: dnn]
"""

import sys

from docopt import docopt
import numpy as np
import tensorflow as tf

import dataset
from training import dnn, tree


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = docopt(__doc__, argv=argv)
    method = args["--with"]

    np.random.seed(42)

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    tokenizer = dataset.get_tokenizer()

    x_train = tokenizer.sequences_to_matrix(x_train, mode="tfidf")
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = tokenizer.sequences_to_matrix(x_test, mode="tfidf")
    y_test = tf.keras.utils.to_categorical(y_test)

    if method == "dnn":
        model = dnn.fit_model(x_train, y_train, x_test, y_test)
        dnn.save_model(model, "model.h5")
    elif method == "tree":
        model = tree.fit_model(x_train, y_train, x_test, y_test)
        tree.save_model(model, "model.pkl")


if __name__ == "__main__":
    main()
