"""Create a model

Usage: train.py [options]

Options:
    -w MODEL --with=MODEL   Machine learning model [default: dnn]
    -d --debug              Debug mode [default: False]
    -o FILE --output=FILE   Path to file for saving a model [default: model.pkl]
"""

import sys

from docopt import docopt
import numpy as np
import tensorflow as tf

import dataset
import training


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = docopt(__doc__, argv=argv)

    method = args["--with"]
    debug = args["--debug"]
    output = args["--output"]

    if debug:
        training.utils.enable_debug_mode()

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    tokenizer = dataset.get_tokenizer()

    x_train = tokenizer.sequences_to_matrix(x_train, mode="tfidf")
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = tokenizer.sequences_to_matrix(x_test, mode="tfidf")
    y_test = tf.keras.utils.to_categorical(y_test)

    module = getattr(training, method)
    model = module.fit_model(x_train, y_train, x_test, y_test)
    module.save_model(model, output)


if __name__ == "__main__":
    main()
