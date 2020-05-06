"""Create a model for news classification.

Usage:
    news_classifier get
    news_classifier fit [options]

Arguments:
    get     Get dataset
    fit     Fit a model

Options:
    -o FILE --output=FILE       Path to output file to store model [default: news_classifier.h5]
    -m METHOD --method=METHOD   Choose a machine learning algorithm [default: dnn]
"""

from docopt import docopt
import tensorflow as tf

from .livedoor_news import save_data, load_data, get_tokenizer
from .nn import DeepNeuralNetwork


def main(argv=None):
    args = docopt(__doc__, argv=argv)

    if args["get"]:
        save_data()
    elif args["fit"]:
        if args["--method"] == "dnn":
            (x_train, y_train), (x_test, y_test) = load_data()
            tokenizer = get_tokenizer()

            x_train = tokenizer.sequences_to_matrix(x_train, mode="tfidf")
            y_train = tf.keras.utils.to_categorical(y_train)
            x_test = tokenizer.sequences_to_matrix(x_test, mode="tfidf")
            y_test = tf.keras.utils.to_categorical(y_test)

            dnn = DeepNeuralNetwork(
                num_words=x_train.shape[1], num_labels=y_train.shape[1], tokenizer=tokenizer
            )

            dnn.fit(x_train, y_train, x_test, y_test)
            dnn.save(args["--output"])


if __name__ == "__main__":
    main()
