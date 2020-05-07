"""Create a model for news classification.

Usage:
    news_classifier get
    news_classifier fit [options]

Arguments:
    get     Get dataset
    fit     Fit a model

Options:
    -o FILE --output=FILE       Path to output file to store model [default: model.h5]
    -m METHOD --method=METHOD   Choose a machine learning algorithm [default: dnn]
"""

from docopt import docopt
import tensorflow as tf

from . import livedoor, dnn


def main(argv=None):
    args = docopt(__doc__, argv=argv)

    if args["get"]:
        livedoor.save_data()
    elif args["fit"]:
        if args["--method"] == "dnn":
            model = dnn.fit_model()
            dnn.save_model(model, args["--output"])


if __name__ == "__main__":
    main()
