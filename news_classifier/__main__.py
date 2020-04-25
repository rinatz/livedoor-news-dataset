"""Create a model for news classification.

Usage:
    news_classifier get
    news_classifier fit

Arguments:
    get     Get dataset
    fit     Fit a model
"""

from docopt import docopt

from .livedoor_news import save_data
from .training import fit_model


def main(argv=None):
    args = docopt(__doc__, argv=argv)

    if args["get"]:
        save_data()
    elif args["fit"]:
        fit_model()


if __name__ == "__main__":
    main()
