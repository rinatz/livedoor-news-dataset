"""News Classifier.

Usage:
    news_classifier save
    news_classifier create
    news_classifier serve

Options:
    -h --help       Show this message.
"""
import sys

from docopt import docopt

from .livedoor_news import save_data
from .model import create_model
from .server import api


def main(argv=None):
    argv = argv or sys.argv[1:]
    args = docopt(__doc__, argv=argv)

    if args["save"]:
        save_data()
    elif args["create"]:
        create_model()
    elif args["serve"]:
        api.run(address="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
