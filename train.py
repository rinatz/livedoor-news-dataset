#!/usr/bin/env python

from livedoor.tokenizer import create_data
from livedoor.model import create_model


def main():
    create_data()
    create_model()


if __name__ == "__main__":
    main()
