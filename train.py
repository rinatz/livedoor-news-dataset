#!/usr/bin/env python

from livedoor.tokenizer import save_data
from livedoor.model import create_model


def main():
    save_data()
    create_model()


if __name__ == "__main__":
    main()
