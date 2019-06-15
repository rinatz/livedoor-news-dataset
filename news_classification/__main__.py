import sys

from .server import api
from .model import create_model


def main(args=None):
    args = args or sys.argv[1:]

    try:
        action = args[0]
    except IndexError:
        print(
            "Usage: python -m news_classification <create-model|server>",
            file=sys.stderr,
        )
        sys.exit(1)

    if action == "create-model":
        create_model()
    elif action == "server":
        api.run(address="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
