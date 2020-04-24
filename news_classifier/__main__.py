from .livedoor_news import save_data
from .training import fit_model


def main():
    save_data()
    fit_model()


if __name__ == "__main__":
    main()
