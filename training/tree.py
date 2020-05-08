import pickle

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from . import utils


def fit_model(x_train, y_train, x_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    report = classification_report(
        utils.categorical_to_labels(y_test), utils.categorical_to_labels(y_pred)
    )

    print(report)

    return model


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
