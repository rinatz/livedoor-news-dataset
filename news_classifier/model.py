import keras
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .livedoor_news import load_data, get_classifications, get_tokenizer


class ClassificationReport(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, x_test, y_test, labels):
        self.labels = labels
        self.x_val = x_val
        self.y_val = self._one_hot_matrix_to_labels(y_val)
        self.x_test = x_test
        self.y_test = self._one_hot_matrix_to_labels(y_test)

    def _one_hot_matrix_to_labels(self, matrix):
        return [self.labels[np.argmax(one_hot)] for one_hot in matrix]

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self._one_hot_matrix_to_labels(self.model.predict(self.x_val))
        report = classification_report(self.y_val, y_pred, labels=self.labels)
        print("val_classification_report:")
        print(report)

    def on_train_end(self, logs=None):
        y_pred = self._one_hot_matrix_to_labels(self.model.predict(self.x_test))
        report = classification_report(self.y_test, y_pred, labels=self.labels)
        print("test_classification_report:")
        print(report)


def build_model(num_words, num_labels):
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(num_words,)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_labels, activation="softmax"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()

    return model


def fit_model(path="news_classifier_model.h5"):
    (x_train, y_train), (x_test, y_test) = load_data()
    tokenizer = get_tokenizer()

    x_train = tokenizer.sequences_to_matrix(x_train, mode="tfidf")
    y_train = keras.utils.to_categorical(y_train)
    x_test = tokenizer.sequences_to_matrix(x_test, mode="tfidf")
    y_test = keras.utils.to_categorical(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    model = build_model(num_words=x_train.shape[1], num_labels=y_train.shape[1])

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=16,
        validation_data=(x_val, y_val),
        callbacks=[
            ClassificationReport(
                x_val,
                y_val,
                x_test,
                y_test,
                labels=list(get_classifications().values()),
            )
        ],
    )

    model.save(path)


def load_model(path="news_classifier_model.h5"):
    return keras.models.load_model(path)
