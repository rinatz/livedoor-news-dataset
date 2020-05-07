import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from . import livedoor


class ClassificationReport(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, x_test, y_test, labels):
        self.labels = labels
        self.x_val = x_val
        self.y_val = self._categorical_to_labels(y_val)
        self.x_test = x_test
        self.y_test = self._categorical_to_labels(y_test)

    def _categorical_to_labels(self, matrix):
        return [self.labels[np.argmax(categorical)] for categorical in matrix]

    def _classification_report(self, x, y):
        y_pred = self._categorical_to_labels(self.model.predict(x))
        return classification_report(y, y_pred, labels=self.labels)

    def on_epoch_end(self, epoch, logs=None):
        report = self._classification_report(self.x_val, self.y_val)
        print("val_classification_report:")
        print(report)

    def on_train_end(self, logs=None):
        report = self._classification_report(self.x_test, self.y_test)
        print("test_classification_report:")
        print(report)


def build_model(num_words=1, num_labels=1):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(num_words,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_labels, activation="softmax"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])

    return model


def fit_model():
    (x_train, y_train), (x_test, y_test) = livedoor.load_data()
    tokenizer = livedoor.get_tokenizer()
    labels = list(livedoor.CATEGORIES.values())

    x_train = tokenizer.sequences_to_matrix(x_train, mode="tfidf")
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = tokenizer.sequences_to_matrix(x_test, mode="tfidf")
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    model = tf.keras.wrappers.scikit_learn.KerasClassifier(
        build_model,
        num_words=x_train.shape[1],
        num_labels=y_train.shape[1],
        epochs=5,
        batch_size=16,
        validation_data=(x_val, y_val),
        callbacks=[ClassificationReport(x_val, y_val, x_test, y_test, labels=labels)],
    )

    model.fit(x_train, y_train)

    return model


def save_model(model, path):
    model.model.save(path)


def load_model(path):
    return tf.keras.models.load_model(path)
