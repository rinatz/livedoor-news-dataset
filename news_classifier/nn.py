import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from .livedoor_news import load_data, get_classes, get_tokenizer


class ClassificationReport(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, x_test, y_test, labels):
        self.labels = labels
        self.x_val = x_val
        self.y_val = self._one_hot_matrix_to_labels(y_val)
        self.x_test = x_test
        self.y_test = self._one_hot_matrix_to_labels(y_test)

    def _one_hot_matrix_to_labels(self, matrix):
        return [self.labels[np.argmax(one_hot)] for one_hot in matrix]

    def _classification_report(self, x, y):
        y_pred = self._one_hot_matrix_to_labels(self.model.predict(x))
        return classification_report(y, y_pred, labels=self.labels)

    def on_epoch_end(self, epoch, logs=None):
        report = self._classification_report(self.x_val, self.y_val)
        print("val_classification_report:")
        print(report)

    def on_train_end(self, logs=None):
        report = self._classification_report(self.x_test, self.y_test)
        print("test_classification_report:")
        print(report)


class DeepNeuralNetwork:
    def __init__(self, num_words, num_labels, tokenizer):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=(num_words,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_labels, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"]
        )

        self.model = model
        self.tokenizer = tokenizer

    def fit(self, x, y, x_test, y_test):
        x, x_val, y, y_val = train_test_split(x, y, test_size=0.1)

        self.model.summary()

        self.model.fit(
            x,
            y,
            epochs=5,
            batch_size=16,
            validation_data=(x_val, y_val),
            callbacks=[
                ClassificationReport(
                    x_val, y_val, x_test, y_test, labels=list(get_classes().values()),
                )
            ],
        )

    def save(self, path):
        self.model.save(path)


def load_model(path):
    return tf.keras.models.load_model(path)
