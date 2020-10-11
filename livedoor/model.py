import numpy as np
from rich import print
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from livedoor.config import CATEGORIES, MODEL_PATH
from livedoor.tokenizer import load_data, get_tokenizer


def categorical_to_labels(y):
    return [CATEGORIES.iloc[np.argmax(categorical)].label for categorical in y]


class ClassificationLogger(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, x_test, y_test):
        self.x_val = x_val
        self.y_val = categorical_to_labels(y_val)
        self.x_test = x_test
        self.y_test = categorical_to_labels(y_test)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = categorical_to_labels(self.model.predict(self.x_val))
        report = classification_report(y_pred, self.y_val)
        print("\n\nval_classification_report:")
        print(report)

    def on_train_end(self, logs=None):
        y_pred = categorical_to_labels(self.model.predict(self.x_test))
        report = classification_report(y_pred, self.y_test)
        print("\n\ntest_classification_report:")
        print(report)


class LivedoorNewsModel:
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.model = None

    @staticmethod
    def _build_model(num_words=1, num_labels=1):
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

        return model

    def fit_model(self, x_train, y_train, x_test, y_test):
        x_train = self.tokenizer.sequences_to_matrix(x_train)
        y_train = tf.keras.utils.to_categorical(y_train)
        x_test = self.tokenizer.sequences_to_matrix(x_test)
        y_test = tf.keras.utils.to_categorical(y_test)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1
        )
        logger = ClassificationLogger(x_val, y_val, x_test, y_test)

        model = tf.keras.wrappers.scikit_learn.KerasClassifier(
            self._build_model,
            num_words=x_train.shape[1],
            num_labels=y_train.shape[1],
            epochs=5,
            batch_size=16,
            validation_data=(x_val, y_val),
            callbacks=[logger],
        )

        model.fit(x_train, y_train)

        self.model = model

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        matrix = self.tokenizer.texts_to_matrix(texts)
        prediction = self.model.predict(matrix)

        if len(texts) == 1:
            return prediction[0]

        return prediction

    def save(self, path):
        self.model.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


def create_model():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = LivedoorNewsModel()
    model.fit_model(x_train, y_train, x_test, y_test)
    model.save(MODEL_PATH)


def load_model():
    model = LivedoorNewsModel()
    model.load(MODEL_PATH)
    return model
