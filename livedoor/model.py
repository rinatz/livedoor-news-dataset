import numpy as np
import pandas as pd
from rich import print
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from livedoor.config import CATEGORIES, MODEL_PATH
from livedoor.tokenizer import load_data, get_tokenizer


def categorical_to_genre(y):
    return [CATEGORIES.iloc[np.argmax(categorical)].genre for categorical in y]


class ClassificationLogger(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, x_test, y_test):
        self.x_val = x_val
        self.y_val = categorical_to_genre(y_val)
        self.x_test = x_test
        self.y_test = categorical_to_genre(y_test)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = categorical_to_genre(self.model.predict(self.x_val))
        report = classification_report(y_pred, self.y_val)
        print("\n\nval_classification_report:")
        print(report)

    def on_train_end(self, logs=None):
        y_pred = categorical_to_genre(self.model.predict(self.x_test))
        report = classification_report(y_pred, self.y_test)
        print("\n\ntest_classification_report:")
        print(report)


class LivedoorNewsModel:
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.model = None

    def fit_model(self, x_train, y_train, x_test, y_test):
        x_train = self.tokenizer.sequences_to_matrix(x_train)
        y_train = tf.keras.utils.to_categorical(y_train)
        x_test = self.tokenizer.sequences_to_matrix(x_test)
        y_test = tf.keras.utils.to_categorical(y_test)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1
        )
        logger = ClassificationLogger(x_val, y_val, x_test, y_test)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(y_train.shape[1], activation="softmax"),
            ]
        )
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"]
        )

        model.fit(
            x_train,
            y_train,
            epochs=5,
            batch_size=16,
            validation_data=(x_val, y_val),
            callbacks=[logger],
        )

        self.model = model

    def predict(self, text):
        matrix = self.tokenizer.texts_to_matrix([text])
        prediction = self.model.predict(matrix)[0]

        categories = pd.DataFrame(
            {
                "label": CATEGORIES.label,
                "genre": CATEGORIES.genre,
                "confidence": prediction,
            }
        ).sort_values("confidence", ascending=False)

        return categories

    def save(self, path):
        self.model.save(path)

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
