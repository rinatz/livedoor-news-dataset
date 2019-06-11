from pathlib import Path

import tensorflow as tf

from .livedoor_news import load_data


def create_model(path="news_classification_model.h5"):
    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    input_shape = (x_train.shape[1],)
    units = y_train.shape[1]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units, activation="softmax"),
        ]
    )
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(2000, 128, input_length=x_train.shape[1]),
    #     tf.keras.layers.Conv1D(32, 7, activation="relu"),
    #     tf.keras.layers.MaxPooling1D(5),
    #     tf.keras.layers.Conv1D(32, 7, activation="relu"),
    #     tf.keras.layers.GlobalMaxPooling1D(),
    #     tf.keras.layers.Dense(units, activation="softmax"),
    # ])
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}")

    model.save(path)

    return model


def load_model(path="news_classification_model.h5"):
    return tf.keras.models.load_model(path)
