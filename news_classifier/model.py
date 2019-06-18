import keras

from .livedoor_news import load_data


def create_model(path="news_classifier_model.h5"):
    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    input_shape = (x_train.shape[1],)
    units = y_train.shape[1]

    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=input_shape),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units, activation="softmax"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}")

    model.save(path)

    return model


def load_model(path="news_classifier_model.h5"):
    return keras.models.load_model(path)
