import keras
import keras_metrics as km
import pandas as pd

from .livedoor_news import load_data, get_classifications


def metrics():
    functions = ["acc"]

    for label, _ in enumerate(get_classifications()):
        functions.append(km.categorical_precision(label=label))
        functions.append(km.categorical_recall(label=label))
        functions.append(km.categorical_f1_score(label=label))

    return functions


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
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=metrics()
    )
    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

    test_metrics = model.evaluate(x_test, y_test)
    print(f"test_loss: {test_metrics[0]:.4f}")
    print(f"test_acc: {test_metrics[1]:.4f}")

    df = pd.DataFrame(
        data=[test_metrics[i : i + 3] for i in range(2, len(test_metrics), 3)],
        index=[description for _, description in get_classifications()],
        columns=["test_precision", "test_recall", "test_f1_score"],
    )
    print(df)

    model.save(path)

    return model


def load_model(path="news_classifier_model.h5"):
    return keras.models.load_model(
        path,
        custom_objects={
            "categorical_precision": km.categorical_precision(),
            "categorical_recall": km.categorical_recall(),
            "categorical_f1_score": km.categorical_f1_score(),
        },
    )
