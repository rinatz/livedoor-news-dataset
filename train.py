import tensorflow as tf

import dataset
from training import dnn


def main():
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    tokenizer = dataset.get_tokenizer()

    x_train = tokenizer.sequences_to_matrix(x_train, mode="tfidf")
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = tokenizer.sequences_to_matrix(x_test, mode="tfidf")
    y_test = tf.keras.utils.to_categorical(y_test)

    model = dnn.fit_model(x_train, y_train, x_test, y_test)
    dnn.save_model(model, "model.h5")


if __name__ == "__main__":
    main()
