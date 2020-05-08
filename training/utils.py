import os
import random

import numpy as np
import tensorflow as tf

import dataset


def categorical_to_labels(y):
    return [dataset.LABELS[np.argmax(categorical)] for categorical in y]


def enable_debug_mode():
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    session = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=config
    )
    tf.compat.v1.keras.backend.set_session(session)
