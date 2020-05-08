import numpy as np

import dataset


def categorical_to_labels(y):
    return [dataset.LABELS[np.argmax(categorical)] for categorical in y]
