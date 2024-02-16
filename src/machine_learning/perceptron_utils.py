from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data')


def sign(value):
    if value > 0:
        return 1
    return -1


def predict(x: np.ndarray, w: np.ndarray):
    return sign(w @ x)


def PLA_forward(X: np.ndarray, w: np.ndarray, Y: np.ndarray, eta: None | float):
    if eta is None:
        eta = 1.0

    iters = 0
    for i in range(len(X)):
        if predict(X[i], w) == Y[i]:
            # print(f'Iter {i} passed')
            continue
        w = w + eta * Y[i] * X[i]
        # print(f'Iter {i} updated w:\n', w)
        iters += 1
        continue
    print('Iters:', iters)
    return iters


def pocket_forward(X: np.ndarray, w: np.ndarray, Y: np.ndarray):
    pass
