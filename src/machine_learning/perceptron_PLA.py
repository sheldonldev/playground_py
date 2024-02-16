import numpy as np
from perceptron_utils import DATA_DIR, PLA_forward


def run(test_count: int, **kwargs):
    data = np.loadtxt(str(DATA_DIR.joinpath('perceptron_PLA_example.txt')))
    # print(data[:5])

    total_iters = 0

    for i in range(test_count):
        np.random.shuffle(data)
        # print(data[:5])

        b = np.ones((len(data), 1))
        X = np.hstack((b, data[:, :4]))
        # print('X:\n', X[:5])

        Y = data[:, 4]
        # print('Y:\n', Y[:5])

        w = np.zeros(X.shape[1])
        # print('w:\n', w)

        total_iters += PLA_forward(X, w, Y, eta=kwargs.get('eta'))
    print('Total iters:', total_iters)
    print('Everage iters:', total_iters / test_count)


if __name__ == "__main__":
    run(2000, eta=0.1)
