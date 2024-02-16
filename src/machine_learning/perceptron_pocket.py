import numpy as np
from perceptron_utils import DATA_DIR, pocket_forward


def run(test_count: int):
    train_data = np.loadtxt(str(DATA_DIR.joinpath('perceptron_pocket_example_train.txt')))
    test_data = np.loadtxt(str(DATA_DIR.joinpath('perceptron_pocket_example_test.txt')))

    total_iters = 0

    for i in range(test_count):

        b = np.ones((len(train_data), 1))
        X = np.hstack((b, train_data[:, :4]))
        # print('X:\n', X[:5])

        Y = train_data[:, 4]
        # print('Y:\n', Y[:5])

        w = np.zeros(X.shape[1])
        # print('w:\n', w)

        total_iters += pocket_forward(X, w, Y)
    print('Total iters:', total_iters)
    print('Everage iters:', total_iters / test_count)


if __name__ == "__main__":
    run(2000)
