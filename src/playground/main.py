import numpy as np


def cal(H):
    y = np.array([1, 2, 3, 4], dtype=np.float64)
    w = np.array([0.1, 0.35, 0.15, 0.4], dtype=np.float64)

    diff = np.abs(H - y)
    result = diff.T @ w
    return result


def run():
    print(cal(2.5))
    print(cal(2.85))
    print(cal(3))
    print(cal(4))


if __name__ == '__main__':
    run()
