import math


def get_finit_hoeffeding_bound(error_bound, data_num, hypothesis_num=1):
    return 2 * hypothesis_num * math.exp(-2 * data_num * math.pow(error_bound, 2))


if __name__ == '__main__':
    print(get_finit_hoeffeding_bound(0.8, 10))
