import pickle as pk
import matplotlib.pyplot as plt
import numpy as np


def convert_X_to_hashX(x):
    if not isinstance(x, list):
        x = x.tolist()
    x = insert_to_list_x(x)
    x = remove_values_from_list_x(x, 'I')
    hashX = ''.join(x)
    return hashX


def insert_to_list_x(x):
    added = ['|', '|', '|']
    indices = [4, 8, 12]

    acc = 0
    for i in range(len(added)):
        x.insert(indices[i]+acc, added[i])
        acc += 1
    return x


def remove_values_from_list_x(x, val):
    return [value for value in x if value != val]


if __name__ == '__main__':
    # pf = pk.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
    # pf_X, pf_hashX, pf_F = pk.load(open(
    #     'D:/files/running/nas101_nsga_100_2X_False_False_0_False_False_0_usingDominatedIdv_09_m11_H13_M42/0/pareto_front.p',
    #     'rb'))
    # print(len(pf_X), len(pf_hashX), len(np.unique(pf_F, axis=0)))
    # plt.scatter(pf_F[:, 0], pf_F[:, 1])
    # plt.show()
    data = pk.load(open('D:/files/101_benchmark/nas101.p', 'rb'))
    count = 0
    total = 0
    for key in data.keys():
        if 1 - data[key]['val_acc'] < 0.067:
            count += 1
        total += 1
    print(count / total * 100, '%')
    exit()
