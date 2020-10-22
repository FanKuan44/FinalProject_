import pickle
import matplotlib.pyplot as plt
import numpy as np

# true_pf = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
# pf_approx_1 = pickle.load(open('50_50_False_False_10_10_2020_11_09_00/0/pf_eval/pf_and_evaluated_gen_2001.p', 'rb'))[0]
# pf_approx_1 = np.unique(pf_approx_1, axis=0)
# print(len(pf_approx_1))
# pf_approx_2 = pickle.load(open('50_50_True_False_10_10_2020_11_19_43/0/pf_eval/pf_and_evaluated_gen_122.p', 'rb'))[0]
# pf_approx_2 = np.unique(pf_approx_2, axis=0)
# print(len(pf_approx_2))
#
# plt.scatter(true_pf[:, 1], true_pf[:, 0], s=160, facecolors='none', label='PF True', edgecolors='#318609')
# plt.scatter(pf_approx_1[:, 1], pf_approx_1[:, 0], s=80, facecolors='none', label='PF Approx 1', edgecolors='red')
# plt.scatter(pf_approx_2[:, 1], pf_approx_2[:, 0], s=40, facecolors='none', label='PF Approx 2', edgecolors='blue')
#
# plt.grid()
# plt.legend()
# plt.show()

import pygmo as pg


def cal_hyper_volume(pf):
    hp = 0
    # pf = pf[np.argsort(pf[:, 0])]
    rf = [pf[0, 0] + 1e-3, pf[-1, 1] + 1e-3]
    tmp = [rf[0] - pf[-1, 0], rf[1] - pf[0, 1]]
    print(tmp)
    test_list = np.zeros((3, 2))
    test_list[0] = pf[0]
    test_list[-1] = pf[-1]
    test_list[1] = np.array([pf[-1, 0], pf[0, 1]])
    plt.scatter(test_list[:, 1], test_list[:, 0])
    plt.scatter(rf[1], rf[0])
    plt.show()
    hp = pg.hypervolume(test_list.tolist())
    hp = hp.compute(rf) / np.prod(tmp)
    print(hp)
    # hp += np.prod([rf[0] - pf[0, 0], rf[1] - pf[0, 1]])
    # for i in range(1, len(pf)):
    #     hp += np.prod([rf[0] - pf[i, 0], pf[i-1, 1] - pf[i, 1]])
    # print(hp)
    # plt.scatter(rf[0], rf[1])
    # plt.scatter(pf[:, 0], pf[:, 1], c='red')
    # plt.show()


# pareto_front = pickle.load(open('cifar100_popsize_100_False_True_13_10_2020_08_55_27/3/pf_eval/pf_and_evaluated_gen_0.p', 'rb'))[0]
# cal_hyper_volume(pareto_front)
x1 = np.array([1, 1, 1])
x2 = np.array([5, 5, 5])
x1[2:], x2[2:] = x2[2:], x1[2:].copy()
print(x1, x2)
# 0.0054013799651327805
