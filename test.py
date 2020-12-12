import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from wrap_pymoo.model.population import MyPopulation as Population


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


def encoding_nas201(X):
    # C = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    X = X.split('+')
    X_new = list()
    X_new.append(X[0][1:-3])

    node = X[1].split('|')[1:-1]

    for edge in node:
        X_new.append(edge[0:-2])

    node = (X[2].split('|'))[1:-1]

    for edge in node:
        X_new.append(edge[0:-2])

    for i in range(len(X_new)):
        if X_new[i] == 'none':
            X_new[i] = '0'
        elif X_new[i] == 'skip_connect':
            X_new[i] = '1'
        elif X_new[i] == 'nor_conv_1x1':
            X_new[i] = '2'
        elif X_new[i] == 'nor_conv_3x3':
            X_new[i] = '3'
        elif X_new[i] == 'avg_pool_3x3':
            X_new[i] = '4'
    X_new = ''.join(X_new)
    return X_new


def convert_to_hashX_NAS201(X):
    if not isinstance(X, list):
        X = X.tolist()
    hashX = ''.join(X)
    return hashX


def evaluate(X):
    F = np.full(2, fill_value=np.nan)

    if BENCHMARK_NAME == 'nas201':
        hashX = convert_to_hashX_NAS201(X)

        F[0] = (encode_nas201[hashX]['FLOP'] - min_max[0]) / (min_max[1] - min_max[0])
        F[1] = 1 - encode_nas201[hashX]['test-accuracy'] / 100

    return F


def sampling(n_samples):
    P = Population(n_samples)
    P_hashX = []
    i = 0
    allowed_choices = ['0', '1', '2', '3', '4']
    while i < n_samples:
        X = np.random.choice(allowed_choices, 6)
        hashX = convert_to_hashX_NAS201(X)
        if hashX not in P_hashX:
            P_hashX.append(hashX)

            F = evaluate(X=X)

            P[i].set('X', X)
            P[i].set('hashX', hashX)
            P[i].set('F', F)

            i += 1
    return P


def better_soln(f_soln1, f_soln2):
    if (f_soln1[0] <= f_soln2[0] and f_soln1[1] < f_soln2[1]) or (f_soln1[0] < f_soln2[0] and f_soln1[1] <= f_soln2[1]):
        return 1
    if (f_soln2[0] <= f_soln1[0] and f_soln2[1] < f_soln1[1]) or (f_soln2[0] < f_soln1[0] and f_soln2[1] <= f_soln1[1]):
        return 2
    return 0


def find_true_pf(X):
    rank = np.zeros(len(X))
    for i in range(len(X)):
        if rank[i] == 0:
            for j in range(i + 1, len(X)):
                better = better_soln(X[i], X[j])
                if better == 1:
                    rank[j] += 1
                elif better == 2:
                    rank[i] += 1
                    break
    pf = X[rank == 0]
    return pf


if __name__ == '__main__':
    # NAS201 = pk.load(open('D:/Files/NAS-Bench-201/CIFAR-10/encode_data.p', 'rb'))
    # F = []
    # for key in NAS201.keys():
    #     F.append(NAS201[key]['test-accuracy'])
    # F = sorted(F, reverse=True)
    # print(1 - F[len(F)*10//100] / 100)

    # mi_ma = pk.load(open('D:/Files/MacroNAS/CIFAR-10/mi_ma_MMACs.p', 'rb'))
    data = pk.load(open('D:/Files/MacroNAS/CIFAR-10/data.p', 'rb'))
    for key in data:
        print(data[key])
        break
    # print('Load data - Done')
    # F = []
    # for key in data.keys():
    #     F.append([data[key]['MMACs'], data[key]['val_acc']])
    # F = np.array(F)
    # F = np.round(F, 6)
    # print(F)
    # F = []
    # for key in data.keys():
    #     data[key]['val_acc'] = np.round(data[key]['val_acc'], 6)
    #     data[key]['MMACs'] = np.round(data[key]['MMACs'], 6)
    #     F.append([data[key]['MMACs'], data[key]['val_acc']])
    #
    # F = np.array(F)
    # F[:, 0] = (F[:, 0] - mi_ma[0]) / (mi_ma[1] - mi_ma[0])
    # F[:, 1] = 1 - F[:, 1]/100
    # F = np.round(F, 6)
    # F = np.unique(F, axis=0)
    # rank = np.zeros(len(F))
    #
    # for i in range(len(F)):
    #     if rank[i] == 0:
    #         for j in range(i + 1, len(F)):
    #             better = better_soln(F[i], F[j])
    #             if better == 1:
    #                 rank[j] += 1
    #             elif better == 2:
    #                 rank[i] += 1
    #                 break
    # pf = F[rank == 0]
    # pk.dump(pf, open('D:/Files/MacroNAS/CIFAR-10/pf.p', 'wb'))
    # pk.dump(data, open('D:/Files/MacroNAS/CIFAR-10/data.p', 'wb'))
    # plt.scatter(pf[:, 0], pf[:, 1])
    # plt.show()
    # pk.dump(pf, open('D:/Files/NAS-Bench-101/PF(nor)_Params-ValidAcc.p', 'wb'))

    # NAS101[:, 0], NAS101[:, 1] = NAS101[:, 1], NAS101[:, 0].copy()
    # print(NAS101)
    # pk.dump(NAS101, open('D:/Files/MacroNAS/CIFAR-100/PF(nor)_MMACs-ValidAcc.p', 'wb'))
    # CIFAR100 = pk.load(open('D:/Files/MacroNAS/CIFAR-100/PF(nor)_MMACs-ValidAcc.p', 'rb'))
    # print(CIFAR100 - NAS101)

    # NAS101[:, 0], NAS101[:, 1] = NAS101[:, 1], NAS101[:, 0].copy()
    # print(NAS101)
    # pk.dump(NAS101, open('D:/Files/NAS-Bench-101/PF(nor)_Params-ValidAcc.p', 'wb'))
    # NAS1011 = pk.load(open('D:/Files/NAS-Bench-101/PF(nor)_Params-ValidAcc.p', 'rb'))
    # print(NAS1011)



