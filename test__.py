import numpy as np
import pickle as pk


def better_soln(f_soln1, f_soln2):
    if (f_soln1[0] <= f_soln2[0] and f_soln1[1] < f_soln2[1]) or (f_soln1[0] < f_soln2[0] and f_soln1[1] <= f_soln2[1]):
        return 1
    if (f_soln2[0] <= f_soln1[0] and f_soln2[1] < f_soln1[1]) or (f_soln2[0] < f_soln1[0] and f_soln2[1] <= f_soln1[1]):
        return 2
    return 0


def convert_to_hashX(X):
    if not isinstance(X, list):
        X = X.tolist()
    hashX = ''.join(X)
    return hashX


def evaluate(X):
    F = np.full(2, fill_value=np.nan)

    hashX = convert_to_hashX(X)

    F[0] = (BENCHMARK_DATA[hashX]['FLOP'] - BENCHMARK_MIN_MAX[0]) / \
           (BENCHMARK_MIN_MAX[1] - BENCHMARK_MIN_MAX[0])
    F[1] = 1 - BENCHMARK_DATA[hashX]['test-accuracy'] / 100

    return F


def find_all_better_neighbors(X, F_X):
    count = 0
    for i in range(len(X)):
        choices = opt.copy()
        choices.remove(X[i])
        neighbors = X.copy()
        for choice in choices:
            neighbors[i] = choice
            F_neighbors = evaluate(neighbors)
            better = better_soln(F_X, F_neighbors)

            if better == 2 or better == 0:
                count += 1
    return count


if __name__ == '__main__':
    PATH_DATA = 'D:/Files'
    BENCHMARK_DATA = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-10/encode_data.p', 'rb'))
    BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-10/mi_ma_FLOPs.p', 'rb'))

    n_better_neighbors = 0
    n_soln = 0
    # opt = ['I', '1', '2']
    # idv = ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I']
    opt = ['0', '1', '2', '3', '4']
    idv = ['0', '0', '0', '0', '0', '0']

    for x0 in range(len(opt)):
        idv[0] = opt[x0]
        for x1 in range(len(opt)):
            idv[1] = opt[x1]
            for x2 in range(len(opt)):
                idv[2] = opt[x2]
                for x3 in range(len(opt)):
                    idv[3] = opt[x3]
                    for x4 in range(len(opt)):
                        idv[4] = opt[x4]
                        for x5 in range(len(opt)):
                            n_soln += 1
                            idv[5] = opt[x5]
                            F_idv = evaluate(X=idv)
                            # print(F_idv)
                            n_better_neighbors += find_all_better_neighbors(idv, F_idv)
                            # for x6 in range(3):
                            #     idv[6] = opt[x6]
                            #     for x7 in range(3):
                            #         idv[7] = opt[x7]
                            #         for x8 in range(3):
                            #             idv[8] = opt[x8]
                            #             for x9 in range(3):
                            #                 idv[9] = opt[x9]
                            #                 for x10 in range(3):
                            #                     idv[10] = opt[x10]
                            #                     for x11 in range(3):
                            #                         idv[11] = opt[x11]
                            #                         for x12 in range(3):
                            #                             idv[12] = opt[x12]
                            #                             for x13 in range(3):
                            #                                 n_soln += 1
                            #                                 idv[13] = opt[x13]
                            #                                 F_idv = evaluate(X=idv)
                            #                                 # print(F_idv)
                            #                                 n_better_neighbors += find_all_better_neighbors(idv, F_idv)
    print(n_soln)
    print(n_better_neighbors / n_soln)
