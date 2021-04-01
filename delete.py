import pickle as p
from wrap_pymoo.util.compare import find_better_idv
import numpy as np
from wrap_pymoo.util.igd_calculating import calc_igd
import os

if __name__ == '__main__':
    f_data = open('D:/Files/benchmarks/MacroNAS/C100/data.p', 'rb')
    data = p.load(f_data)
    f_data.close()

    f_mi_ma = open('D:/Files/benchmarks/MacroNAS/C100/mi_ma.p', 'rb')
    mi_ma = p.load(f_mi_ma)
    f_mi_ma.close()

    print('load data - done')

    path = 'D:/Files/RESULTS/processing'
    files = os.listdir(path)
    for result in files:
        folders = os.listdir(path + '/' + result)
        for folder in folders:
            f_eval = open(path + '/' + result + '/' + folder + '/no_eval_and_IGD.p', 'rb')
            n_evals = p.load(f_eval)[0]
            f_eval.close()

            os.mkdir(path + '/' + result + '/' + folder + '/pf1_eval')

            path_ = path + '/' + result + '/' + folder + '/elitist_archive'
            gens = os.listdir(path_)
            gens = sorted(gens, key=lambda x: int(x.split('_')[-1][:-2]))
            for gen in gens:
                f = open(path_ + '/' + gen, 'rb')
                ea = p.load(f)
                f.close()
                n_gens = int(gen[4:-2])
                arch = [''.join(idv) for idv in ea]
                metrics = [[data[x]['MMACs'], data[x]['test_acc']] for x in arch]
                metrics = np.array(metrics)
                metrics[:, 0] = np.round((metrics[:, 0] - mi_ma['MMACs']['min']) /
                                         (mi_ma['MMACs']['max'] - mi_ma['MMACs']['min']), 6)
                metrics[:, 1] = np.round(1 - metrics[:, 1], 4)

                l = len(metrics)
                r = np.zeros(l)
                for i in range(l):
                    if r[i] == 0:
                        for j in range(i + 1, l):
                            better_idv = find_better_idv(f0_0=metrics[i][0], f0_1=metrics[i][1],
                                                         f1_0=metrics[j][0], f1_1=metrics[j][1])
                            if better_idv == 0:
                                r[j] += 1
                            elif better_idv == 1:
                                r[i] += 1
                                break
                pf_test_ = metrics[r == 0]
                pf_test_ = np.unique(pf_test_, axis=0)

                path__ = path + '/' + result + '/' + folder + f'/pf1_eval/pf_and_evaluated_gen_{n_gens}.p'
                p.dump([pf_test_, n_evals[n_gens]], open(path__, 'wb'))
        print('done')
