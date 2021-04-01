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

    f_pf = open('D:/Files/benchmarks/MacroNAS/C100/pf_test(error).p', 'rb')
    pf = p.load(f_pf)
    f_pf.close()

    path = 'D:/Files/RESULTS/processing'
    files = os.listdir(path)
    for result in files:
        folders = os.listdir(path + '/' + result)
        for folder in folders:
            f_eval = open(path + '/' + result + '/' + folder + '/no_eval_and_IGD.p', 'rb')
            n_evals = p.load(f_eval)[0]
            f_eval.close()

            path_ = path + '/' + result + '/' + folder + '/elitist_archive'
            gens = os.listdir(path_)
            gens = sorted(gens, key=lambda x: int(x.split('_')[-1][:-2]))
            IGD_ = []
            worst_f0 = -np.inf
            worst_f1 = -np.inf
            for gen in gens:
                f = open(path_ + '/' + gen, 'rb')
                ea = p.load(f)
                f.close()
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
                IGD_.append(calc_igd(pareto_front=pf, pareto_s=pf_test_))
                worst_f0 = max(worst_f0, np.max(pf_test_[:, 0]))
                worst_f1 = max(worst_f1, np.max(pf_test_[:, 1]))
            p.dump([n_evals, IGD_], open(path + '/' + result + '/' + folder + '/no_eval_and_IGD_.p', 'wb'))
            p.dump([worst_f0, worst_f1], open(path + '/' + result + '/' + folder + '/reference_point_.p', 'wb'))