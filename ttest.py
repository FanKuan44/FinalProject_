import pickle as pk
import os
import numpy as np

from scipy import stats
from statistics import stdev


if __name__ == '__main__':
    alpha = 0.01
    folder = 'D:/Files/FINAL-RESULTS/MicroNAS/101_C10'
    points = [1000, 2000, 5000, 10000, 30000]
    # points = [100, 200, 500, 1000, 3000]

    for folder_ in os.listdir(folder):
        if folder_ == 'IGD' or folder_ == 'HP':
            i = 0
            avg_dpfs_lst = []
            ori = []
            rs1 = []
            rs2 = []
            rs3 = []
            rs4 = []
            rs5 = []
            for path in os.listdir(folder + '/' + folder_):
                print(path)
                dpfs, evals = pk.load(open(f'{folder}/{folder_}/{path}', 'rb'))
                for value in points:
                    # print(value)
                    tmp = dpfs[:, np.where(evals[-1] == value)]
                    tmp = tmp.reshape(21)
                    # print(tmp)
                    if i == 0:
                        ori.append(tmp)
                    elif i == 1:
                        rs1.append(tmp)
                    elif i == 2:
                        rs2.append(tmp)
                    elif i == 3:
                        rs3.append(tmp)
                    elif i == 4:
                        rs4.append(tmp)
                    elif i == 5:
                        rs5.append(tmp)
                i += 1
            tmp_0 = []
            tmp_1 = []
            tmp_2 = []
            tmp_3 = []
            tmp_4 = []
            tmp_5 = []
            for i in range(len(ori)):
                print(points[i])
                # print('Ori')
                tmp_0.append(f'{np.round(np.mean(ori[i]), 6)} ({np.round(stdev(ori[i]), 6)})')
                # print('RS')
                tmp_1.append(f'{np.round(np.mean(rs1[i]), 6)} ({np.round(stdev(rs1[i]), 6)})')
                p_value1 = stats.ttest_ind(ori[i], rs1[i])[-1]
                # print(p_value1)
                if p_value1 > alpha:
                    print('Accept null hypothesis that the means are equal.')
                else:
                    print('Reject the null hypothesis that the means are equal.')
                tmp_2.append(f'{np.round(np.mean(rs2[i]), 6)} ({np.round(stdev(rs2[i]), 6)})')
                p_value2 = stats.ttest_ind(ori[i], rs2[i])[-1]
                # print(p_value2)
                if p_value2 > alpha:
                    print('Accept null hypothesis that the means are equal.')
                else:
                    print('Reject the null hypothesis that the means are equal.')
                tmp_3.append(f'{np.round(np.mean(rs3[i]), 6)} ({np.round(stdev(rs3[i]), 6)})')
                p_value3 = stats.ttest_ind(ori[i], rs3[i])[-1]
                # print(p_value3)
                if p_value3 > alpha:
                    print('Accept null hypothesis that the means are equal.')
                else:
                    print('Reject the null hypothesis that the means are equal.')
                tmp_4.append(f'{np.round(np.mean(rs4[i]), 6)} ({np.round(stdev(rs4[i]), 6)})')
                p_value4 = stats.ttest_ind(ori[i], rs4[i])[-1]
                # print(p_value4)
                if p_value4 > alpha:
                    print('Accept null hypothesis that the means are equal.')
                else:
                    print('Reject the null hypothesis that the means are equal.')
                tmp_5.append(f'{np.round(np.mean(rs5[i]), 6)} ({np.round(stdev(rs5[i]), 6)})')
                p_value5 = stats.ttest_ind(ori[i], rs5[i])[-1]
                # print(p_value5)
                if p_value5 > alpha:
                    print('Accept null hypothesis that the means are equal.')
                else:
                    print('Reject the null hypothesis that the means are equal.')
            for result in tmp_0:
                print(result)
            print('----')
            for result in tmp_1:
                print(result)
            print('----')
            for result in tmp_2:
                print(result)
            print('----')
            for result in tmp_3:
                print(result)
            print('----')
            for result in tmp_4:
                print(result)
            print('----')
            for result in tmp_5:
                print(result)
            print('----')