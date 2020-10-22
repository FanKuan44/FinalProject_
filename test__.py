def check_better(x1, x2):
    if x1[0] <= x2[0] and x1[1] < x2[1]:
        return 'obj1'
    if x1[1] <= x2[1] and x1[0] < x2[0]:
        return 'obj1'
    if x2[0] <= x1[0] and x2[1] < x1[1]:
        return 'obj2'
    if x2[1] <= x1[1] and x2[0] < x1[0]:
        return 'obj2'
    """-----------------------------------"""
    # if x1[0] >= x2[0] and x1[1] > x2[1]:
    #     return 'obj1'
    # if x1[1] >= x2[1] and x1[0] > x2[0]:
    #     return 'obj1'
    # if x2[0] >= x1[0] and x2[1] > x1[1]:
    #     return 'obj2'
    # if x2[1] >= x1[1] and x2[0] > x1[0]:
    #     return 'obj2'
    """-----------------------------------"""
    # if x1[0] <= x2[0] and x1[1] > x2[1]:
    #     return 'obj1'
    # if x1[1] >= x2[1] and x1[0] < x2[0]:
    #     return 'obj1'
    # if x2[0] <= x1[0] and x2[1] > x1[1]:
    #     return 'obj2'
    # if x2[1] >= x1[1] and x2[0] < x1[0]:
    #     return 'obj2'
    """-----------------------------------"""
    # if x1[0] >= x2[0] and x1[1] < x2[1]:
    #     return 'obj1'
    # if x1[1] <= x2[1] and x1[0] > x2[0]:
    #     return 'obj1'
    # if x2[0] >= x1[0] and x2[1] < x1[1]:
    #     return 'obj2'
    # if x2[1] <= x1[1] and x2[0] > x1[0]:
    #     return 'obj2'
    return 'none'


# Validation accuracy - MMACs - Parameters
import pickle
import numpy as np
import matplotlib.pyplot as plt

# f_value = pickle.load(open('data.p', 'rb'))
# value = pickle.load(open('nas_101.p', 'rb'))
# benchmark = {}
# for i in range(len(f_value)):
#     benchmark[f_value[i][0]] = {'val_acc': 0, 'training_time': 0, 'params': 0}
#
# for i in range(len(f_value)):
#     benchmark[f_value[i][0]]['val_acc'] += value[i][0]
#     benchmark[f_value[i][0]]['training_time'] += value[i][1]
#     benchmark[f_value[i][0]]['params'] += value[i][2]
#
# for key in benchmark.keys():
#     benchmark[key]['val_acc'] /= 3
#     benchmark[key]['training_time'] /= 3
#     benchmark[key]['params'] /= 3
#
# pickle.dump(benchmark, open('nas_101_.p', 'wb'))
benchmark = pickle.load(open('data.p', 'rb'))
print(benchmark)
# min_max = pickle.load(open('bosman_benchmark/cifar100/min_max_cifar100.p', 'rb'))
#
# f_value_2_obj = np.zeros((len(f_value), 2))
# f_value_2_obj[:, 0] = 1 - f_value[:, 0]
# f_value_2_obj[:, 1] = f_value[:, 1]
#
# f_value_2_obj[:, 1] = (f_value_2_obj[:, 1] - min_max['min_MMACs']) / (min_max['max_MMACs'] - min_max['min_MMACs'])
#
# f_value_2_obj = f_value_2_obj[np.argsort(f_value_2_obj[:, 1])]
# f_value_2_obj = np.unique(f_value_2_obj, axis=0)
#
# pf = [f_value_2_obj[0]]
# rank = np.zeros(len(pf))
# for i in range(1, len(f_value_2_obj)):
#     print('i =', i, 'len_pf', len(pf))
#     flag = True
#     for j in range(len(pf)):
#         better = check_better(f_value_2_obj[i], pf[j])
#         if better == 'obj2':
#             flag = False
#             break
#         elif better == 'obj1':
#             rank[j] += 1
#     idx = (rank == 0)
#     pf = (np.array(pf)[rank == 0]).tolist()
#     rank = rank[rank == 0]
#     if flag:
#         pf.append(f_value_2_obj[i])
#         rank = rank.tolist()
#         rank.append(0.0)
#         rank = np.array(rank)
# pickle.dump(pf, open('pf_validation_MMACs_cifar100.p', 'wb'))
# pf = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
#
# plt.scatter(pf[:, 1], pf[:, 0], s=20, edgecolors='blue', facecolors='none', label='True-PF')
# plt.title('PF-MMACs-ValidationError-CIFAR10')
# plt.xlabel('MMACs')
# plt.ylabel('Validation Error')
# plt.legend()
# plt.grid()
# plt.show()
