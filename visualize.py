# INPUT = 'input'
# OUTPUT = 'output'
# CONV3X3 = 'conv3x3-bn-relu'
# CONV1X1 = 'conv1x1-bn-relu'
# MAXPOOL3X3 = 'maxpool3x3'
#
# from nasbench import api
#
# # Use nasbench_full.tfrecord for full dataset (run download command above).
# nasbench = api.NASBench('C:/Users/ADMIN/Desktop/nsga-net-master/nasbench/nasbench_only108.tfrecord')
# cell = api.ModelSpec(
#   matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
#           [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
#           [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
#           [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
#           [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
#           [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
#           [0, 0, 0, 0, 0, 0, 0]],   # output layer
#   # Operations at the vertices of the module, matches order of matrix.
#   ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])
#
# # Querying multiple times may yield different results. Each cell is evaluated 3
# # times at each epoch budget and querying will sample one randomly.
# data = nasbench.query(cell)


"""Find the solutions whose rank equal 0"""
# f_value = np.array(pickle.load(open('f_value.p', 'rb')))
# f_value[:, 0] = 1 - f_value[:, 0]
# print("LOAD DATA DONE!!")
# rank = np.zeros(len(f_value))
# for i in range(len(f_value) - 1):
#     if rank[i] == 0:
#         for j in range(i + 1, len(f_value)):
#             better = check_better(f_value[i], f_value[j])
#             if better == 'obj1':
#                 rank[j] += 1
#             elif better == 'obj2':
#                 rank[i] += 1
#                 break
# pareto_front = []
# for i in range(len(rank)):
#     if rank[i] == 0:
#         pareto_front.append(f_value[i])
#         print(f_value[i])


# from deap.tools._hypervolume.pyhv import _HyperVolume
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import pickle
import pygmo as pg
import os
import numpy as np

now = datetime.now()
dir_name = now.strftime("%d_%m_%Y_%H_%M_%S")
try:
    os.mkdir('fig/' + dir_name)
except FileExistsError:
    pass


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
    # return 'none'


def visualize_2d(objective_0, objective_1, place_to_plot, label, axis_labels=('x', 'y'), color='blue', legend=False):
    place_to_plot.plot(objective_0, objective_1, c=color, label=label)
    place_to_plot.set_xlabel(axis_labels[0])
    place_to_plot.set_ylabel(axis_labels[1])
    if legend:
        place_to_plot.legend()


""" ------------------- Visualize "Pareto Front Approximate" -----------------------"""


def get_pfs(path, visualize_all=False):
    pfs_lst = []
    if visualize_all:
        number_of_pfs_to_visualize = len(os.listdir(path))
    else:
        number_of_pfs_to_visualize = 1
    for number_of_file in range(number_of_pfs_to_visualize):
        path_ = path + f'/{number_of_file}/pf_eval'
        pf_and_eval_folder = os.listdir(path_)

        # Sort files by name
        pf_and_eval_folder = sorted(pf_and_eval_folder, key=lambda x: int(x.split('_')[-1][:-2]))

        pf, _ = pickle.load(open(path_ + '/' + pf_and_eval_folder[-1], 'rb'))
        pfs_lst.append(pf)
    return pfs_lst


def visualize_pf(objective_0, objective_1, place_to_plot, label, axis_labels=('x', 'y'), color='blue', legend=False,
                 visualize_pf_true=False, marker='.', size=10):
    if visualize_pf_true:
        pareto_front_true = np.array(pickle.load(open('search/pf_val.p', 'rb')))
        first = True
        pf_objective_0 = pareto_front_true[:, 0]
        pf_objective_1 = pareto_front_true[:, 1]
        for i in range(len(pareto_front_true) - 1):
            if first:
                tmp_point = np.array([pf_objective_0[i + 1], pf_objective_1[i]])
                place_to_plot.plot([pf_objective_0[i], tmp_point[0]], [pf_objective_1[i], tmp_point[1]],
                                   c='#318609', label='PF True')
                place_to_plot.plot([tmp_point[0], pf_objective_0[i + 1]], [tmp_point[1], pf_objective_1[i + 1]],
                                   c='#318609')
                first = False
            else:
                tmp_point = np.array([pf_objective_0[i + 1], pf_objective_1[i]])
                place_to_plot.plot([pf_objective_0[i], tmp_point[0]], [pf_objective_1[i], tmp_point[1]],
                                   c='#318609')
                place_to_plot.plot([tmp_point[0], pf_objective_0[i + 1]], [tmp_point[1], pf_objective_1[i + 1]],
                                   c='#318609')

    first = True
    for i in range(len(objective_0) - 1):
        if first:
            tmp_point = np.array([objective_0[i + 1], objective_1[i]])
            place_to_plot.plot([objective_0[i], tmp_point[0]], [objective_1[i], tmp_point[1]], linestyle='--',
                               c=color, label=label, linewidth=1)
            place_to_plot.plot([tmp_point[0], objective_0[i + 1]], [tmp_point[1], objective_1[i + 1]], c=color,
                               linestyle='--', linewidth=1)
            first = False
        else:
            tmp_point = np.array([objective_0[i + 1], objective_1[i]])
            place_to_plot.plot([objective_0[i], tmp_point[0]], [objective_1[i], tmp_point[1]],
                               c=color, linestyle='--', linewidth=1)
            place_to_plot.plot([tmp_point[0], objective_0[i + 1]], [tmp_point[1], objective_1[i + 1]], c=color,
                               linestyle='--', linewidth=1)
    place_to_plot.scatter(objective_0, objective_1, c=color, marker=marker, s=size)
    place_to_plot.set_xlabel(axis_labels[0])
    place_to_plot.set_ylabel(axis_labels[1])
    if legend:
        place_to_plot.legend()


def visualize_scatter_pf(objective_0, objective_1, size, label, place_to_plot, benchmark=None,
                         color='blue', axis_labels=('x', 'y'), legend=False, visualize_pf_true=False):
    if visualize_pf_true:
        pf_true = None
        if benchmark == 'nas101':
            pf_true = pickle.load(open('101_benchmark/pf_valerror_trainingtime.p', 'rb'))
        elif benchmark == 'cifar10':
            pf_true = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
        elif benchmark == 'cifar100':
            pf_true = pickle.load(open('bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))
        plt.scatter(pf_true[:, 1], pf_true[:, 0], s=20, label='PF True', c='#318609')
    place_to_plot.scatter(objective_0, objective_1, s=size, facecolors='none', edgecolors=color, label=label)
    place_to_plot.set_xlabel(axis_labels[0])
    place_to_plot.set_ylabel(axis_labels[1])
    if legend:
        place_to_plot.legend()


def visualize_pf_approximate_2_algorithm(path1, path2, benchmark=None, visualize_all=False, plot_scatter=False,
                                         show_fig=False, save_fig=False, visualize_pf_true=False):
    try:
        os.mkdir('fig/' + dir_name + '/pf')
    except FileExistsError:
        pass

    pfs_lst_1 = get_pfs(path1, visualize_all=visualize_all)
    pfs_lst_2 = get_pfs(path2, visualize_all=visualize_all)

    number_of_pfs_visualize = len(pfs_lst_1)
    for pfs_i in range(number_of_pfs_visualize):
        fig, ax = plt.subplots(1)

        axis_lbs = []
        if benchmark == 'nas101':
            axis_lbs = ['Norm #Training time', 'Validation error']
        elif benchmark == 'cifar10' or benchmark == 'cifar100':
            axis_lbs = ['Norm #MMACs', 'Validation error']

        label1 = path1.split('_')[2:5]
        label2 = path2.split('_')[2:5]

        if plot_scatter:
            visualize_scatter_pf(objective_0=pfs_lst_1[pfs_i][:, 1], objective_1=pfs_lst_1[pfs_i][:, 0], size=100,
                                 place_to_plot=ax, axis_labels=axis_lbs, color='blue', label=label1,
                                 benchmark=benchmark, legend=True, visualize_pf_true=visualize_pf_true)
            visualize_scatter_pf(objective_0=pfs_lst_2[pfs_i][:, 1], objective_1=pfs_lst_2[pfs_i][:, 0], size=50,
                                 place_to_plot=ax, axis_labels=axis_lbs, color='red', label=label2,
                                 benchmark=benchmark, legend=True, visualize_pf_true=False)
        else:
            visualize_pf(objective_0=pfs_lst_1[pfs_i][:, 0], objective_1=pfs_lst_1[pfs_i][:, 1],
                         place_to_plot=ax, axis_labels=axis_lbs, color='blue', label=label1, legend=True,
                         visualize_pf_true=False, marker='s', size=20)
            visualize_pf(objective_0=pfs_lst_2[pfs_i][:, 0], objective_1=pfs_lst_2[pfs_i][:, 1],
                         place_to_plot=ax, axis_labels=axis_lbs, color='red', label=label2, legend=True,
                         visualize_pf_true=False, marker='^')
        plt.title(f'Elitist Archive Exp {pfs_i}')
        if save_fig:
            plt.savefig('fig/' + dir_name + '/pf/' + f'elitist_archive_exp_{pfs_i}')
            print('Figures are saved on ' + 'fig/' + dir_name + '/pf')
        if show_fig:
            plt.show()
        plt.clf()


"""------ Visualize "Distance from Pareto Front True to Pareto Front Approximate" and "Number of Evaluations" ------"""


def cal_euclid_distance(x1, x2):
    e_dis = np.sqrt(np.sum((x1 - x2) ** 2))
    return e_dis


def cal_dpf(pareto_front, pareto_s):
    d = 0
    for solution in pareto_front:
        d_ = np.inf
        for solution_ in pareto_s:
            d_ = min(cal_euclid_distance(solution, solution_), d_)
        d += d_
    return d / len(pareto_front)


def get_avg_dpf_and_evaluate(paths, pf_true):
    dpf_avg_each_path = []
    no_eval_avg_each_path = []

    for path in paths:
        dpf_each_exp = []
        no_eval_each_exp = []

        number_of_folders = len(os.listdir(path))
        min_eval = np.inf
        eval_gen_0 = 0
        for i in range(number_of_folders):
            path_ = path + f'/{i}/pf_eval'
            pf_and_eval_folder = os.listdir(path_)

            # Sort file by name follow alphabet
            pf_and_eval_folder = sorted(pf_and_eval_folder, key=lambda x: int(x.split('_')[-1][:-2]))

            _, no_eval = pickle.load(open(path_ + '/' + pf_and_eval_folder[-1], 'rb'))
            _, eval_gen_0 = pickle.load(open(path_ + '/' + pf_and_eval_folder[0], 'rb'))

            min_eval = min(min_eval, no_eval)

        upper = min_eval // eval_gen_0 * eval_gen_0

        for i in range(number_of_folders):
            path_ = path + f'/{i}/pf_eval'
            pf_and_eval_folder = os.listdir(path_)

            # Sort files by name
            pf_and_eval_folder = sorted(pf_and_eval_folder, key=lambda x: int(x.split('_')[-1][:-2]))

            dpf_each_gen = []
            no_eval_each_gen = []

            for file in pf_and_eval_folder:
                pf_approximate, no_eval = pickle.load(open(path_ + '/' + file, 'rb'))

                dpf = cal_dpf(pareto_front=pf_true, pareto_s=pf_approximate)

                dpf_each_gen.append(dpf)
                no_eval_each_gen.append(no_eval)

            # Interpolation
            new_no_eval_each_gen = np.arange(eval_gen_0, upper + eval_gen_0, eval_gen_0)

            f = interp1d(no_eval_each_gen, dpf_each_gen)
            new_dpf_each_gen = f(new_no_eval_each_gen)

            dpf_each_exp.append(new_dpf_each_gen)
            no_eval_each_exp.append(new_no_eval_each_gen)

        dpf_each_exp = np.array(dpf_each_exp)
        no_eval_each_exp = np.array(no_eval_each_exp)

        dpf_avg = np.sum(dpf_each_exp, axis=0) / len(dpf_each_exp)
        no_eval_avg = np.sum(no_eval_each_exp, axis=0) / len(no_eval_each_exp)

        dpf_avg_each_path.append(dpf_avg)
        no_eval_avg_each_path.append(no_eval_avg)

    return dpf_avg_each_path, no_eval_avg_each_path


def visualize_dpf_and_no_evaluations_2_algorithm(paths, benchmark=None, show_fig=False, save_fig=False):
    pf_true = None
    if benchmark == 'nas101':
        pf_true = pickle.load(open('101_benchmark/pf_valerror_trainingtime.p', 'rb'))
    elif benchmark == 'cifar10':
        pf_true = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
    elif benchmark == 'cifar100':
        pf_true = pickle.load(open('bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))

    dpf_avg_each_path, no_eval_avg_each_path = get_avg_dpf_and_evaluate(paths, pf_true)
    fig, ax = plt.subplots(1)
    axis_lbs = ['No.Evaluations', 'DPF']
    colors = ['red', 'blue', 'green']

    for i in range(len(paths)):
        label = paths[i].split('_')[2:5]

        visualize_2d(objective_0=no_eval_avg_each_path[i], objective_1=dpf_avg_each_path[i], place_to_plot=ax,
                     axis_labels=axis_lbs, color=colors[i], label=label, legend=True)

    plt.grid()
    plt.title(f'{benchmark}')
    if save_fig:
        plt.savefig('fig/' + dir_name + '/' + 'dpfs_eval')
        print('Figures are saved on ' + 'fig/' + dir_name)
    if show_fig:
        plt.show()
    plt.clf()


"""------------ Visualize "Hyper-Volume of Pareto Front Approximate" and "Number of Evaluations" ------------"""


def find_max_f0_f1_min_f0_f1(paths):
    min_f0 = np.inf
    max_f0 = -np.inf

    min_f1 = np.inf
    max_f1 = -np.inf

    for path in paths:
        number_of_folders = len(os.listdir(path))
        for folder_i in range(number_of_folders):
            path_ = path + f'/{folder_i}/pf_eval'

            pf_and_eval_folder = os.listdir(path_)

            # Sort files by name
            pf_and_eval_folder = sorted(pf_and_eval_folder, key=lambda x: int(x.split('_')[-1][:-2]))

            number_of_files = len(pf_and_eval_folder)

            for file_i in range(number_of_files):
                pf_approximate, _ = pickle.load(open(path_ + '/' + pf_and_eval_folder[file_i], 'rb'))

                min_f0 = min(min_f0, pf_approximate[-1, 0])
                max_f0 = max(max_f0, pf_approximate[0, 0])

                min_f1 = min(min_f1, pf_approximate[0, 1])
                max_f1 = max(max_f1, pf_approximate[-1, 1])

    return min_f0, max_f0, min_f1, max_f1


def cal_hyper_volume(pareto_front, min_f0, max_f0, min_f1, max_f1):
    rf = [max_f0 + 1e-5, max_f1 + 1e-5]
    tmp = [rf[0] - min_f0, rf[1] - min_f1]

    hyper_volume = pg.hypervolume(pareto_front.tolist())
    hyper_volume = hyper_volume.compute(rf)
    return 1 - hyper_volume / np.prod(tmp)


def get_avg_hp_and_evaluate(paths, min_f0, max_f0, min_f1, max_f1):
    hp_avg_each_path = []
    no_eval_avg_each_path = []

    for path in paths:
        hp_each_exp = []
        no_eval_each_exp = []

        min_eval = np.inf

        eval_gen_0 = 0

        number_of_folders = len(os.listdir(path))

        for folder_i in range(number_of_folders):
            path_ = path + f'/{folder_i}/pf_eval'

            pf_and_eval_folder = os.listdir(path_)

            # Sort files by name
            pf_and_eval_folder = sorted(pf_and_eval_folder, key=lambda x: int(x.split('_')[-1][:-2]))

            _, no_eval = pickle.load(open(path_ + '/' + pf_and_eval_folder[-1], 'rb'))

            _, eval_gen_0 = pickle.load(open(path_ + '/' + pf_and_eval_folder[0], 'rb'))

            min_eval = min(min_eval, no_eval)

        upper = min_eval // eval_gen_0 * eval_gen_0

        for folder_i in range(number_of_folders):
            path_ = path + f'/{folder_i}/pf_eval'

            pf_and_eval_folder = os.listdir(path_)

            # Sort files by name
            pf_and_eval_folder = sorted(pf_and_eval_folder, key=lambda x: int(x.split('_')[-1][:-2]))

            number_of_files = len(pf_and_eval_folder)

            hp_each_gen = []
            no_eval_each_gen = []

            for file_i in range(number_of_files):
                pf_approximate, no_eval = pickle.load(open(path_ + '/' + pf_and_eval_folder[file_i], 'rb'))

                hp = cal_hyper_volume(pf_approximate, min_f0, max_f0, min_f1, max_f1)

                hp_each_gen.append(hp)
                no_eval_each_gen.append(no_eval)

            # Interpolation
            new_no_eval_each_gen = np.arange(eval_gen_0, upper + eval_gen_0, eval_gen_0)

            f = interp1d(no_eval_each_gen, hp_each_gen)
            new_hp_each_gen = f(new_no_eval_each_gen)

            hp_each_exp.append(new_hp_each_gen)
            no_eval_each_exp.append(new_no_eval_each_gen)

        hp_each_exp = np.array(hp_each_exp)
        no_eval_each_exp = np.array(no_eval_each_exp)

        hp_avg = np.sum(hp_each_exp, axis=0) / len(hp_each_exp)
        no_eval_avg = np.sum(no_eval_each_exp, axis=0) / len(no_eval_each_exp)

        hp_avg_each_path.append(hp_avg)
        no_eval_avg_each_path.append(no_eval_avg)

    return hp_avg_each_path, no_eval_avg_each_path


def visualize_hp_and_no_evaluations_2_algorithm(paths, benchmark=None, show_fig=False, save_fig=False):
    min_f0, max_f0, min_f1, max_f1 = find_max_f0_f1_min_f0_f1(paths)
    hp_avg_each_path, no_eval_avg_each_path = get_avg_hp_and_evaluate(paths, min_f0, max_f0, min_f1, max_f1)

    fig, ax = plt.subplots(1)
    axis_lbs = ['No.Evaluations', '1 - Hypervolume']
    colors = ['red', 'blue', 'green']

    for i in range(len(paths)):
        label = paths[i].split('_')[2:5]
        visualize_2d(objective_0=no_eval_avg_each_path[i], objective_1=hp_avg_each_path[i], place_to_plot=ax,
                     axis_labels=axis_lbs, color=colors[i], label=label, legend=True)

    plt.grid()
    plt.title(f'{benchmark}')
    if save_fig:
        plt.savefig('fig/' + dir_name + '/' + 'hp_eval')
        print('Figures are saved on ' + 'fig/' + dir_name)
    if show_fig:
        plt.show()
    plt.clf()


"""------------------------------------ Main ------------------------------------"""


def main():
    path_1 = 'cifar10_popsize_100_True_False_24_10_2020_13_32_34'
    path_2 = 'cifar10_popsize_100_False_False_24_10_2020_13_46_12'
    path_3 = 'nas101_popsize_100_True_False_13_10_2020_10_05_31'

    paths = [path_1, path_2]

    benchmark = path_1.split('_')[0]

    visualize_dpf_and_no_evaluations_2_algorithm(paths=paths, benchmark=benchmark,
                                                 show_fig=False, save_fig=True)

    # visualize_pf_approximate_2_algorithm(path1=path_1, path2=path_2, benchmark=benchmark, visualize_all=True,
    #                                      plot_scatter=True, show_fig=False, save_fig=True, visualize_pf_true=True)

    visualize_hp_and_no_evaluations_2_algorithm(paths=paths, benchmark=benchmark, show_fig=False, save_fig=True)


if __name__ == '__main__':
    main()

