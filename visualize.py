import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from datetime import datetime
from pymoo.indicators.hv import _HyperVolume
from scipy.interpolate import interp1d


def visualize_2d(objective_0, objective_1, place_to_plot, label, axis_labels=('x', 'y'), legend=False):
    place_to_plot.plot(objective_0, objective_1, label=label)
    place_to_plot.set_xlabel(axis_labels[0])
    place_to_plot.set_ylabel(axis_labels[1])
    if legend:
        place_to_plot.legend()


# """ ------------------- Visualize "Pareto Front Approximate" -----------------------"""
#
#
# def get_pfs(path, visualize_all=False):
#     pfs_lst = []
#     if visualize_all:
#         number_of_pfs_to_visualize = len(os.listdir(path))
#     else:
#         number_of_pfs_to_visualize = 1
#     for number_of_file in range(number_of_pfs_to_visualize):
#         path_ = path + f'/{number_of_file}/pf_eval'
#         pf_and_eval_folder = os.listdir(path_)
#
#         # Sort files by name
#         pf_and_eval_folder = sorted(pf_and_eval_folder, key=lambda x: int(x.split('_')[-1][:-2]))
#
#         pf, _ = pickle.load(open(path_ + '/' + pf_and_eval_folder[-1], 'rb'))
#         pfs_lst.append(pf)
#     return pfs_lst
#
#
# def visualize_pf(objective_0, objective_1, place_to_plot, label, axis_labels=('x', 'y'), color='blue', legend=False,
#                  visualize_pf_true=False, marker='.', size=10):
#     if visualize_pf_true:
#         pareto_front_true = np.array(pickle.load(open('search/pf_val.p', 'rb')))
#         first = True
#         pf_objective_0 = pareto_front_true[:, 0]
#         pf_objective_1 = pareto_front_true[:, 1]
#         for i in range(len(pareto_front_true) - 1):
#             if first:
#                 tmp_point = np.array([pf_objective_0[i + 1], pf_objective_1[i]])
#                 place_to_plot.plot([pf_objective_0[i], tmp_point[0]], [pf_objective_1[i], tmp_point[1]],
#                                    c='#318609', label='PF True')
#                 place_to_plot.plot([tmp_point[0], pf_objective_0[i + 1]], [tmp_point[1], pf_objective_1[i + 1]],
#                                    c='#318609')
#                 first = False
#             else:
#                 tmp_point = np.array([pf_objective_0[i + 1], pf_objective_1[i]])
#                 place_to_plot.plot([pf_objective_0[i], tmp_point[0]], [pf_objective_1[i], tmp_point[1]],
#                                    c='#318609')
#                 place_to_plot.plot([tmp_point[0], pf_objective_0[i + 1]], [tmp_point[1], pf_objective_1[i + 1]],
#                                    c='#318609')
#
#     first = True
#     for i in range(len(objective_0) - 1):
#         if first:
#             tmp_point = np.array([objective_0[i + 1], objective_1[i]])
#             place_to_plot.plot([objective_0[i], tmp_point[0]], [objective_1[i], tmp_point[1]], linestyle='--',
#                                c=color, label=label, linewidth=1)
#             place_to_plot.plot([tmp_point[0], objective_0[i + 1]], [tmp_point[1], objective_1[i + 1]], c=color,
#                                linestyle='--', linewidth=1)
#             first = False
#         else:
#             tmp_point = np.array([objective_0[i + 1], objective_1[i]])
#             place_to_plot.plot([objective_0[i], tmp_point[0]], [objective_1[i], tmp_point[1]],
#                                c=color, linestyle='--', linewidth=1)
#             place_to_plot.plot([tmp_point[0], objective_0[i + 1]], [tmp_point[1], objective_1[i + 1]], c=color,
#                                linestyle='--', linewidth=1)
#     place_to_plot.scatter(objective_0, objective_1, c=color, marker=marker, s=size)
#     place_to_plot.set_xlabel(axis_labels[0])
#     place_to_plot.set_ylabel(axis_labels[1])
#     if legend:
#         place_to_plot.legend()
#
#
# def visualize_scatter_pf(objective_0, objective_1, size, label, place_to_plot, benchmark=None,
#                          color='blue', axis_labels=('x', 'y'), legend=False, visualize_pf_true=False):
#     if visualize_pf_true:
#         pf_true = None
#         if benchmark == 'nas101':
#             pf_true = pickle.load(open('101_benchmark/pf_valerror_trainingtime.p', 'rb'))
#         elif benchmark == 'cifar10':
#             pf_true = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
#         elif benchmark == 'cifar100':
#             pf_true = pickle.load(open('bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))
#         plt.scatter(pf_true[:, 1], pf_true[:, 0], s=20, label='PF True', c='#318609')
#     place_to_plot.scatter(objective_0, objective_1, s=size, facecolors='none', edgecolors=color, label=label)
#     place_to_plot.set_xlabel(axis_labels[0])
#     place_to_plot.set_ylabel(axis_labels[1])
#     if legend:
#         place_to_plot.legend()
#
#
# def visualize_pf_approximate_2_algorithm(path1, path2, benchmark=None, visualize_all=False, plot_scatter=False,
#                                          show_fig=False, save_fig=False, visualize_pf_true=False):
#     try:
#         os.mkdir('fig/' + dir_name + '/pf')
#     except FileExistsError:
#         pass
#
#     pfs_lst_1 = get_pfs(path1, visualize_all=visualize_all)
#     pfs_lst_2 = get_pfs(path2, visualize_all=visualize_all)
#
#     number_of_pfs_visualize = len(pfs_lst_1)
#     for pfs_i in range(number_of_pfs_visualize):
#         fig, ax = plt.subplots(1)
#
#         axis_lbs = []
#         if benchmark == 'nas101':
#             axis_lbs = ['Norm #Training time', 'Validation error']
#         elif benchmark == 'cifar10' or benchmark == 'cifar100':
#             axis_lbs = ['Norm #MMACs', 'Validation error']
#
#         label1 = path1.split('_')[2:5]
#         label2 = path2.split('_')[2:5]
#
#         if plot_scatter:
#             visualize_scatter_pf(objective_0=pfs_lst_1[pfs_i][:, 1], objective_1=pfs_lst_1[pfs_i][:, 0], size=100,
#                                  place_to_plot=ax, axis_labels=axis_lbs, color='blue', label=label1,
#                                  benchmark=benchmark, legend=True, visualize_pf_true=visualize_pf_true)
#             visualize_scatter_pf(objective_0=pfs_lst_2[pfs_i][:, 1], objective_1=pfs_lst_2[pfs_i][:, 0], size=50,
#                                  place_to_plot=ax, axis_labels=axis_lbs, color='red', label=label2,
#                                  benchmark=benchmark, legend=True, visualize_pf_true=False)
#         else:
#             visualize_pf(objective_0=pfs_lst_1[pfs_i][:, 0], objective_1=pfs_lst_1[pfs_i][:, 1],
#                          place_to_plot=ax, axis_labels=axis_lbs, color='blue', label=label1, legend=True,
#                          visualize_pf_true=False, marker='s', size=20)
#             visualize_pf(objective_0=pfs_lst_2[pfs_i][:, 0], objective_1=pfs_lst_2[pfs_i][:, 1],
#                          place_to_plot=ax, axis_labels=axis_lbs, color='red', label=label2, legend=True,
#                          visualize_pf_true=False, marker='^')
#         plt.title(f'Elitist Archive Exp {pfs_i}')
#         if save_fig:
#             plt.savefig('fig/' + dir_name + '/pf/' + f'elitist_archive_exp_{pfs_i}')
#             print('Figures are saved on ' + 'fig/' + dir_name + '/pf')
#         if show_fig:
#             plt.show()
#         plt.clf()


'''------ Visualize "Distance from True Pareto Front to Approximate Pareto Front" and "Number of Evaluations" ------'''


def get_avg_dpfs_and_no_evaluations(paths):
    dpfs_avg_each_path = []
    no_evaluations_avg_each_path = []

    for path in paths:
        no_evaluations_each_exp = []
        dpfs_each_exp = []

        number_of_experiments = len(os.listdir(path))
        min_total_no_evaluations = np.inf
        no_evaluations_gen_0 = 0

        for i in range(number_of_experiments):
            path_ = path + f'/{i}'
            total_no_evaluations, _ = pickle.load(open(f'{path_}/no_eval_and_dpfs.p', 'rb'))
            no_evaluations_gen_0 = total_no_evaluations[0]
            min_total_no_evaluations = min(min_total_no_evaluations, total_no_evaluations[-1])

        max_total_no_evaluations = min_total_no_evaluations // no_evaluations_gen_0 * no_evaluations_gen_0

        for i in range(number_of_experiments):
            path_ = path + f'/{i}'

            total_no_evaluations, total_dpfs = pickle.load(open(f'{path_}/no_eval_and_dpfs.p', 'rb'))

            # Interpolation
            new_no_eval_each_gen = np.arange(no_evaluations_gen_0, max_total_no_evaluations + 1, no_evaluations_gen_0//2)

            f = interp1d(total_no_evaluations, total_dpfs)
            new_dpf_each_gen = f(new_no_eval_each_gen)

            dpfs_each_exp.append(new_dpf_each_gen)
            no_evaluations_each_exp.append(new_no_eval_each_gen)

            print(new_no_eval_each_gen)

        dpfs_each_exp = np.array(dpfs_each_exp)
        no_evaluations_each_exp = np.array(no_evaluations_each_exp)

        pickle.dump([dpfs_each_exp, no_evaluations_each_exp], open(f'{FOLDER}/DPFS/{path.split("/")[-1]}_dpfs.p', 'wb'))

        dpf_avg = np.sum(dpfs_each_exp, axis=0) / len(dpfs_each_exp)
        no_eval_avg = np.sum(no_evaluations_each_exp, axis=0) / len(no_evaluations_each_exp)

        dpfs_avg_each_path.append(dpf_avg)
        no_evaluations_avg_each_path.append(no_eval_avg)

    return dpfs_avg_each_path, no_evaluations_avg_each_path


def visualize_dpfs_and_no_evaluations_algorithms(paths, show_fig=False, save_fig=False, log_x=True, log_y=True):
    dpfs_avg_each_path, no_evaluations_avg_each_path = get_avg_dpfs_and_no_evaluations(paths)
    fig, ax = plt.subplots(1)
    axis_lbs = ['No.Evaluations', 'DPFS']

    for i in range(len(paths)):
        print(paths[i].split('_'))
        label = paths[i].split('_')[1:]
        # label = paths[i].split('_')[1:-4]
        # if label[3] == 'False' and label[4] == 'False' and label[6] == 'False' and label[7] == 'False':
        #     label_ = 'NSGA-II original'
        # elif label[3] == 'False' and label[4] == 'False' and label[6] == 'False' and label[7] == 'True':
        #     label_ = f'NSGA-II with Using Surrogate Model'
        # elif label[3] == 'False' and label[4] == 'True' and label[6] == 'False' and label[7] == 'False':
        #     label_ = f'NSGA-II with IPS k = {int(label[5])}'
        # elif label[3] == 'False' and label[4] == 'True' and label[6] == 'False' and label[7] == 'True':
        #     label_ = f'NSGA-II with Using Surrogate Model + IPS k = {int(label[5])}'
        # else:
        label_ = label
        visualize_2d(objective_0=no_evaluations_avg_each_path[i], objective_1=dpfs_avg_each_path[i], place_to_plot=ax,
                     axis_labels=axis_lbs, label=label_, legend=True)

    plt.grid()
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    title = paths[-1].split('_')[0].split('/')[-1]
    pop_size = paths[-1].split('_')[2]
    plt.title(title + f' - pop size: {pop_size}')
    if save_fig:
        plt.savefig('fig/' + dir_name + '/' + 'dpfs_eval')
        print('Figures are saved on ' + 'fig/' + dir_name)
        plt.clf()
    if show_fig:
        plt.show()


'''------------ Visualize "Hyper-Volume of Pareto Front Approximate" and "Number of Evaluations" ------------'''


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

                min_f0 = min(min_f0, pf_approximate[0, 0])
                max_f0 = max(max_f0, pf_approximate[-1, 0])

                min_f1 = min(min_f1, pf_approximate[-1, 1])
                max_f1 = max(max_f1, pf_approximate[0, 1])

    return min_f0, max_f0, min_f1, max_f1


def cal_hyper_volume(hp, pareto_front, rf):
    hyper_volume = hp.compute(pareto_front)
    return hyper_volume / np.prod(rf)


def get_avg_hp_and_evaluate(paths, hp_calculate, rf):
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

                hp = cal_hyper_volume(hp=hp_calculate, pareto_front=pf_approximate, rf=rf)

                if len(hp_each_gen) == 0:
                    hp_each_gen.append(hp)
                    no_eval_each_gen.append(no_eval)
                else:
                    if no_eval == no_eval_each_gen[-1]:
                        hp_each_gen[-1] = hp
                    else:
                        hp_each_gen.append(hp)
                        no_eval_each_gen.append(no_eval)

            # Interpolation
            new_no_eval_each_gen = np.arange(eval_gen_0, upper + 1, eval_gen_0//2)

            f = interp1d(no_eval_each_gen, hp_each_gen)
            new_hp_each_gen = f(new_no_eval_each_gen)

            hp_each_exp.append(new_hp_each_gen)
            no_eval_each_exp.append(new_no_eval_each_gen)

        hp_each_exp = np.array(hp_each_exp)
        no_eval_each_exp = np.array(no_eval_each_exp)

        pickle.dump([hp_each_exp, no_eval_each_exp], open(f'{FOLDER}/HP/{path.split("/")[-1]}_hp.p', 'wb'))

        hp_avg = np.sum(hp_each_exp, axis=0) / len(hp_each_exp)

        no_eval_avg = np.sum(no_eval_each_exp, axis=0) / len(no_eval_each_exp)

        hp_avg_each_path.append(hp_avg)
        no_eval_avg_each_path.append(no_eval_avg)

    return hp_avg_each_path, no_eval_avg_each_path


def visualize_hp_and_no_evaluations_algorithms(paths, show_fig=False, save_fig=False, log_x=True, log_y=True):
    _, max_f0, _, max_f1 = find_max_f0_f1_min_f0_f1(paths)
    rf = [max_f0 + 1e-5, max_f1 + 1e-5]
    hp_calculate = _HyperVolume(rf)

    hp_avg_each_path, no_eval_avg_each_path = get_avg_hp_and_evaluate(paths, hp_calculate, rf)

    fig, ax = plt.subplots(1)
    axis_lbs = ['No.Evaluations', 'Hypervolume']

    for i in range(len(paths)):
        label = paths[i].split('_')[1:]
        # label = paths[i].split('_')[1:-4]
        # if label[3] == 'False' and label[4] == 'False' and label[6] == 'False' and label[7] == 'False':
        #     label_ = 'NSGA-II original'
        # elif label[3] == 'False' and label[4] == 'False' and label[6] == 'False' and label[7] == 'True':
        #     label_ = f'NSGA-II with Using Surrogate Model'
        # elif label[3] == 'False' and label[4] == 'True' and label[6] == 'False' and label[7] == 'False':
        #     label_ = f'NSGA-II with IPS k = {int(label[5])}'
        # elif label[3] == 'False' and label[4] == 'True' and label[6] == 'False' and label[7] == 'True':
        #     label_ = f'NSGA-II with Using Surrogate Model + IPS k = {int(label[5])}'
        # else:
        label_ = label
        visualize_2d(objective_0=no_eval_avg_each_path[i], objective_1=hp_avg_each_path[i], place_to_plot=ax,
                     axis_labels=axis_lbs, label=label_, legend=True)
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    plt.grid()
    title = paths[-1].split('_')[0].split('/')[-1]
    pop_size = paths[-1].split('_')[2]
    plt.title(title + f' - pop size: {pop_size}')
    if save_fig:
        plt.savefig('fig/' + dir_name + '/' + 'hp_eval')
        print('Figures are saved on ' + 'fig/' + dir_name)
        plt.clf()
    if show_fig:
        plt.show()


''' ------------------------------------ Main ------------------------------------ '''


def main():
    paths = []
    for path in os.listdir(FOLDER):
        paths.append(FOLDER + '/' + path)
    os.mkdir(FOLDER + '/' + 'DPFS')
    os.mkdir(FOLDER + '/' + 'HP')

    visualize_dpfs_and_no_evaluations_algorithms(paths=paths, show_fig=True, save_fig=False, log_x=LOG_X, log_y=LOG_Y)

    # visualize_pf_approximate_2_algorithm(path1=path_1, path2=path_2, benchmark=benchmark, visualize_all=True,
    #                                      plot_scatter=True, show_fig=False, save_fig=True, visualize_pf_true=True)

    visualize_hp_and_no_evaluations_algorithms(paths=paths, show_fig=True, save_fig=False, log_x=LOG_X, log_y=LOG_Y)


if __name__ == '__main__':
    SAVE = False
    LOG_X = True
    LOG_Y = False
    FOLDER = 'D:/Files/running/NAS-Bench-201-ImageNet16-120-Final'
    if SAVE:
        now = datetime.now()
        dir_name = now.strftime('%d_%m_%Y_%H_%M_%S')
        os.mkdir('fig/' + dir_name)
    main()
