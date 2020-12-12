import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure


def visualize_multi_results(path):
    handles, labels = None, None
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(100, 100))

    i = 0
    j = 0
    for folder in os.listdir(path):
        # print(folder)
        sub_folder = path + '/' + folder
        for folder_ in os.listdir(sub_folder):
            if folder_ == 'HP' or folder_ == 'DPFS':
                print(folder_)

                sub_sub_folder = sub_folder + '/' + folder_
                title = None
                for file in os.listdir(sub_sub_folder):
                    label = file.split('_')[4:8]
                    print(label)
                    title = file.split('_')[0]
                    if label[0] == 'False' and label[-2] == 'False':
                        label_ = 'NSGA-II'
                    elif label[0] == 'False' and label[-2] == 'True':
                        label_ = f'NSGA-II w/ USM'
                    elif label[0] == 'True' and label[-2] == 'False':
                        label_ = f'NSGA-II w/ IPS k={int(label[1])}'
                    elif label[0] == 'True' and label[-2] == 'True':
                        label_ = f'NSGA-II w/ USM + IPS k={int(label[1])}'
                    else:
                        label_ = label
                    path_file = sub_sub_folder + '/' + file
                    data1, data2 = pk.load(open(path_file, 'rb'))
                    data1 = np.sum(data1, axis=0) / len(data1)
                    ax[i][j].plot(data2[0], data1, label=label_)

                if i == 0:
                    if folder_ == 'HP':
                        title_ = '$Hypervolume$'
                    else:
                        title_ = r'$D_{P_F \rightarrow S}$'
                    ax[i][j].set_title(title_, fontsize=14)
                print(title)
                if j == 0:
                    label__ = None
                    if title == 'NAS-Bench-101':
                        label__ = 'NAS-101'
                    elif title == 'NAS-Bench-201-CIFAR-10':
                        label__ = 'NAS-201-C10'
                    elif title == 'NAS-Bench-201-CIFAR-100':
                        label__ = 'NAS-201-C100'
                    elif title == 'NAS-Bench-201-ImageNet16-120':
                        label__ = 'NAS-201-INet16'
                    elif title == 'MacroNAS-CIFAR-10':
                        label__ = 'MacroNAS-C10'
                    elif title == 'MacroNAS-CIFAR-100':
                        label__ = 'MacroNAS-C100'
                    ax[i][j].set_ylabel(label__, fontsize=11, fontstyle='italic')
                ax[i][j].set_xscale('log')
                ax[i][j].set_yscale('log')
                ax[i][j].grid()
                handles, labels = ax[i][j].get_legend_handles_labels()
                j += 1
                if j == 2:
                    j = 0
                    i += 1
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=11)
    plt.show()


def visualize_per_result(path):
    for folder in os.listdir(path):
        # print(folder)
        sub_folder = path + '/' + folder
        for folder_ in os.listdir(sub_folder):
            if folder_ == 'HP' or folder_ == 'DPFS':
                print(folder_)

                sub_sub_folder = sub_folder + '/' + folder_
                title = None
                for file in os.listdir(sub_sub_folder):
                    label = file.split('_')[4:8]
                    title = file.split('_')[0]
                    if label[0] == 'False' and label[-2] == 'False':
                        label_ = 'NSGA-II'
                    elif label[0] == 'False' and label[-2] == 'True':
                        label_ = f'NSGA-II w/ USM'
                    elif label[0] == 'True' and label[-2] == 'False':
                        label_ = f'NSGA-II w/ IPS k={int(label[1])}'
                    elif label[0] == 'True' and label[-2] == 'True':
                        label_ = f'NSGA-II w/ USM + IPS k={int(label[1])}'
                    else:
                        label_ = label
                    path_file = sub_sub_folder + '/' + file
                    data1, data2 = pk.load(open(path_file, 'rb'))
                    data1 = np.sum(data1, axis=0) / len(data1)
                    plt.plot(data2[0], data1, label=label_)

                if folder_ == 'HP':
                    ylabel = '$Hypervolume$'
                else:
                    ylabel = r'$D_{P_F \rightarrow S}$'
                plt.ylabel(ylabel, fontsize=11, fontstyle='italic')
                plt.xlabel('#Evaluations', fontsize=11, fontstyle='italic')
                plt.title(title, fontsize=14)
                plt.xscale('log')
                plt.grid()
                plt.legend()
                plt.show()


if __name__ == '__main__':
    PATH = 'D:/Files/FINAL-RESULTS/MicroNAS'
    visualize_multi_results(PATH)
    # visualize_per_result(PATH)
