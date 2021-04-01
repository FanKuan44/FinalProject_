import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure


def visualize_multi_results(path):
    handles, labels = None, None
    fig, ax = plt.subplots(nrows=2, ncols=2)#, figsize=(100, 100))

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
                        label_ = f'NSGA-II w/ SM'
                    elif label[0] == 'True' and label[-2] == 'False':
                        label_ = f'NSGA-II w/ LS (k = {int(label[1])})'
                    elif label[0] == 'True' and label[-2] == 'True':
                        label_ = f'NSGA-II w/ SM + LS (k = {int(label[1])})'
                    else:
                        label_ = label
                    path_file = sub_sub_folder + '/' + file
                    data1, data2 = p.load(open(path_file, 'rb'))
                    data1 = np.sum(data1, axis=0) / len(data1)
                    if label_ == 'NSGA-II w/ LS (k = 2)' or label_ == 'NSGA-II w/ SM + LS (k = 2)':
                    # if label_ == 'NSGA-II w/ SM' or label_ == 'NSGA-II':
                        continue
                    else:
                        if label_ == 'NSGA-II':
                            ax[i][j].plot(data2[0], data1, 'k--', label=label_)
                        else:
                            ax[i][j].plot(data2[0], data1, label=label_)
                    # ax[j].plot(data2[0], data1, label=label_)
                if i == 0:
                    if folder_ == 'HP':
                        title_ = 'Hypervolume'
                    else:
                        title_ = 'IGD'
                    # ax[i][j].set_title(title_, fontsize=20, fontname='Times New Roman', fontweight='bold')
                    # ax[j].set_title(title_, fontsize=20, fontname='Times New Roman', fontweight='bold')
                print(title)
                if j == 0:
                    label__ = None
                    if title == 'NAS-Bench-101':
                        label__ = 'NAS-101'
                    elif title == 'NAS-Bench-201-CIFAR-10':
                        label__ = 'NAS-201-1'
                    elif title == 'NAS-Bench-201-CIFAR-100':
                        label__ = 'NAS-201-2'
                    elif title == 'NAS-Bench-201-ImageNet16-120':
                        label__ = 'NAS-201-3'
                    elif title == 'MacroNAS-CIFAR-10':
                        label__ = 'MacroNAS-C10'
                    elif title == 'MacroNAS-CIFAR-100':
                        label__ = 'MacroNAS-C100'
                    ax[i][j].set_ylabel(label__, fontsize=20, fontname='Times New Roman', fontweight='bold')
                    # ax[j].set_ylabel(label__, fontsize=20, fontname='Times New Roman', fontweight='bold')
                ax[i][j].set_xscale('log')
                # ax[j].set_xscale('log')
                ax[i][j].grid()
                # ax[j].grid()
                for label in (ax[i][j].get_xticklabels() + ax[i][j].get_yticklabels()):
                    label.set_fontsize(10)
                handles, labels = ax[i][j].get_legend_handles_labels()
                # handles, labels = ax[j].get_legend_handles_labels()
                j += 1
                if j == 2:
                    j = 0
                    i += 1
    plt.rc('font', family='Times New Roman')
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18, frameon=False)
    plt.show()


def visualize_per_result(path):
    # 28, 30, 28
    i = 0
    for folder_ in os.listdir(path):
        if folder_ == 'HP' or folder_ == 'DPFS':
            print(folder_)
            fig, ax = plt.subplots()
            sub_sub_folder = path + '/' + folder_
            title = None
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(12)
            for file in os.listdir(sub_sub_folder):
                label = file.split('_')[5:9]
                title = file.split('_')[0] + '-' + file.split('_')[1]
                bm = file.split('_')[0]

                if label[0] == 'False' and label[-2] == 'False':
                    label_ = 'NSGA-II'
                elif label[0] == 'False' and label[-2] == 'True':
                    label_ = f'NSGA-II w/ SM'
                elif label[0] == 'True' and label[-2] == 'False':
                    label_ = f'NSGA-II w/ LS (k = {int(label[1])})'
                elif label[0] == 'True' and label[-2] == 'True':
                    label_ = f'NSGA-II w/ SM + LS (k = {int(label[1])})'
                else:
                    label_ = label
                path_file = sub_sub_folder + '/' + file
                data1, data2 = p.load(open(path_file, 'rb'))
                data1 = np.sum(data1, axis=0) / len(data1)
                if label_ == 'NSGA-II w/ LS (k = 2)' or label_ == 'NSGA-II w/ SM + LS (k = 2)':
                    continue
                else:
                    if bm == 'MacroNAS':
                        if label_ == 'NSGA-II':
                            ax.plot(data2[0], data1, 'k--', label=label_)
                        elif label_ == 'NSGA-II w/ SM':
                            ax.plot(data2[0], data1, label=label_, c='#1f77b4')
                        elif label_ == 'NSGA-II w/ LS (k = 1)':
                            ax.plot(data2[0], data1, label=label_, c='orange')
                        elif label_ == 'NSGA-II w/ SM + LS (k = 1)':
                            ax.plot(data2[0], data1, label=label_, c='green')
                    else:
                        if label_ == 'NSGA-II':
                            ax.plot(data2[0], data1, 'k--')
                        elif label_ == 'NSGA-II w/ SM':
                            ax.plot(data2[0], data1, c='#1f77b4')
                        elif label_ == 'NSGA-II w/ LS (k = 1)':
                            ax.plot(data2[0], data1, c='orange')
                        elif label_ == 'NSGA-II w/ SM + LS (k = 1)':
                            ax.plot(data2[0], data1, c='green')
                    #     label_ == 'NSGA-II w/ LS (k = 1)' or label_ == 'NSGA-II w/ SM + LS (k = 1)':
                    #     ax.plot(data2[0], data1, label=label_)
                    # elif label_ == 'NSGA-II':
                    #     ax.plot(data2[0], data1, 'k--', label=label_)
                    # else:
                    #     ax.plot(data2[0], data1, label=label_)
            handles, labels = ax.get_legend_handles_labels()
            if folder_ == 'HP' or folder_ == 'HP_':
                ylabel = 'Hypervolume'
            else:
                ylabel = 'IGD'
            plt.ylabel(ylabel, fontsize=26,  fontweight='bold', fontname='Times New Roman')
            if i % 2 == 0:
                if title == 'NAS-Bench-101':
                    title = 'NAS-101'
                elif title == 'NAS-Bench-201-CIFAR-10':
                    title = 'NAS-201-1'
                elif title == 'NAS-Bench-201-CIFAR-100':
                    title = 'NAS-201-2'
                elif title == 'NAS-Bench-201-ImageNet16-120':
                    title = 'NAS-201-3'
                elif title == 'MacroNAS-CIFAR-10':
                    title = 'MacroNAS-'
                elif title == 'MacroNAS-CIFAR-100':
                    title = 'MacroNAS-2'
                plt.title(title, fontsize=26, fontweight='bold', fontname='Times New Roman')
            else:
                # plt.xlabel('#Evals', fontsize=20,  fontweight='bold', fontname='Times New Roman')
                # plt.legend(prop={'size': 20, 'family': 'Times New Roman'})
                plt.rc('font', family='Times New Roman')
                fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18, frameon=False)
            plt.xscale('log')
            # plt.yscale('log')
            plt.grid()
            # plt.legend(prop={'size': 20, 'family': 'Times New Roman'})
            # plt.rc('font', family='Times New Roman')
            # fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18, frameon=False)
            plt.show()
            i += 1


if __name__ == '__main__':
    benchmark = 'C100'
    PATH = f'D:/Files/RESULTS/MacroNAS_{benchmark}'
    # PATH = 'D:/Files/test'
    # visualize_multi_results(PATH)
    visualize_per_result(PATH)
