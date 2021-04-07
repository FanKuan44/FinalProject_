import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure


def visualize_multi_results(path):
    plt.rc('font', family='Times New Roman')
    handles, labels = None, None
    fig, ax = plt.subplots(nrows=2, ncols=2)

    i = 0
    j = 0
    for folder in os.listdir(path):
        # print(folder)
        sub_folder = path + '/' + folder
        for folder_ in os.listdir(sub_folder):
            if folder_ == 'HP' or folder_ == 'IGD':
                if folder_ == 'HP':
                    j = 1
                else:
                    j = 0
                print(folder_)

                sub_sub_folder = sub_folder + '/' + folder_
                title = None

                for file in os.listdir(sub_sub_folder):
                    attrs = file.split('_')
                    label = attrs[5:9]
                    title = attrs[0] + '-' + attrs[1]
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
                        if label_ == 'NSGA-II':
                            ax[i][j].plot(data2[0], data1, 'k--', label=label_)
                        else:
                            ax[i][j].plot(data2[0], data1, label=label_)
                if i == 0:
                    if folder_ == 'HP':
                        title_ = 'Hypervolume'
                    else:
                        title_ = 'IGD'
                    ax[i][j].set_title(title_, fontsize=20, fontname='Times New Roman', fontweight='bold')
                print(title)
                if j == 0:
                    label__ = None
                    if title == '101-C10':
                        label__ = 'NAS-101'
                    elif title == '201-C10':
                        label__ = 'NAS-201-1'
                    elif title == '201-C100':
                        label__ = 'NAS-201-2'
                    elif title == '201-IN16-120':
                        label__ = 'NAS-201-3'
                    elif title == 'MacroNAS-C10':
                        label__ = 'MacroNAS-C10'
                    elif title == 'MacroNAS-C100':
                        label__ = 'MacroNAS-C100'
                    ax[i][j].set_ylabel(label__, fontsize=20, fontname='Times New Roman', fontweight='bold')
                ax[i][j].set_xscale('log')
                # ax[j].set_xscale('log')
                ax[i][j].grid()
                for label in (ax[i][j].get_xticklabels() + ax[i][j].get_yticklabels()):
                    label.set_fontsize(13)
                handles, labels = ax[i][j].get_legend_handles_labels()
        i += 1
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18, frameon=False)
    plt.show()


def visualize_per_result(path):
    # 28, 30, 28
    i = 0
    plt.rc('font', family='Times New Roman')
    for folder_ in os.listdir(path):
        if folder_ == 'HP' or folder_ == 'IGD':
            print(folder_)
            fig, ax = plt.subplots()
            sub_sub_folder = path + '/' + folder_
            title = None
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(12)
            for file in os.listdir(sub_sub_folder):
                attr = file.split('_')
                print(attr)
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
                    if bm == 'MacroNAS' or bm == '201' or bm == '101':
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
            handles, labels = ax.get_legend_handles_labels()
            if folder_ == 'HP':
                ylabel = 'Hypervolume'
                fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18, frameon=False)
            else:
                ylabel = 'IGD'
                if title == '101-C10':
                    title = 'NAS-101'
                elif title == '201-C10':
                    title = 'NAS-201-1'
                elif title == '201-C100':
                    title = 'NAS-201-2'
                elif title == '201-IN16-120':
                    title = 'NAS-201-3'
                elif title == 'MacroNAS-C10':
                    title = 'MacroNAS-'
                elif title == 'MacroNAS-C100':
                    title = 'MacroNAS-2'
                plt.title(title, fontsize=26, fontweight='bold', fontname='Times New Roman')
            plt.ylabel(ylabel, fontsize=26,  fontweight='bold', fontname='Times New Roman')
            plt.xscale('log')
            plt.grid()
            plt.show()
            i += 1


if __name__ == '__main__':
    PATH = f'D:/Files/RESULTS/processing/101_C10'
    # PATH = 'D:/Files/RESULTS/201_C10'
    # visualize_multi_results(PATH)
    visualize_per_result(PATH)
