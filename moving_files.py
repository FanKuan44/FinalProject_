import os

if __name__ == '__main__':
    path_new = 'D:/Files/RESULTS/MacroNAS_C100/MacroNAS_C100_NSGA-II_100_2X_True_2_False_0_d01_m04_H00_M29_S41'
    path_old = 'D:/Files/RESULTS/old_C100/MacroNAS_C100_NSGA-II_100_2X_True_2_False_0_d30_m03_H09_M11_S45'

    for i in range(21):
        file_1_old = path_old + '/' + f'{i}' + '/no_eval_and_IGD_.p'
        file_2_old = path_old + '/' + f'{i}' + '/reference_point_.p'
        file_1_new = path_new + '/' + f'{i}' + '/no_eval_and_IGD_.p'
        file_2_new = path_new + '/' + f'{i}' + '/reference_point_.p'
        os.rename(file_1_old, file_1_new)
        os.rename(file_2_old, file_2_new)