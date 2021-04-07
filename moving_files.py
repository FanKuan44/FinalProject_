import os

if __name__ == '__main__':
    path_old = 'D:/Files/RESULTS/processing/101_C10/IGD'
    path_new = 'D:/Files/RESULTS/processing/101_C10/IGD'

    files = os.listdir(path_new)
    for f in files:
        attrs = f.split('_')
        print(attrs)
        attrs[0] = '101!'
        old_f = path_old + '/' + f
        new_f = path_new + '/' + '_'.join(attrs)
        print(old_f, new_f)
        os.rename(old_f, new_f)
