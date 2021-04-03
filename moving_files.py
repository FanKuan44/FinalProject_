import os

if __name__ == '__main__':
    path_old = 'D:/Files/RESULTS/201_C10/HP'
    path_new = 'D:/Files/RESULTS/201_C10/HP'

    files = os.listdir(path_new)
    for f in files:
        attrs = f.split('_')
        print(attrs)
        attrs[0] = '201!'
        old_f = path_old + '/' + f
        new_f = path_new + '/' + '_'.join(attrs)
        print(old_f, new_f)
        os.rename(old_f, new_f)
