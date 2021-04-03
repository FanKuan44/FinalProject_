import pickle as p
from wrap_pymoo.util.compare import find_better_idv
import numpy as np
from wrap_pymoo.util.IGD_calculating import calc_IGD
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    f_data = open(f'D:/Files/BENCHMARKS/101/mi_ma.p', 'rb')
    data = p.load(f_data)
    f_data.close()
    print(data)
    # plt.show()
