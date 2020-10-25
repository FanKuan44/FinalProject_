import pickle
import numpy as np
import matplotlib.pyplot as plt


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


tmp = 14.284508917973143


def cal_angle():
    p_top = np.array([0, 1])
    p_bot = np.array([0.5, -0.5])
    p_middle = np.array([-0.1, 0])
    dot = np.array([p_top, p_middle, p_bot])
    plt.plot(dot[:, 0], dot[:, 1], 'o')
    plt.plot(dot[:, 0], dot[:, 1])

    x1 = p_top - p_middle
    x2 = p_bot - p_middle
    cosine_angle = (x1[0] * x2[0] + x1[1] * x2[1]) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)))

    angle = np.arccos(cosine_angle)
    plt.show()
    return 360 - np.degrees(angle)


if __name__ == '__main__':
    # cal_angle()
    pf = pickle.load(open('101_benchmark/nas101.p', 'rb'))
    print(pf)
    min_max = pickle.load(open('101_benchmark/min_max_NAS101.p', 'rb'))
    print(min_max)
    # pf[:, 1] = (pf[:, 1] - min_max['min_model_params']) / (min_max['max_model_params'] - min_max['min_model_params'])
    # pf = np.unique(pf, axis=0)
    # pickle.dump(pf, open('101_benchmark/pf_validation_parameters.p', 'wb'))
    # plt.scatter(pf[:, 1], pf[:, 0])
    # plt.show()
