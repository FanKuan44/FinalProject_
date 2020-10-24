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
    # c = np.sqrt(np.sum((p_top - p_middle) ** 2))
    # a = np.sqrt(np.sum((p_bot - p_middle) ** 2))
    # b = np.sqrt(np.sum((p_bot - p_top) ** 2))
    #
    # cosine_angle = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    angle = np.arccos(cosine_angle)
    print(p_middle, p_top, p_bot, np.degrees(angle), 180 - np.degrees(angle))
    plt.show()
    return 360 - np.degrees(angle)


if __name__ == '__main__':
    cal_angle()
