import numpy as np


def kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3(p1, p2, p3):
    v_cp = p3 - p2
    dt = -v_cp[1] * (p1[0] - p2[0]) + v_cp[0] * (p1[1] - p2[1])
    if dt > 0:
        return 'tren'
    return 'duoi'


def cal_angle(p_middle, p_top, p_bot):
    x1 = p_top - p_middle
    x2 = p_bot - p_middle
    cosine_angle = (x1[0] * x2[0] + x1[1] * x2[1]) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)))

    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)
