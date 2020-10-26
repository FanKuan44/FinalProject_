import numpy as np


def cal_euclid_distance(x1, x2):
    e_dis = np.sqrt(np.sum((x1 - x2) ** 2))
    return e_dis


def cal_dpfs(pareto_front, pareto_s):
    pareto_s = np.unique(pareto_s, axis=0)
    d = 0
    for solution in pareto_front:
        d_ = np.inf
        for solution_ in pareto_s:
            d_ = min(cal_euclid_distance(solution, solution_), d_)
        d += d_
    return d / len(pareto_front)
