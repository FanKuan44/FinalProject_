def find_better_idv(f0_0, f0_1, f1_0, f1_1, pos=None):
    """
    - Find the better individual between two individuals input.\n
    - If the individual has the highest f1 or f2, just check value f1 or f2 to find better individual.

    :param f0_0: the first objective value of first individual
    :param f0_1: the second objective value of first individual
    :param f1_0: the first objective value of second individual
    :param f1_1: the second objective value of second individual
    :param pos: position of the individual in pareto front or set of knee solutions (first, last, none)

    :return: -1: non-dominated || 0: individual 1 || 1: individual 2
    """
    if pos is None:
        if (f0_0 <= f1_0 and f0_1 < f1_1) or (f0_0 < f1_0 and f0_1 <= f1_1):
            return 0
        if (f1_0 <= f0_0 and f1_1 < f0_1) or (f1_0 < f0_0 and f1_1 <= f0_1):
            return 1
        return -1
    elif pos == 0:
        return 0 if f0_0 < f1_0 else 1
    else:
        return 0 if f0_1 < f1_1 else 1


def find_better_idv_bosman_ver(alpha, f0_0, f0_1, f1_1, f1_0):
    """
    - Find the better individual between two individuals input (followed Bosman's paper).
    https://arxiv.org/pdf/2004.08996.pdf

    :param alpha: value of alpha
    :param f0_0: the first objective value of first individual
    :param f0_1: the second objective value of first individual
    :param f1_0: the first objective value of second individual
    :param f1_1: the second objective value of second individual

    :return: 0: individual 1 || 1: individual 2
    """
    f1_new = alpha * f0_0 + (1 - alpha) * f0_1
    f2_new = alpha * f1_0 + (1 - alpha) * f1_1
    return 0 if f1_new < f2_new else 1

