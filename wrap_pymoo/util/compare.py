def find_better_idv(f1, f2, position=None):
    """
    - Find the better individual between two individuals input.\n
    - If the individual has the highest f1 or f2, just check value f1 or f2 to find better individual.

    :param f1: fitness value of the first individual
    :param f2: fitness value of the second individual
    :param position: position of the individual in pareto front or set of knee solutions (first, last, none)

    :return: 0: non-dominated || 1: individual 1 || 2: individual 2
    """
    if position is None:
        if (f1[0] <= f2[0] and f1[1] < f2[1]) or (f1[0] < f2[0] and f1[1] <= f2[1]):
            return 1
        if (f2[0] <= f1[0] and f2[1] < f1[1]) or (f2[0] < f1[0] and f2[1] <= f1[1]):
            return 2
    elif position == 'first':
        if f1[0] < f2[0]:
            return 1
        if f2[0] <= f1[0]:
            return 2
    else:
        if f1[1] < f2[1]:
            return 1
        if f2[1] <= f1[1]:
            return 2
    return 0


def find_better_idv_bosman_ver(alpha, f1, f2):
    """
    - Find the better individual between two individuals input (followed Bosman's paper).
    https://arxiv.org/pdf/2004.08996.pdf

    :param alpha: value of alpha
    :param f1: the old fitness value of first individual
    :param f2: the old fitness value of first individual

    :return: 1: individual 1 || 2: individual 2
    """
    f1_new = alpha * f1[0] + (1 - alpha) * f1[1]
    f2_new = alpha * f2[0] + (1 - alpha) * f2[1]
    if f1_new < f2_new:
        return 1
    return 2
