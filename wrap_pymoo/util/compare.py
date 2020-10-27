def find_better_idv(f1, f2, position=None):
    """
    Kiem tra xem ca the nao tot hon.
    Neu ca the do nam o vi tri dau hoac cuoi thi chi xet 1 trong 2 fitness value, con lai thi xet binh thuong.
    =========================================================================================================

    Parameters:
    ----------
    :param f1: fitness value of first individual
    :param f2: fitness value of second individual
    :param position: position of individual in pareto front or set of knee solutions (first, last, none)
    =========================================================================================================

    Returns:
    ------
    :return:
    1: individual 1
    2: individual 2
    0: non-dominated
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
    Kiem tra xem ca the nao tot hon theo paper cua Bosman.
    Ngau nhien 1 gia tri alpha, tinh lai fitness value cua tung individual theo alpha va so sanh.
    =========================================================================================================

    Parameters:
    -----------
    :param alpha: value of alpha
    :param f1: the old fitness value of first individual
    :param f2: the old fitness value of first individual
    =========================================================================================================

    Returns:
    :return:
    1: individual 1
    2: individual 2
    """
    f1_new = alpha * f1[0] + (1 - alpha) * f1[1]
    f2_new = alpha * f2[0] + (1 - alpha) * f2[1]
    if f1_new < f2_new:
        return 1
    return 2
