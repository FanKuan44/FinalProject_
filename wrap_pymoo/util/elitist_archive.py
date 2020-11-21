import numpy as np
from wrap_pymoo.util.compare import find_better_idv


def update_elitist_archive(new_idv_X_lst, new_idv_hashX_lst, new_idv_F_lst,
                           elitist_archive_X, elitist_archive_hashX, elitist_archive_F,
                           isDominated_hashX, first=False):
    """
    Update elitist archive and get the dominated individuals
    :param new_idv_X_lst: List of Individuals need to checked (X)
    :param new_idv_hashX_lst: List of Individuals need to checked (hashX)
    :param new_idv_F_lst: List of Individuals need to checked (F)
    :param elitist_archive_X: Current Elitist Archive (X)
    :param elitist_archive_hashX: Current Elitist Archive (hashX)
    :param elitist_archive_F: Current Elitist Archive (F)
    :param isDominated_hashX: Current List of Dominated Individuals
    :param first: Using to check type of current Elitist Archive is 'list' or not?
    :returns: current_elitist_archive_X, current_elitist_archive_X, current_elitist_archive_X, current_elitist_archive_hashX, current_elitist_archive_F, isDominated_hashX
    """
    if first:
        current_elitist_archive_X = elitist_archive_X.copy()
        current_elitist_archive_hashX = elitist_archive_hashX.copy()
        current_elitist_archive_F = elitist_archive_F.copy()
    else:
        current_elitist_archive_X = elitist_archive_X.copy().tolist()
        current_elitist_archive_hashX = elitist_archive_hashX.copy().tolist()
        current_elitist_archive_F = elitist_archive_F.copy().tolist()

    rank = np.zeros(len(current_elitist_archive_X))
    current_isDominated_hashX = isDominated_hashX

    for i in range(len(new_idv_X_lst)):  # Duyet cac phan tu trong list can check
        if new_idv_hashX_lst[i] not in current_elitist_archive_hashX:
            flag = True  # Check xem co bi dominated khong?
            for j in range(len(current_elitist_archive_X)):  # Duyet cac phan tu trong elitist archive hien tai
                better_idv = find_better_idv(new_idv_F_lst[i],
                                             current_elitist_archive_F[j])  # Kiem tra xem tot hon hay khong?
                if better_idv == 1:
                    rank[j] += 1
                    if current_elitist_archive_hashX[j] not in current_isDominated_hashX:
                        current_isDominated_hashX.append(current_elitist_archive_hashX[j])
                elif better_idv == 2:
                    flag = False
                    if new_idv_hashX_lst[i] not in current_isDominated_hashX:
                        current_isDominated_hashX.append(new_idv_hashX_lst[i])
                    break
            if flag:
                current_elitist_archive_X.append(np.array(new_idv_X_lst[i]))
                current_elitist_archive_hashX.append(np.array(new_idv_hashX_lst[i]))
                current_elitist_archive_F.append(np.array(new_idv_F_lst[i]))
                rank = np.append(rank, 0)

    current_elitist_archive_X = np.array(current_elitist_archive_X)[rank == 0]
    current_elitist_archive_hashX = np.array(current_elitist_archive_hashX)[rank == 0]
    current_elitist_archive_F = np.array(current_elitist_archive_F)[rank == 0]

    return current_elitist_archive_X, \
           current_elitist_archive_hashX, \
           current_elitist_archive_F, \
           current_isDominated_hashX

