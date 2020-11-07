import numpy as np

from pymoo.operators.selection.tournament_selection import compare
from pymoo.util.dominator import Dominator
# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")
    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        if tournament_type == 'comp_by_dom_and_crowding':
            rel = Dominator.get_relation(pop[a].F, pop[b].F)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b

        elif tournament_type == 'comp_by_rank_and_crowding':
            S[i] = compare(a, pop[a].rank, b, pop[b].rank,
                           method='smaller_is_better')

        else:
            raise Exception("Unknown tournament type.")

        if np.isnan(S[i]):
            S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                           method='larger_is_better', return_random_if_equal=True)
    return S[:, None].astype(np.int)
