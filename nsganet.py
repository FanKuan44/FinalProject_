import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import timeit
import torch

from acc_predictor.factory import get_acc_predictor
from datetime import datetime

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.util.display import disp_multi_objective
from pymoo.util.non_dominated_sorting import NonDominatedSorting
# =========================================================================================================
# Implementation based on nsga2 from https://github.com/msu-coinlab/pymoo
# =========================================================================================================
from nasbench import wrap_api as api

from wrap_pymoo.algorithms.genetic_algorithm import GeneticAlgorithm

from wrap_pymoo.model.individual import MyIndividual as Individual
from wrap_pymoo.model.population import MyPopulation as Population

from wrap_pymoo.util.compare import find_better_idv
from wrap_pymoo.util.dpfs_calculating import cal_dpfs
from wrap_pymoo.util.find_knee_solutions import cal_angle, kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3
from wrap_pymoo.util.survial_selection import RankAndCrowdingSurvival
from wrap_pymoo.util.tournament_selection import binary_tournament

from wrap_pymoo.factory_nasbench import combine_matrix1D_and_opsINT, split_to_matrix1D_and_opsINT, create_model
from wrap_pymoo.factory_nasbench import encoding_ops, decoding_ops, encoding_matrix, decoding_matrix


def encode(X, benchmark):
    if isinstance(X, list):
        X = np.array(X)
    if benchmark == 'MacroNAS-CIFAR-10' or benchmark == 'MacroNAS-CIFAR-100':
        X = np.where(X == 'I', 0, X)
    encode_X = np.array(X, dtype=np.int)
    return encode_X


def encode_(X):
    hashX = []
    for x in X:
        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x)

        matrix_2D = encoding_matrix(matrix_1D)
        ops_STRING = encoding_ops(ops_INT)

        modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
        hashX.append(BENCHMARK_API.get_module_hash(modelspec))
    return np.array(hashX)


# Using for MarcoNAS-C10/C100
def insert_to_list_x(x):
    added = ['|', '|', '|']
    indices = [4, 8, 12]

    acc = 0
    for i in range(len(added)):
        x.insert(indices[i]+acc, added[i])
        acc += 1
    return x


def remove_values_from_list_X(X, val):
    return [value for value in X if value != val]


def convert_to_hashX(X, benchmark):
    if not isinstance(X, list):
        X = X.tolist()
    if benchmark == 'MacroNAS-CIFAR-10' or benchmark == 'MacroNAS-CIFAR-100':
        X = insert_to_list_x(X)
        X = remove_values_from_list_X(X, 'I')
    hashX = ''.join(X)
    return hashX


'''------------------------------------------------------------------------------------------------------------------'''


def crossover(P1, P2, typeC):
    O1, O2 = P1.copy(), P2.copy()

    if typeC == '1X':
        pt = np.random.randint(1, len(O1))

        O1[pt:], O2[pt:] = O2[pt:], O1[pt:].copy()

    elif typeC == '2X':
        pts = np.random.choice(range(1, len(O1) - 1), 2, replace=False)

        l = min(pts)
        u = max(pts)

        O1[l:u], O2[l:u] = O2[l:u], O1[l:u].copy()

    elif typeC == 'UX':
        pts = np.random.randint(0, 2, O1.shape, dtype=np.bool)

        O1[pts], O2[pts] = O2[pts], O1[pts].copy()

    return O1, O2


class NSGANet(GeneticAlgorithm):
    def __init__(self,
                 m_nEs,
                 typeC,
                 local_search_on_n_vars,
                 using_surrogate_model,
                 update_model_after_n_gens,
                 path,
                 **kwargs):

        set_if_none(kwargs, 'individual', Individual(rank=np.inf, crowding=-1))
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        super().__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective

        ''' Custom '''
        self.typeC = typeC

        self.alpha = 1

        self.DS = []
        self.A_X, self.A_hashX, self.A_F = [], [], []
        self.tmp_A_X, self.tmp_A_hashX, self.tmp_A_F = [], [], []

        self.dpfs = []
        self.no_eval = []

        self.m_nEs = m_nEs

        self.n_vars = local_search_on_n_vars
        self.using_surrogate_model = using_surrogate_model
        self.update_model_after_n_gens = update_model_after_n_gens
        self.surrogate_model = None
        self.update_model = True
        self.n_updates = 0
        self.training_data = []

        self.nEs = 0
        self.path = path

        self.F_total = []
        self.worst_f0 = -np.inf
        self.worst_f1 = -np.inf

    def update_fake_A(self, new_solution):
        X = new_solution.get('X')
        hashX = new_solution.get('hashX')
        F = new_solution.get('F')

        rank = np.zeros(len(self.tmp_A_X))
        if hashX not in self.tmp_A_hashX:
            flag = True
            for j in range(len(self.tmp_A_X)):

                better_idv = find_better_idv(F, self.tmp_A_F[j])
                if better_idv == 1:
                    rank[j] += 1

                elif better_idv == 2:
                    flag = False
                    break

            if flag:
                self.tmp_A_X.append(np.array(X))
                self.tmp_A_hashX.append(np.array(hashX))
                self.tmp_A_F.append(np.array(F))
                rank = np.append(rank, 0)

        self.tmp_A_X = np.array(self.tmp_A_X)[rank == 0].tolist()
        self.tmp_A_hashX = np.array(self.tmp_A_hashX)[rank == 0].tolist()
        self.tmp_A_F = np.array(self.tmp_A_F)[rank == 0].tolist()

    def update_A(self, new_solution):
        X = new_solution.get('X')
        hashX = new_solution.get('hashX')
        F = new_solution.get('F')

        rank = np.zeros(len(self.A_X))
        if hashX not in self.A_hashX:
            flag = True
            for j in range(len(self.A_X)):

                better_idv = find_better_idv(F, self.A_F[j])
                if better_idv == 1:
                    rank[j] += 1
                    if self.A_hashX[j] not in self.DS:
                        self.DS.append(self.A_hashX[j])

                elif better_idv == 2:
                    flag = False
                    if hashX not in self.DS:
                        self.DS.append(hashX)
                    break

            if flag:
                self.A_X.append(np.array(X))
                self.A_hashX.append(np.array(hashX))
                self.A_F.append(np.array(F))
                rank = np.append(rank, 0)

        self.A_X = np.array(self.A_X)[rank == 0].tolist()
        self.A_hashX = np.array(self.A_hashX)[rank == 0].tolist()
        self.A_F = np.array(self.A_F)[rank == 0].tolist()

    def evaluate(self, X, using_surrogate_model=False, count_nE=True):
        F = np.full(2, fill_value=np.nan)

        twice = False  # --> using on 'surrogate model method'
        if not using_surrogate_model:
            if BENCHMARK_NAME == 'MacroNAS-CIFAR-10' or BENCHMARK_NAME == 'MacroNAS-CIFAR-100':
                hashX = ''.join(X.tolist())

                F[0] = np.round((BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX[0]) /
                                (BENCHMARK_MIN_MAX[1] - BENCHMARK_MIN_MAX[0]), 6)
                F[1] = np.round(1 - BENCHMARK_DATA[hashX]['val_acc'] / 100, 6)

            elif BENCHMARK_NAME == 'NAS-Bench-101':
                matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X)

                matrix_2D = encoding_matrix(matrix_1D)
                ops_STRING = encoding_ops(ops_INT)

                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                hashX = BENCHMARK_API.get_module_hash(modelspec)

                F[0] = np.round((BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX[0]) /
                                (BENCHMARK_MIN_MAX[1] - BENCHMARK_MIN_MAX[0]), 6)
                F[1] = np.round(1 - BENCHMARK_DATA[hashX]['val_acc'], 6)

            elif BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-10' or BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-100' \
                    or BENCHMARK_NAME == 'NAS-Bench-201-ImageNet16-120':
                hashX = convert_to_hashX(X, BENCHMARK_NAME)

                F[0] = np.round((BENCHMARK_DATA[hashX]['FLOP'] - BENCHMARK_MIN_MAX[0]) /
                                (BENCHMARK_MIN_MAX[1] - BENCHMARK_MIN_MAX[0]), 6)
                F[1] = np.round(1 - BENCHMARK_DATA[hashX]['test-accuracy'] / 100, 6)

            if count_nE:
                self.nEs += 1
            self.F_total.append(F[1])
        else:
            if BENCHMARK_NAME == 'MacroNAS-CIFAR-10' or BENCHMARK_NAME == 'MacroNAS-CIFAR-100':
                encode_X = encode(X, BENCHMARK_NAME)
                hashX = ''.join(X.tolist())

                F[0] = np.round((BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX[0]) /
                                (BENCHMARK_MIN_MAX[1] - BENCHMARK_MIN_MAX[0]), 6)
                F[1] = self.surrogate_model.predict(np.array([encode_X]))[0][0]

                if F[1] < self.alpha:
                    twice = True
                    F[1] = np.round(1 - BENCHMARK_DATA[hashX]['val_acc'] / 100, 6)
                    self.nEs += 1

            elif BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-10' or BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-100' \
                    or BENCHMARK_NAME == 'NAS-Bench-201-ImageNet16-120':
                hashX = convert_to_hashX(X, BENCHMARK_NAME)
                encode_X = encode(X, BENCHMARK_NAME)

                F[0] = np.round((BENCHMARK_DATA[hashX]['FLOP'] - BENCHMARK_MIN_MAX[0]) /
                                (BENCHMARK_MIN_MAX[1] - BENCHMARK_MIN_MAX[0]), 6)
                F[1] = self.surrogate_model.predict(np.array([encode_X]))[0][0]

                if F[1] < self.alpha:
                    twice = True
                    F[1] = np.round(1 - BENCHMARK_DATA[hashX]['test-accuracy'] / 100, 6)
                    self.nEs += 1

            else:
                matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X)

                matrix_2D = encoding_matrix(matrix_1D)
                ops_STRING = encoding_ops(ops_INT)
                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                hashX = BENCHMARK_API.get_module_hash(modelspec)

                F[0] = np.round((BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX[0]) /
                                (BENCHMARK_MIN_MAX[1] - BENCHMARK_MIN_MAX[0]), 6)
                F[1] = self.surrogate_model.predict(np.array([X]))[0][0]

                if F[1] < self.alpha:
                    twice = True
                    F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc']
                    self.nEs += 1
            if twice:
                self.F_total.append(F[1])
        return F, twice

    @staticmethod
    def _create_surrogate_model(inputs, targets):
        surrogate_model = get_acc_predictor('mlp', inputs, targets, verbose=False)
        return surrogate_model

    def _sampling(self, n_samples):
        P = Population(n_samples)
        P_hashX = []
        i = 0

        if BENCHMARK_NAME == 'NAS-Bench-101':
            while i < n_samples:
                matrix_2D, ops_STRING = create_model()
                MS = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)

                if BENCHMARK_API.is_valid(MS):
                    hashX = BENCHMARK_API.get_module_hash(MS)

                    if hashX not in P_hashX:
                        P_hashX.append(hashX)

                        matrix_1D = decoding_matrix(matrix_2D)
                        ops_INT = decoding_ops(ops_STRING)

                        X = combine_matrix1D_and_opsINT(matrix=matrix_1D, ops=ops_INT)
                        F, _ = self.evaluate(X=X, using_surrogate_model=False)

                        P[i].set('X', X)
                        P[i].set('hashX', hashX)
                        P[i].set('F', F)
                        self.update_A(P[i])

                        i += 1

        else:
            if BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-10' or BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-100'\
                    or BENCHMARK_NAME == 'NAS-Bench-201-ImageNet16-120':
                l = 6
                opt = ['0', '1', '2', '3', '4']
            else:
                l = 14
                opt = ['I', '1', '2']

            while i < n_samples:
                X = np.random.choice(opt, l)
                hashX = convert_to_hashX(X, BENCHMARK_NAME)
                if hashX not in P_hashX:
                    P_hashX.append(hashX)

                    F, _ = self.evaluate(X=X)

                    P[i].set('X', X)
                    P[i].set('hashX', hashX)
                    P[i].set('F', F)
                    self.update_A(P[i])

                    i += 1

        return P

    def _crossover(self, P, pC=0.9):
        O = Population(len(P))
        O_hashX = []

        nCOs = 0  # --> Avoid to stuck

        i = 0
        full = False
        while not full:
            idx = np.random.choice(len(P), size=(len(P) // 2, 2), replace=False)
            P_ = P[idx]

            if BENCHMARK_NAME == 'NAS-Bench-101':
                for j in range(len(P_)):
                    if np.random.random() < pC:
                        new_O1_X, new_O2_X = crossover(P_[j][0].get('X'), P_[j][1].get('X'), self.typeC)

                        matrix1_1D, ops1_INT = split_to_matrix1D_and_opsINT(new_O1_X)
                        matrix2_1D, ops2_INT = split_to_matrix1D_and_opsINT(new_O2_X)

                        matrix1_2D = encoding_matrix(matrix1_1D)
                        matrix2_2D = encoding_matrix(matrix2_1D)

                        ops1_STRING = encoding_ops(ops1_INT)
                        ops2_STRING = encoding_ops(ops2_INT)

                        new_MS1 = api.ModelSpec(matrix=matrix1_2D, ops=ops1_STRING)
                        new_MS2 = api.ModelSpec(matrix=matrix2_2D, ops=ops2_STRING)

                        new_MS_lst = [new_MS1, new_MS2]
                        new_O_X_lst = [new_O1_X, new_O2_X]

                        for m in range(2):
                            if BENCHMARK_API.is_valid(new_MS_lst[m]):
                                new_O_hashX = BENCHMARK_API.get_module_hash(new_MS_lst[m])
                                if nCOs <= 100:
                                    if (new_O_hashX not in O_hashX) and (new_O_hashX not in self.DS):
                                        O_hashX.append(new_O_hashX)

                                        new_O_F, twice = self.evaluate(X=new_O_X_lst[m],
                                                                       using_surrogate_model=self.using_surrogate_model)
                                        O[i].set('X', new_O_X_lst[m])
                                        O[i].set('hashX', new_O_hashX)
                                        O[i].set('F', new_O_F)

                                        if not self.using_surrogate_model:
                                            self.update_A(O[i])
                                        else:
                                            if twice:
                                                self.update_A(O[i])
                                            else:
                                                self.training_data.append(O[i])
                                                self.update_fake_A(O[i])

                                        i += 1
                                        if i == len(P):
                                            full = True
                                            break
                                else:
                                    O_hashX.append(new_O_hashX)

                                    new_O_F, twice = self.evaluate(X=new_O_X_lst[m],
                                                                   using_surrogate_model=self.using_surrogate_model)
                                    O[i].set('X', new_O_X_lst[m])
                                    O[i].set('hashX', new_O_hashX)
                                    O[i].set('F', new_O_F)

                                    if not self.using_surrogate_model:
                                        self.update_A(O[i])
                                    else:
                                        if twice:
                                            self.update_A(O[i])
                                        else:
                                            self.training_data.append(O[i])
                                            self.update_fake_A(O[i])

                                    i += 1
                                    if i == len(P):
                                        full = True
                                        break
                    else:
                        for m in range(2):
                            O[i].set('X', P_[j][m].get('X'))
                            O[i].set('hashX', P_[j][m].get('hashX'))
                            O[i].set('F', P_[j][m].get('F'))
                            i += 1
                            if i == len(P):
                                full = True
                                break
                    if full:
                        break
            else:
                for j in range(len(P_)):
                    if np.random.random() < pC:
                        o1_X, o2_X = crossover(P_[j][0].get('X'), P_[j][1].get('X'), self.typeC)

                        o_X = [o1_X, o2_X]
                        o_hashX = [convert_to_hashX(o1_X, BENCHMARK_NAME), convert_to_hashX(o2_X, BENCHMARK_NAME)]

                        if nCOs <= 100:
                            for m in range(2):
                                if (o_hashX[m] not in O_hashX) and (o_hashX[m] not in self.DS):
                                    O_hashX.append(o_hashX[m])
                                    o_F, twice = self.evaluate(X=o_X[m], using_surrogate_model=self.using_surrogate_model)

                                    O[i].set('X', o_X[m])
                                    O[i].set('hashX', o_hashX[m])
                                    O[i].set('F', o_F)

                                    if not self.using_surrogate_model:
                                        self.update_A(O[i])
                                    else:
                                        if twice:
                                            self.update_A(O[i])
                                        else:
                                            self.training_data.append(O[i])
                                            self.update_fake_A(O[i])

                                    i += 1
                                    if i == len(P):
                                        full = True
                                        break
                        else:
                            for m in range(2):
                                O_hashX.append(o_hashX[m])
                                o_F, twice = self.evaluate(X=o_X[m], using_surrogate_model=self.using_surrogate_model)

                                O[i].set('X', o_X[m])
                                O[i].set('hashX', o_hashX[m])
                                O[i].set('F', o_F)

                                if not self.using_surrogate_model:
                                    self.update_A(O[i])
                                else:
                                    if twice:
                                        self.update_A(O[i])
                                    else:
                                        self.training_data.append(O[i])
                                        self.update_fake_A(O[i])
                                i += 1
                                if i == len(P):
                                    full = True
                                    break

                    else:
                        for m in range(2):
                            O[i].set('X', P_[j][m].get('X'))
                            O[i].set('hashX', P_[j][m].get('hashX'))
                            O[i].set('F', P_[j][m].get('F'))
                            i += 1
                            if i == len(P):
                                full = True
                                break

                    if full:
                        break
            nCOs += 1
        return O

    def _mutation(self, P, O):
        P_hashX = P.get('hashX')

        new_O = Population(len(O))

        new_O_hashX = []

        old_O_X = O.get('X')

        i = 0
        full = False
        pM = 1/len(old_O_X[0])
        while not full:
            if BENCHMARK_NAME == 'NAS-Bench-101':
                for x in old_O_X:
                    matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x)

                    new_matrix_1D = matrix_1D.copy()
                    new_ops_INT = ops_INT.copy()

                    pM_idxs_matrix = np.random.rand(len(new_matrix_1D))
                    for j in range(len(pM_idxs_matrix)):
                        if pM_idxs_matrix[j] <= pM:
                            new_matrix_1D[j] = 1 - new_matrix_1D[j]

                    pM_idxs_ops = np.random.rand(len(new_ops_INT))
                    for j in range(len(pM_idxs_ops)):
                        if pM_idxs_ops[j] <= pM:
                            choices = [0, 1, 2]
                            choices.remove(new_ops_INT[j])
                            new_ops_INT[j] = np.random.choice(choices)

                    matrix_2D = encoding_matrix(new_matrix_1D)
                    ops_STRING = encoding_ops(new_ops_INT)

                    new_MS = api.ModelSpec(matrix_2D, ops_STRING)

                    if BENCHMARK_API.is_valid(new_MS):
                        hashX = BENCHMARK_API.get_module_hash(new_MS)
                        if (hashX not in new_O_hashX) and (hashX not in P_hashX) and (hashX not in self.DS):
                            X = combine_matrix1D_and_opsINT(new_matrix_1D, new_ops_INT)
                            new_O_hashX.append(hashX)

                            F, twice = self.evaluate(X=X, using_surrogate_model=self.using_surrogate_model)

                            new_O[i].set('X', X)
                            new_O[i].set('hashX', hashX)
                            new_O[i].set('F', F)

                            if not self.using_surrogate_model:
                                self.update_A(new_O[i])
                            else:
                                if twice:
                                    self.update_A(new_O[i])
                                else:
                                    self.training_data.append(new_O[i])
                                    self.update_fake_A(new_O[i])

                            i += 1
                            if i == len(P):
                                full = True
                                break

            else:
                if BENCHMARK_NAME == 'MacroNAS-CIFAR-10' or BENCHMARK_NAME == 'MacroNAS-CIFAR-100':
                    opt = ['I', '1', '2']
                else:
                    opt = ['0', '1', '2', '3', '4']

                pM_idxs = np.random.rand(old_O_X.shape[0], old_O_X.shape[1])

                for m in range(len(old_O_X)):
                    X = old_O_X[m].copy()

                    for n in range(pM_idxs.shape[1]):
                        if pM_idxs[m][n] <= pM:
                            allowed_opt = opt.copy()
                            allowed_opt.remove(X[n])

                            X[n] = np.random.choice(allowed_opt)

                    hashX = convert_to_hashX(X, BENCHMARK_NAME)

                    if (hashX not in new_O_hashX) and (hashX not in P_hashX) and (hashX not in self.DS):
                        new_O_hashX.append(hashX)

                        F, twice = self.evaluate(X=X, using_surrogate_model=self.using_surrogate_model)

                        new_O[i].set('X', X)
                        new_O[i].set('hashX', hashX)
                        new_O[i].set('F', F)

                        if not self.using_surrogate_model:
                            self.update_A(new_O[i])
                        else:
                            if twice:
                                self.update_A(new_O[i])
                            else:
                                self.training_data.append(new_O[i])
                                self.update_fake_A(new_O[i])

                        i += 1
                        if i == len(P):
                            full = True
                            break

        return new_O

    def _initialize_custom(self):
        pop = self._sampling(self.pop_size)
        if self.using_surrogate_model:
            if BENCHMARK_NAME == 'NAS-Bench-101':
                self.surrogate_model = self._create_surrogate_model(inputs=pop.get('X'),
                                                                    targets=pop.get('F')[:, 1])
            else:
                self.surrogate_model = self._create_surrogate_model(inputs=encode(pop.get('X'), BENCHMARK_NAME),
                                                                    targets=pop.get('F')[:, 1])

        pop = self.survival.do(pop, self.pop_size)

        return pop

    def local_search_on_X(self, P, X, ls_on_knee_solutions=False):
        P_hashX = P.get('hashX')
        len_soln = len(P_hashX[0])
        S = P.new()
        S = S.merge(X)

        # Using for local search on knee solutions
        first, last = 0, 0
        if ls_on_knee_solutions:
            first, last = len(S) - 2, len(S) - 1

        m_searches = len_soln
        if BENCHMARK_NAME == 'MacroNAS-CIFAR-10' or BENCHMARK_NAME == 'MacroNAS-CIFAR-100':
            ops = ['I', '1', '2']
        else:
            ops = ['0', '1', '2', '3', '4']

        for i in range(len(S)):
            # Avoid stuck because don't find any better architecture
            tmp_m_searches = 100
            tmp_n_searches = 0

            checked = [S[i].get('hashX')]
            n_searches = 0

            while (n_searches < m_searches) and (tmp_n_searches < tmp_m_searches):
                tmp_n_searches += 1
                o = S[i].copy()  # --> neighboring solution

                if BENCHMARK_NAME == 'NAS-Bench-101':
                    ''' Local search on edges '''
                    matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(o.get('X'))

                    ops_STRING = encoding_ops(ops_INT)

                    while True:
                        idxs = np.random.choice(range(len(matrix_1D)), size=self.n_vars, replace=False)
                        matrix_1D_new = matrix_1D.copy()
                        matrix_1D_new[idxs] = 1 - matrix_1D[idxs]

                        matrix_2D_new = encoding_matrix(matrix_1D_new)

                        modelspec = api.ModelSpec(matrix=matrix_2D_new, ops=ops_STRING)
                        if BENCHMARK_API.is_valid(modelspec):
                            o_hashX = BENCHMARK_API.get_module_hash(modelspec)
                            o_X = combine_matrix1D_and_opsINT(matrix_1D_new, ops_INT)
                            break
                else:
                    idxs = np.random.choice(range(len(o.get('X'))), size=self.n_vars, replace=False)
                    o_X = o.get('X').copy()
                    for idx in idxs:
                        allowed_ops = ops.copy()
                        allowed_ops.remove(o.get('X')[idx])
                        new_op = np.random.choice(allowed_ops)
                        o_X[idx] = new_op
                    o_hashX = convert_to_hashX(o_X, BENCHMARK_NAME)

                if (o_hashX not in checked) and (o_hashX not in P_hashX) and (o_hashX not in self.DS):
                    checked.append(o_hashX)
                    n_searches += 1

                    o_F, twice = self.evaluate(o_X, using_surrogate_model=self.using_surrogate_model)

                    if i == first and LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                        better_idv = find_better_idv(o_F, S[i].get('F'), 'first')
                    elif i == last and LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                        better_idv = find_better_idv(o_F, S[i].get('F'), 'last')
                    else:
                        better_idv = find_better_idv(o_F, S[i].get('F'))

                    if better_idv == 1:  # --> neighboring solutions is better
                        S[i].set('X', o_X)
                        S[i].set('hashX', o_hashX)
                        S[i].set('F', o_F)

                        if not self.using_surrogate_model:
                            self.update_A(S[i])
                        else:
                            if twice:
                                self.update_A(S[i])
                            else:
                                self.training_data.append(S[i])
                                self.update_fake_A(S[i])

                    else:  # --> no one is better || current solution is better
                        o.set('X', o_X)
                        o.set('hashX', o_hashX)
                        o.set('F', o_F)

                        if not self.using_surrogate_model:
                            self.update_A(o)
                        else:
                            if twice:
                                self.update_A(o)
                            else:
                                self.training_data.append(o)
                                self.update_fake_A(o)

        return S

    def improve_potential_solutions(self, P):
        P_F = P.get('F')

        front_0 = NonDominatedSorting().do(P_F, n_stop_if_ranked=len(P), only_non_dominated_front=True)

        PF = P[front_0].copy()
        PF_F = P_F[front_0].copy()

        # normalize val_error for calculating angle between two individuals
        nPF_F = P_F[front_0].copy()

        mi_f0 = np.min(PF_F[:, 0])
        ma_f0 = np.max(PF_F[:, 0])

        mi_f1 = np.min(PF_F[:, 1])
        ma_f1 = np.max(PF_F[:, 1])

        nPF_F[:, 0] = (PF_F[:, 0] - mi_f0) / (ma_f0 - mi_f0)
        nPF_F[:, 1] = (PF_F[:, 1] - mi_f1) / (ma_f1 - mi_f1)

        new_idx = np.argsort(PF_F[:, 0])  # --> use for sort

        PF = PF[new_idx]
        PF_F = PF_F[new_idx]
        nPF_F = nPF_F[new_idx]
        front_0 = front_0[new_idx]

        angle = [np.array([360, 0])]
        for i in range(1, len(PF_F) - 1):
            l = None
            u = None
            for m in range(i - 1, -1, -1):
                if np.sum(np.abs(PF_F[m] - PF_F[i])) != 0:
                    l = m
                    break
            for m in range(i + 1, len(PF_F), 1):
                if np.sum(np.abs(PF_F[m] - PF_F[i])) != 0:
                    u = m
                    break

            if l is None or u is None:
                angle.append(np.array([0, i]))
            else:
                tren_hay_duoi = kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3(PF_F[i], PF_F[l], PF_F[u])
                if tren_hay_duoi == 'duoi':
                    angle.append(np.array([cal_angle(p_middle=nPF_F[i], p_top=nPF_F[l], p_bot=nPF_F[u]), i]))
                else:
                    angle.append(np.array([0, i]))

        angle.append(np.array([360, len(PF) - 1]))
        angle = np.array(angle)
        angle = angle[np.argsort(angle[:, 0])]

        angle = angle[angle[:, 0] > 210]

        idx_S = np.array(angle[:, 1], dtype=np.int)
        S = PF[idx_S].copy()

        S = self.local_search_on_X(P, X=S, ls_on_knee_solutions=True)

        PF[idx_S] = S

        P[front_0] = PF

        return P

    def _mating(self, pop):
        # crossover
        off = self._crossover(P=pop)

        # mutation
        off = self._mutation(P=pop, O=off)

        return off

    def _next(self, pop):
        # mating
        off = self._mating(pop)

        # merge the offsprings with the current population
        pop = pop.merge(off)

        # selecting
        pop = self.survival.do(pop, self.pop_size)

        if LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
            pop = self.improve_potential_solutions(P=pop)

        return pop

    def solve_custom(self):
        self.n_gen = 0

        # initialize
        self.pop = self._initialize_custom()

        self._do_each_gen(first=True)

        while self.nEs < self.m_nEs:
            self.n_gen += 1

            if self.dpfs[-1] != 0.0:
                self.pop = self._next(self.pop)
            else:
                self.nEs += self.pop_size

            self._do_each_gen()
        self._finalize()
        return

    def _do_each_gen(self, first=False):
        if self.using_surrogate_model:
            self.alpha = np.mean(self.F_total)

            if self.m_nEs - self.nEs < 2 * self.m_nEs // 3 or self.n_updates == 15:
                self.update_model = False

            if not first:
                tmp_set_X = np.array(self.tmp_A_X)
                tmp_set_hashX = np.array(self.tmp_A_hashX)

                tmp_set = Population(len(self.tmp_A_X))
                tmp_set.set('X', tmp_set_X)
                tmp_set.set('hashX', tmp_set_hashX)
                for i in range(len(tmp_set)):
                    if (tmp_set[i].get('hashX') not in self.A_hashX) and (tmp_set[i].get('hashX') not in self.DS):
                        F, _ = self.evaluate(tmp_set[i].get('X'), using_surrogate_model=False, count_nE=True)
                        tmp_set[i].set('F', F)
                        self.update_A(tmp_set[i])

            if self.n_gen % self.update_model_after_n_gens == 0:
                data = np.array(self.training_data)
                self.training_data = []

                X = []
                Y = []
                checked = []
                for i in range(len(data)):
                    if BENCHMARK_NAME == 'NAS-Bench-101':
                        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(data[i].get('X'))

                        matrix_2D = encoding_matrix(matrix_1D)
                        ops_STRING = encoding_ops(ops_INT)
                        modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                        hashX = BENCHMARK_API.get_module_hash(modelspec)
                    else:
                        hashX = convert_to_hashX(data[i].get('X'), BENCHMARK_NAME)
                    if (hashX not in checked) and (hashX not in self.DS) and (hashX not in self.A_hashX):
                        checked.append(hashX)
                        F, _ = self.evaluate(data[i].get('X'), using_surrogate_model=False, count_nE=True)
                        data[i].set('F', F)
                        self.update_A(data[i])
                        X.append(data[i].get('X'))
                        Y.append(F[1])
                for i in range(len(self.A_X)):
                    X.append(self.A_X[i])
                    Y.append(self.A_F[i][1])
                X = np.array(X)
                Y = np.array(Y)
                if self.update_model:
                    self.n_updates += 1
                    if BENCHMARK_NAME == 'NAS-Bench-101':
                        self.surrogate_model.fit(x=X, y=Y)
                    else:
                        self.surrogate_model.fit(x=encode(X, BENCHMARK_NAME), y=Y, verbose=False)

        if DEBUG:
            print(f'Number of evaluations used: {self.nEs}/{self.m_nEs}')

        if SAVE:
            pf = np.array(self.A_F)
            pf = np.unique(pf, axis=0)
            pf = pf[np.argsort(pf[:, 0])]
            pk.dump([pf, self.nEs], open(f'{self.path}/pf_eval/pf_and_evaluated_gen_{self.n_gen}.p', 'wb'))
            self.worst_f0 = max(self.worst_f0, np.max(pf[:, 0]))
            self.worst_f1 = max(self.worst_f1, np.max(pf[:, 1]))

            dpfs = round(cal_dpfs(pareto_s=pf, pareto_front=BENCHMARK_PF_TRUE), 6)
            print(self.nEs, dpfs)
            if len(self.no_eval) == 0:
                self.dpfs.append(dpfs)
                self.no_eval.append(self.nEs)
            else:
                if self.nEs == self.no_eval[-1]:
                    self.dpfs[-1] = dpfs
                else:
                    self.dpfs.append(dpfs)
                    self.no_eval.append(self.nEs)

    def _finalize(self):
        self.A_X = np.array(self.A_X)
        self.A_hashX = np.array(self.A_hashX)
        self.A_F = np.array(self.A_F)

        pk.dump([self.A_X, self.A_hashX, self.A_F],
                open(self.path + '/pareto_front.p', 'wb'))
        if SAVE:
            pk.dump([self.worst_f0, self.worst_f1], open(f'{self.path}/reference_point.p', 'wb'))
            # visualize DPFS
            pk.dump([self.no_eval, self.dpfs], open(f'{self.path}/no_eval_and_dpfs.p', 'wb'))
            plt.plot(self.no_eval, self.dpfs)
            plt.xlabel('No.Evaluations')
            plt.ylabel('DPFS')
            plt.grid()
            plt.savefig(f'{self.path}/dpfs_and_no_evaluations')
            plt.clf()

            # visualize elitist archive
            plt.scatter(BENCHMARK_PF_TRUE[:, 0], BENCHMARK_PF_TRUE[:, 1], facecolors='none', edgecolors='blue', s=40,
                        label='true pf')
            plt.scatter(self.A_F[:, 0], self.A_F[:, 1], c='red', s=15,
                        label='elitist archive')

            if BENCHMARK_NAME == 'NAS-Bench-101':
                plt.xlabel('params (normalize)')
                plt.ylabel('valid-error')
            elif BENCHMARK_NAME == 'MacroNAS-CIFAR-10' or BENCHMARK_NAME == 'MacroNAS-CIFAR-100':
                plt.xlabel('MMACs (normalize)')
                plt.ylabel('valid-error')
            else:
                plt.xlabel('FLOP (normalize)')
                plt.ylabel('test-error')

            plt.legend()
            plt.grid()
            plt.savefig(f'{self.path}/final_pf')
            plt.clf()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('NSGAII for NAS')
    #
    # # hyper-parameters for problem
    # parser.add_argument('--benchmark_name', type=str, default='cifar10',
    #                     help='the benchmark used for optimizing')
    # parser.add_argument('--max_no_evaluations', type=int, default=10000)
    #
    # # hyper-parameters for main
    # parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument('--number_of_runs', type=int, default=1, help='number of runs')
    # parser.add_argument('--save', type=int, default=1, help='save log file')
    #
    # # hyper-parameters for algorithm (NSGAII)
    # parser.add_argument('--algorithm_name', type=str, default='nsga', help='name of algorithm used')
    # parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
    # parser.add_argument('--crossover_type', type=str, default='UX')
    #
    # parser.add_argument('--local_search_on_pf', type=int, default=0, help='local search on pareto front')
    # parser.add_argument('--local_search_on_knees', type=int, default=0, help='local search on knee solutions')
    # parser.add_argument('--local_search_on_n_points', type=int, default=1)
    # parser.add_argument('--followed_bosman_paper', type=int, default=0, help='local search followed by bosman paper')
    #
    # parser.add_argument('--using_surrogate_model', type=int, default=0)
    # parser.add_argument('--update_model_after_n_gens', type=int, default=10)
    # args = parser.parse_args()
    ''' ------- '''
    SAVE = True
    DEBUG = False

    ALGORITHM_NAME = 'NSGA-II'

    NUMBER_OF_RUNS = 21
    INIT_SEED = 0

    problems = ['NAS-Bench-101']

    for BENCHMARK_NAME in problems:
        user_input = [[1, 1, '2X', 1, 10]]

        PATH_DATA = 'D:/Files'

        if BENCHMARK_NAME == 'NAS-Bench-101':
            BENCHMARK_API = api.NASBench_()
            BENCHMARK_DATA = pk.load(open(PATH_DATA + '/NAS-Bench-101/data.p', 'rb'))
            BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/NAS-Bench-101/mi_ma_Params.p', 'rb'))
            BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/NAS-Bench-101/PF(nor)_Params-ValidAcc.p', 'rb'))

        elif BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-10':
            BENCHMARK_DATA = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-10/encode_data.p', 'rb'))
            BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-10/mi_ma_FLOPs.p', 'rb'))
            BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-10/PF(nor)_FLOPs-TestAcc.p', 'rb'))

        elif BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-100':
            BENCHMARK_DATA = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-100/encode_data.p', 'rb'))
            BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-100/mi_ma_FLOPs.p', 'rb'))
            BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/NAS-Bench-201/CIFAR-100/PF(nor)_FLOPs-TestAcc.p', 'rb'))

        elif BENCHMARK_NAME == 'NAS-Bench-201-ImageNet16-120':
            BENCHMARK_DATA = pk.load(open(PATH_DATA + '/NAS-Bench-201/ImageNet16-120/encode_data.p', 'rb'))
            BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/NAS-Bench-201/ImageNet16-120/mi_ma_FLOPs.p', 'rb'))
            BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/NAS-Bench-201/ImageNet16-120/PF(nor)_FLOPs-TestAcc.p', 'rb'))

        elif BENCHMARK_NAME == 'MacroNAS-CIFAR-10':
            BENCHMARK_DATA = pk.load(open(PATH_DATA + '/MacroNAS/CIFAR-10/data.p', 'rb'))
            BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/MacroNAS/CIFAR-10/mi_ma_MMACs.p', 'rb'))
            BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/MacroNAS/CIFAR-10/PF(nor)_MMACs-ValidAcc.p', 'rb'))

        elif BENCHMARK_NAME == 'MacroNAS-CIFAR-100':
            BENCHMARK_DATA = pk.load(open(PATH_DATA + '/MacroNAS/CIFAR-100/data.p', 'rb'))
            BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/MacroNAS/CIFAR-100/mi_ma_MMACs.p', 'rb'))
            BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/MacroNAS/CIFAR-100/PF(nor)_MMACs-ValidAcc.p', 'rb'))

        print('--> Load benchmark - Done')

        if BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-10' or BENCHMARK_NAME == 'NAS-Bench-201-CIFAR-100' \
                or BENCHMARK_NAME == 'NAS-Bench-201-ImageNet16-120':
            POP_SIZE = 20
            MAX_NO_EVALUATIONS = 3000
        else:
            POP_SIZE = 100
            MAX_NO_EVALUATIONS = 30000

        for _input in user_input:
            CROSSOVER_TYPE = _input[2]

            USING_SURROGATE_MODEL = bool(_input[-2])
            UPDATE_MODEL_AFTER_N_GENS = _input[-1]
            if not USING_SURROGATE_MODEL:
                UPDATE_MODEL_AFTER_N_GENS = 0

            LOCAL_SEARCH_ON_KNEE_SOLUTIONS = bool(_input[0])
            LOCAL_SEARCH_ON_N_POINTS = _input[1]
            if not LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                LOCAL_SEARCH_ON_N_POINTS = 0

            now = datetime.now()
            dir_name = now.strftime(f'{BENCHMARK_NAME}_{ALGORITHM_NAME}_{POP_SIZE}_{CROSSOVER_TYPE}_'
                                    f'{LOCAL_SEARCH_ON_KNEE_SOLUTIONS}_{LOCAL_SEARCH_ON_N_POINTS}_'
                                    f'{USING_SURROGATE_MODEL}_{UPDATE_MODEL_AFTER_N_GENS}_'
                                    f'd%d_m%m_H%H_M%M_S%S')
            ROOT_PATH = PATH_DATA + '/results/' + dir_name

            # Create root folder
            os.mkdir(ROOT_PATH)
            print(f'--> Create folder {ROOT_PATH} - Done\n')

            for i_run in range(16, NUMBER_OF_RUNS):
                SEED = INIT_SEED + i_run * 100
                np.random.seed(SEED)
                torch.random.manual_seed(SEED)

                SUB_PATH = ROOT_PATH + f'/{i_run}'

                # Create new folder (i_run) in root folder
                os.mkdir(SUB_PATH)
                print(f'--> Create folder {SUB_PATH} - Done')

                # Create new folder (pf_eval) in 'i_run' folder
                os.mkdir(SUB_PATH + '/pf_eval')
                print(f'--> Create folder {SUB_PATH}/pf_eval - Done')

                # Create new folder (visualize_pf_each_gen) in 'i_run' folder
                os.mkdir(SUB_PATH + '/visualize_pf_each_gen')
                print(f'--> Create folder {SUB_PATH}/visualize_pf_each_gen - Done\n')

                net = NSGANet(
                    m_nEs=MAX_NO_EVALUATIONS,
                    pop_size=POP_SIZE,
                    selection=TournamentSelection(func_comp=binary_tournament),
                    survival=RankAndCrowdingSurvival(),
                    typeC=CROSSOVER_TYPE,
                    local_search_on_n_vars=LOCAL_SEARCH_ON_N_POINTS,
                    using_surrogate_model=USING_SURROGATE_MODEL,
                    update_model_after_n_gens=UPDATE_MODEL_AFTER_N_GENS,
                    path=SUB_PATH)

                start = timeit.default_timer()
                print(f'--> Experiment {i_run + 1} is running.')
                net.solve_custom()
                end = timeit.default_timer()

                print(f'--> The number of runs done: {i_run + 1}/{NUMBER_OF_RUNS}')
                print(f'--> Took {end - start} seconds.\n')

            print(f'All {NUMBER_OF_RUNS} runs - Done\nResults are saved on folder {ROOT_PATH}.\n')
