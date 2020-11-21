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
from pymoo.rand import random
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


# Using for Cifar10 and Cifar100
def encode(X):
    if isinstance(X, list):
        X = np.array(X)
    encode_X = np.where(X == 'I', 0, X)
    encode_X = np.array(encode_X, dtype=np.int)
    return encode_X


def insert_to_list_x(x):
    added = ['|', '|', '|']
    indices = [4, 8, 12]

    acc = 0
    for i in range(len(added)):
        x.insert(indices[i]+acc, added[i])
        acc += 1
    return x


def remove_values_from_list_x(x, val):
    return [value for value in x if value != val]


def convert_X_to_hashX(x):
    if not isinstance(x, list):
        x = x.tolist()
    x = insert_to_list_x(x)
    x = remove_values_from_list_x(x, 'I')
    hashX = ''.join(x)
    return hashX


'''------------------------------------------------------------------------------------------------------------------'''


class NSGANet(GeneticAlgorithm):
    def __init__(self,
                 m_nEs,
                 typeC,
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

        self.DS = []
        self.EA_X, self.EA_hashX, self.EA_F = [], [], []

        self.dpfs = []
        self.no_eval = []

        self.m_nEs = m_nEs

        self.using_surrogate_model = using_surrogate_model
        self.update_model_after_n_gens = update_model_after_n_gens
        self.surrogate_model = None
        self.models_for_training = []

        self.nEs = 0
        self.path = path

    def update_EA(self, new_solution):
        X = new_solution.get('X')
        hashX = new_solution.get('hashX')
        F = new_solution.get('F')

        rank = np.zeros(len(self.EA_X))
        if hashX not in self.EA_hashX:
            flag = True
            for j in range(len(self.EA_X)):

                better_idv = find_better_idv(F, self.EA_F[j])
                if better_idv == 1:
                    rank[j] += 1
                    if self.EA_hashX[j] not in self.DS:
                        self.DS.append(self.EA_hashX[j])

                elif better_idv == 2:
                    flag = False
                    if hashX not in self.DS:
                        self.DS.append(hashX)
                    break

            if flag:
                self.EA_X.append(np.array(X))
                self.EA_hashX.append(np.array(hashX))
                self.EA_F.append(np.array(F))
                rank = np.append(rank, 0)

        self.EA_X = np.array(self.EA_X)[rank == 0].tolist()
        self.EA_hashX = np.array(self.EA_hashX)[rank == 0].tolist()
        self.EA_F = np.array(self.EA_F)[rank == 0].tolist()

    def evaluate(self, X, using_surrogate_model=False, count_nE=True):
        F = np.full(2, fill_value=np.nan)

        if not using_surrogate_model:
            if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                hashX = ''.join(X.tolist())

                F[0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
                       / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
                F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc'] / 100

            else:
                matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X)

                matrix_2D = encoding_matrix(matrix_1D)
                ops_STRING = encoding_ops(ops_INT)

                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                hashX = BENCHMARK_API.get_module_hash(modelspec)

                F[0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                        BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc']

            if count_nE:
                self.nEs += 1

        else:
            self.models_for_training.append(X)
            if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                encode_X = encode(X)

                hashX = ''.join(X.tolist())

                F[0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
                       / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
                F[1] = self.surrogate_model.predict(np.array([encode_X]))[0][0]
                if BENCHMARK_NAME == 'cifar10':
                    if F[1] < 0.0835:
                        F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc'] / 100
                        self.nEs += 1
                else:
                    if F[1] < 0.304:
                        F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc'] / 100
                        self.nEs += 1

            else:
                matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X)

                matrix_2D = encoding_matrix(matrix_1D)
                ops_STRING = encoding_ops(ops_INT)
                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                hashX = BENCHMARK_API.get_module_hash(modelspec)

                F[0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                        BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                F[1] = self.surrogate_model.predict(np.array([X]))[0][0]
                if F[1] < 0.067:
                    F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc']
                    self.nEs += 1

        return F

    @staticmethod
    def _create_surrogate_model(inputs, targets):
        surrogate_model = get_acc_predictor('mlp', inputs, targets)
        return surrogate_model

    def _sampling(self, n_samples):
        P = Population(n_samples)
        P_hashX = []
        i = 0
        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            allowed_choices = ['I', '1', '2']
            while i < n_samples:
                X = np.random.choice(allowed_choices, 14)
                hashX = convert_X_to_hashX(X)
                if hashX not in P_hashX:
                    P_hashX.append(hashX)

                    F = self.evaluate(X=X, using_surrogate_model=False)

                    P[i].set('X', X)
                    P[i].set('hashX', hashX)
                    P[i].set('F', F)
                    self.update_EA(P[i])

                    i += 1

        else:
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
                        F = self.evaluate(X=X, using_surrogate_model=False)

                        P[i].set('X', X)
                        P[i].set('hashX', hashX)
                        P[i].set('F', F)
                        self.update_EA(P[i])

                        i += 1

        return P

    def _crossover(self, P, typeC='UX'):
        O = Population(len(P))
        O_hashX = []

        nCOs = 0  # --> Avoid to stuck

        i = 0
        full = False
        while not full:
            idx = np.random.choice(len(P), size=(len(P) // 2, 2), replace=False)
            P_X_ = P.get('X')[idx]

            if BENCHMARK_NAME == 'nas101':
                for j in range(len(P_X_)):
                    new_O1_X, new_O2_X = P_X_[j][0].copy(), P_X_[j][1].copy()

                    if typeC == '1X':
                        pt = np.random.randint(1, len(new_O1_X))

                        new_O1_X[pt:], new_O2_X[pt:] = new_O2_X[pt:], new_O1_X[pt:].copy()

                    elif typeC == '2X':
                        pts = np.random.choice(range(1, len(new_O1_X) - 1), 2, replace=False)

                        l = min(pts)
                        u = max(pts)

                        new_O1_X[l:u], new_O2_X[l:u] = new_O2_X[l:u], new_O1_X[l:u].copy()

                    elif typeC == 'UX':
                        pts = np.random.randint(0, 2, new_O1_X.shape, dtype=np.bool)

                        new_O1_X[pts], new_O2_X[pts] = new_O2_X[pts], new_O1_X[pts].copy()

                    else:
                        raise Exception('Crossover method is not available!')

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

                                    new_O_F = self.evaluate(X=new_O_X_lst[m],
                                                            using_surrogate_model=self.using_surrogate_model)
                                    O[i].set('X', new_O_X_lst[m])
                                    O[i].set('hashX', new_O_hashX)
                                    O[i].set('F', new_O_F)

                                    if not self.using_surrogate_model:
                                        self.update_EA(O[i])
                                    else:
                                        tmp_O = O[i].copy()
                                        tmp_O_F = self.evaluate(X=tmp_O.get('X'),
                                                                using_surrogate_model=False, count_nE=False)
                                        tmp_O.set('F', tmp_O_F)
                                        self.update_EA(tmp_O)

                                    i += 1
                                    if i == len(P):
                                        full = True
                                        break
                            else:
                                O_hashX.append(new_O_hashX)

                                new_O_F = self.evaluate(X=new_O_X_lst[m],
                                                        using_surrogate_model=self.using_surrogate_model)
                                O[i].set('X', new_O_X_lst[m])
                                O[i].set('hashX', new_O_hashX)
                                O[i].set('F', new_O_F)

                                if not self.using_surrogate_model:
                                    self.update_EA(O[i])
                                else:
                                    tmp_O = O[i].copy()
                                    tmp_O_F = self.evaluate(X=tmp_O.get('X'),
                                                            using_surrogate_model=False, count_nE=False)
                                    tmp_O.set('F', tmp_O_F)
                                    self.update_EA(tmp_O)

                                i += 1
                                if i == len(P):
                                    full = True
                                    break
                    if full:
                        break

            elif BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                for j in range(len(P_X_)):
                    new_O1_X, new_O2_X = P_X_[j][0].copy(), P_X_[j][1].copy()

                    if typeC == '1X':
                        pt = np.random.randint(1, len(new_O1_X))

                        new_O1_X[pt:], new_O2_X[pt:] = new_O2_X[pt:], new_O1_X[pt:].copy()

                    elif typeC == '2X':
                        pts = np.random.choice(range(1, len(new_O1_X) - 1), 2, replace=False)

                        l = min(pts)
                        u = max(pts)

                        new_O1_X[l:u], new_O2_X[l:u] = new_O2_X[l:u], new_O1_X[l:u].copy()

                    elif typeC == 'UX':
                        pts = np.random.randint(0, 2, new_O1_X.shape, dtype=np.bool)

                        new_O1_X[pts], new_O2_X[pts] = new_O2_X[pts], new_O1_X[pts].copy()

                    else:
                        raise Exception('Crossover method is not available!')

                    new_O1_hashX = convert_X_to_hashX(new_O1_X)
                    new_O2_hashX = convert_X_to_hashX(new_O2_X)

                    if nCOs <= 100:
                        if (new_O1_hashX not in O_hashX) and \
                                (new_O1_hashX not in self.DS):
                            O_hashX.append(new_O1_hashX)

                            new_O1_F = self.evaluate(X=new_O1_X,
                                                     using_surrogate_model=self.using_surrogate_model)
                            O[i].set('X', new_O1_X)
                            O[i].set('hashX', new_O1_hashX)
                            O[i].set('F', new_O1_F)

                            if not self.using_surrogate_model:
                                self.update_EA(O[i])
                            else:
                                tmp_O = O[i].copy()
                                tmp_O_F = self.evaluate(X=tmp_O.get('X'),
                                                        using_surrogate_model=False, count_nE=False)
                                tmp_O.set('F', tmp_O_F)
                                self.update_EA(tmp_O)

                            i += 1
                            if i == len(P):
                                full = True
                                break

                        if (new_O2_hashX not in O_hashX) and \
                                (new_O2_hashX not in self.DS):
                            O_hashX.append(new_O2_hashX)

                            new_O2_F = self.evaluate(X=new_O2_X,
                                                     using_surrogate_model=self.using_surrogate_model)
                            O[i].set('X', new_O2_X)
                            O[i].set('hashX', new_O2_hashX)
                            O[i].set('F', new_O2_F)

                            if not self.using_surrogate_model:
                                self.update_EA(O[i])
                            else:
                                tmp_O = O[i].copy()
                                tmp_O_F = self.evaluate(X=tmp_O.get('X'),
                                                        using_surrogate_model=False, count_nE=False)
                                tmp_O.set('F', tmp_O_F)
                                self.update_EA(tmp_O)

                            i += 1
                            if i == len(P):
                                full = True
                                break
                    else:
                        O_hashX.append(new_O1_hashX)

                        new_O1_F = self.evaluate(X=new_O1_X,
                                                 using_surrogate_model=self.using_surrogate_model)
                        O[i].set('X', new_O1_X)
                        O[i].set('hashX', new_O1_hashX)
                        O[i].set('F', new_O1_F)

                        if not self.using_surrogate_model:
                            self.update_EA(O[i])
                        else:
                            tmp_O = O[i].copy()
                            tmp_O_F = self.evaluate(X=tmp_O.get('X'),
                                                    using_surrogate_model=False, count_nE=False)
                            tmp_O.set('F', tmp_O_F)
                            self.update_EA(tmp_O)

                        i += 1
                        if i == len(P):
                            full = True
                            break

                        O_hashX.append(new_O2_hashX)

                        new_O2_F = self.evaluate(X=new_O2_X,
                                                 using_surrogate_model=self.using_surrogate_model)
                        O[i].set('X', new_O2_X)
                        O[i].set('hashX', new_O2_hashX)
                        O[i].set('F', new_O2_F)

                        if not self.using_surrogate_model:
                            self.update_EA(O[i])
                        else:
                            tmp_O = O[i].copy()
                            tmp_O_F = self.evaluate(X=tmp_O.get('X'),
                                                    using_surrogate_model=False, count_nE=False)
                            tmp_O.set('F', tmp_O_F)
                            self.update_EA(tmp_O)

                        i += 1
                        if i == len(P):
                            full = True
                            break

            nCOs += 1

        return O

    def _mutation(self, P, O, pM=0.05):
        P_hashX = P.get('hashX')

        new_O = Population(len(O))

        new_O_hashX = []

        old_O_X = O.get('X')

        i = 0
        full = False
        while not full:
            if BENCHMARK_NAME == 'nas101':
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

                            F = self.evaluate(X=X, using_surrogate_model=self.using_surrogate_model)
                            new_O[i].set('X', X)
                            new_O[i].set('hashX', hashX)
                            new_O[i].set('F', F)

                            if not self.using_surrogate_model:
                                self.update_EA(new_O[i])
                            else:
                                tmp_new_O = new_O[i].copy()
                                tmp_new_O_F = self.evaluate(X=tmp_new_O.get('X'),
                                                            using_surrogate_model=False, count_nE=False)
                                tmp_new_O.set('F', tmp_new_O_F)
                                self.update_EA(tmp_new_O)

                            i += 1
                            if i == len(P):
                                full = True
                                break

            elif BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                pM_idxs = np.random.rand(old_O_X.shape[0], old_O_X.shape[1])

                for m in range(len(old_O_X)):
                    X = old_O_X[m].copy()

                    for n in range(pM_idxs.shape[1]):
                        if pM_idxs[m][n] <= pM:
                            allowed_choices = ['I', '1', '2']
                            allowed_choices.remove(X[n])

                            X[n] = np.random.choice(allowed_choices)

                    hashX = convert_X_to_hashX(X)

                    if (hashX not in new_O_hashX) and (hashX not in P_hashX) and (hashX not in self.DS):
                        new_O_hashX.append(hashX)

                        F = self.evaluate(X=X, using_surrogate_model=self.using_surrogate_model)

                        new_O[i].set('X', X)
                        new_O[i].set('hashX', hashX)
                        new_O[i].set('F', F)

                        if not self.using_surrogate_model:
                            self.update_EA(new_O[i])
                        else:
                            tmp_new_O = new_O[i].copy()
                            tmp_new_O_F = self.evaluate(X=tmp_new_O.get('X'),
                                                        using_surrogate_model=False, count_nE=False)
                            tmp_new_O.set('F', tmp_new_O_F)
                            self.update_EA(tmp_new_O)

                        i += 1
                        if i == len(P):
                            full = True
                            break

        return new_O

    def _initialize_custom(self):
        pop = self._sampling(self.pop_size)
        if self.using_surrogate_model:
            if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                self.surrogate_model = self._create_surrogate_model(inputs=encode(pop.get('X')),
                                                                    targets=pop.get('F')[:, 1])
            else:
                self.surrogate_model = self._create_surrogate_model(inputs=pop.get('X'),
                                                                    targets=pop.get('F')[:, 1])

        pop = self.survival.do(pop, self.pop_size)

        return pop

    def local_search_on_X(self, P, X, ls_on_knee_solutions=False):
        P_hashX = P.get('hashX')

        S = P.new()
        S = S.merge(X)

        # Using for local search on knee solutions
        first, last = 0, 0
        if ls_on_knee_solutions:
            first, last = len(S) - 2, len(S) - 1

        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            m_searches = 14
        else:
            m_searches = 26

        if LOCAL_SEARCH_ON_N_POINTS == 1:
            for i in range(len(S)):
                # Avoid stuck because don't find any better architecture
                tmp_m_searches = 100
                tmp_n_searches = 0

                checked = [S[i].get('hashX')]
                n_searches = 0

                while (n_searches < m_searches) and (tmp_n_searches < tmp_m_searches):
                    tmp_n_searches += 1
                    o = S[i].copy()  # --> neighboring solution

                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.randint(0, 14)
                        ops = ['I', '1', '2']
                        ops.remove(o.get('X')[idx])

                        new_op = np.random.choice(ops)

                        o_X = o.get('X').copy()
                        o_X[idx] = new_op
                        o_hashX = convert_X_to_hashX(o_X)

                    else:
                        ''' Local search on ops '''
                        # matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])
                        #
                        # matrix_2D = encoding_matrix(matrix_1D)
                        # ops_STRING = encoding_ops(ops_INT)
                        #
                        # idx = np.random.randint(1, 6)
                        # ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        # ops.remove(ops_STRING[idx])
                        #
                        # new_op = np.random.choice(ops)
                        #
                        # ops_STRING[idx] = new_op
                        #
                        # modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                        # x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)
                        #
                        # ops_INT = decoding_ops(ops_STRING)
                        #
                        # x_new_X = combine_matrix1D_and_opsINT(matrix_1D, ops_INT)

                        ''' Local search on edges '''
                        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(o.get('X'))

                        ops_STRING = encoding_ops(ops_INT)

                        while True:
                            idx = np.random.randint(len(matrix_1D))
                            matrix_1D_new = matrix_1D.copy()
                            matrix_1D_new[idx] = 1 - matrix_1D[idx]

                            matrix_2D_new = encoding_matrix(matrix_1D_new)

                            modelspec = api.ModelSpec(matrix=matrix_2D_new, ops=ops_STRING)
                            if BENCHMARK_API.is_valid(modelspec):
                                o_hashX = BENCHMARK_API.get_module_hash(modelspec)
                                o_X = combine_matrix1D_and_opsINT(matrix_1D_new, ops_INT)
                                break

                    if (o_hashX not in checked) and (o_hashX not in P_hashX)\
                            and (o_hashX not in self.DS):
                        checked.append(o_hashX)
                        n_searches += 1

                        o_F = self.evaluate(o_X, using_surrogate_model=self.using_surrogate_model)

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
                                self.update_EA(S[i])
                            else:
                                tmp_o = S[i].copy()
                                tmp_o_F = self.evaluate(tmp_o.get('X'), using_surrogate_model=False, count_nE=False)
                                tmp_o.set('F', tmp_o_F)
                                self.update_EA(tmp_o)

                        else:  # --> no one is better || current solution is better
                            o.set('X', o_X)
                            o.set('hashX', o_hashX)
                            o.set('F', o_F)

                            if not self.using_surrogate_model:
                                self.update_EA(o)
                            else:
                                tmp_o = o.copy()
                                tmp_o_F = self.evaluate(tmp_o.get('X'), using_surrogate_model=False, count_nE=False)
                                tmp_o.set('F', tmp_o_F)
                                self.update_EA(tmp_o)

        elif LOCAL_SEARCH_ON_N_POINTS == 2:
            for i in range(len(S)):
                # Avoid stuck because don't find any better architecture
                tmp_m_searches = 100
                tmp_n_searches = 0

                checked = [S[i].get('hashX')]
                n_searches = 0

                while (n_searches < m_searches) and (tmp_n_searches < tmp_m_searches):
                    tmp_n_searches += 1
                    o = S[i].copy()  # --> neighboring solution

                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.choice(14, size=2, replace=False)
                        ops1, ops2 = ['I', '1', '2'], ['I', '1', '2']
                        ops1.remove(o.get('X')[idx[0]])
                        ops2.remove(o.get('X')[idx[1]])

                        new_op1, new_op2 = np.random.choice(ops1), np.random.choice(ops2)

                        o_X = o.get('X').copy()
                        o_X[idx[0]], o_X[idx[1]] = new_op1, new_op2
                        o_hashX = convert_X_to_hashX(o_X)

                    else:
                        ''' Local search on edges '''
                        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(o.get('X'))

                        ops_STRING = encoding_ops(ops_INT)

                        while True:
                            idxs = np.random.choice(len(matrix_1D), size=4, replace=False)
                            matrix_1D_new = matrix_1D.copy()
                            matrix_1D_new[idxs] = 1 - matrix_1D[idxs]

                            matrix_2D = encoding_matrix(matrix_1D_new)

                            modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                            if BENCHMARK_API.is_valid(modelspec):
                                o_hashX = BENCHMARK_API.get_module_hash(modelspec)
                                o_X = combine_matrix1D_and_opsINT(matrix_1D_new, ops_INT)
                                break

                    if (o_hashX not in checked) and (o_hashX not in P_hashX) \
                            and (o_hashX not in self.DS):
                        checked.append(o_hashX)
                        n_searches += 1

                        o_F = self.evaluate(o_X, using_surrogate_model=self.using_surrogate_model)

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
                                self.update_EA(S[i])
                            else:
                                tmp_o = S[i].copy()
                                tmp_o_F = self.evaluate(tmp_o.get('X'), using_surrogate_model=False, count_nE=False)
                                tmp_o.set('F', tmp_o_F)
                                self.update_EA(tmp_o)

                        else:  # --> no one is better || current solution is better
                            o.set('X', o_X)
                            o.set('hashX', o_hashX)
                            o.set('F', o_F)

                            if not self.using_surrogate_model:
                                self.update_EA(o)
                            else:
                                tmp_o = o.copy()
                                tmp_o_F = self.evaluate(tmp_o.get('X'), using_surrogate_model=False, count_nE=False)
                                tmp_o.set('F', tmp_o_F)
                                self.update_EA(tmp_o)

        return S

    def improve_potential_solutions(self, P):
        P_F = P.get('F')

        front_0 = NonDominatedSorting().do(P_F, n_stop_if_ranked=len(P), only_non_dominated_front=True)

        PF = P[front_0].copy()
        F_PF = P_F[front_0].copy()

        # normalize val_error for calculating angle between two individuals
        nF_PF = P_F[front_0].copy()

        min_f1 = np.min(F_PF[:, 1])
        max_f1 = np.max(F_PF[:, 1])
        nF_PF[:, 1] = (F_PF[:, 1] - min_f1) / (max_f1 - min_f1)

        new_idx = np.argsort(F_PF[:, 0])  # --> use for sort

        PF = PF[new_idx]
        F_PF = F_PF[new_idx]
        nF_PF = nF_PF[new_idx]
        front_0 = front_0[new_idx]

        angle = [np.array([360, 0])]
        for i in range(1, len(F_PF) - 1):
            if (np.sum(F_PF[i - 1] - F_PF[i]) == 0) or (np.sum(F_PF[i] - F_PF[i + 1]) == 0):
                angle.append(np.array([0, i]))
            else:
                tren_hay_duoi = kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3(F_PF[i], F_PF[i - 1], F_PF[i + 1])
                if tren_hay_duoi == 'duoi':
                    angle.append(np.array([cal_angle(p_middle=nF_PF[i], p_top=nF_PF[i - 1], p_bot=nF_PF[i + 1]), i]))
                else:
                    angle.append(np.array([0, i]))

        angle.append(np.array([360, len(PF) - 1]))
        angle = np.array(angle)
        angle = angle[np.argsort(angle[:, 0])]

        angle = angle[angle[:, 0] > 210]

        idx_S = np.array(angle[:, 1], dtype=np.int)
        S = PF[idx_S].copy()

        # f_knee_solutions = f_pareto_front[idx_knee_solutions]
        # plt.scatter(f_pareto_front[:, 0], f_pareto_front[:, 1], s=30, edgecolors='blue',
        #             facecolors='none', label='True PF')
        # plt.scatter(f_knee_solutions[:, 0], f_knee_solutions[:, 1], c='red', s=15,
        #             label='Knee Solutions')

        S = self.local_search_on_X(P, X=S, ls_on_knee_solutions=True)

        PF[idx_S] = S

        P[front_0] = PF

        return P

    def _mating(self, P):
        # crossover
        O = self._crossover(P=P, typeC=self.typeC)

        # mutation
        O = self._mutation(P=P, O=O, pM=0.1)

        return O

    def _next(self, P):
        # mating
        O = self._mating(P)

        # merge the offsprings with the current population
        P = P.merge(O)

        # select best individuals
        P = self.survival.do(P, self.pop_size)

        # # local search on pareto front
        # if LOCAL_SEARCH_ON_PARETO_FRONT:
        #     pop_F = pop.get('F')
        #
        #     front_0 = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size, only_non_dominated_front=True)
        #
        #     pareto_front = pop[front_0].copy()
        #
        #     if LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER:
        #         pareto_front, non_dominance_X, non_dominance_hashX, non_dominance_F = \
        #             self.local_search_on_X_bosman(pop, X=pareto_front)
        #     else:
        #         pareto_front = self.local_search_on_X(pop, X=pareto_front)
        #     pop[front_0] = pareto_front
        #
        #     # update elitist archive - local search on pareto front
        #     self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F, self.dominated_idv = \
        #         update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
        #                                self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
        #                                self.dominated_idv)

        # local search on knee solutions
        if LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
            P = self.improve_potential_solutions(P)

        return P

    def solve_custom(self):
        self.n_gen = 1

        # initialize
        self.pop = self._initialize_custom()

        self._do_each_gen()

        while (self.nEs < self.m_nEs) and (self.dpfs[-1] != 0.0):
            self.n_gen += 1

            self.pop = self._next(self.pop)

            self._do_each_gen()

        self._finalize()
        return

    def _do_each_gen(self):
        if self.using_surrogate_model \
                and (self.m_nEs - self.nEs > self.m_nEs // 3) \
                and (self.n_gen % self.update_model_after_n_gens == 0):

            if len(self.models_for_training) < 500:
                X = np.array(self.models_for_training)
                self.models_for_training = []
            else:
                idxs = random.perm(len(self.models_for_training))
                X = np.array(self.models_for_training)[idxs[:500]]
                self.models_for_training = np.array(self.models_for_training)[idxs[500:]].tolist()

            Y = []
            for x in X:
                Y.append(self.evaluate(x, using_surrogate_model=False, count_nE=True)[1])

            Y = np.array(Y)
            if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                self.surrogate_model.fit(x=encode(X), y=Y)
            else:
                self.surrogate_model.fit(x=X, y=Y)

            if DEBUG:
                print('Update surrogate model - Done')

        if DEBUG:
            print(f'Number of evaluations used: {self.nEs}/{self.m_nEs}')

        if SAVE:
            pf = np.array(self.EA_F)
            pf = pf[np.argsort(pf[:, 0])]
            pk.dump([pf, self.nEs], open(f'{self.path}/pf_eval/pf_and_evaluated_gen_{self.n_gen}.p', 'wb'))

            dpfs = round(cal_dpfs(pareto_s=pf, pareto_front=BENCHMARK_PF_TRUE), 5)
            if len(self.no_eval) == 0:
                self.dpfs.append(dpfs)
                self.no_eval.append(self.nEs)
            else:
                if self.nEs == self.no_eval[-1]:
                    self.dpfs[-1] = dpfs
                else:
                    if DEBUG:
                        print('number of local searches:', self.nEs - self.no_eval[-1] - 200)
                    self.dpfs.append(dpfs)
                    self.no_eval.append(self.nEs)

    def _finalize(self):
        self.EA_X = np.array(self.EA_X)
        self.EA_hashX = np.array(self.EA_hashX)
        self.EA_F = np.array(self.EA_F)

        pk.dump([self.EA_X, self.EA_hashX, self.EA_F],
                open(self.path + '/pareto_front.p', 'wb'))
        if SAVE:
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
            plt.scatter(self.EA_F[:, 0], self.EA_F[:, 1], c='red', s=15,
                        label='elitist archive')
            if BENCHMARK_NAME == 'nas101':
                plt.xlabel('params (normalize)')
            else:
                plt.xlabel('MMACs (normalize)')
            plt.ylabel('validation error')
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
    # user_input = [[0, 0, 0, 0],
    #               [1, 0, 1, 0],
    #               [1, 0, 2, 0],
    #               [0, 1, 1, 0],
    #               [0, 1, 2, 0],
    #               [1, 0, 1, 1],
    #               [1, 0, 2, 1],
    #               [0, 1, 1, 1],
    #               [0, 1, 2, 1]]
    user_input = [[0, 0, 0, 0, '2X', 0, 0],
                  [0, 1, 2, 0, '2X', 0, 0],
                  [0, 0, 0, 0, '2X', 1, 10],
                  [0, 1, 2, 0, '2X', 1, 10]]

    PATH_DATA = 'D:/Files'

    BENCHMARK_NAME = 'nas101'

    if BENCHMARK_NAME == 'nas101':
        BENCHMARK_API = api.NASBench_()
        BENCHMARK_DATA = pk.load(open(PATH_DATA + '/101_benchmark/nas101.p', 'rb'))
        BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/101_benchmark/min_max_NAS101.p', 'rb'))
        BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/101_benchmark/pf_validation_parameters.p', 'rb'))

    elif BENCHMARK_NAME == 'cifar10':
        BENCHMARK_DATA = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar10/cifar10.p', 'rb'))
        BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar10/min_max_cifar10.p', 'rb'))
        BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))

    else:
        BENCHMARK_DATA = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar100/cifar100.p', 'rb'))
        BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar100/min_max_cifar100.p', 'rb'))
        BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))

    BENCHMARK_PF_TRUE[:, 0], BENCHMARK_PF_TRUE[:, 1] = BENCHMARK_PF_TRUE[:, 1], BENCHMARK_PF_TRUE[:, 0].copy()
    print('--> Load benchmark - Done')

    SAVE = True
    DEBUG = False

    ALGORITHM_NAME = 'nsga'
    POP_SIZE = 100

    NUMBER_OF_RUNS = 10
    INIT_SEED = 0

    for _input in user_input:
        # BENCHMARK_NAME = args.benchmark_name
        # BENCHMARK_DATA = None
        # BENCHMARK_MIN_MAX = None
        # BENCHMARK_PF_TRUE = None
        #
        # if BENCHMARK_NAME == 'nas101':
        #     BENCHMARK_API = api.NASBench_()
        #     BENCHMARK_DATA = pk.load(open(PATH_DATA + '/101_benchmark/nas101.p', 'rb'))
        #     BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/101_benchmark/min_max_NAS101.p', 'rb'))
        #     BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/101_benchmark/pf_validation_parameters.p', 'rb'))
        #
        # elif BENCHMARK_NAME == 'cifar10':
        #     BENCHMARK_DATA = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar10/cifar10.p', 'rb'))
        #     BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar10/min_max_cifar10.p', 'rb'))
        #     BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
        #
        # else:
        #     BENCHMARK_DATA = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar100/cifar100.p', 'rb'))
        #     BENCHMARK_MIN_MAX = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar100/min_max_cifar100.p', 'rb'))
        #     BENCHMARK_PF_TRUE = pk.load(open(PATH_DATA + '/bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))
        #
        # BENCHMARK_PF_TRUE[:, 0], BENCHMARK_PF_TRUE[:, 1] = BENCHMARK_PF_TRUE[:, 1], BENCHMARK_PF_TRUE[:, 0].copy()
        # print('--> Load benchmark - Done')
        #
        # SAVE = bool(args.save)
        #
        # MAX_NO_EVALUATIONS = args.max_no_evaluations
        #
        # ALGORITHM_NAME = args.algorithm_name
        #
        # POP_SIZE = args.pop_size
        # CROSSOVER_TYPE = args.crossover_type
        #
        # LOCAL_SEARCH_ON_PARETO_FRONT = bool(args.local_search_on_pf)
        # LOCAL_SEARCH_ON_KNEE_SOLUTIONS = bool(args.local_search_on_knees)
        # LOCAL_SEARCH_ON_N_POINTS = args.local_search_on_n_points
        # LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER = bool(args.followed_bosman_paper)
        #
        # USING_SURROGATE_MODEL = bool(args.using_surrogate_model)
        # UPDATE_MODEL_AFTER_N_GENS = args.update_model_after_n_gens
        #
        # NUMBER_OF_RUNS = args.number_of_runs
        # INIT_SEED = args.seed

        CROSSOVER_TYPE = _input[4]

        USING_SURROGATE_MODEL = bool(_input[-2])
        UPDATE_MODEL_AFTER_N_GENS = _input[-1]

        MAX_NO_EVALUATIONS = 30000

        LOCAL_SEARCH_ON_PARETO_FRONT = bool(_input[0])
        LOCAL_SEARCH_ON_KNEE_SOLUTIONS = bool(_input[1])
        LOCAL_SEARCH_ON_N_POINTS = _input[2]
        LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER = bool(_input[3])

        now = datetime.now()
        dir_name = now.strftime(f'{BENCHMARK_NAME}_{ALGORITHM_NAME}_{POP_SIZE}_{CROSSOVER_TYPE}_'
                                f'{LOCAL_SEARCH_ON_PARETO_FRONT}_{LOCAL_SEARCH_ON_KNEE_SOLUTIONS}_'
                                f'{LOCAL_SEARCH_ON_N_POINTS}_{LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER}_'
                                f'{USING_SURROGATE_MODEL}_{UPDATE_MODEL_AFTER_N_GENS}_'
                                f'd%d_m%m_H%H_M%M')
        ROOT_PATH = PATH_DATA + '/results/' + dir_name

        # Create root folder
        os.mkdir(ROOT_PATH)
        print(f'--> Create folder {ROOT_PATH} - Done\n')

        for i_run in range(NUMBER_OF_RUNS):
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
