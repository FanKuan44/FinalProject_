import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import timeit
import torch

from acc_predictor.factory import get_acc_predictor
from datetime import datetime

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.cython.decomposition import Tchebicheff
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.rand import random
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.display import disp_multi_objective
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

from scipy.spatial.distance import cdist

from nasbench import wrap_api as api

from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.util.compare import find_better_idv
from wrap_pymoo.util.IGD_calculating import calc_IGD
# from wrap_pymoo.util.elitist_archive import update_elitist_archive
from wrap_pymoo.util.find_knee_solutions import cal_angle, kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3

from wrap_pymoo.factory_nasbench import combine_matrix1D_and_opsINT, split_to_matrix1D_and_opsINT, create_model
from wrap_pymoo.factory_nasbench import encoding_ops, decoding_ops, encoding_matrix, decoding_matrix


def encode(X, benchmark):
    if isinstance(X, list):
        X = np.array(X)
    if benchmark == 'MacroNAS':
        X = np.where(X == 'I', 0, X)
    encode_X = np.array(X, dtype=np.int)
    return encode_X


# Using for 101
def encode_(X):
    hashX = []
    for x in X:
        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x)

        matrix_2D = encoding_matrix(matrix_1D)
        ops_STRING = encoding_ops(ops_INT)

        modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
        hashX.append(BENCHMARK_API.get_module_hash(modelspec))
    return np.array(hashX)


def encode_for_evaluating(x, benchmark):
    if benchmark == '101':
        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x)

        matrix_2D = encoding_matrix(matrix_1D)
        ops_STRING = encoding_ops(ops_INT)

        modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
        hashX = BENCHMARK_API.get_module_hash(modelspec)
    else:
        if not isinstance(x, list):
            x = x.tolist()
        hashX = ''.join(x)
    return hashX


def encode_for_predicting(x, benchmark):
    if benchmark == '101':
        return np.array([x])
    else:
        if isinstance(x, list):
            x = np.array(x)
        if benchmark == 'MacroNAS':
            x = np.where(x == 'I', 0, x)
        encodeX = np.array(x, dtype=np.int)
    return np.array([encodeX])


# Using for MarcoNAS
def insert_to_list_x(x):
    added = ['|', '|', '|']
    indices = [4, 8, 12]

    acc = 0
    for i in range(len(added)):
        x.insert(indices[i] + acc, added[i])
        acc += 1
    return x


def remove_values_from_list_X(X, val):
    return [value for value in X if value != val]


def convert_to_hashX(X, benchmark):
    if not isinstance(X, list):
        X = X.tolist()
    if benchmark == 'MacroNAS':
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


class MOEADNET(GeneticAlgorithm):
    def __init__(self,
                 m_nEs,
                 typeC,
                 local_search_on_n_vars,
                 using_surrogate_model,
                 update_model_after_n_gens,
                 path,
                 ref_dirs,
                 n_neighbors=20,
                 prob_neighbor_mating=0.7,
                 **kwargs):

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating

        self._decomposition = None
        self.ideal_point = None

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print('Setting number of neighbors to number of reference directions : %s' % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

        ''' Custom '''
        self.NIS = 0
        self.nEs_converging = 0
        self.converging = False

        self.typeC = typeC

        self.alpha = 1

        self.DS = []
        self.A_X, self.A_hashX, self.A_F = [], [], []
        self.tmp_A_X, self.tmp_A_hashX, self.tmp_A_F = [], [], []

        self.IGD = []
        self.no_eval = []

        self.m_nEs = m_nEs

        self.n_vars = local_search_on_n_vars
        self.using_surrogate_model = using_surrogate_model
        self.update_model_after_n_gens = update_model_after_n_gens
        self.surrogate_model = None
        self.update_model = True
        self.n_updates = 0
        self.training_data = []

        self.Y_hat = []
        self.Y = []

        self.nEs = 0
        self.path = path

        self.F_total = []
        self.worst_f0 = -np.inf
        self.worst_f1 = -np.inf

    def update_fake_A(self, new_solution):
        X = new_solution.get('X')
        hashX = new_solution.get('hashX')
        F = new_solution.get('F')

        l = len(self.tmp_A_X)
        r = np.zeros(l)
        if hashX not in self.tmp_A_hashX:
            flag = True
            for i in range(l):
                better_idv = find_better_idv(f0_0=F[0], f0_1=F[1],
                                             f1_0=self.tmp_A_F[i][0], f1_1=self.tmp_A_F[i][1])
                if better_idv == 0:
                    r[i] += 1
                elif better_idv == 1:
                    flag = False
                    break

            if flag:
                self.tmp_A_X.append(np.array(X))
                self.tmp_A_hashX.append(np.array(hashX))
                self.tmp_A_F.append(np.array(F))
                r = np.append(r, 0)

        self.tmp_A_X = np.array(self.tmp_A_X)[r == 0].tolist()
        self.tmp_A_hashX = np.array(self.tmp_A_hashX)[r == 0].tolist()
        self.tmp_A_F = np.array(self.tmp_A_F)[r == 0].tolist()

    def update_A(self, new_solution):
        X = new_solution.get('X')
        hashX = new_solution.get('hashX')
        F = new_solution.get('F')

        l = len(self.A_X)
        r = np.zeros(l)
        if hashX not in self.A_hashX:
            flag = True
            for i in range(l):
                better_idv = find_better_idv(f0_0=F[0], f0_1=F[1],
                                             f1_0=self.A_F[i][0], f1_1=self.A_F[i][1])
                if better_idv == 0:
                    r[i] += 1
                    if self.A_hashX[i] not in self.DS:
                        self.DS.append(self.A_hashX[i])

                elif better_idv == 1:
                    flag = False
                    if hashX not in self.DS:
                        self.DS.append(hashX)
                    break

            if flag:
                self.A_X.append(np.array(X))
                self.A_hashX.append(np.array(hashX))
                self.A_F.append(np.array(F))
                r = np.append(r, 0)

        self.A_X = np.array(self.A_X)[r == 0].tolist()
        self.A_hashX = np.array(self.A_hashX)[r == 0].tolist()
        self.A_F = np.array(self.A_F)[r == 0].tolist()

    def evaluate(self, X, using_surrogate_model=False, count_nE=True):
        F = np.full(2, fill_value=np.nan)

        hashX = encode_for_evaluating(X, BENCHMARK_NAME)
        F[0] = round((BENCHMARK_DATA[hashX][OBJECTIVE_1] - BENCHMARK_MIN_MAX[OBJECTIVE_1]['min']) /
                     (BENCHMARK_MIN_MAX[OBJECTIVE_1]['max'] - BENCHMARK_MIN_MAX[OBJECTIVE_1]['min']), 6)

        twice = False  # --> using on 'surrogate model method'
        if not using_surrogate_model:
            F[1] = round(1 - BENCHMARK_DATA[hashX][OBJECTIVE_2], 4)
            if count_nE:
                self.nEs += 1
            self.F_total.append(F[1])
        else:
            encodeX = encode_for_predicting(X, BENCHMARK_NAME)
            F[1] = self.surrogate_model.predict(encodeX)[0][0]
            old_F = F[1]
            if F[1] < self.alpha:
                twice = True
                F[1] = round(1 - BENCHMARK_DATA[hashX][OBJECTIVE_2], 4)
                if self.update_model:
                    self.Y_hat.append(old_F)
                    self.Y.append(F[1])
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

        if BENCHMARK_NAME == '101':
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
            if BENCHMARK_NAME == '201':
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

    def _crossover(self, idx, P, pC=0.9):
        O = Population(len(idx))
        O_hashX = []

        nCOs = 0

        i = 0
        P_ = P[idx]

        while True:
            if BENCHMARK_NAME == '101':
                if np.random.random() < pC:
                    new_O1_X, new_O2_X = crossover(P_[0].get('X'), P_[1].get('X'), self.typeC)

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
                                        return O
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
                                if i == len(P_):
                                    return O
                else:
                    for m in range(2):
                        O[i].set('X', P_[m].get('X'))
                        O[i].set('hashX', P_[m].get('hashX'))
                        O[i].set('F', P_[m].get('F'))
                        i += 1
                        if i == len(P_):
                            return O
            else:
                if np.random.random() < pC:
                    o1_X, o2_X = crossover(P_[0].get('X'), P_[1].get('X'), self.typeC)

                    o_X = [o1_X, o2_X]
                    o_hashX = [convert_to_hashX(o1_X, BENCHMARK_NAME), convert_to_hashX(o2_X, BENCHMARK_NAME)]

                    if nCOs <= 100:
                        for m in range(2):
                            if (o_hashX[m] not in O_hashX) and (o_hashX[m] not in self.DS):
                                O_hashX.append(o_hashX[m])
                                o_F, twice = self.evaluate(X=o_X[m],
                                                           using_surrogate_model=self.using_surrogate_model)

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
                                if i == len(P_):
                                    return O
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
                            if i == len(P_):
                                return O

                else:
                    for m in range(2):
                        O[i].set('X', P_[m].get('X'))
                        O[i].set('hashX', P_[m].get('hashX'))
                        O[i].set('F', P_[m].get('F'))
                        i += 1
                        if i == len(P_):
                            return O
            nCOs += 1

    def _mutation(self, P, O):
        P_hashX = P.get('hashX')

        new_O = Population(len(O))

        new_O_hashX = []

        old_O_X = O.get('X')

        i = 0
        pM = 1 / len(old_O_X[0])
        while True:
            if BENCHMARK_NAME == '101':
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
                            if i == len(O):
                                return new_O

            else:
                if BENCHMARK_NAME == 'MacroNAS':
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
                        if i == len(O):
                            return new_O

    def _initialize_custom(self):
        self._decomposition = Tchebicheff()

        pop = self._sampling(self.pop_size)
        if self.using_surrogate_model:
            if BENCHMARK_NAME == '101':
                self.surrogate_model = self._create_surrogate_model(inputs=pop.get('X'),
                                                                    targets=pop.get('F')[:, 1])
            else:
                self.surrogate_model = self._create_surrogate_model(inputs=encode(pop.get('X'), BENCHMARK_NAME),
                                                                    targets=pop.get('F')[:, 1])

        self.ideal_point = np.min(pop.get('F'), axis=0)

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
        if BENCHMARK_NAME == 'MacroNAS':
            ops = ['I', '1', '2']
        else:
            ops = ['0', '1', '2', '3', '4']

        for i in range(len(S)):
            # Avoid stuck because do not find any better architectures
            tmp_m_searches = 100
            tmp_n_searches = 0

            checked = [S[i].get('hashX')]
            n_searches = 0

            while (n_searches < m_searches) and (tmp_n_searches < tmp_m_searches):
                tmp_n_searches += 1
                o = S[i].copy()  # --> neighboring solution

                if BENCHMARK_NAME == '101':
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
                        better_idv = find_better_idv(f0_0=o_F[0], f0_1=o_F[1],
                                                     f1_0=S[i].get('F')[0], f1_1=S[i].get('F')[1],
                                                     pos='first')
                    elif i == last and LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                        better_idv = find_better_idv(f0_0=o_F[0], f0_1=o_F[1],
                                                     f1_0=S[i].get('F')[0], f1_1=S[i].get('F')[1],
                                                     pos='last')
                    else:
                        better_idv = find_better_idv(f0_0=o_F[0], f0_1=o_F[1],
                                                     f1_0=S[i].get('F')[0], f1_1=S[i].get('F')[1])

                    if better_idv == 0:  # --> neighboring solutions is better
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
        # iterate for each member of the population in random order
        idxs = random.perm(len(pop))

        for idx in idxs:
            # all neighbors of this individual and corresponding weights
            N = self.neighbors[idx, :]
            '''
            N: bieu dien vi tri neighbor cua ca the thu idx trong quan the

            Ngau nhien phat sinh 1 so tu [0, 1]
            - neu nho hon prob_neighbor_mating thi tien hanh lai ghep giua 2 
            neighbor ngau nhien
            - nguoc lai thi chon ngau nhien 2 ca the trong quan the va tien hanh lai ghep
            '''
            if random.random() < self.prob_neighbor_mating:
                idx_ = N[random.perm(self.n_neighbors)][:self.crossover.n_parents]
            else:
                idx_ = random.perm(self.pop_size)[:self.crossover.n_parents]

            # crossover
            off = self._crossover(idx=idx_, P=pop)

            # mutation
            off = self._mutation(P=pop, O=off)

            off = off[random.randint(0, len(off))]
            off_F = off.get('F')

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off_F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get('F'), weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)
            off_FV = self._decomposition.do(off_F, weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = off

        return pop

    def _next(self, pop):
        # mating
        pop = self._mating(pop)

        if LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
            pop = self.improve_potential_solutions(P=pop)

        return pop

    def solve_custom(self):
        self.n_gen = 1

        # initialize
        self.pop = self._initialize_custom()

        self._do_each_gen(first=True)

        while self.nEs < self.m_nEs:
            self.n_gen += 1

            if not self.converging:
                self.pop = self._next(self.pop)
            else:
                self.nEs += (self.m_nEs - self.nEs_converging) // 10

            self._do_each_gen()
        self._finalize()

    def _do_each_gen(self, first=False):
        if self.using_surrogate_model:
            self.alpha = np.mean(self.F_total)

            # if self.m_nEs - self.nEs < 2 * self.m_nEs // 3 or self.n_updates == 15:
            #     self.update_model = False

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

            if self.update_model and self.n_gen % self.update_model_after_n_gens == 0:
                self.Y_hat = np.array(self.Y_hat)
                self.Y = np.array(self.Y)
                error = 1/len(self.Y) * np.sum((self.Y - self.Y_hat)**2)
                if error <= 1e-3:
                    self.update_model = False
                else:
                    self.Y_hat, self.Y = [], []
                    data = np.array(self.training_data)
                    self.training_data = []

                    X = []
                    Y = []
                    checked = []
                    for i in range(len(data)):
                        if BENCHMARK_NAME == '101':
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

                    self.n_updates += 1
                    if BENCHMARK_NAME == '101':
                        self.surrogate_model.fit(x=X, y=Y)
                    else:
                        self.surrogate_model.fit(x=encode(X, BENCHMARK_NAME), y=Y, verbose=False)

        pf = np.array(self.A_F)
        pf = np.unique(pf, axis=0)
        pf = pf[np.argsort(pf[:, 0])]

        p.dump([pf, self.nEs], open(f'{self.path}/pf_eval/pf_and_evaluated_gen_{self.n_gen}.p', 'wb'))
        p.dump(self.A_X, open(f'{self.path}/elitist_archive/gen_{self.n_gen}.p', 'wb'))

        self.worst_f0 = max(self.worst_f0, np.max(pf[:, 0]))
        self.worst_f1 = max(self.worst_f1, np.max(pf[:, 1]))

        IGD = calc_IGD(pareto_s=pf, pareto_front=BENCHMARK_PF_TRUE)

        if len(self.no_eval) == 0:
            self.IGD.append(IGD)
            self.no_eval.append(self.nEs)
        else:
            if self.nEs == self.no_eval[-1]:
                self.IGD[-1] = IGD
            else:
                self.IGD.append(IGD)
                self.no_eval.append(self.nEs)

        if not first:
            if self.IGD[-1] - self.IGD[-2] == 0:
                self.NIS += 1
            else:
                self.NIS = 0

        if (self.NIS == 100 or self.IGD[-1] == 0) and not self.converging:
            self.converging = True
            self.nEs_converging = self.nEs

        if DEBUG:
            print(f'Number of evaluations used: {self.nEs}/{self.m_nEs}')
            print(IGD)

    def _finalize(self):
        self.A_X = np.array(self.A_X)
        self.A_hashX = np.array(self.A_hashX)
        self.A_F = np.array(self.A_F)

        p.dump([self.A_X, self.A_hashX, self.A_F], open(self.path + '/pareto_front.p', 'wb'))
        if SAVE:
            p.dump([self.worst_f0, self.worst_f1], open(f'{self.path}/reference_point.p', 'wb'))
            # visualize IGD
            p.dump([self.no_eval, self.IGD], open(f'{self.path}/#Evals_IGD.p', 'wb'))
            plt.plot(self.no_eval, self.IGD)
            plt.xlabel('#Evals')
            plt.ylabel('IGD')
            plt.grid()
            plt.savefig(f'{self.path}/#Evals-IGD')
            plt.clf()

            # visualize elitist archive
            plt.scatter(BENCHMARK_PF_TRUE[:, 0], BENCHMARK_PF_TRUE[:, 1], facecolors='none', edgecolors='blue', s=40,
                        label='true pf')
            plt.scatter(self.A_F[:, 0], self.A_F[:, 1], c='red', s=15,
                        label='elitist archive')

            plt.xlabel(OBJECTIVE_1 + '(normalize)')
            plt.ylabel('Validation Error')

            plt.legend()
            plt.grid()
            plt.savefig(f'{self.path}/final_pf')
            plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='benchmark name')
    parser.add_argument('--search_space', type=str, default='C10', help='search space')
    parser.add_argument('--path', type=str, help='path for loading data and saving results')
    parser.add_argument('--n_runs', type=int, default=21, help='number of experiments runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    ''' ------- '''
    SAVE = True
    DEBUG = True

    ALGORITHM_NAME = 'MOEAD'
    N_POINTS = 15

    NUMBER_OF_RUNS = args.n_runs
    INIT_SEED = args.seed

    BENCHMARK_NAME = args.benchmark
    SEARCH_SPACE = args.search_space

    if BENCHMARK_NAME == 'MacroNAS':
        OBJECTIVE_1 = 'MMACs'
    elif BENCHMARK_NAME == '101':
        OBJECTIVE_1 = 'Params'
    else:
        OBJECTIVE_1 = 'FLOPs'
    OBJECTIVE_2 = 'valid_acc'

    user_input = [[0, 0, '2X', 1, 10]]

    PATH_DATA = args.path + '/BENCHMARKS'

    if BENCHMARK_NAME == '101':
        BENCHMARK_API = api.NASBench_()
    f_data = open(PATH_DATA + f'/{BENCHMARK_NAME}/{SEARCH_SPACE}/data.p', 'rb')
    f_mi_ma = open(PATH_DATA + f'/{BENCHMARK_NAME}/{SEARCH_SPACE}/mi_ma.p', 'rb')
    f_pf = open(PATH_DATA + f'/{BENCHMARK_NAME}/{SEARCH_SPACE}/pf_valid(error).p', 'rb')

    BENCHMARK_DATA = p.load(f_data)
    BENCHMARK_MIN_MAX = p.load(f_mi_ma)
    BENCHMARK_PF_TRUE = p.load(f_pf)

    f_data.close()
    f_mi_ma.close()
    f_pf.close()
    print('--> Load benchmark - Done')

    if BENCHMARK_NAME == '201':
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
        dir_name = now.strftime(
            f'{BENCHMARK_NAME}_{SEARCH_SPACE}_{ALGORITHM_NAME}_{POP_SIZE}_{N_POINTS}_{CROSSOVER_TYPE}_'
            f'{LOCAL_SEARCH_ON_KNEE_SOLUTIONS}_{LOCAL_SEARCH_ON_N_POINTS}_'
            f'{USING_SURROGATE_MODEL}_{UPDATE_MODEL_AFTER_N_GENS}_'
            f'd%d_m%m_H%H_M%M_S%S')
        ROOT_PATH = args.path + '/RESULTS/' + dir_name
        # Create root folder
        os.mkdir(ROOT_PATH)
        print(f'--> Create folder {ROOT_PATH} - Done\n')

        f_log = open(ROOT_PATH + '/log_file.txt', 'w')
        f_log.write(f'- #RUNS: {NUMBER_OF_RUNS}\n- BENCHMARK: {BENCHMARK_NAME}\n- SEARCH_SPACE: {SEARCH_SPACE}\n'
                    f'- OBJECTIVE 1: {OBJECTIVE_1}\n- OBJECTIVE 2: Validation error\n'
                    f'- ALGORITHM: {ALGORITHM_NAME}\n- POP_SIZE: {POP_SIZE}\n- #MAX_EVALS: {MAX_NO_EVALUATIONS}\n'
                    f'- N_POINTS: {N_POINTS}\n- CROSSOVER_TYPE: {CROSSOVER_TYPE}\n- LOCAL_SEARCH_ON_KNEE_SOLUTIONS: '
                    f'{bool(LOCAL_SEARCH_ON_KNEE_SOLUTIONS)}\n- LOCAL_SEARCH_ON_#POINTS: {LOCAL_SEARCH_ON_N_POINTS}\n'
                    f'- USING_SURROGATE_MODEL: {bool(USING_SURROGATE_MODEL)}\n- UPDATE_MODEL_AFTER_#GENS: '
                    f'{UPDATE_MODEL_AFTER_N_GENS}')
        f_log.close()

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

            # Create new folder (elitist_archive) in 'i_run' folder
            os.mkdir(SUB_PATH + '/elitist_archive')
            print(f'--> Create folder {SUB_PATH}/elitist_archive - Done\n')

            INIT_REF_DIRS = UniformReferenceDirectionFactory(n_dim=2, n_points=N_POINTS).do()

            net = MOEADNET(
                m_nEs=MAX_NO_EVALUATIONS,
                typeC=CROSSOVER_TYPE,
                local_search_on_n_vars=LOCAL_SEARCH_ON_N_POINTS,
                using_surrogate_model=USING_SURROGATE_MODEL,
                update_model_after_n_gens=UPDATE_MODEL_AFTER_N_GENS,
                path=SUB_PATH,
                ref_dirs=INIT_REF_DIRS,
                n_neighbors=10,
                prob_neighbor_mating=1.0)

            start = timeit.default_timer()
            print(f'--> Experiment {i_run + 1} is running.')
            net.solve_custom()
            end = timeit.default_timer()

            print(f'--> The number of runs done: {i_run + 1}/{NUMBER_OF_RUNS}')
            print(f'--> Took {end - start} seconds.\n')

        print(f'All {NUMBER_OF_RUNS} runs - Done\n'
              f'Results are saved on folder {ROOT_PATH}.\n')
