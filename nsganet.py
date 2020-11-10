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

from wrap_pymoo.util.compare import find_better_idv, find_better_idv_bosman_ver
from wrap_pymoo.util.dpfs_calculating import cal_dpfs
from wrap_pymoo.util.elitist_archive import update_elitist_archive
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
                 max_no_evaluations,
                 crossover_type,
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
        self.crossover_type = crossover_type

        self.dominated_idv = []
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = [], [], []

        self.dpfs = []
        self.no_eval = []

        self.max_no_evaluations = max_no_evaluations

        self.using_surrogate_model = using_surrogate_model
        self.update_model_after_n_gens = update_model_after_n_gens
        self.surrogate_model = None
        self.models_for_training = []

        self.no_evaluations = 0
        self.path = path

    def true_evaluate(self, X, single=False, count_n_evaluations=True):
        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            if single:
                F = np.full(2, fill_value=np.nan)

                hashX = ''.join(X.tolist())

                F[0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
                       / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
                F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc'] / 100

                if count_n_evaluations:
                    self.no_evaluations += 1

            else:
                F = np.full(shape=(X.shape[0], 2), fill_value=np.nan)
                for i in range(X.shape[0]):
                    hashX = ''.join(X[i].tolist())

                    F[i][0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
                              / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
                    F[i][1] = 1 - BENCHMARK_DATA[hashX]['val_acc'] / 100

                    if count_n_evaluations:
                        self.no_evaluations += 1

        else:
            if single:
                F = np.full(2, fill_value=np.nan)
                matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X)

                matrix_2D = encoding_matrix(matrix_1D)
                ops_STRING = encoding_ops(ops_INT)

                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                hashX = BENCHMARK_API.get_module_hash(modelspec)

                F[0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                        BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc']

                if count_n_evaluations:
                    self.no_evaluations += 1
            else:
                F = np.full(shape=(len(X), 2), fill_value=np.nan)

                for i in range(len(X)):
                    matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X[i])

                    matrix_2D = encoding_matrix(matrix_1D)
                    ops_STRING = encoding_ops(ops_INT)
                    modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                    hashX = BENCHMARK_API.get_module_hash(modelspec)

                    F[i, 0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                            BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                    F[i, 1] = 1 - BENCHMARK_DATA[hashX]['val_acc']

                    if count_n_evaluations:
                        self.no_evaluations += 1

        return F

    def fake_evaluate(self, X, single=False):
        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            encode_X = encode(X)

            if single:
                F = np.full(2, fill_value=np.nan)

                hashX = ''.join(X.tolist())

                F[0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
                       / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
                F[1] = self.surrogate_model.predict(np.array([encode_X]))[0][0]

            else:
                F = np.full(shape=(X.shape[0], 2), fill_value=np.nan)
                for i in range(encode_X.shape[0]):
                    hashX = ''.join(X[i].tolist())

                    F[i][0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
                              / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
                f1 = self.surrogate_model.predict(encode_X).reshape(X.shape[0])
                F[:, 1] = f1
        else:
            if single:
                F = np.full(2, fill_value=np.nan)

                matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X)

                matrix_2D = encoding_matrix(matrix_1D)
                ops_STRING = encoding_ops(ops_INT)
                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                hashX = BENCHMARK_API.get_module_hash(modelspec)

                F[0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                        BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                F[1] = self.surrogate_model.predict(np.array([X]))[0][0]

            else:
                F = np.full(shape=(X.shape[0], 2), fill_value=np.nan)
                for i in range(len(X)):
                    matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(X[i])

                    matrix_2D = encoding_matrix(matrix_1D)
                    ops_STRING = encoding_ops(ops_INT)
                    modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                    hashX = BENCHMARK_API.get_module_hash(modelspec)

                    F[i, 0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                            BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                f1 = self.surrogate_model.predict(X).reshape(X.shape[0])
                F[:, 1] = f1
        return F

    @staticmethod
    def _create_surrogate_model(inputs, targets):
        surrogate_model = get_acc_predictor('mlp', inputs, targets)
        return surrogate_model

    @staticmethod
    def _sampling(n_samples):
        pop = Population(n_samples)
        pop_X, pop_hashX = [], []

        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            allowed_choices = ['I', '1', '2']

            while len(pop_X) < n_samples:
                new_X = np.random.choice(allowed_choices, 14)
                new_hashX = convert_X_to_hashX(new_X)
                if new_hashX not in pop_hashX:
                    pop_X.append(new_X)
                    pop_hashX.append(new_hashX)

        else:
            while len(pop_X) < n_samples:
                matrix_2D, ops_STRING = create_model()
                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)

                if BENCHMARK_API.is_valid(modelspec):
                    hashX = BENCHMARK_API.get_module_hash(modelspec)

                    if hashX not in pop_hashX:
                        matrix_1D = decoding_matrix(matrix_2D)
                        ops_INT = decoding_ops(ops_STRING)

                        X = combine_matrix1D_and_opsINT(matrix=matrix_1D, ops=ops_INT)
                        pop_X.append(X)
                        pop_hashX.append(hashX)

        pop.set('X', pop_X)
        pop.set('hashX', pop_hashX)

        return pop

    def _crossover(self, pop, type_crossover='UX'):
        pop_X = pop.get('X')

        offsprings_X, offsprings_hashX = [], []

        n_crossovers = 0
        n_counts = 0
        while len(offsprings_X) < len(pop_X):
            idx = np.random.choice(len(pop_X), size=(len(pop_X) // 2, 2), replace=False)
            pop_X_ = pop.get('X')[idx]

            if BENCHMARK_NAME == 'nas101':
                for i in range(len(pop_X_)):
                    tmp_offspring1_X, tmp_offspring2_X = pop_X_[i][0].copy(), pop_X_[i][1].copy()

                    if type_crossover == 'UX':
                        crossover_pts = np.random.randint(0, 2, tmp_offspring1_X.shape, dtype=np.bool)

                        tmp_offspring1_X[crossover_pts], tmp_offspring2_X[crossover_pts] = \
                            tmp_offspring2_X[crossover_pts], tmp_offspring1_X[crossover_pts].copy()

                    elif type_crossover == '1X':
                        crossover_pt = np.random.randint(1, len(tmp_offspring1_X))

                        tmp_offspring1_X[crossover_pt:], tmp_offspring2_X[crossover_pt:] = \
                            tmp_offspring2_X[crossover_pt:], tmp_offspring1_X[crossover_pt:].copy()

                    elif type_crossover == '2X':
                        crossover_pts = np.random.choice(range(1, len(tmp_offspring1_X) - 1), 2, replace=False)
                        lower = min(crossover_pts)
                        upper = max(crossover_pts)

                        tmp_offspring1_X[lower:upper], tmp_offspring2_X[lower:upper] = \
                            tmp_offspring2_X[lower:upper], tmp_offspring1_X[lower:upper].copy()
                    else:
                        raise Exception('Crossover method is not available!')

                    matrix1_1D, ops1_INT = split_to_matrix1D_and_opsINT(tmp_offspring1_X)
                    matrix2_1D, ops2_INT = split_to_matrix1D_and_opsINT(tmp_offspring2_X)

                    matrix1_2D = encoding_matrix(matrix1_1D)
                    matrix2_2D = encoding_matrix(matrix2_1D)

                    ops1_STRING = encoding_ops(ops1_INT)
                    ops2_STRING = encoding_ops(ops2_INT)

                    tmp_modelspec1 = api.ModelSpec(matrix=matrix1_2D, ops=ops1_STRING)
                    tmp_modelspec2 = api.ModelSpec(matrix=matrix2_2D, ops=ops2_STRING)

                    list_tmp_modelspec = [tmp_modelspec1, tmp_modelspec2]
                    list_tmp_offspring_X = [tmp_offspring1_X, tmp_offspring2_X]

                    for j in range(2):
                        if BENCHMARK_API.is_valid(list_tmp_modelspec[j]):
                            n_counts += 1
                            tmp_offspring_hashX = BENCHMARK_API.get_module_hash(list_tmp_modelspec[j])
                            if n_crossovers <= 100:
                                if (tmp_offspring_hashX not in offsprings_hashX) and \
                                        (tmp_offspring_hashX not in self.dominated_idv):
                                    offsprings_X.append(list_tmp_offspring_X[j])
                                    offsprings_hashX.append(tmp_offspring_hashX)
                            else:
                                offsprings_X.append(list_tmp_offspring_X[j])
                                offsprings_hashX.append(tmp_offspring_hashX)

            elif BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                for i in range(len(pop_X_)):
                    tmp_offspring1_X, tmp_offspring2_X = pop_X_[i][0].copy(), pop_X_[i][1].copy()

                    if type_crossover == '1X':
                        crossover_pt = np.random.randint(1, len(tmp_offspring1_X))

                        tmp_offspring1_X[crossover_pt:], tmp_offspring2_X[crossover_pt:] = \
                            tmp_offspring2_X[crossover_pt:], tmp_offspring1_X[crossover_pt:].copy()

                    elif type_crossover == '2X':
                        crossover_pts = np.random.choice(range(1, len(tmp_offspring1_X) - 1), 2, replace=False)
                        lower = min(crossover_pts)
                        upper = max(crossover_pts)

                        tmp_offspring1_X[lower:upper], tmp_offspring2_X[lower:upper] = \
                            tmp_offspring2_X[lower:upper], tmp_offspring1_X[lower:upper].copy()

                    elif type_crossover == 'UX':
                        crossover_pts = np.random.randint(0, 2, tmp_offspring1_X.shape, dtype=np.bool)

                        tmp_offspring1_X[crossover_pts], tmp_offspring2_X[crossover_pts] = \
                            tmp_offspring2_X[crossover_pts], tmp_offspring1_X[crossover_pts].copy()
                    else:
                        raise Exception('Crossover method is not available!')

                    tmp_offspring1_hashX = convert_X_to_hashX(tmp_offspring1_X)
                    tmp_offspring2_hashX = convert_X_to_hashX(tmp_offspring2_X)

                    if n_crossovers <= 100:
                        if (tmp_offspring1_hashX not in offsprings_hashX) and \
                                (tmp_offspring1_hashX not in self.dominated_idv):
                            offsprings_X.append(tmp_offspring1_X)
                            offsprings_hashX.append(tmp_offspring1_hashX)

                        if (tmp_offspring2_hashX not in offsprings_hashX) and \
                                (tmp_offspring2_hashX not in self.dominated_idv):
                            offsprings_X.append(tmp_offspring2_X)
                            offsprings_hashX.append(tmp_offspring2_hashX)
                    else:
                        offsprings_X.append(tmp_offspring1_X)
                        offsprings_hashX.append(tmp_offspring1_hashX)

                        offsprings_X.append(tmp_offspring2_X)
                        offsprings_hashX.append(tmp_offspring2_hashX)

            n_crossovers += 1

        idxs = random.perm(len(offsprings_X))

        offspring_X = np.array(offsprings_X)[idxs[:len(pop_X)]]
        offspring_hashX = np.array(offsprings_hashX)[idxs[:len(pop_X)]]

        offsprings = Population(len(pop))

        offsprings.set('X', offspring_X)
        offsprings.set('hashX', offspring_hashX)
        return offsprings

    def _mutation(self, pop, old_offsprings, prob_mutation=0.05):
        pop_hashX = pop.get('hashX')

        new_offsprings = Population(len(old_offsprings))

        new_offsprings_X = []
        new_offsprings_hashX = []

        old_offsprings_X = old_offsprings.get('X')

        while len(new_offsprings_X) < len(old_offsprings):
            if BENCHMARK_NAME == 'nas101':
                for x in old_offsprings_X:
                    matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x)

                    new_matrix_1D = matrix_1D.copy()
                    new_ops_INT = ops_INT.copy()

                    prob_mutation_idxs_matrix = np.random.rand(len(new_matrix_1D))
                    for i in range(len(prob_mutation_idxs_matrix)):
                        if prob_mutation_idxs_matrix[i] <= prob_mutation:
                            new_matrix_1D[i] = 1 - new_matrix_1D[i]

                    prob_mutation_idxs_ops = np.random.rand(len(new_ops_INT))
                    for i in range(len(prob_mutation_idxs_ops)):
                        if prob_mutation_idxs_ops[i] <= prob_mutation:
                            choices = [0, 1, 2]
                            choices.remove(new_ops_INT[i])
                            new_ops_INT[i] = np.random.choice(choices)

                    matrix_2D = encoding_matrix(new_matrix_1D)
                    ops_STRING = encoding_ops(new_ops_INT)

                    new_modelspec = api.ModelSpec(matrix_2D, ops_STRING)

                    if BENCHMARK_API.is_valid(new_modelspec):
                        hashX = BENCHMARK_API.get_module_hash(new_modelspec)
                        if (hashX not in new_offsprings_hashX) and \
                                (hashX not in pop_hashX) and (hashX not in self.dominated_idv):
                            X = combine_matrix1D_and_opsINT(new_matrix_1D, new_ops_INT)
                            new_offsprings_X.append(X)
                            new_offsprings_hashX.append(hashX)

            elif BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                prob_mutation_idxs = np.random.rand(old_offsprings_X.shape[0], old_offsprings_X.shape[1])

                for i in range(len(old_offsprings_X)):
                    tmp_new_offspring_X = old_offsprings_X[i].copy()

                    for j in range(prob_mutation_idxs.shape[1]):
                        if prob_mutation_idxs[i][j] <= prob_mutation:
                            allowed_choices = ['I', '1', '2']
                            allowed_choices.remove(tmp_new_offspring_X[j])

                            tmp_new_offspring_X[j] = np.random.choice(allowed_choices)

                    tmp_new_offspring_hashX = convert_X_to_hashX(tmp_new_offspring_X)

                    if (tmp_new_offspring_hashX not in new_offsprings_hashX) and \
                            (tmp_new_offspring_hashX not in pop_hashX) and \
                            (tmp_new_offspring_hashX not in self.dominated_idv):
                        new_offsprings_X.append(tmp_new_offspring_X)
                        new_offsprings_hashX.append(tmp_new_offspring_hashX)

        idxs = random.perm(len(new_offsprings_X))

        new_offsprings_X = np.array(new_offsprings_X)[idxs[:len(pop)]]
        new_offspring_hashX = np.array(new_offsprings_hashX)[idxs[:len(pop)]]

        new_offsprings.set('X', new_offsprings_X)
        new_offsprings.set('hashX', new_offspring_hashX)

        return new_offsprings

    def _initialize_custom(self):
        pop = self._sampling(self.pop_size)
        pop_F = self.true_evaluate(X=pop.get('X'))
        pop.set('F', pop_F)
        if self.using_surrogate_model:
            if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                self.surrogate_model = self._create_surrogate_model(inputs=encode(pop.get('X')),
                                                                    targets=pop_F[:, 1])
            else:
                self.surrogate_model = self._create_surrogate_model(inputs=pop.get('X'),
                                                                    targets=pop_F[:, 1])

        pop = self.survival.do(pop, self.pop_size)

        return pop

    def local_search_on_X(self, pop, X, ls_on_knee_solutions=False):
        off_ = pop.new()
        off_ = off_.merge(X)

        x_old_X, x_old_hashX, x_old_F = off_.get('X'), off_.get('hashX'), off_.get('F')

        non_dominance_X, non_dominance_hashX, non_dominance_F = [], [], []

        # Using for local search on knee solutions
        first, last = 0, 0
        if ls_on_knee_solutions:
            first, last = len(x_old_X) - 2, len(x_old_X) - 1

        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            max_true_n_searches = 14
        else:
            max_true_n_searches = 26

        if LOCAL_SEARCH_ON_N_POINTS == 1:
            for i in range(len(x_old_X)):
                # Avoid stuck because don't find any better architecture
                max_tmp_n_searches = 100
                tmp_n_searches = 0

                checked = [x_old_hashX[i]]
                true_n_searches = 0

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < max_tmp_n_searches):
                    tmp_n_searches += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.randint(0, 14)
                        ops = ['I', '1', '2']
                        ops.remove(x_old_X[i][idx])

                        new_op = np.random.choice(ops)

                        x_new_X = x_old_X[i].copy()
                        x_new_X[idx] = new_op
                        x_new_hashX = convert_X_to_hashX(x_new_X)

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
                        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])

                        ops_STRING = encoding_ops(ops_INT)

                        while True:
                            idx = np.random.randint(len(matrix_1D))
                            matrix_1D_new = matrix_1D.copy()
                            matrix_1D_new[idx] = 1 - matrix_1D[idx]

                            matrix_2D_new = encoding_matrix(matrix_1D_new)

                            modelspec = api.ModelSpec(matrix=matrix_2D_new, ops=ops_STRING)
                            if BENCHMARK_API.is_valid(modelspec):
                                x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)
                                x_new_X = combine_matrix1D_and_opsINT(matrix_1D_new, ops_INT)
                                break

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX)\
                            and (x_new_hashX not in self.dominated_idv):
                        checked.append(x_new_hashX)
                        true_n_searches += 1

                        true_x_new_F = None
                        if self.using_surrogate_model:
                            x_new_F = self.fake_evaluate(x_new_X, single=True)
                            if BENCHMARK_NAME == 'cifar10':
                                if x_new_F[1] < 0.0835:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            elif BENCHMARK_NAME == 'cifar100':
                                if x_new_F[1] < 0.304:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            else:
                                if x_new_F[1] < 0.067:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)

                            self.models_for_training.append(x_new_X)
                            true_x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=False)
                        else:
                            x_new_F = self.true_evaluate(x_new_X, single=True)

                        if i == first and LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                            better_idv = find_better_idv(x_new_F, x_old_F[i], 'first')
                        elif i == last and LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                            better_idv = find_better_idv(x_new_F, x_old_F[i], 'last')
                        else:
                            better_idv = find_better_idv(x_new_F, x_old_F[i])

                        if better_idv == 1:
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

                        elif better_idv == 0:
                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

                        else:
                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

        elif LOCAL_SEARCH_ON_N_POINTS == 2:
            for i in range(len(x_old_X)):
                # Avoid stuck because don't find any better architecture
                max_tmp_n_searches = 100
                tmp_n_searches = 0

                checked = [x_old_hashX[i]]
                true_n_searches = 0

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < max_tmp_n_searches):
                    tmp_n_searches += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.choice(14, size=2, replace=False)
                        ops1, ops2 = ['I', '1', '2'], ['I', '1', '2']
                        ops1.remove(x_old_X[i][idx[0]])
                        ops2.remove(x_old_X[i][idx[1]])

                        new_op1, new_op2 = np.random.choice(ops1), np.random.choice(ops2)

                        x_new_X = x_old_X[i].copy()

                        x_new_X[idx[0]], x_new_X[idx[1]] = new_op1, new_op2

                        x_new_hashX = convert_X_to_hashX(x_new_X)

                    else:
                        ''' Local search on ops '''
                        # matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])
                        #
                        # matrix_2D = encoding_matrix(matrix_1D)
                        # ops_STRING = encoding_ops(ops_INT)
                        #
                        # idxs = np.random.choice(range(1, 6), size=2, replace=False)
                        #
                        # ops1 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        # ops2 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        #
                        # ops1.remove(ops_STRING[idxs[0]])
                        # ops2.remove(ops_STRING[idxs[1]])
                        #
                        # new_op1 = np.random.choice(ops1)
                        # new_op2 = np.random.choice(ops2)
                        #
                        # ops_STRING[idxs[0]] = new_op1
                        # ops_STRING[idxs[1]] = new_op2
                        #
                        # modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                        # x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)
                        #
                        # ops_INT = decoding_ops(ops_STRING)
                        #
                        # x_new_X = combine_matrix1D_and_opsINT(matrix_1D, ops_INT)

                        ''' Local search on edges '''
                        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])

                        ops_STRING = encoding_ops(ops_INT)

                        while True:
                            idxs = np.random.choice(len(matrix_1D), size=4, replace=False)
                            matrix_1D_new = matrix_1D.copy()
                            matrix_1D_new[idxs] = 1 - matrix_1D[idxs]

                            matrix_2D = encoding_matrix(matrix_1D_new)

                            modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                            if BENCHMARK_API.is_valid(modelspec):
                                x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)
                                x_new_X = combine_matrix1D_and_opsINT(matrix_1D_new, ops_INT)
                                break

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX)\
                            and (x_new_hashX not in self.dominated_idv):
                        true_n_searches += 1
                        checked.append(x_new_hashX)

                        true_x_new_F = None
                        if self.using_surrogate_model:
                            x_new_F = self.fake_evaluate(x_new_X, single=True)
                            if BENCHMARK_NAME == 'cifar10':
                                if x_new_F[1] < 0.0835:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            elif BENCHMARK_NAME == 'cifar100':
                                if x_new_F[1] < 0.304:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            else:
                                if x_new_F[1] < 0.067:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)

                            self.models_for_training.append(x_new_X)
                            true_x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=False)
                        else:
                            x_new_F = self.true_evaluate(x_new_X, single=True)

                        if i == first and LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                            better_idv = find_better_idv(x_new_F, x_old_F[i], 'first')
                        elif i == last and LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
                            better_idv = find_better_idv(x_new_F, x_old_F[i], 'last')
                        else:
                            better_idv = find_better_idv(x_new_F, x_old_F[i])

                        if better_idv == 1:
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

                        elif better_idv == 0:
                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

                        else:
                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hashX = np.array(non_dominance_hashX)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hashX', x_old_hashX)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hashX, non_dominance_F

    def local_search_on_X_bosman(self, pop, X):
        off_ = pop.new()
        off_ = off_.merge(X)

        x_old_X, x_old_hashX, x_old_F = off_.get('X'), off_.get('hashX'), off_.get('F')

        non_dominance_X, non_dominance_hashX, non_dominance_F = [], [], []

        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            max_true_n_searches = 14
        else:
            max_true_n_searches = 26

        if LOCAL_SEARCH_ON_N_POINTS == 1:
            for i in range(len(x_old_X)):
                # Avoid stuck because don't find any better architecture
                max_tmp_n_searches = 100
                tmp_n_searches = 0

                checked = [x_old_hashX[i]]
                true_n_searches = 0
                alpha = np.random.rand()

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < max_tmp_n_searches):
                    tmp_n_searches += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.randint(0, 14)
                        ops = ['I', '1', '2']
                        ops.remove(x_old_X[i][idx])

                        new_op = np.random.choice(ops)

                        x_new_X = x_old_X[i].copy()
                        x_new_X[idx] = new_op
                        x_new_hashX = convert_X_to_hashX(x_new_X)
                    else:
                        ''' Local search on ops '''
                        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])

                        matrix_2D = encoding_matrix(matrix_1D)
                        ops_STRING = encoding_ops(ops_INT)

                        idx = np.random.randint(1, 6)
                        ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops.remove(ops_STRING[idx])

                        new_op = np.random.choice(ops)

                        ops_STRING[idx] = new_op

                        modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                        x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)

                        ops_INT = decoding_ops(ops_STRING)

                        x_new_X = combine_matrix1D_and_opsINT(matrix_1D, ops_INT)

                        ''' Local search on edges '''
                        # matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])
                        #
                        # ops_STRING = encoding_ops(ops_INT)
                        #
                        # while True:
                        #     idx = np.random.randint(len(matrix_1D))
                        #     matrix_1D_new = matrix_1D.copy()
                        #     matrix_1D_new[idx] = 1 - matrix_1D[idx]
                        #
                        #     matrix_2D_new = encoding_matrix(matrix_1D_new)
                        #
                        #     modelspec = ModelSpec(matrix=matrix_2D_new, ops=ops_STRING)
                        #     if BENCHMARK_API.is_valid(modelspec):
                        #         x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)
                        #         x_new_X = combine_matrix1D_and_opsINT(matrix_1D_new, ops_INT)
                        #         break

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        true_n_searches += 1
                        checked.append(x_new_hashX)

                        true_x_new_F = None
                        if self.using_surrogate_model:
                            x_new_F = self.fake_evaluate(x_new_X)
                            if BENCHMARK_NAME == 'cifar10':
                                if x_new_F[1] < 0.0835:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            elif BENCHMARK_NAME == 'cifar100':
                                if x_new_F[1] < 0.304:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            else:
                                if x_new_F[1] < 0.067:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)

                            self.models_for_training.append(x_new_X)
                            true_x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=False)
                        else:
                            x_new_F = self.true_evaluate(x_new_X, single=True)

                        better_idv = find_better_idv(f1=x_new_F, f2=x_old_F[i])

                        if better_idv == 0:  # Non-dominated solution
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

                        else:
                            better_idv_ = find_better_idv_bosman_ver(alpha=alpha, f1=x_new_F, f2=x_old_F[i])
                            if better_idv_ == 1:  # Improved solution
                                x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                                non_dominance_X.append(x_new_X)
                                non_dominance_hashX.append(x_new_hashX)
                                if self.using_surrogate_model:
                                    non_dominance_F.append(true_x_new_F)
                                else:
                                    non_dominance_F.append(x_new_F)

        elif LOCAL_SEARCH_ON_N_POINTS == 2:
            for i in range(len(x_old_X)):
                # Avoid stuck because don't find any better architecture
                max_tmp_n_searches = 100
                tmp_n_searches = 0

                checked = [x_old_hashX[i]]
                true_n_searches = 0
                alpha = np.random.rand()

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < max_tmp_n_searches):
                    tmp_n_searches += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.choice(14, size=2, replace=False)
                        ops1, ops2 = ['I', '1', '2'], ['I', '1', '2']
                        ops1.remove(x_old_X[i][idx[0]])
                        ops2.remove(x_old_X[i][idx[1]])

                        new_op1, new_op2 = np.random.choice(ops1), np.random.choice(ops2)

                        x_new_X = x_old_X[i].copy()

                        x_new_X[idx[0]], x_new_X[idx[1]] = new_op1, new_op2

                        x_new_hashX = convert_X_to_hashX(x_new_X)
                    else:
                        ''' Local search on ops '''
                        matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])

                        matrix_2D = encoding_matrix(matrix_1D)
                        ops_STRING = encoding_ops(ops_INT)

                        idxs = np.random.choice(range(1, 6), size=2, replace=False)

                        ops1 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops2 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']

                        ops1.remove(ops_STRING[idxs[0]])
                        ops2.remove(ops_STRING[idxs[1]])

                        new_op1 = np.random.choice(ops1)
                        new_op2 = np.random.choice(ops2)

                        ops_STRING[idxs[0]] = new_op1
                        ops_STRING[idxs[1]] = new_op2

                        modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                        x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)

                        ops_INT = decoding_ops(ops_STRING)

                        x_new_X = combine_matrix1D_and_opsINT(matrix_1D, ops_INT)

                        ''' Local search on edges '''
                        # matrix_1D, ops_INT = split_to_matrix1D_and_opsINT(x_old_X[i])
                        #
                        # ops_STRING = encoding_ops(ops_INT)
                        #
                        # while True:
                        #     idxs = np.random.choice(len(matrix_1D), size=2, replace=False)
                        #     matrix_1D_new = matrix_1D.copy()
                        #     matrix_1D_new[idxs] = 1 - matrix_1D[idxs]
                        #
                        #     matrix_2D = encoding_matrix(matrix_1D_new)
                        #
                        #     modelspec = ModelSpec(matrix=matrix_2D, ops=ops_STRING)
                        #     if BENCHMARK_API.is_valid(modelspec):
                        #         x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)
                        #         x_new_X = combine_matrix1D_and_opsINT(matrix_1D_new, ops_INT)
                        #         break

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        true_n_searches += 1
                        checked.append(x_new_hashX)

                        true_x_new_F = None
                        if self.using_surrogate_model:
                            x_new_F = self.fake_evaluate(x_new_X)
                            if BENCHMARK_NAME == 'cifar10':
                                if x_new_F[1] < 0.0835:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            elif BENCHMARK_NAME == 'cifar100':
                                if x_new_F[1] < 0.304:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)
                            else:
                                if x_new_F[1] < 0.067:
                                    x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=True)

                            self.models_for_training.append(x_new_X)
                            true_x_new_F = self.true_evaluate(x_new_X, single=True, count_n_evaluations=False)
                        else:
                            x_new_F = self.true_evaluate(x_new_X, single=True)

                        better_idv = find_better_idv(f1=x_new_F, f2=x_old_F[i])

                        if better_idv == 0:  # Non-dominated solution
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if self.using_surrogate_model:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

                        else:
                            better_idv_ = find_better_idv_bosman_ver(alpha=alpha, f1=x_new_F, f2=x_old_F[i])
                            if better_idv_ == 1:  # Improved solution
                                x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                                non_dominance_X.append(x_new_X)
                                non_dominance_hashX.append(x_new_hashX)
                                if self.using_surrogate_model:
                                    non_dominance_F.append(true_x_new_F)
                                else:
                                    non_dominance_F.append(x_new_F)

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hashX = np.array(non_dominance_hashX)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hashX', x_old_hashX)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hashX, non_dominance_F

    def _mating(self, pop):
        # crossover
        offsprings = self._crossover(pop=pop, type_crossover=self.crossover_type)

        # evaluate offsprings - crossover
        if self.using_surrogate_model:
            offsprings_predict_F = self.fake_evaluate(X=offsprings.get('X'))

            if BENCHMARK_NAME == 'cifar10':
                idxs = np.where(offsprings_predict_F[:, 1] < 0.0835)[0]
            elif BENCHMARK_NAME == 'cifar100':
                idxs = np.where(offsprings_predict_F[:, 1] < 0.304)[0]
            else:
                idxs = np.where(offsprings_predict_F[:, 1] < 0.067)[0]

            offsprings_predict_F[idxs] = self.true_evaluate(X=offsprings.get('X')[idxs])
            offsprings.set('F', offsprings_predict_F)
            self.models_for_training.extend(offsprings.get('X').tolist())
            offsprings_true_F = self.true_evaluate(X=offsprings.get('X'), count_n_evaluations=False)
        else:
            offsprings_true_F = self.true_evaluate(X=offsprings.get('X'))
            offsprings.set('F', offsprings_true_F)

        # update elitist archive - crossover
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F, self.dominated_idv = \
            update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings_true_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                   self.dominated_idv)

        # mutation
        offsprings = self._mutation(pop=pop, old_offsprings=offsprings, prob_mutation=0.1)

        # evaluate offsprings - mutation
        if self.using_surrogate_model:
            offsprings_predict_F = self.fake_evaluate(X=offsprings.get('X'))
            if BENCHMARK_NAME == 'cifar10':
                idxs = np.where(offsprings_predict_F[:, 1] < 0.0835)[0]
            elif BENCHMARK_NAME == 'cifar100':
                idxs = np.where(offsprings_predict_F[:, 1] < 0.304)[0]
            else:
                idxs = np.where(offsprings_predict_F[:, 1] < 0.067)[0]

            offsprings_predict_F[idxs] = self.true_evaluate(X=offsprings.get('X')[idxs])
            offsprings.set('F', offsprings_predict_F)
            self.models_for_training.extend(offsprings.get('X').tolist())
            offsprings_true_F = self.true_evaluate(X=offsprings.get('X'), count_n_evaluations=False)
        else:
            offsprings_true_F = self.true_evaluate(X=offsprings.get('X'))
            offsprings.set('F', offsprings_true_F)

        # update elitist archive - mutation
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F, self.dominated_idv = \
            update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings_true_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                   self.dominated_idv)

        return offsprings

    def _next(self, pop):
        # mating
        offsprings = self._mating(pop)

        # merge the offsprings with the current population
        pop = pop.merge(offsprings)

        # select best individuals
        pop = self.survival.do(pop, self.pop_size)

        # local search on pareto front
        if LOCAL_SEARCH_ON_PARETO_FRONT:
            pop_F = pop.get('F')

            front_0 = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size, only_non_dominated_front=True)

            pareto_front = pop[front_0].copy()

            if LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER:
                pareto_front, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X_bosman(pop, X=pareto_front)
            else:
                pareto_front, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X(pop, X=pareto_front)
            pop[front_0] = pareto_front

            # update elitist archive - local search on pareto front
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F, self.dominated_idv = \
                update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                       self.dominated_idv)

        # local search on knee solutions
        if LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
            pop_F = pop.get('F')

            front_0 = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size, only_non_dominated_front=True)

            pareto_front = pop[front_0].copy()
            f_pareto_front = pop_F[front_0].copy()

            # normalize val_error for calculating angle between two individuals
            f_pareto_front_normalize = pop_F[front_0].copy()

            min_f1 = np.min(f_pareto_front[:, 1])
            max_f1 = np.max(f_pareto_front[:, 1])
            f_pareto_front_normalize[:, 1] = (f_pareto_front[:, 1] - min_f1) / (max_f1 - min_f1)

            new_idx = np.argsort(f_pareto_front[:, 0])

            pareto_front = pareto_front[new_idx]
            f_pareto_front = f_pareto_front[new_idx]
            f_pareto_front_normalize = f_pareto_front_normalize[new_idx]
            front_0 = front_0[new_idx]

            angle = [np.array([360, 0])]
            for i in range(1, len(f_pareto_front) - 1):
                if (np.sum(f_pareto_front[i - 1] - f_pareto_front[i]) == 0) or (
                        np.sum(f_pareto_front[i] - f_pareto_front[i + 1]) == 0):
                    angle.append(np.array([0, i]))
                else:
                    tren_hay_duoi = kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3(f_pareto_front[i], f_pareto_front[i - 1],
                                                                             f_pareto_front[i + 1])
                    if tren_hay_duoi == 'duoi':
                        angle.append(
                            np.array(
                                [cal_angle(p_middle=f_pareto_front_normalize[i], p_top=f_pareto_front_normalize[i - 1],
                                           p_bot=f_pareto_front_normalize[i + 1]), i]))
                    else:
                        angle.append(np.array([0, i]))

            angle.append(np.array([360, len(pareto_front) - 1]))
            angle = np.array(angle)
            angle = angle[np.argsort(angle[:, 0])]

            angle = angle[angle[:, 0] > 210]

            idx_knee_solutions = np.array(angle[:, 1], dtype=np.int)
            knee_solutions = pareto_front[idx_knee_solutions].copy()

            # f_knee_solutions = f_pareto_front[idx_knee_solutions]
            # plt.scatter(f_pareto_front[:, 0], f_pareto_front[:, 1], s=30, edgecolors='blue',
            #             facecolors='none', label='True PF')
            # plt.scatter(f_knee_solutions[:, 0], f_knee_solutions[:, 1], c='red', s=15,
            #             label='Knee Solutions')

            if LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER:
                knee_solutions, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X_bosman(pop, X=knee_solutions)
            else:
                knee_solutions, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X(pop, X=knee_solutions, ls_on_knee_solutions=True)

            # update elitist archive - local search on knee solutions
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F, self.dominated_idv = \
                update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                       self.dominated_idv)

            pareto_front[idx_knee_solutions] = knee_solutions

            pop[front_0] = pareto_front

        return pop

    def solve_custom(self):
        self.n_gen = 1

        # initialize
        self.pop = self._initialize_custom()

        # update elitist archive - initialize
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F, self.dominated_idv = \
            update_elitist_archive(self.pop.get('X'), self.pop.get('hashX'), self.pop.get('F'),
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                   self.dominated_idv, first=True)
        self._do_each_gen()

        while self.no_evaluations < self.max_no_evaluations:
            self.n_gen += 1

            self.pop = self._next(self.pop)

            self._do_each_gen()

        self._finalize()
        return

    def _do_each_gen(self):
        if self.using_surrogate_model \
                and (self.max_no_evaluations - self.no_evaluations > self.max_no_evaluations // 3) \
                and (self.n_gen % self.update_model_after_n_gens == 0):

            if len(self.models_for_training) < 500:
                x = np.array(self.models_for_training)
                self.models_for_training = []
            else:
                idxs = random.perm(len(self.models_for_training))
                x = np.array(self.models_for_training)[idxs[:500]]
                self.models_for_training = np.array(self.models_for_training)[idxs[500:]].tolist()

            y = self.true_evaluate(x, count_n_evaluations=True)[:, 1]
            if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                self.surrogate_model.fit(x=encode(x), y=y)
            else:
                self.surrogate_model.fit(x=x, y=y)

            if DEBUG:
                print('Update surrogate model - Done')

        if DEBUG:
            print(f'Number of evaluations used: {self.no_evaluations}/{self.max_no_evaluations}')

        if SAVE:
            pf = self.elitist_archive_F
            pf = pf[np.argsort(pf[:, 0])]
            pk.dump([pf, self.no_evaluations], open(f'{self.path}/pf_eval/pf_and_evaluated_gen_{self.n_gen}.p', 'wb'))

            dpfs = round(cal_dpfs(pareto_s=self.elitist_archive_F, pareto_front=BENCHMARK_PF_TRUE), 5)
            if len(self.no_eval) == 0:
                self.dpfs.append(dpfs)
                self.no_eval.append(self.no_evaluations)
            else:
                if self.no_evaluations == self.no_eval[-1]:
                    self.dpfs[-1] = dpfs
                else:
                    if DEBUG:
                        print('number of local searches:', self.no_evaluations - self.no_eval[-1] - 200)
                    self.dpfs.append(dpfs)
                    self.no_eval.append(self.no_evaluations)

    def _finalize(self):
        pk.dump([self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F],
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
            plt.scatter(self.elitist_archive_F[:, 0], self.elitist_archive_F[:, 1], c='red', s=15,
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
    user_input = [[0, 0, 0, 0, '2X', 1, 10]]

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
    DEBUG = True

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

        if USING_SURROGATE_MODEL:
            MAX_NO_EVALUATIONS = 10000
        else:
            MAX_NO_EVALUATIONS = 10000

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
                max_no_evaluations=MAX_NO_EVALUATIONS,
                pop_size=POP_SIZE,
                selection=TournamentSelection(func_comp=binary_tournament),
                survival=RankAndCrowdingSurvival(),
                crossover_type=CROSSOVER_TYPE,
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
