import argparse
import os
import pickle as pk
import timeit
import matplotlib.pyplot as plt
import numpy as np
import torch

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

from acc_predictor.factory import get_acc_predictor
from nasbench import wrap_api as api
from nasbench.lib import model_spec
from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.util.compare import find_better_idv, find_better_idv_bosman_ver
from wrap_pymoo.util.dpfs_calculating import cal_dpfs
from wrap_pymoo.util.elitist_archive import update_elitist_archive
from wrap_pymoo.util.find_knee_solutions import cal_angle, kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3

ModelSpec = model_spec.ModelSpec

# =========================================================================================================
# Implementation
# =========================================================================================================


def encode(X):
    if isinstance(X, list):
        X = np.array(X)
    encode_X = np.where(X == 'I', 0, X)
    encode_X = np.array(encode_X, dtype=np.int)
    return encode_X


class MOEADNET(GeneticAlgorithm):
    def __init__(self,
                 max_no_evaluations,
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

        ''' customize '''
        self.elitist_archive_X = []
        self.elitist_archive_hashX = []
        self.elitist_archive_F = []

        self.dpfs = []
        self.no_eval = []

        self.max_no_evaluations = max_no_evaluations

        self.using_surrogate_model = using_surrogate_model
        self.update_model_after_n_gens = update_model_after_n_gens
        self.surrogate_model = None
        self.models_for_training = []

        self.no_evaluations = 0
        self.path = path

    def true_evaluate(self, X, count_n_evaluations=True):
        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            if len(X.shape) == 1:
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
            if len(X.shape) == 2:
                F = np.full(2, fill_value=np.nan)

                modelspec = api.ModelSpec(matrix=np.array(X[:-1], dtype=np.int), ops=X[-1].tolist())
                hashX = BENCHMARK_API.get_module_hash(modelspec)

                F[0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                        BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                F[1] = 1 - BENCHMARK_DATA[hashX]['val_acc']

                if count_n_evaluations:
                    self.no_evaluations += 1

            else:
                F = np.full(shape=(X.shape[0], 2), fill_value=np.nan)

                for i in range(X.shape[0]):
                    modelspec = api.ModelSpec(matrix=np.array(X[i][:-1], dtype=np.int), ops=X[i][-1].tolist())
                    hashX = BENCHMARK_API.get_module_hash(modelspec)

                    F[i, 0] = (BENCHMARK_DATA[hashX]['params'] - BENCHMARK_MIN_MAX['min_model_params']) / (
                            BENCHMARK_MIN_MAX['max_model_params'] - BENCHMARK_MIN_MAX['min_model_params'])
                    F[i, 1] = 1 - BENCHMARK_DATA[hashX]['val_acc']

                    if count_n_evaluations:
                        self.no_evaluations += 1

        return F

    def fake_evaluate(self, X):
        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            encode_X = encode(X)

            if len(encode_X.shape) == 1:
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
            F = None
        return F

    @staticmethod
    def _sampling(n_samples):
        pop = Population(n_samples)
        pop_X, pop_hashX = [], []

        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
            allowed_choices = ['I', '1', '2']

            while len(pop_X) < n_samples:
                new_X = np.random.choice(allowed_choices, 14)
                new_hashX = ''.join(new_X.tolist())
                if new_hashX not in pop_hashX:
                    pop_X.append(new_X)
                    pop_hashX.append(new_hashX)
        else:
            INPUT = 'input'
            OUTPUT = 'output'
            conv3x3 = 'conv3x3-bn-relu'
            conv1x1 = 'conv1x1-bn-relu'
            maxpool3x3 = 'maxpool3x3'
            num_vertices = 7
            allowed_ops = [conv1x1, conv3x3, maxpool3x3]
            allowed_edges = [0, 1]
            while len(pop_X) < n_samples:
                matrix = np.random.choice(allowed_edges, size=(num_vertices, num_vertices))
                matrix = np.triu(matrix, 1)
                ops = np.random.choice(allowed_ops, size=(1, num_vertices))
                ops[0][0] = INPUT
                ops[0][-1] = OUTPUT

                modelspec = api.ModelSpec(matrix=matrix, ops=ops[0].tolist())

                if BENCHMARK_API.is_valid(modelspec):
                    hashX = BENCHMARK_API.get_module_hash(modelspec)
                    if hashX not in pop_hashX:
                        X = np.concatenate((matrix, ops), axis=0)
                        pop_X.append(X)
                        pop_hashX.append(hashX)

        pop.set('X', pop_X)
        pop.set('hashX', pop_hashX)

        return pop

    def _initialize_custom(self):
        self._decomposition = Tchebicheff()

        if self.using_surrogate_model:
            # Khoi tao 1 so luong kien truc mang de train surrogate model
            models_sampling = self._sampling(500)
            models_sampling_F = self.true_evaluate(models_sampling.get('X'))
            models_sampling.set('F', models_sampling_F)

            self.surrogate_model, _ = self._fit_acc_predictor(inputs=encode(models_sampling.get('X')),
                                                              targets=models_sampling_F[:, 1])
            print('-> initialize surrogate model - done')
            idxs = random.perm(500)
            pop = models_sampling[idxs[:self.pop_size]]
        else:
            pop = self._sampling(self.pop_size)
            pop_F = self.true_evaluate(X=pop.get('X'))
            pop.set('F', pop_F)

        self.ideal_point = np.min(pop.get('F'), axis=0)

        return pop

    @staticmethod
    def _crossover(pop, parents_idx):
        pop_hashX = pop.get('hashX')

        parents_X = pop[parents_idx].get('X')

        offsprings = Population(len(parents_idx))
        offsprings_X = []
        offsprings_hashX = []

        crossover_count = 0
        while len(offsprings_X) < len(parents_idx):
            crossover_count += 1

            crossover_pt = np.random.randint(1, len(parents_X[0]) - 1)

            tmp_offsprings_X = parents_X.copy()

            tmp_offsprings_X[0][crossover_pt:], tmp_offsprings_X[1][crossover_pt:] = \
                tmp_offsprings_X[1][crossover_pt:], tmp_offsprings_X[0][crossover_pt:].copy()

            tmp_offsprings_hashX = [''.join(tmp_offsprings_X[0].tolist()), ''.join(tmp_offsprings_X[0].tolist())]

            for i in range(len(tmp_offsprings_hashX)):
                if crossover_count < 100:
                    if (tmp_offsprings_hashX[i] not in pop_hashX) and (tmp_offsprings_hashX[i] not in offsprings_hashX):
                        offsprings_X.append(tmp_offsprings_X[i])
                        offsprings_hashX.append(tmp_offsprings_hashX[i])
                else:
                    offsprings_X.append(tmp_offsprings_X[i])
                    offsprings_hashX.append(tmp_offsprings_hashX[i])

        offsprings.set('X', offsprings_X[:len(parents_idx)])
        offsprings.set('hashX', offsprings_hashX[:len(parents_idx)])
        return offsprings

    @staticmethod
    def _mutation(pop, old_offsprings, prob_mutation):
        pop_hashX = pop.get('hashX')

        new_offsprings = Population(len(old_offsprings))
        new_offsprings_X = []
        new_offsprings_hashX = []

        old_offsprings_X = old_offsprings.get('X')

        while len(new_offsprings_X) < len(old_offsprings):
            for i in range(len(old_offsprings_X)):
                tmp_new_offspring_X = old_offsprings_X[i].copy()

                prob_mutation_idxs = np.random.rand(len(old_offsprings_X[i]))
                for j in range(len(prob_mutation_idxs)):
                    if prob_mutation_idxs[j] < prob_mutation:
                        allowed_choices = ['I', '1', '2']
                        allowed_choices.remove(tmp_new_offspring_X[j])

                        tmp_new_offspring_X[j] = np.random.choice(allowed_choices)

                tmp_new_offspring_hashX = ''.join(tmp_new_offspring_X)
                if (tmp_new_offspring_hashX not in pop_hashX) and (tmp_new_offspring_hashX not in new_offsprings_hashX):
                    new_offsprings_X.append(tmp_new_offspring_X)
                    new_offsprings_hashX.append(tmp_new_offspring_hashX)

        new_offsprings.set('X', new_offsprings_X[:len(old_offsprings)])
        new_offsprings.set('hashX', new_offsprings_hashX[:len(old_offsprings)])
        return new_offsprings

    @staticmethod
    def _fit_acc_predictor(inputs, targets):
        acc_predictor = get_acc_predictor('mlp', inputs, targets)
        return acc_predictor, acc_predictor.predict(inputs)

    def local_search_on_X(self, pop, X, ls_on_knee_solutions=False):
        off_ = pop.new()
        off_ = off_.merge(X)

        x_old_X, x_old_hashX, x_old_F = off_.get('X'), off_.get('hashX'), off_.get('F')

        non_dominance_X, non_dominance_hashX, non_dominance_F = [], [], []

        # Using for local search on knee solutions
        first, last = 0, 0
        if ls_on_knee_solutions:
            first, last = len(x_old_X) - 2, len(x_old_X) - 1

        if LOCAL_SEARCH_ON_N_POINTS == 1:
            stop_iter = 30

            for i in range(len(x_old_X)):
                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.randint(0, 14)
                        ops = ['I', '1', '2']
                        ops.remove(x_old_X[i][idx])
                    else:
                        idx = np.random.randint(1, 6)
                        ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    x_new_X = x_old_X[i].copy()

                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        x_new_X[idx] = new_op

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx] = new_op

                        modelspec = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                              ops=x_new_X[-1].tolist())
                        x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        checked.append(x_new_hashX)
                        j += 1

                        true_x_new_F = None
                        if USING_SURROGATE_MODEL:
                            x_new_F = self.fake_evaluate(x_new_X)
                            if x_new_F[1] < 0.1:
                                x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                            self.models_for_training.append(x_new_X)

                            true_x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=False)
                        else:
                            x_new_F = self.true_evaluate(x_new_X)

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
                            if USING_SURROGATE_MODEL:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

                        elif better_idv == 0:
                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            if USING_SURROGATE_MODEL:
                                non_dominance_F.append(true_x_new_F)
                            else:
                                non_dominance_F.append(x_new_F)

        elif LOCAL_SEARCH_ON_N_POINTS == 2:
            stop_iter = 30

            for i in range(len(x_old_X)):

                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.choice(14, size=2, replace=False)
                        ops1, ops2 = ['I', '1', '2'], ['I', '1', '2']
                        ops1.remove(x_old_X[i][idx[0]])
                        ops2.remove(x_old_X[i][idx[1]])
                    else:
                        idx = np.random.choice(range(1, 6), size=2, replace=False)
                        ops1 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops2 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops1.remove(x_old_X[i][-1][idx[0]])
                        ops2.remove(x_old_X[i][-1][idx[1]])

                    new_op1, new_op2 = np.random.choice(ops1), np.random.choice(ops2)

                    x_new_X = x_old_X[i].copy()
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        x_new_X[idx[0]], x_new_X[idx[1]] = new_op1, new_op2

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx[0]], x_new_X[-1][idx[1]] = new_op1, new_op2

                        module = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                           ops=x_new_X[-1].tolist())
                        x_new_hashX = BENCHMARK_API.get_module_hash(module)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        j += 1
                        checked.append(x_new_hashX)

                        true_x_new_F = None
                        if USING_SURROGATE_MODEL:
                            x_new_F = self.fake_evaluate(x_new_X)
                            if x_new_F[1] < 0.1:
                                x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                            self.models_for_training.append(x_new_X)

                            true_x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=False)
                        else:
                            x_new_F = self.true_evaluate(x_new_X)

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
                            if USING_SURROGATE_MODEL:
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

        if LOCAL_SEARCH_ON_N_POINTS == 1:
            stop_iter = 30

            for i in range(len(x_old_X)):
                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0
                alpha = np.random.rand()

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.randint(0, 14)
                        ops = ['I', '1', '2']
                        ops.remove(x_old_X[i][idx])
                    else:
                        idx = np.random.randint(1, 6)
                        ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    x_new_X = x_old_X[i].copy()
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        x_new_X[idx] = new_op

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx] = new_op

                        modelspec = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                              ops=x_new_X[-1].tolist())
                        x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        checked.append(x_new_hashX)

                        x_new_F = self.true_evaluate(x_new_X)

                        better_idv = find_better_idv(f1=x_new_F, f2=x_old_F[i])

                        if better_idv == 0:  # Non-dominated solution
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F)

                        else:
                            better_idv_ = find_better_idv_bosman_ver(alpha=alpha, f1=x_new_F, f2=x_old_F[i])
                            if better_idv_ == 1:  # Improved solution
                                x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                                non_dominance_X.append(x_new_X)
                                non_dominance_hashX.append(x_new_hashX)
                                non_dominance_F.append(x_new_F)
                        j += 1

        elif LOCAL_SEARCH_ON_N_POINTS == 2:
            stop_iter = 30

            for i in range(len(x_old_X)):
                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0
                alpha = np.random.rand()

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        idx = np.random.choice(14, size=2, replace=False)
                        ops1, ops2 = ['I', '1', '2'], ['I', '1', '2']
                        ops1.remove(x_old_X[i][idx[0]])
                        ops2.remove(x_old_X[i][idx[1]])
                    else:
                        idx = np.random.choice(range(1, 6), size=2, replace=False)
                        ops1 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops2 = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops1.remove(x_old_X[i][-1][idx[0]])
                        ops2.remove(x_old_X[i][-1][idx[1]])

                    new_op1, new_op2 = np.random.choice(ops1), np.random.choice(ops2)

                    x_new_X = x_old_X[i].copy()
                    if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
                        x_new_X[idx[0]], x_new_X[idx[1]] = new_op1, new_op2

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx[0]], x_new_X[-1][idx[1]] = new_op1, new_op2

                        modelspec = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                              ops=x_new_X[-1].tolist())
                        x_new_hashX = BENCHMARK_API.get_module_hash(modelspec)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        checked.append(x_new_hashX)

                        x_new_F = self.true_evaluate(x_new_X)

                        better_idv = find_better_idv(f1=x_new_F, f2=x_old_F[i])

                        if better_idv == 0:  # Non-dominated solution
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F)

                        else:
                            better_idv_ = find_better_idv_bosman_ver(alpha=alpha, f1=x_new_F, f2=x_old_F[i])
                            if better_idv_ == 1:  # Improved solution
                                x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F

                                non_dominance_X.append(x_new_X)
                                non_dominance_hashX.append(x_new_hashX)
                                non_dominance_F.append(x_new_F)
                        j += 1

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hashX = np.array(non_dominance_hashX)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hashX', x_old_hashX)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hashX, non_dominance_F

    def _next(self, pop):
        if self.using_surrogate_model:
            if self.n_gen % self.update_model_after_n_gens == 0:
                if len(self.models_for_training) < 500:
                    x = np.array(self.models_for_training)
                    self.models_for_training = []
                else:
                    idxs = random.perm(len(self.models_for_training))
                    x = np.array(self.models_for_training)[idxs[:500]]
                    self.models_for_training = self.models_for_training[idxs[500:]]
                y = self.true_evaluate(x, count_n_evaluations=True)[:, 1]
                self.surrogate_model.fit(x=encode(x), y=y)
                print('Update surrogate model - Done')

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
                parents_idx = N[random.perm(self.n_neighbors)][:self.crossover.n_parents]
            else:
                parents_idx = random.perm(self.pop_size)[:self.crossover.n_parents]

            ''' crossover '''
            offsprings = self._crossover(pop=pop, parents_idx=parents_idx)

            if self.using_surrogate_model:
                offsprings_fake_F = self.fake_evaluate(X=offsprings.get('X'))

                tmp = offsprings[offsprings_fake_F[:, 1] < 0.1].get('X')
                if len(tmp) != 0:
                    self.models_for_training.extend(tmp)
                offsprings.set('F', offsprings_fake_F)

                offsprings_true_F = self.true_evaluate(X=offsprings.get('X'), count_n_evaluations=False)
            else:
                offsprings_true_F = self.true_evaluate(X=offsprings.get('X'))
                offsprings.set('F', offsprings_true_F)

            ''' update elitist archive (crossover) '''
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings_true_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

            ''' mutation '''
            offsprings = self._mutation(pop=pop, old_offsprings=offsprings, prob_mutation=0.1)

            if self.using_surrogate_model:
                offsprings_fake_F = self.fake_evaluate(X=offsprings.get('X'))

                tmp = offsprings[offsprings_fake_F[:, 1] < 0.1].get('X')
                if len(tmp) != 0:
                    self.models_for_training.extend(tmp)
                offsprings.set('F', offsprings_fake_F)

                offsprings_true_F = self.true_evaluate(X=offsprings.get('X'), count_n_evaluations=False)
            else:
                offsprings_true_F = self.true_evaluate(X=offsprings.get('X'))
                offsprings.set('F', offsprings_true_F)

            ''' update elitist archive (mutation) '''
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings_true_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

            offspring = offsprings[random.randint(0, len(offsprings))]
            offspring_F = offspring.get('F')

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, offspring_F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get('F'), weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)
            off_FV = self._decomposition.do(offspring_F, weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = offspring

        ''' Local Search on PF '''
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

            ''' UPDATE ELITIST ARCHIVE AFTER LOCAL SEARCH '''
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

        ''' Local Search on Knee Solutions '''
        if LOCAL_SEARCH_ON_KNEE_SOLUTIONS:
            pop_F = pop.get('F')

            front_0 = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size, only_non_dominated_front=True)

            pareto_front = pop[front_0].copy()
            f_pareto_front = pop_F[front_0].copy()

            # Normalize val_error for calculating angle between two individuals
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

            ''' UPDATE ELITIST ARCHIVE AFTER LOCAL SEARCH '''
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

            pareto_front[idx_knee_solutions] = knee_solutions

            pop[front_0] = pareto_front

        return pop

    def solve_custom(self):
        self.n_gen = 1

        # initialize
        self.pop = self._initialize_custom()

        # update elitist archive - initialize
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(self.pop.get('X'), self.pop.get('hashX'), self.pop.get('F'),
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                   first=True)

        self._do_each_gen()

        while self.no_evaluations < self.max_no_evaluations:
            self.n_gen += 1

            self.pop = self._next(self.pop)

            self._do_each_gen()

            if self.n_gen == 20:
                self.using_surrogate_model = False

        self._finalize()
        return

    def _do_each_gen(self):
        # print(f'Number of evaluations used: {self.no_evaluations}/{self.max_no_evaluations}')
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
                    self.dpfs.append(dpfs)
                    self.no_eval.append(self.no_evaluations)

    def _finalize(self):
        if SAVE:
            pk.dump([self.no_eval, self.dpfs], open(f'{self.path}/no_eval_and_dpfs.p', 'wb'))
            plt.plot(self.no_eval, self.dpfs)
            plt.xlabel('No.Evaluations')
            plt.ylabel('DPFS')
            plt.grid()
            plt.savefig(f'{self.path}/dpfs_and_no_evaluations')
            plt.clf()

            plt.scatter(BENCHMARK_PF_TRUE[:, 0], BENCHMARK_PF_TRUE[:, 1], facecolors='none', edgecolors='blue', s=40,
                        label='true pf')
            plt.scatter(self.elitist_archive_F[:, 0], self.elitist_archive_F[:, 1], c='red', s=15, label='elitist archive')
            plt.xlabel('MMACs (normalize)')
            plt.ylabel('validation error')
            plt.legend()
            plt.grid()
            plt.savefig(f'{self.path}/final_pf')
            plt.clf()

    # def reset_params(self):
    #     self._decomposition = None
    #     self.ideal_point = None
    #
    #     self.elitist_archive_X = []
    #     self.elitist_archive_hashX = []
    #     self.elitist_archive_F = []
    #
    #     self.dpfs = []
    #     self.no_eval = []
    #
    #     self.surrogate_model = None
    #     self.models_for_training = []
    #
    #     self.no_evaluations = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MOEAD for NAS')

    # hyper-parameters for problem
    parser.add_argument('--benchmark_name', type=str, default='cifar10',
                        help='the benchmark used for optimizing')
    parser.add_argument('--max_no_evaluations', type=int, default=10000)

    # hyper-parameters for main
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--number_of_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--save', type=int, default=1, help='save log file')

    # hyper-parameters for algorithm (MOEAD)
    parser.add_argument('--algorithm_name', type=str, default='moead', help='name of algorithm used')
    parser.add_argument('--n_points', type=int, default=100)

    parser.add_argument('--using_surrogate_model', type=int, default=0)
    parser.add_argument('--update_model_after_n_gens', type=int, default=10)

    parser.add_argument('--local_search_on_pf', type=int, default=0, help='local search on pareto front')
    parser.add_argument('--local_search_on_knees', type=int, default=0, help='local search on knee solutions')
    parser.add_argument('--local_search_on_n_points', type=int, default=1)
    parser.add_argument('--followed_bosman_paper', type=int, default=0, help='local search followed by bosman paper')

    args = parser.parse_args()

    BENCHMARK_NAME = args.benchmark_name
    BENCHMARK_DATA = None
    BENCHMARK_MIN_MAX = None
    BENCHMARK_PF_TRUE = None

    if BENCHMARK_NAME == 'nas101':
        nasbench_tfrecord = 'nasbench/nasbench_only108.tfrecord'
        BENCHMARK_API = api.NASBench_(nasbench_tfrecord)
        BENCHMARK_DATA = pk.load(open('101_benchmark/nas101.p', 'rb'))
        BENCHMARK_MIN_MAX = pk.load(open('101_benchmark/min_max_NAS101.p', 'rb'))
        BENCHMARK_PF_TRUE = pk.load(open('101_benchmark/pf_validation_parameters.p', 'rb'))

    elif BENCHMARK_NAME == 'cifar10':
        BENCHMARK_DATA = pk.load(open('bosman_benchmark/cifar10/cifar10.p', 'rb'))
        BENCHMARK_MIN_MAX = pk.load(open('bosman_benchmark/cifar10/min_max_cifar10.p', 'rb'))
        BENCHMARK_PF_TRUE = pk.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))

    elif BENCHMARK_NAME == 'cifar100':
        BENCHMARK_DATA = pk.load(open('bosman_benchmark/cifar100/cifar100.p', 'rb'))
        BENCHMARK_MIN_MAX = pk.load(open('bosman_benchmark/cifar100/min_max_cifar100.p', 'rb'))
        BENCHMARK_PF_TRUE = pk.load(open('bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))

    BENCHMARK_PF_TRUE[:, 0], BENCHMARK_PF_TRUE[:, 1] = BENCHMARK_PF_TRUE[:, 1], BENCHMARK_PF_TRUE[:, 0].copy()
    print('--> Load benchmark - Done')

    SAVE = bool(args.save)

    ALGORITHM_NAME = args.algorithm_name

    N_POINTS = args.n_points
    MAX_NO_EVALUATIONS = args.max_no_evaluations

    LOCAL_SEARCH_ON_PARETO_FRONT = bool(args.local_search_on_pf)
    LOCAL_SEARCH_ON_KNEE_SOLUTIONS = bool(args.local_search_on_knees)
    LOCAL_SEARCH_ON_N_POINTS = args.local_search_on_n_points
    LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER = bool(args.followed_bosman_paper)

    USING_SURROGATE_MODEL = bool(args.using_surrogate_model)
    UPDATE_MODEL_AFTER_N_GENS = args.update_model_after_n_gens

    NUMBER_OF_RUNS = args.number_of_runs
    INIT_SEED = args.seed

    now = datetime.now()
    dir_name = now.strftime(f'{BENCHMARK_NAME}_{ALGORITHM_NAME}_{N_POINTS}_'
                            f'{LOCAL_SEARCH_ON_PARETO_FRONT}_{LOCAL_SEARCH_ON_KNEE_SOLUTIONS}_'
                            f'{LOCAL_SEARCH_ON_N_POINTS}_{LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER}_'
                            f'{USING_SURROGATE_MODEL}_{UPDATE_MODEL_AFTER_N_GENS}_'
                            f'%d_%m_%H_%M')
    root_path = dir_name

    # Create root folder
    os.mkdir(root_path)
    print(f'--> Create folder {root_path} - Done\n')

    for i_run in range(NUMBER_OF_RUNS):
        SEED = INIT_SEED + i_run * 100
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)

        sub_path = root_path + f'/{i_run}'

        # Create new folder (i_run) in root folder
        os.mkdir(sub_path)
        print(f'--> Create folder {sub_path} - Done')

        # Create new folder (pf_eval) in 'i_run' folder
        os.mkdir(sub_path + '/pf_eval')
        print(f'--> Create folder {sub_path}/pf_eval - Done')

        # Create new folder (visualize_pf_each_gen) in 'i_run' folder
        os.mkdir(sub_path + '/visualize_pf_each_gen')
        print(f'--> Create folder {sub_path}/visualize_pf_each_gen - Done\n')

        INIT_REF_DIRS = UniformReferenceDirectionFactory(n_dim=2, n_points=N_POINTS).do()
        net = MOEADNET(
            max_no_evaluations=MAX_NO_EVALUATIONS,
            using_surrogate_model=USING_SURROGATE_MODEL,
            update_model_after_n_gens=UPDATE_MODEL_AFTER_N_GENS,
            path=sub_path,
            ref_dirs=INIT_REF_DIRS,
            n_neighbors=10,
            prob_neighbor_mating=1.0)

        start = timeit.default_timer()
        print(f'--> Experiment {i_run + 1} running')
        net.solve_custom()
        end = timeit.default_timer()

        print(f'--> The number of runs DONE: {i_run + 1}/{NUMBER_OF_RUNS}')
        print(f'--> Took {end - start} seconds\n')

    print(f'All {NUMBER_OF_RUNS} runs - Done\nResults are saved on folder {root_path}')