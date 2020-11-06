import argparse
import keras
import kerop
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import timeit
import tensorflow as tf
# import torch

# from acc_predictor.factory import get_acc_predictor
from datetime import datetime

from keras_preprocessing.image.image_data_generator import ImageDataGenerator

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import Dominator
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

# =========================================================================================================
# Implementation based on nsga2 from https://github.com/msu-coinlab/pymoo
# =========================================================================================================

from wrap_pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from wrap_pymoo.model.individual import MyIndividual as Individual
from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.util.compare import find_better_idv, find_better_idv_bosman_ver
from wrap_pymoo.util.elitist_archive import update_elitist_archive
from wrap_pymoo.util.find_knee_solutions import cal_angle, kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.optimizer.set_jit(True)


def encode(X):
    if isinstance(X, list):
        X = np.array(X)
    encode_X = np.where(X == 'I', 0, X)
    encode_X = np.array(encode_X, dtype=np.int)
    return encode_X


def insert_to_list_x(x):
    added = ['|', '|', '|', '|']
    indices = [4, 7, 10, 14]

    acc = 0
    for i in range(len(added)):
        x.insert(indices[i]+acc, added[i])
        acc += 1
    return x


def convert_X_to_hashX(x):
    if not isinstance(x, list):
        x = x.tolist()
    x = insert_to_list_x(x)
    x = remove_values_from_list_x(x, 'I')
    hashX = ''.join(x)
    return hashX


def remove_values_from_list_x(x, val):
    return [value for value in x if value != val]


def create_and_evaluate_model(list_of_layers):
    """
    :param list_of_layers: hashX
    :return:
    """
    keras.backend.clear_session()
    TF_CONFIG_ = tf.compat.v1.ConfigProto()
    TF_CONFIG_.gpu_options.per_process_gpu_memory_fraction = 0.9
    TF_CONFIG_.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=TF_CONFIG_)
    keras.backend.set_session(sess)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(32, 32, 3)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    filters = 32
    for layer in list_of_layers:
        if layer == '1':
            model.add(keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
        if layer == '2':
            model.add(keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu'))
        if layer == '|':
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Dropout(0.3))
            filters *= 2
        else:
            model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(filters, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))

    early_stopping = keras.callbacks.EarlyStopping(patience=8)
    optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit model in order to make predictions
    history = model.fit_generator(generator=data.flow(X_train, y_train, batch_size=128),
                                  epochs=N_EPOCHS,
                                  validation_data=(X_val, y_val), callbacks=[early_stopping])
    # _, test_acc = model.evaluate(x=X_test, y=y_test, verbose=0)
    _, layer_flops, _, _ = kerop.profile(model)

    FLOPs = sum(layer_flops) / 1e6

    return FLOPs, max(history.history['val_accuracy'])


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

        ''' Customize '''
        self.data = dict()

        self.crossover_type = crossover_type
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = [], [], []

        self.no_eval = []

        self.max_no_evaluations = max_no_evaluations

        self.using_surrogate_model = using_surrogate_model
        self.update_model_after_n_gens = update_model_after_n_gens
        self.surrogate_model = None
        self.models_for_training = []

        self.no_evaluations = 0
        self.path = path

    def true_evaluate(self, X, single=False, count_n_evaluations=True):
        if single:
            F = np.full(2, fill_value=np.nan)

            FLOPs, val_acc = create_and_evaluate_model(X)
            try:
                self.data[X].append(val_acc)
            except KeyError:
                self.data[X] = [val_acc]

            F[0] = FLOPs
            F[1] = 1 - sum(self.data[X]) / len(self.data[X])

            if count_n_evaluations:
                self.no_evaluations += 1

        else:
            F = np.full(shape=(len(X), 2), fill_value=np.nan)
            for i in range(len(X)):
                FLOPs, val_acc = create_and_evaluate_model(X[i])

                try:
                    self.data[X[i]].append(val_acc)
                except KeyError:
                    self.data[X[i]] = [val_acc]

                F[i][0] = FLOPs
                F[i][1] = 1 - sum(self.data[X[i]]) / len(self.data[X[i]])

                if count_n_evaluations:
                    self.no_evaluations += 1

        return F

    # def fake_evaluate(self, X):
    #     if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
    #         encode_X = encode(X)
    #
    #         if len(encode_X.shape) == 1:
    #             F = np.full(2, fill_value=np.nan)
    #
    #             hashX = ''.join(X.tolist())
    #
    #             F[0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
    #                    / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
    #             F[1] = self.surrogate_model.predict(np.array([encode_X]))[0][0]
    #         else:
    #             F = np.full(shape=(X.shape[0], 2), fill_value=np.nan)
    #             for i in range(encode_X.shape[0]):
    #                 hashX = ''.join(X[i].tolist())
    #
    #                 F[i][0] = (BENCHMARK_DATA[hashX]['MMACs'] - BENCHMARK_MIN_MAX['min_MMACs']) \
    #                           / (BENCHMARK_MIN_MAX['max_MMACs'] - BENCHMARK_MIN_MAX['min_MMACs'])
    #             f1 = self.surrogate_model.predict(encode_X).reshape(X.shape[0])
    #             F[:, 1] = f1
    #     else:
    #         F = None
    #     return F

    # @staticmethod
    # def _create_surrogate_model(inputs, targets):
    #     surrogate_model = get_acc_predictor('mlp', inputs, targets)
    #     return surrogate_model

    @staticmethod
    def _sampling(n_samples):
        pop = Population(n_samples)
        pop_X, pop_hashX = [], []

        allowed_choices = ['I', '1', '2']

        while len(pop_X) < n_samples:
            new_X = np.random.choice(allowed_choices, 13)
            new_hashX = convert_X_to_hashX(new_X)
            if new_hashX not in pop_hashX:
                pop_X.append(new_X)
                pop_hashX.append(new_hashX)

        pop.set('X', pop_X)
        pop.set('hashX', pop_hashX)

        return pop

    @staticmethod
    def _crossover(pop, type_crossover='UX'):
        pop_X = pop.get('X')

        offsprings_X, offsprings_hashX = [], []

        n_crossovers = 0

        while len(offsprings_X) < len(pop_X):
            idx = np.random.choice(len(pop_X), size=(len(pop_X) // 2, 2), replace=False)
            pop_X_ = pop.get('X')[idx]

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
                    raise ValueError('Crossover method is not available!')

                tmp_offspring1_hashX = convert_X_to_hashX(tmp_offspring1_X)
                tmp_offspring2_hashX = convert_X_to_hashX(tmp_offspring2_X)

                if n_crossovers <= 100:
                    if tmp_offspring1_hashX not in offsprings_hashX:
                        offsprings_X.append(tmp_offspring1_X)
                        offsprings_hashX.append(tmp_offspring1_hashX)

                    if tmp_offspring2_hashX not in offsprings_hashX:
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

    @staticmethod
    def _mutation(pop, old_offsprings, prob_mutation=0.05):
        pop_hashX = pop.get('hashX')

        new_offsprings = Population(len(old_offsprings))

        new_offsprings_X = []
        new_offsprings_hashX = []

        old_offsprings_X = old_offsprings.get('X')

        n_mutations = 0

        while len(new_offsprings_X) < len(old_offsprings):

            prob_mutation_idxs = np.random.rand(old_offsprings_X.shape[0], old_offsprings_X.shape[1])

            for i in range(len(old_offsprings_X)):
                tmp_new_offspring_X = old_offsprings_X[i].copy()

                for j in range(prob_mutation_idxs.shape[1]):
                    if prob_mutation_idxs[i][j] <= prob_mutation:
                        allowed_choices = ['I', '1', '2']
                        allowed_choices.remove(tmp_new_offspring_X[j])

                        tmp_new_offspring_X[j] = np.random.choice(allowed_choices)

                tmp_new_offspring_hashX = convert_X_to_hashX(tmp_new_offspring_X)

                if n_mutations <= 100:
                    if (tmp_new_offspring_hashX not in new_offsprings_hashX) and \
                            (tmp_new_offspring_hashX not in pop_hashX):
                        new_offsprings_X.append(tmp_new_offspring_X)
                        new_offsprings_hashX.append(tmp_new_offspring_hashX)
                else:
                    new_offsprings_X.append(tmp_new_offspring_X)
                    new_offsprings_hashX.append(tmp_new_offspring_hashX)

            n_mutations += 1
        idxs = random.perm(len(new_offsprings_X))

        new_offsprings_X = np.array(new_offsprings_X)[idxs[:len(pop)]]
        new_offspring_hashX = np.array(new_offsprings_hashX)[idxs[:len(pop)]]

        new_offsprings.set('X', new_offsprings_X)
        new_offsprings.set('hashX', new_offspring_hashX)

        return new_offsprings

    def _initialize_custom(self):
        pop = self._sampling(self.pop_size)

        pop_F = self.true_evaluate(X=pop.get('hashX'))
        pop.set('F', pop_F)
        # if self.using_surrogate_model:
        #     self.surrogate_model = self._create_surrogate_model(inputs=encode(pop.get('X')),
        #                                                         targets=pop_F[:, 1])
        #     print('-> initialize surrogate model - done')
        #
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

        max_true_n_searches = 14

        if LOCAL_SEARCH_ON_N_POINTS == 1:
            for i in range(len(x_old_X)):
                true_n_searches = 0
                checked = [x_old_hashX[i]]

                tmp_max_n_searches = 100  # Using for avoiding stuck
                tmp_n_searches = 0  # Using for avoiding stuck

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < tmp_max_n_searches):
                    tmp_n_searches += 1

                    idx = np.random.randint(0, 13)
                    ops = ['I', '1', '2']
                    ops.remove(x_old_X[i][idx])

                    new_op = np.random.choice(ops)

                    x_new_X = x_old_X[i].copy()
                    x_new_X[idx] = new_op
                    x_new_hashX = convert_X_to_hashX(x_new_X)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        checked.append(x_new_hashX)
                        true_n_searches += 1

                        true_x_new_F = None
                        # if self.using_surrogate_model:
                        #     x_new_F = self.fake_evaluate(x_new_X)
                        #     if BENCHMARK_NAME == 'cifar10':
                        #         if x_new_F[1] < 0.085:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #     elif BENCHMARK_NAME == 'cifar100':
                        #         if x_new_F[1] < 0.305:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #
                        #     self.models_for_training.append(x_new_X)
                        #     true_x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=False)
                        # else:
                        x_new_F = self.true_evaluate(x_new_hashX, single=True)

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

        elif LOCAL_SEARCH_ON_N_POINTS == 2:
            for i in range(len(x_old_X)):
                true_n_searches = 0
                checked = [x_old_hashX[i]]

                tmp_max_n_searches = 100  # Using for avoiding stuck
                tmp_n_searches = 0  # Using for avoiding stuck

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < tmp_max_n_searches):
                    tmp_n_searches += 1

                    idxs = np.random.choice(13, size=2, replace=False)
                    ops1, ops2 = ['I', '1', '2'], ['I', '1', '2']
                    ops1.remove(x_old_X[i][idxs[0]])
                    ops2.remove(x_old_X[i][idxs[1]])

                    new_op1, new_op2 = np.random.choice(ops1), np.random.choice(ops2)

                    x_new_X = x_old_X[i].copy()
                    x_new_X[idxs[0]], x_new_X[idxs[1]] = new_op1, new_op2
                    x_new_hashX = convert_X_to_hashX(x_new_X)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        true_n_searches += 1
                        checked.append(x_new_hashX)

                        true_x_new_F = None
                        # if self.using_surrogate_model:
                        #     x_new_F = self.fake_evaluate(x_new_X)
                        #     if BENCHMARK_NAME == 'cifar10':
                        #         if x_new_F[1] < 0.085:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #     elif BENCHMARK_NAME == 'cifar100':
                        #         if x_new_F[1] < 0.305:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #
                        #     self.models_for_training.append(x_new_X)
                        #     true_x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=False)
                        # else:
                        x_new_F = self.true_evaluate(x_new_hashX, single=True)

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

        max_true_n_searches = 14

        if LOCAL_SEARCH_ON_N_POINTS == 1:
            for i in range(len(x_old_X)):
                checked = [x_old_hashX[i]]
                true_n_searches = 0

                tmp_max_n_searches = 100  # Using for avoiding stuck
                tmp_n_searches = 0  # Using for avoiding stuck

                alpha = np.random.rand()

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < tmp_max_n_searches):
                    tmp_n_searches += 1

                    idx = np.random.randint(0, 13)
                    ops = ['I', '1', '2']
                    ops.remove(x_old_X[i][idx])

                    new_op = np.random.choice(ops)

                    x_new_X = x_old_X[i].copy()
                    x_new_X[idx] = new_op

                    x_new_hashX = convert_X_to_hashX(x_new_X)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        true_n_searches += 1

                        checked.append(x_new_hashX)

                        true_x_new_F = None
                        # if self.using_surrogate_model:
                        #     x_new_F = self.fake_evaluate(x_new_X)
                        #     if BENCHMARK_NAME == 'cifar10':
                        #         if x_new_F[1] < 0.085:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #     elif BENCHMARK_NAME == 'cifar100':
                        #         if x_new_F[1] < 0.305:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #
                        #     self.models_for_training.append(x_new_X)
                        #     true_x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=False)
                        # else:
                        x_new_F = self.true_evaluate(x_new_hashX, single=True)

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
                checked = [x_old_hashX[i]]
                true_n_searches = 0

                tmp_max_n_searches = 100  # Using for avoiding stuck
                tmp_n_searches = 0  # Using for avoiding stuck

                alpha = np.random.rand()

                while (true_n_searches < max_true_n_searches) and (tmp_n_searches < tmp_max_n_searches):
                    tmp_n_searches += 1

                    idxs = np.random.choice(13, size=2, replace=False)
                    ops1, ops2 = ['I', '1', '2'], ['I', '1', '2']
                    ops1.remove(x_old_X[i][idxs[0]])
                    ops2.remove(x_old_X[i][idxs[1]])

                    new_op1, new_op2 = np.random.choice(ops1), np.random.choice(ops2)

                    x_new_X = x_old_X[i].copy()
                    x_new_X[idxs[0]], x_new_X[idxs[1]] = new_op1, new_op2
                    x_new_hashX = convert_X_to_hashX(x_new_X)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        true_n_searches += 1

                        checked.append(x_new_hashX)

                        true_x_new_F = None
                        # if self.using_surrogate_model:
                        #     x_new_F = self.fake_evaluate(x_new_X)
                        #     if BENCHMARK_NAME == 'cifar10':
                        #         if x_new_F[1] < 0.085:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #     elif BENCHMARK_NAME == 'cifar100':
                        #         if x_new_F[1] < 0.305:
                        #             x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=True)
                        #
                        #     self.models_for_training.append(x_new_X)
                        #     true_x_new_F = self.true_evaluate(x_new_X, count_n_evaluations=False)
                        # else:
                        x_new_F = self.true_evaluate(x_new_hashX, single=True)

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
        print('crossover - done')

        ''' evaluate offsprings - crossover '''
        # if self.using_surrogate_model:
        #     offsprings_predict_F = self.fake_evaluate(X=offsprings.get('X'))
        #
        #     if BENCHMARK_NAME == 'cifar10':
        #         idxs = np.where(offsprings_predict_F[:, 1] < 0.085)[0]
        #     elif BENCHMARK_NAME == 'cifar100':
        #         idxs = np.where(offsprings_predict_F[:, 1] < 0.305)[0]
        #     else:
        #         idxs = np.where(offsprings_predict_F[:, 1] < 0.305)[0]
        #
        #     offsprings_predict_F[idxs] = self.true_evaluate(X=offsprings.get('X')[idxs])
        #     offsprings.set('F', offsprings_predict_F)
        #     self.models_for_training.extend(offsprings.get('X').tolist())
        #     offsprings_true_F = self.true_evaluate(X=offsprings.get('X'), count_n_evaluations=False)
        # else:
        offsprings_true_F = self.true_evaluate(X=offsprings.get('hashX'))
        offsprings.set('F', offsprings_true_F)

        # update elitist archive - crossover
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings_true_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

        # mutation
        offsprings = self._mutation(pop=pop, old_offsprings=offsprings, prob_mutation=0.1)
        print('mutation - done')

        ''' evaluate offsprings - mutation '''
        # if self.using_surrogate_model:
        #     offsprings_predict_F = self.fake_evaluate(X=offsprings.get('X'))
        #     if BENCHMARK_NAME == 'cifar10':
        #         idxs = np.where(offsprings_predict_F[:, 1] < 0.085)[0]
        #     elif BENCHMARK_NAME == 'cifar100':
        #         idxs = np.where(offsprings_predict_F[:, 1] < 0.305)[0]
        #     else:
        #         idxs = np.where(offsprings_predict_F[:, 1] < 0.305)[0]
        #
        #     offsprings_predict_F[idxs] = self.true_evaluate(X=offsprings.get('X')[idxs])
        #     offsprings.set('F', offsprings_predict_F)
        #     self.models_for_training.extend(offsprings.get('X').tolist())
        #     offsprings_true_F = self.true_evaluate(X=offsprings.get('X'), count_n_evaluations=False)
        # else:
        offsprings_true_F = self.true_evaluate(X=offsprings.get('hashX'))
        offsprings.set('F', offsprings_true_F)

        # update elitist archive - mutation
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings_true_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

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
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

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
        print('-> initialize - done')

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

        self._finalize()
        return

    def _do_each_gen(self):
        # if self.using_surrogate_model \
        #         and (self.max_no_evaluations - self.no_evaluations > self.max_no_evaluations // 3) \
        #         and (self.n_gen % self.update_model_after_n_gens == 0):
        #
        #     if len(self.models_for_training) < 500:
        #         x = np.array(self.models_for_training)
        #         self.models_for_training = []
        #     else:
        #         idxs = random.perm(len(self.models_for_training))
        #         x = np.array(self.models_for_training)[idxs[:500]]
        #         self.models_for_training = np.array(self.models_for_training)[idxs[500:]].tolist()
        #
        #     y = self.true_evaluate(x, count_n_evaluations=True)[:, 1]
        #     self.surrogate_model.fit(x=encode(x), y=y)
        #
        #     if DEBUG:
        #         print('Update surrogate model - Done')

        if DEBUG:
            print(f'Number of evaluations used: {self.no_evaluations}/{self.max_no_evaluations}')

        if SAVE:
            pf = self.elitist_archive_F
            pf = pf[np.argsort(pf[:, 0])]
            pk.dump([pf, self.no_evaluations], open(f'{self.path}/pf_eval/pf_and_evaluated_gen_{self.n_gen}.p', 'wb'))

            plt.scatter(self.elitist_archive_F[:, 0], self.elitist_archive_F[:, 1], c='blue', s=15,
                        label='elitist archive')
            plt.xlabel('FLOPs')
            plt.ylabel('validation error')
            plt.legend()
            plt.grid()
            plt.savefig(f'{self.path}/pf_visualize/{self.n_gen}')
            plt.clf()

    def _finalize(self):
        if SAVE:
            pk.dump([self.elitist_archive_hashX, self.elitist_archive_F], open(f'{self.path}/elitist_archive.p', 'wb'))
            pk.dump(self.data, open(f'{self.path}/data.p', 'wb'))

            # visualize elitist archive
            plt.scatter(self.elitist_archive_F[:, 0], self.elitist_archive_F[:, 1], c='blue', s=15,
                        label='elitist archive')
            plt.xlabel('FLOPs')
            plt.ylabel('validation error')
            plt.legend()
            plt.grid()
            plt.savefig(f'{self.path}/final_pf')
            plt.clf()


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


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival:

    @staticmethod
    def do(pop, n_survive):
        # get the objective space values and objects
        F = pop.get('F')

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
        return pop[survivors]


def calc_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity
    return crowding


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NSGAII for NAS')

    # hyper-parameters for problem
    parser.add_argument('--max_no_evaluations', type=int, default=10000)
    parser.add_argument('--n_epochs', type=int, default=1)

    # hyper-parameters for main
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--save', type=int, default=1, help='save log file')

    # hyper-parameters for algorithm (NSGAII)
    parser.add_argument('--algorithm_name', type=str, default='nsga', help='name of algorithm used')
    parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
    parser.add_argument('--crossover_type', type=str, default='UX')

    parser.add_argument('--local_search_on_pf', type=int, default=0, help='local search on pareto front')
    parser.add_argument('--local_search_on_knees', type=int, default=0, help='local search on knee solutions')
    parser.add_argument('--local_search_on_n_points', type=int, default=1)
    parser.add_argument('--followed_bosman_paper', type=int, default=0, help='local search followed by bosman paper')

    parser.add_argument('--using_surrogate_model', type=int, default=0)
    parser.add_argument('--update_model_after_n_gens', type=int, default=10)
    args = parser.parse_args()

    BENCHMARK_NAME = 'SVHN'

    X_train, y_train = pk.load(open('SVHN_dataset/training_data.p', 'rb'))
    print('Load training data - done')
    X_val, y_val = pk.load(open('SVHN_dataset/validating_data.p', 'rb'))
    print('Load validation data - done')
    X_test, y_test = pk.load(open('SVHN_dataset/testing_data.p', 'rb'))
    print('Load testing data - done')

    DEBUG = True

    SAVE = bool(args.save)

    MAX_NO_EVALUATIONS = args.max_no_evaluations
    N_EPOCHS = args.n_epochs

    ALGORITHM_NAME = args.algorithm_name

    POP_SIZE = args.pop_size
    CROSSOVER_TYPE = args.crossover_type

    LOCAL_SEARCH_ON_PARETO_FRONT = bool(args.local_search_on_pf)
    LOCAL_SEARCH_ON_KNEE_SOLUTIONS = bool(args.local_search_on_knees)
    LOCAL_SEARCH_ON_N_POINTS = args.local_search_on_n_points
    LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER = bool(args.followed_bosman_paper)

    USING_SURROGATE_MODEL = bool(args.using_surrogate_model)
    UPDATE_MODEL_AFTER_N_GENS = args.update_model_after_n_gens

    SEED = args.seed

    data = ImageDataGenerator(rotation_range=8,
                              zoom_range=[0.95, 1.05],
                              height_shift_range=0.10,
                              shear_range=0.15)

    now = datetime.now()
    dir_name = now.strftime(f'{BENCHMARK_NAME}_{ALGORITHM_NAME}_{POP_SIZE}_{CROSSOVER_TYPE}_'
                            f'{LOCAL_SEARCH_ON_PARETO_FRONT}_{LOCAL_SEARCH_ON_KNEE_SOLUTIONS}_'
                            f'{LOCAL_SEARCH_ON_N_POINTS}_{LOCAL_SEARCH_FOLLOWED_BOSMAN_PAPER}_'
                            f'{USING_SURROGATE_MODEL}_{UPDATE_MODEL_AFTER_N_GENS}_'
                            f'd%d_m%m_H%H_M%M')
    ROOT_PATH = dir_name

    # Create root folder
    os.mkdir(ROOT_PATH)
    print(f'--> Create folder {ROOT_PATH} - Done\n')

    np.random.seed(SEED)
    # torch.random.manual_seed(SEED)

    # Create new folder (pf_eval) in root folder
    os.mkdir(ROOT_PATH + '/pf_eval')
    print(f'--> Create folder {ROOT_PATH}/pf_eval - Done')

    # Create new folder (pf_visualize) in root folder
    os.mkdir(ROOT_PATH + '/pf_visualize')
    print(f'--> Create folder {ROOT_PATH}/pf_visualize - Done')

    net = NSGANet(
        max_no_evaluations=MAX_NO_EVALUATIONS,
        pop_size=POP_SIZE,
        selection=TournamentSelection(func_comp=binary_tournament),
        survival=RankAndCrowdingSurvival(),
        crossover_type=CROSSOVER_TYPE,
        using_surrogate_model=USING_SURROGATE_MODEL,
        update_model_after_n_gens=UPDATE_MODEL_AFTER_N_GENS,
        path=ROOT_PATH)

    start = timeit.default_timer()
    net.solve_custom()
    end = timeit.default_timer()

    print(f'--> Took {end - start} seconds.\n')

    print(f'Results are saved on folder {ROOT_PATH}.\n')
