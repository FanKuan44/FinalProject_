import argparse
import os
import pickle as pk
import timeit
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.cython.decomposition import Tchebicheff
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from scipy.spatial.distance import cdist

from acc_predictor.factory import get_acc_predictor
from nasbench import wrap_api as api
from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.util.dpfs_calculating import cal_dpfs
from wrap_pymoo.util.elitist_archive import update_elitist_archive


# =========================================================================================================
# Implementation
# =========================================================================================================
def encode(X):
    if isinstance(X, list):
        X = np.array(X)
    encode_X = np.where(X == 'I', 0, X)
    encode_X = np.array(encode_X, dtype=np.int)
    return encode_X


class MOEAD_net(GeneticAlgorithm):
    def __init__(self,
                 max_no_evaluations,
                 surrogate_model_using,
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

        ''' Phan duoi day la customize '''
        self.elitist_archive_X = []
        self.elitist_archive_hashX = []
        self.elitist_archive_F = []

        self.dpfs = []
        self.no_eval = []

        self.max_no_evaluations = max_no_evaluations

        self.surrogate_model_using = surrogate_model_using
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
            F = None
        return F

    def fake_evaluate(self, X):
        encode_X = encode(X)
        if BENCHMARK_NAME == 'cifar10' or BENCHMARK_NAME == 'cifar100':
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
    def _sampling_custom(n_samples):
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

        pop.set('X', pop_X)
        pop.set('hashX', pop_hashX)

        return pop

    def _initialize_custom(self):
        self._decomposition = Tchebicheff()

        if self.surrogate_model_using:
            # Khoi tao 1 so luong kien truc mang de train surrogate model
            models_sampling = self._sampling_custom(500)
            models_sampling_F = self.true_evaluate(models_sampling.get('X'))
            models_sampling.set('F', models_sampling_F)

            self.surrogate_model, _ = self._fit_acc_predictor(inputs=encode(models_sampling.get('X')),
                                                              targets=models_sampling_F[:, 1])
            print('-> initialize surrogate model - done')
            pop = models_sampling[:self.pop_size]
        else:
            pop = self._sampling_custom(self.pop_size)
            pop_F = self.true_evaluate(X=pop.get('X'))
            pop.set('F', pop_F)

        self.ideal_point = np.min(pop.get('F'), axis=0)

        return pop

    @staticmethod
    def crossover_custom(pop, parents_idx):
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
    def mutation_custom(pop, old_offsprings, prob_mutation):
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

    def _next(self, pop):
        if self.surrogate_model_using:
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
                self.models_for_training = []

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
            offsprings = self.crossover_custom(pop=pop, parents_idx=parents_idx)

            if self.surrogate_model_using:
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
            offsprings = self.mutation_custom(pop=pop, old_offsprings=offsprings, prob_mutation=0.1)

            if self.surrogate_model_using:
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

        return pop

    def solve_custom(self):
        self.n_gen = 1

        self.pop = self._initialize_custom()
        print('-> initialize population - done')

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
                self.surrogate_model_using = False
        self._finalize()

        return

    def _do_each_gen(self):
        print(f'Number of evaluations used: {self.no_evaluations}/{self.max_no_evaluations}')

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

    def reset_params(self):
        self._decomposition = None
        self.ideal_point = None

        self.elitist_archive_X = []
        self.elitist_archive_hashX = []
        self.elitist_archive_F = []

        self.dpfs = []
        self.no_eval = []

        self.surrogate_model = None
        self.models_for_training = []

        self.no_evaluations = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MOEAD for NAS')

    # hyper-parameters for problem
    parser.add_argument('--benchmark_name', type=str, default='cifar10',
                        help='the benchmark used for optimizing')
    parser.add_argument('--max_no_evaluations', type=int, default=10000)

    # hyper-parameters for main
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--number_of_runs', type=int, default=1, help='number of runs')

    # hyper-parameters for algorithm (MOEAD)
    parser.add_argument('--algorithm_name', type=str, default='moead', help='name of algorithm used')
    parser.add_argument('--n_points', type=int, default=100)
    parser.add_argument('--surrogate_model_using', type=int, default=0)
    parser.add_argument('--update_model_after_n_gens', type=int, default=10)
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

    ALGORITHM_NAME = args.algorithm_name

    N_POINTS = args.n_points
    MAX_NO_EVALUATIONS = args.max_no_evaluations
    SURROGATE_MODEL_USING = bool(args.surrogate_model_using)
    UPDATE_MODEL_AFTER_N_GENS = args.update_model_after_n_gens

    NUMBER_OF_RUNS = args.number_of_runs
    SEED = args.seed

    now = datetime.now()
    dir_name = now.strftime(f'{BENCHMARK_NAME}_{ALGORITHM_NAME}_{N_POINTS}_%d_%m_%H_%M')
    root_path = dir_name

    # Create root folder
    os.mkdir(root_path)
    print(f'--> Create folder {root_path} - Done\n')

    for i_run in range(NUMBER_OF_RUNS):
        seed = SEED + i_run * 100
        np.random.seed(seed)
        torch.random.manual_seed(seed)

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

        init_ref_dirs = UniformReferenceDirectionFactory(n_dim=2, n_points=N_POINTS).do()
        net = MOEAD_net(
            max_no_evaluations=MAX_NO_EVALUATIONS,
            surrogate_model_using=SURROGATE_MODEL_USING,
            update_model_after_n_gens=UPDATE_MODEL_AFTER_N_GENS,
            path=sub_path,
            ref_dirs=init_ref_dirs,
            n_neighbors=10,
            prob_neighbor_mating=1.0)

        start = timeit.default_timer()
        net.solve_custom()
        end = timeit.default_timer()

        print(f'--> The number of runs: {i_run + 1}/{NUMBER_OF_RUNS}')
        print(f'--> Took {end - start} seconds\n')

    print(f'All {NUMBER_OF_RUNS} runs - Done')
