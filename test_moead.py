import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import timeit


from datetime import datetime

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm

from pymoo.cython.decomposition import Tchebicheff

# from pymoo.docs import parse_doc_string

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling

from pymoo.rand import random

from pymoo.util.display import disp_multi_objective
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

from scipy.spatial.distance import cdist

from nasbench import wrap_api as api

from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.util.dpfs_calculating import cal_dpfs
from wrap_pymoo.util.elitist_archive import update_elitist_archive


# =========================================================================================================
# Implementation
# =========================================================================================================


class MOEAD_net(GeneticAlgorithm):
    def __init__(self,
                 path,
                 ref_dirs,
                 n_neighbors=20,
                 prob_neighbor_mating=0.7,
                 **kwargs):

        self.pop_hash = None
        self.no_evaluations = 0

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating

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

        self.elitist_archive_X = []
        self.elitist_archive_hashX = []
        self.elitist_archive_F = []

        self.dpf = []
        self.no_eval = []

        self.path = path

    def _evaluate(self, X):
        if benchmark_name == 'cifar10' or benchmark_name == 'cifar100':
            if len(X.shape) == 1:
                F = np.full(2, fill_value=np.nan)

                hashX = ''.join(X.tolist())

                F[0] = (benchmark_data[hashX]['MMACs'] - benchmark_min_max['min_MMACs'])\
                        / (benchmark_min_max['max_MMACs'] - benchmark_min_max['min_MMACs'])
                F[1] = 1 - benchmark_data[hashX]['val_acc'] / 100
                self.no_evaluations += 1
            else:
                F = np.full(shape=(X.shape[0], 2), fill_value=np.nan)
                for i in range(X.shape[0]):
                    hashX = ''.join(X[i].tolist())

                    F[i][0] = (benchmark_data[hashX]['MMACs'] - benchmark_min_max['min_MMACs'])\
                       / (benchmark_min_max['max_MMACs'] - benchmark_min_max['min_MMACs'])
                    F[i][1] = 1 - benchmark_data[hashX]['val_acc'] / 100
                    self.no_evaluations += 1
        else:
            F = None
        return F

    @staticmethod
    def _sampling(n_samples):
        pop = Population(n_samples)
        pop_X = []
        pop_hashX = []

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

    def _initialize(self):
        self._decomposition = Tchebicheff()

        pop = self._sampling(self.pop_size)
        pop_F = self._evaluate(X=pop.get('X'))
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

    def _solve(self, problem, termination):
        # generation counter
        self.n_gen = 1

        # initialize the first population and evaluate it
        self.pop = self._initialize()

        self._each_iteration(self, first=True)

        # while termination criteria not fulfilled
        while self.no_evaluations < 1e4:
            self.n_gen += 1

            # do the next iteration
            self.pop = self._next(self.pop)

            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()

        return self.pop

    def _next(self, pop):
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

            # do recombination and create an offspring
            offsprings = self._crossover(pop=pop, parents_idx=parents_idx)

            offsprings_F = self._evaluate(X=offsprings.get('X'))
            offsprings.set('F', offsprings_F)
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings.get('F'),
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

            offsprings = self._mutation(pop=pop, old_offsprings=offsprings, prob_mutation=0.1)

            offsprings_F = self._evaluate(X=offsprings.get('X'))
            offsprings.set('F', offsprings_F)

            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(offsprings.get('X'), offsprings.get('hashX'), offsprings.get('F'),
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

    def solve_(self):
        self.n_gen = 1

        self.pop = self._initialize()

        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(self.pop.get('X'), self.pop.get('hashX'), self.pop.get('F'),
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                   first=True)

        self._do_each_gen()

        while self.no_evaluations < 3e4:
            self.n_gen += 1

            self.pop = self._next(self.pop)

            self._do_each_gen()

        self._finalize()

    def _do_each_gen(self):
        pf = self.elitist_archive_F
        pf = pf[np.argsort(pf[:, 0])]
        pk.dump([pf, self.no_evaluations], open(f'{self.path}/pf_eval/pf_and_evaluated_gen_{self.n_gen}.p', 'wb'))

        dpfs = round(cal_dpfs(pareto_s=self.elitist_archive_F, pareto_front=benchmark_pf_true), 5)
        self.dpf.append(dpfs)
        self.no_eval.append(self.no_evaluations)

    def _finalize(self):
        pk.dump([self.no_eval, self.dpf], open(f'{self.path}/no_eval_and_dpfs.p', 'wb'))
        plt.plot(self.no_eval, self.dpf)
        plt.xlabel('No.Evaluations')
        plt.ylabel('DPFS')
        plt.grid()
        plt.savefig(f'{self.path}/dpfs_and_no_evaluations')
        plt.clf()

        plt.scatter(benchmark_pf_true[:, 0], benchmark_pf_true[:, 1], facecolors='none', edgecolors='blue', s=40, label='true pf')
        plt.scatter(self.elitist_archive_F[:, 0], self.elitist_archive_F[:, 1], c='red', s=15, label='elitist archive')
        plt.xlabel('MMACs (normalize)')
        plt.ylabel('validation error')
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.path}/final_pf')
        plt.clf()


if __name__ == '__main__':
    benchmark_name = 'cifar100'
    benchmark_data = None
    benchmark_min_max = None
    benchmark_pf_true = None

    if benchmark_name == 'nas101':
        NASBENCH_TFRECORD = 'nasbench/nasbench_only108.tfrecord'
        benchmark_api = api.NASBench_(NASBENCH_TFRECORD)
        benchmark_data = pk.load(open('101_benchmark/nas101.p', 'rb'))
        benchmark_min_max = pk.load(open('101_benchmark/min_max_NAS101.p', 'rb'))
        benchmark_pf_true = pk.load(open('101_benchmark/pf_validation_parameters.p', 'rb'))
    elif benchmark_name == 'cifar10':
        benchmark_data = pk.load(open('bosman_benchmark/cifar10/cifar10.p', 'rb'))
        benchmark_min_max = pk.load(open('bosman_benchmark/cifar10/min_max_cifar10.p', 'rb'))
        benchmark_pf_true = pk.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
    elif benchmark_name == 'cifar100':
        benchmark_data = pk.load(open('bosman_benchmark/cifar100/cifar100.p', 'rb'))
        benchmark_min_max = pk.load(open('bosman_benchmark/cifar100/min_max_cifar100.p', 'rb'))
        benchmark_pf_true = pk.load(open('bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))
    benchmark_pf_true[:, 0], benchmark_pf_true[:, 1] = benchmark_pf_true[:, 1], benchmark_pf_true[:, 0].copy()

    print('--> Load benchmark - Done')
    algorithm_name = 'moead'
    n_points = 100
    number_of_runs = 10
    init_seed = 0

    now = datetime.now()
    dir_name = now.strftime(f'{benchmark_name}_{algorithm_name}_{n_points}_%d_%m_%H_%M')
    root_path = dir_name

    # Create root folder
    os.mkdir(root_path)
    print(f'--> Create folder {root_path} - Done\n')

    for i_run in range(number_of_runs):
        seed = init_seed + i_run * 100
        np.random.seed(seed)

        sub_path = root_path + f'/{i_run}'

        # Create new folder (i_run) in root folder
        os.mkdir(sub_path)
        print(f'--> Create folder {sub_path} - Done')

        # Create new folder (pf_eval) in 'i_run' folder
        os.mkdir(sub_path + '/pf_eval')
        print(f'--> Create folder {sub_path}/pf_eval - Done')

        # Create new folder (visualize_pf_each_gen) in 'i_run' folder
        os.mkdir(sub_path + '/visualize_pf_each_gen')
        print(f'--> Create folder {sub_path}/visualize_pf_each_gen - Done')

        init_ref_dirs = UniformReferenceDirectionFactory(2, n_points=n_points).do()
        net = MOEAD_net(
            path=sub_path,
            ref_dirs=init_ref_dirs,
            n_neighbors=10,
            prob_neighbor_mating=1.0)
        start = timeit.default_timer()
        net.solve_()
        end = timeit.default_timer()
        print(f'--> Run {i_run} - Done')
        print(f'--> Took {end - start} seconds\n')

    print(f'Run {number_of_runs} - Done')

