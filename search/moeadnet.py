import pickle
import matplotlib.pyplot as plt
import numpy as np

from pymoo.util.display import disp_multi_objective
from scipy.spatial.distance import cdist

from nasbench import wrap_api as api

from wrap_pymoo.algorithms.genetic_algorithm import GeneticAlgorithm

from wrap_pymoo.model.individual import MyIndividual as Individual
from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.model.sampling import MySampling as Sampling


from wrap_pymoo.operators.crossover.point_crossover import MyPointCrossover as PointCrossover
from wrap_pymoo.operators.mutation.mutation import MyMutation as Mutation

from wrap_pymoo.util.dpfs_calculating import cal_dpfs
from wrap_pymoo.util.elitist_archive import update_elitist_archive

from pymoo.algorithms.moead import MOEAD

from pymoo.cython.decomposition import Tchebicheff
from pymoo.operators.default_operators import set_if_none

from pymoo.rand import random
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


class MOEADNet(GeneticAlgorithm):
    def __init__(self, benchmark, path,
                 ref_dirs, n_neighbors=20, decomposition='auto', prob_neighbor_mating=0.7,
                 **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)
        super().__init__(**kwargs)

        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print('Setting number of neighbors to number of reference directions : %s' % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective

        self.elitist_archive_X = []
        self.elitist_archive_hashX = []
        self.elitist_archive_F = []

        self.dpf = []
        self.no_eval = []

        self.benchmark = benchmark
        self.path = path

        self.benchmark_api = None
        if benchmark == 'nas101':
            self.pf_true = pickle.load(open('101_benchmark/pf_validation_parameters.p', 'rb'))
            NASBENCH_TFRECORD = 'nasbench/nasbench_only108.tfrecord'
            benchmark_api = api.NASBench_(NASBENCH_TFRECORD)
            self.benchmark_api = benchmark_api

        elif benchmark == 'cifar10':
            self.pf_true = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))

        elif benchmark == 'cifar100':
            self.pf_true = pickle.load(open('bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))
        self.pf_true[:, 0], self.pf_true[:, 1] = self.pf_true[:, 1], self.pf_true[:, 0].copy()

    def _solve(self, problem, termination):
        self.n_gen = 0
        # print('Gen:', self.n_gen)

        ''' Initialization '''
        self.pop = self._initialize()

        dpfs = round(cal_dpfs(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
        self.dpf.append(dpfs)
        self.no_eval.append(self.problem._n_evaluated)

        self._each_iteration(self, first=True)

        # while termination criteria not fulfilled
        while termination.do_continue(self):
            self.n_gen += 1
            # print('Gen:', self.n_gen)

            # do the next iteration
            self.pop = self._next(self.pop)

            dpfs = round(cal_dpfs(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
            self.dpf.append(dpfs)
            self.no_eval.append(self.problem._n_evaluated)

            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()

        return self.pop

    def _initialize(self):
        self._decomposition = Tchebicheff()

        pop = Population(n_individuals=0, individual=self.individual)
        pop = self.sampling.sample(problem=self.problem, pop=pop, n_samples=self.pop_size, algorithm=self)

        ''' Update Elitist Archive '''
        # print('--> UPDATE ELITIST ARCHIVE AFTER INITIALIZE POPULATION')
        pop_X = pop.get('X')
        pop_hashX = pop.get('hashX')
        pop_F = pop.get('F')

        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(pop_X, pop_hashX, pop_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                   first=True)
        # print('--> UPDATE ELITIST ARCHIVE AFTER INITIALIZE POPULATION - DONE')

        self.ideal_point = np.min(pop.get('F'), axis=0)

        return pop

    @staticmethod
    def _crossover(pop, parents_idx):
        offspring = Population(2)
        parents = pop[parents_idx].get('X')
        crossover_pt = np.random.randint(1, len(parents[0]))
        X = parents.copy()
        X[0][crossover_pt:], X[1][crossover_pt:] = X[1][crossover_pt:], X[0][crossover_pt:].copy()
        offspring.set('X', X)
        return offspring

    @staticmethod
    def _mutation(off):
        offspring = Population(2)
        off_X = off.get('X')
        for i in range(len(off_X)):
            mutation_prob = np.random.rand(len(off_X[i]))
            for j in range(len(mutation_prob)):
                if mutation_prob[j] < 0.1:
                    allowed_choices = ['I', '1', '2']
                    allowed_choices.remove(off_X[i][j])
                    off_X[i][j] = np.random.choice(allowed_choices)
        offspring.set('X', off_X)
        return offspring

    def _next(self, pop):
        tmp = random.perm(len(pop))

        for i in tmp:
            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]

            if random.random() < self.prob_neighbor_mating:
                parents = N[random.perm(self.n_neighbors)][:self.crossover.n_parents]
            else:
                parents = random.perm(self.pop_size)[::self.crossover.n_parents]

            # do recombination and create an offspring
            off = self._crossover(pop, parents)
            off = self._mutation(off)

            off = off[np.random.randint(0, len(off))]

            # evaluate the offspring
            off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
            off.set('F', off_F)

            CV = np.zeros(1)
            feasible = np.ones(1, dtype=np.bool)

            off.set('CV', CV)
            off.set('feasible', feasible)

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off_F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get('F'), weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)
            off_FV = self._decomposition.do(off_F, weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = off

        pop_X = pop.get('X')
        pop_hashX = pop.get('hashX')
        pop_F = pop.get('F')

        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(pop_X, pop_hashX, pop_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)
        return pop

    # def _mating(self, pop):
    #     """ CROSSOVER """
    #     # print("--> CROSSOVER")
    #     off = self.crossover.do(problem=self.problem, pop=pop, algorithm=self)
    #     # print("--> CROSSOVER - DONE")
    #
    #     """ EVALUATE OFFSPRINGS AFTER CROSSOVER (USING FOR UPDATING ELITIST ARCHIVE) """
    #     # print("--> EVALUATE AFTER CROSSOVER")
    #     off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
    #     off.set('F', off_F)
    #     # print("--> EVALUATE AFTER CROSSOVER - DONE")
    #
    #     """ UPDATING ELITIST ARCHIVE """
    #     # print('--> UPDATE ELITIST ARCHIVE AFTER CROSSOVER')
    #     off_X = off.get('X')
    #     off_hash_X = off.get('hash_X')
    #     off_F = off.get('F')
    #     self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
    #         update_elitist_archive(off_X, off_hash_X, off_F,
    #                                self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)
    #     # print('--> UPDATE ELITIST ARCHIVE AFTER CROSSOVER - DONE')
    #
    #     """ MUTATION """
    #     # print("--> MUTATION")
    #     off = self.mutation.do(self.problem, off, algorithm=self)
    #     # print("--> MUTATION - DONE")
    #
    #     """ EVALUATE OFFSPRINGS AFTER MUTATION (USING FOR UPDATING ELITIST ARCHIVE) """
    #     # print("--> EVALUATE AFTER MUTATION")
    #     off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
    #     off.set('F', off_F)
    #     # print("--> EVALUATE AFTER MUTATION - DONE")
    #
    #     """ UPDATING ELITIST ARCHIVE """
    #     # print('--> UPDATE ELITIST ARCHIVE AFTER MUTATION')
    #     off_X = off.get('X')
    #     off_hash_X = off.get('hash_X')
    #     off_F = off.get('F')
    #     self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
    #         update_elitist_archive(off_X, off_hash_X, off_F,
    #                                self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)
    #     # print('--> UPDATE ELITIST ARCHIVE AFTER MUTATION - DONE')
    #
    #     # _off_hash_X = _off.get('hash_X').tolist()
    #     #
    #     # """ Check duplicate in pop """
    #     # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, pop_hash_X)
    #     #
    #     # _off = _off[not_duplicate]
    #     # _off_hash_X = _off.get('hash_X').tolist()
    #     #
    #     # """ Check duplicate in new offsprings """
    #     # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, _off_hash_X, True)
    #     # _off = _off[not_duplicate]
    #     #
    #     # _off_hash_X = _off.get('hash_X').tolist()
    #     #
    #     # """ Check duplicate in current offsprings """
    #     # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, off.get('hash_X').tolist())
    #     # _off = _off[not_duplicate]
    #     #
    #     # if len(_off) > self.n_offsprings - len(off):
    #     #     I = random.perm(self.n_offsprings - len(off))
    #     #     _off = _off[I]
    #     # if len(_off) != 0:
    #     #     _off_f = self.evaluator.eval(self.problem, _off.get('X'), check=True, algorithm=self)
    #     #     _off.set('F', _off_f)
    #     #     # add to the offsprings and increase the mating counter
    #     # off = off.merge(_off)
    #
    #     CV = np.zeros((len(off), 1))
    #     feasible = np.ones((len(off), 1), dtype=np.bool)
    #
    #     off.set('CV', CV)
    #     off.set('feasible', feasible)
    #     return off
    #
    # def _local_search_on_x(self, pop, x):
    #     off_ = pop.new()
    #     off_ = off_.merge(x)
    #
    #     x_old_X = x.get('X')
    #     x_old_hash_X = x.get('hash_X')
    #     x_old_F = x.get('F')
    #
    #     non_dominance_X = []
    #     non_dominance_hash_X = []
    #     non_dominance_F = []
    #
    #     if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
    #         for i in range(len(x_old_X)):
    #             checked = []
    #             stop_iter = 14 * 2
    #             j = 0
    #             best_X = x_old_X[i].copy()
    #             best_hash_X = x_old_hash_X[i].copy()
    #             best_F = x_old_F[i].copy()
    #             while j < stop_iter:
    #                 idx = np.random.randint(0, 14)
    #                 ops = ['I', '1', '2']
    #                 ops.remove(x_old_X[i][idx])
    #                 new_op = np.random.choice(ops)
    #                 if [idx, new_op] not in checked:
    #                     checked.append([idx, new_op])
    #
    #                     x_new_X = x_old_X[i].copy()
    #                     x_new_X[idx] = new_op
    #
    #                     x_new_hash_X = ''.join(x_new_X.tolist())
    #
    #                     if x_new_hash_X not in x_old_hash_X:
    #                         x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
    #                                                       check=True, algorithm=self)
    #                         # result = check_better(x_new_F[0], x_old_F[i])
    #                         result = check_better(x_new_F[0], best_F)
    #                         if result == 'obj1':
    #                             best_X = x_new_X
    #                             best_hash_X = x_new_hash_X
    #                             best_F = x_new_F[0]
    #                             non_dominance_X.append(x_new_X)
    #                             non_dominance_hash_X.append(x_new_hash_X)
    #                             non_dominance_F.append(x_new_F[0])
    #                         elif result == 'none':
    #                             non_dominance_X.append(x_new_X)
    #                             non_dominance_hash_X.append(x_new_hash_X)
    #                             non_dominance_F.append(x_new_F[0])
    #                     j += 1
    #             x_old_X[i] = best_X
    #             x_old_hash_X[i] = best_hash_X
    #             x_old_F[i] = best_F
    #     elif self.problem.problem_name == 'nas101':
    #         for i in range(len(x_old_X)):
    #             checked = []
    #             stop_iter = 5 * 2
    #             j = 0
    #             while j < stop_iter:
    #                 idx = np.random.randint(1, 6)
    #                 ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    #                 ops.remove(x_old_X[i][-1][idx])
    #                 new_op = np.random.choice(ops)
    #
    #                 if [idx, new_op] not in checked:
    #                     checked.append([idx, new_op])
    #                     x_new_X = x_old_X[i].copy()
    #                     x_new_X[-1][idx] = new_op
    #                     neighbor = api.ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int), ops=x_new_X[-1].tolist())
    #                     neighbor_hash = self.benchmark.get_module_hash(neighbor)
    #
    #                     if neighbor_hash not in x_old_hash_X:
    #                         neighbor_F = self.evaluator.eval(self.problem, np.array([x_new_X]), check=True,
    #                                                          algorithm=self)
    #                         result = check_better(neighbor_F[0], x_old_F[i])
    #                         if result == 'obj1':
    #                             x_old_X[i] = x_new_X
    #                             x_old_hash_X[i] = neighbor_hash
    #                             x_old_F[i] = neighbor_F
    #                             non_dominance_X.append(x_new_X)
    #                             non_dominance_hash_X.append(neighbor_hash)
    #                             non_dominance_F.append(neighbor_F[0])
    #                             break
    #                         elif result == 'none':
    #                             non_dominance_X.append(x_new_X)
    #                             non_dominance_hash_X.append(neighbor_hash)
    #                             non_dominance_F.append(neighbor_F[0])
    #                     j += 1
    #
    #     non_dominance_X = np.array(non_dominance_X)
    #     non_dominance_hash_X = np.array(non_dominance_hash_X)
    #     non_dominance_F = np.array(non_dominance_F)
    #
    #     off_.set('X', x_old_X)
    #     off_.set('hash_X', x_old_hash_X)
    #     off_.set('F', x_old_F)
    #
    #     return off_, non_dominance_X, non_dominance_hash_X, non_dominance_F
    #
    # def _local_search_on_x_bosman(self, pop, x):
    #     off_ = pop.new()
    #     off_ = off_.merge(x)
    #
    #     x_old_X = x.get('X')
    #     x_old_hash_X = x.get('hash_X')
    #     x_old_F = x.get('F')
    #
    #     non_dominance_X = []
    #     non_dominance_hash_X = []
    #     non_dominance_F = []
    #
    #     if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
    #         for i in range(len(x_old_X)):
    #             checked = [''.join(x_old_X[i].copy().tolist())]
    #             stop_iter = 20
    #             j = 0
    #             alpha = np.random.rand(1)
    #             while j < stop_iter:
    #                 idx = np.random.randint(0, 14)
    #
    #                 ops = ['I', '1', '2']
    #                 ops.remove(x_old_X[i][idx])
    #                 new_op = np.random.choice(ops)
    #
    #                 tmp = x_old_X[i].copy().tolist()
    #                 tmp[idx] = new_op
    #                 tmp_str = ''.join(tmp)
    #
    #                 if tmp_str not in checked:
    #                     checked.append(tmp_str)
    #
    #                     x_new_X = x_old_X[i].copy()
    #                     x_new_X[idx] = new_op
    #
    #                     x_new_hash_X = ''.join(x_new_X.tolist())
    #
    #                     if x_new_hash_X not in x_old_hash_X:
    #                         x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
    #                                                       check=True, algorithm=self)
    #                         result = check_better(x_new_F[0], x_old_F[i])
    #                         if result == 'none':
    #                             # print("Non-dominated solution")
    #                             x_old_X[i] = x_new_X
    #                             x_old_hash_X[i] = x_new_hash_X
    #                             x_old_F[i] = x_new_F[0]
    #                             non_dominance_X.append(x_new_X)
    #                             non_dominance_hash_X.append(x_new_hash_X)
    #                             non_dominance_F.append(x_new_F[0])
    #                         else:
    #                             result_ = check_better_bosman(alpha=alpha, f_obj1=x_new_F[0], f_obj2=x_old_F[i])
    #                             if result_ == 'obj1':
    #                                 # print("Improved solution")
    #                                 x_old_X[i] = x_new_X
    #                                 x_old_hash_X[i] = x_new_hash_X
    #                                 x_old_F[i] = x_new_F[0]
    #                                 non_dominance_X.append(x_new_X)
    #                                 non_dominance_hash_X.append(x_new_hash_X)
    #                                 non_dominance_F.append(x_new_F[0])
    #                 j += 1
    #     elif self.problem.problem_name == 'nas101':
    #         for i in range(len(x_old_X)):
    #             checked = []
    #             stop_iter = 5 * 2
    #             j = 0
    #             while j < stop_iter:
    #                 idx = np.random.randint(1, 6)
    #                 ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    #                 ops.remove(x_old_X[i][-1][idx])
    #                 new_op = np.random.choice(ops)
    #
    #                 if [idx, new_op] not in checked:
    #                     checked.append([idx, new_op])
    #                     x_new_X = x_old_X[i].copy()
    #                     x_new_X[-1][idx] = new_op
    #                     neighbor = api.ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int), ops=x_new_X[-1].tolist())
    #                     neighbor_hash = self.benchmark.get_module_hash(neighbor)
    #
    #                     if neighbor_hash not in x_old_hash_X:
    #                         neighbor_F = self.evaluator.eval(self.problem, np.array([x_new_X]), check=True,
    #                                                          algorithm=self)
    #                         result = check_better(neighbor_F[0], x_old_F[i])
    #                         if result == 'obj1':
    #                             x_old_X[i] = x_new_X
    #                             x_old_hash_X[i] = neighbor_hash
    #                             x_old_F[i] = neighbor_F
    #                             non_dominance_X.append(x_new_X)
    #                             non_dominance_hash_X.append(neighbor_hash)
    #                             non_dominance_F.append(neighbor_F[0])
    #                             break
    #                         elif result == 'none':
    #                             non_dominance_X.append(x_new_X)
    #                             non_dominance_hash_X.append(neighbor_hash)
    #                             non_dominance_F.append(neighbor_F[0])
    #                     j += 1
    #
    #     non_dominance_X = np.array(non_dominance_X)
    #     non_dominance_hash_X = np.array(non_dominance_hash_X)
    #     non_dominance_F = np.array(non_dominance_F)
    #
    #     off_.set('X', x_old_X)
    #     off_.set('hash_X', x_old_hash_X)
    #     off_.set('F', x_old_F)
    #
    #     return off_, non_dominance_X, non_dominance_hash_X, non_dominance_F

    def _finalize(self):
        plt.scatter(self.pf_true[:, 0], self.pf_true[:, 1], facecolors='none', edgecolors='blue', s=30, label='True PF')
        plt.scatter(self.elitist_archive_F[:, 0], self.elitist_archive_F[:, 1], c='red', s=10, label='Elitist Archive')
        if self.problem.problem_name == 'nas101':
            plt.xlabel('Params (normalize)')
        else:
            plt.xlabel('MMACs (normalize)')
        plt.ylabel('Validation Error')
        plt.legend()
        plt.savefig(f'{self.path}/final_pf')
        plt.clf()

        plt.plot(self.no_eval, self.dpf)
        plt.xlabel('No.Evaluations')
        plt.ylabel('DPFS')
        plt.grid()
        plt.savefig(f'{self.path}/dpfs_and_no_evaluations')
        plt.clf()

        pf = np.array(self.elitist_archive_F)
        pf = np.unique(pf, axis=0)
        plt.scatter(pf[:, 1], pf[:, 0], c='blue')
        plt.savefig('xxxxx')
        plt.clf()


# =========================================================================================================
# Interface
# =========================================================================================================


def moeadnet(
        pop_size=100,
        n_neighbors=10,
        prob_neighbor_mating=1.0,
        sampling=Sampling(),
        crossover=PointCrossover(type_crossover='2X'),
        mutation=Mutation(prob=0.05),
        **kwargs):

    ref_dirs = UniformReferenceDirectionFactory(2, n_points=pop_size).do()

    return MOEADNet(ref_dirs=ref_dirs,
                    n_neighbors=n_neighbors,
                    prob_neighbor_mating=prob_neighbor_mating,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                    **kwargs)
