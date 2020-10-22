import numpy as np
import math
from pymoo.rand import random
from pymoo.algorithms.moead import MOEAD
# from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from wrap_pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from wrap_pymoo.model.individual import Individual
from wrap_pymoo.model.population import Population
# from pymoo.model.population import Population

from wrap_pymoo.model.sampling import MySampling
from pymoo.model.survival import Survival
from pymoo.operators.crossover.point_crossover import PointCrossover
from wrap_pymoo.operators.crossover.point_crossover import PointCrossover

from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from wrap_pymoo.operators.mutation.mutation import Mutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import Dominator
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

import pickle
import matplotlib.pyplot as plt
# =========================================================================================================
# Implementation based on nsga2 from https://github.com/msu-coinlab/pymoo
# =========================================================================================================
from nasbench import wrap_api as api

NASBENCH_TFRECORD = 'nasbench/nasbench_only108.tfrecord'
benchmark = api.NASBench_(NASBENCH_TFRECORD)


def check_better(obj1, obj2):
    if obj1[0] <= obj2[0] and obj1[1] < obj2[1]:
        return 'obj1'
    if obj1[1] <= obj2[1] and obj1[0] < obj2[0]:
        return 'obj1'
    if obj2[0] <= obj1[0] and obj2[1] < obj1[1]:
        return 'obj2'
    if obj2[1] <= obj1[1] and obj2[0] < obj1[0]:
        return 'obj2'
    return 'none'


def check_better_bosman(alpha, f_obj1, f_obj2):
    f_obj1_ = alpha * f_obj1[0] + (1 - alpha) * f_obj1[1]
    f_obj2_ = alpha * f_obj2[0] + (1 - alpha) * f_obj2[1]
    if f_obj1_ <= f_obj2_:
        return 'obj1'
    else:
        return 'obj2'


def check_off_not_in(f_off, f_check):
    checked = []
    checked_ = []
    for i in range(len(f_off)):
        flag = True
        for j in f_check:
            if np.sum(f_off[i] - j) == 0:
                flag = False
                break
        if flag:
            flag_ = True
            for x in checked_:
                if np.sum(f_off[i] - x) == 0:
                    flag_ = False
                    break
            if flag_:
                checked.append(i)
                checked_.append(f_off[i])
    return checked


def is_x1_not_duplicate_x2(x1, x2, check_in_self=False):
    not_duplicated = []
    for i in range(len(x1)):
        if check_in_self:
            if x2.count(x1[i]) == 1:
                not_duplicated.append(i)
        else:
            if x1[i] not in x2:
                not_duplicated.append(i)
    return not_duplicated


def kiem_tra_p1_nam_phia_duoi_p2_p3(p1, p2, p3):
    v_cp = p3 - p2
    dt = -v_cp[1] * (p1[0] - p2[0]) + v_cp[0] * (p1[1] - p2[1])
    if dt > 0:
        return True
    return False


def cal_angle(p_middle, p_top, p_bot):
    c = np.sqrt(np.sum((p_top - p_middle) ** 2))
    a = np.sqrt(np.sum((p_bot - p_middle) ** 2))
    b = np.sqrt(np.sum((p_bot - p_top) ** 2))

    cosine_angle = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)


def update_elitist_archive(new_idv_X_lst, new_idv_hash_X_lst, new_idv_F_lst,
                           elitist_archive_X, elitist_archive_hash_X, elitist_archive_F, first=False):
    if first:
        current_elitist_archive_X = elitist_archive_X.copy()
        current_elitist_archive_hash_X = elitist_archive_hash_X.copy()
        current_elitist_archive_F = elitist_archive_F.copy()
    else:
        current_elitist_archive_X = elitist_archive_X.copy().tolist()
        current_elitist_archive_hash_X = elitist_archive_hash_X.copy().tolist()
        current_elitist_archive_F = elitist_archive_F.copy().tolist()

    rank = np.zeros(len(current_elitist_archive_X))

    for i in range(len(new_idv_X_lst)):  # Duyet cac phan tu trong list can check
        if new_idv_hash_X_lst[i] not in current_elitist_archive_hash_X:
            flag = True  # Check xem co bi dominate khong?
            for j in range(len(current_elitist_archive_X)):  # Duyet cac phan tu trong elitist archive hien tai
                better_idv = check_better(new_idv_F_lst[i], current_elitist_archive_F[j])  # Kiem tra xem tot hon hay khong?
                if better_idv == 'obj1':
                    rank[j] += 1
                elif better_idv == 'obj2':
                    flag = False
                    break
            if flag:
                current_elitist_archive_X.append(np.array(new_idv_X_lst[i]))
                current_elitist_archive_hash_X.append(np.array(new_idv_hash_X_lst[i]))
                current_elitist_archive_F.append(np.array(new_idv_F_lst[i]))
                rank = np.append(rank, 0)

    current_elitist_archive_X = np.array(current_elitist_archive_X)[rank == 0]
    current_elitist_archive_hash_X = np.array(current_elitist_archive_hash_X)[rank == 0]
    current_elitist_archive_F = np.array(current_elitist_archive_F)[rank == 0]

    return current_elitist_archive_X,\
           current_elitist_archive_hash_X,\
           current_elitist_archive_F


def cal_euclid_distance(x1, x2):
    e_dis = np.sqrt(np.sum((x1 - x2) ** 2))
    return e_dis


def cal_dpf(pareto_front, pareto_s):
    pareto_s = np.unique(pareto_s, axis=0)
    d = 0
    for solution in pareto_front:
        d_ = np.inf
        for solution_ in pareto_s:
            d_ = min(cal_euclid_distance(solution, solution_), d_)
        d += d_
    return d / len(pareto_front)


class NSGANet(GeneticAlgorithm):
    def __init__(self, local_search_on_pf, local_search_on_knee, path, opt_val_acc_and_training_time=1,
                 **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective

        self.local_search_on_pf = local_search_on_pf
        self.local_search_on_knee = local_search_on_knee
        self.opt_val_acc_and_training_time = opt_val_acc_and_training_time

        self.elitist_archive_X = []
        self.elitist_archive_hash_X = []
        self.elitist_archive_F = []

        self.dpf = []
        self.no_eval = []

        self.benchmark = benchmark
        self.path = path

        self.pf_true = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))

    def _solve(self, problem, termination):
        self.n_gen = 0
        print('Gen:', self.n_gen)

        """ Initialization """
        self.pop = self._initialize()

        dpf = round(cal_dpf(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
        self.dpf.append(dpf)
        self.no_eval.append(self.problem._n_evaluated)

        self._each_iteration(self, first=True)

        # while termination criteria not fulfilled
        while termination.do_continue(self):
            self.n_gen += 1
            print('Gen:', self.n_gen)

            # do the next iteration
            self.pop = self._next(self.pop)

            dpf = round(cal_dpf(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
            self.dpf.append(dpf)
            self.no_eval.append(self.problem._n_evaluated)

            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()

        return self.pop

    def _initialize(self):
        pop = Population(n_individuals=0, individual=self.individual)
        pop = self.sampling.sample(self.problem, pop, self.pop_size, algorithm=self)

        if self.survival:
            pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        """ Update Elitist Archive """
        # print('--> UPDATE ELITIST ARCHIVE AFTER INITIALIZE POPULATION')
        pop_X = pop.get('X')
        pop_hash_X = pop.get('hash_X')
        pop_F = pop.get('F')
        self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
            update_elitist_archive(pop_X, pop_hash_X, pop_F,
                                   self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F,
                                   first=True)
        # print('--> UPDATE ELITIST ARCHIVE AFTER INITIALIZE POPULATION - DONE')

        return pop

    def _next(self, pop):
        """ Mating """
        self.off = self._mating(pop)

        # merge the offsprings with the current population
        pop = pop.merge(self.off)

        # the do survival selection
        pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        """Local Search on PF"""
        if self.local_search_on_pf == 1:
            pop_F = pop.get("F")

            front = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size)

            pareto_front = pop[front[0]].copy()
            f_pareto_front = pop_F[front[0]].copy()

            new_idx = np.argsort(f_pareto_front[:, 1])

            pareto_front = pareto_front[new_idx]

            pareto_front, non_dominance_X, non_dominance_hash_X, non_dominance_F = \
                self._local_search_on_x_bosman(pop, x=pareto_front)
            pop[front[0]] = pareto_front
            # print('non_dominance_F\n', non_dominance_F)
            """ Update Elitist Archive """
            self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hash_X, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)

        """Local Search on Knee Solutions"""
        if self.local_search_on_knee == 1:
            pop_F = pop.get("F")

            front = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size)

            pareto_front = pop[front[0]].copy()
            f_pareto_front = pop_F[front[0]].copy()

            new_idx = np.argsort(f_pareto_front[:, 1])

            pareto_front = pareto_front[new_idx]
            f_pareto_front = f_pareto_front[new_idx]

            angle = [np.array([360, 0])]
            for i in range(1, len(f_pareto_front) - 1):
                if kiem_tra_p1_nam_phia_duoi_p2_p3(f_pareto_front[i], f_pareto_front[i - 1],
                                                   f_pareto_front[i + 1]):
                    angle.append(
                        np.array([cal_angle(f_pareto_front[i], f_pareto_front[i - 1],
                                            f_pareto_front[i + 1]), i]))
                else:
                    angle.append(np.array([0, i]))
            angle.append(np.array([360, len(pareto_front) - 1]))
            angle = np.array(angle)
            angle = angle[np.argsort(angle[:, 0], )]
            angle = angle[angle[:, 0] > 210]

            idx_knee_solutions = np.array(angle[:, 1], dtype=np.int)
            knee_solutions = pareto_front[idx_knee_solutions].copy()

            knee_solutions_, non_dominance_X, non_dominance_hash_X, non_dominance_F =\
                self._local_search_on_x_bosman(pop, x=knee_solutions)

            """ Update Elitist Archive """
            self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hash_X, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)

            pareto_front[idx_knee_solutions] = knee_solutions_

            # f_knee_solutions_ = knee_solutions_.get("F")
            # # plt.scatter(f_knee_solutions_[:, 1], f_knee_solutions_[:, 0], s=15, c='blue')
            pop[front[0]] = pareto_front
        """----------------------------------------------------------------------------"""
        return pop

    def _mating(self, pop):
        """ CROSSOVER """
        # print("--> CROSSOVER")
        off = self.crossover.do(problem=self.problem, pop=pop, algorithm=self)
        # print("--> CROSSOVER - DONE")

        """ EVALUATE OFFSPRINGS AFTER CROSSOVER (USING FOR UPDATING ELITIST ARCHIVE) """
        # print("--> EVALUATE AFTER CROSSOVER")
        off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
        off.set('F', off_F)
        # print("--> EVALUATE AFTER CROSSOVER - DONE")

        """ UPDATING ELITIST ARCHIVE """
        # print('--> UPDATE ELITIST ARCHIVE AFTER CROSSOVER')
        off_X = off.get('X')
        off_hash_X = off.get('hash_X')
        off_F = off.get('F')
        self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
            update_elitist_archive(off_X, off_hash_X, off_F,
                                   self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)
        # print('--> UPDATE ELITIST ARCHIVE AFTER CROSSOVER - DONE')

        """ MUTATION """
        # print("--> MUTATION")
        off = self.mutation.do(self.problem, off, algorithm=self)
        # print("--> MUTATION - DONE")

        """ EVALUATE OFFSPRINGS AFTER MUTATION (USING FOR UPDATING ELITIST ARCHIVE) """
        # print("--> EVALUATE AFTER MUTATION")
        off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
        off.set('F', off_F)
        # print("--> EVALUATE AFTER MUTATION - DONE")

        """ UPDATING ELITIST ARCHIVE """
        # print('--> UPDATE ELITIST ARCHIVE AFTER MUTATION')
        off_X = off.get('X')
        off_hash_X = off.get('hash_X')
        off_F = off.get('F')
        self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
            update_elitist_archive(off_X, off_hash_X, off_F,
                                   self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)
        # print('--> UPDATE ELITIST ARCHIVE AFTER MUTATION - DONE')

        # _off_hash_X = _off.get('hash_X').tolist()
        #
        # """ Check duplicate in pop """
        # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, pop_hash_X)
        #
        # _off = _off[not_duplicate]
        # _off_hash_X = _off.get('hash_X').tolist()
        #
        # """ Check duplicate in new offsprings """
        # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, _off_hash_X, True)
        # _off = _off[not_duplicate]
        #
        # _off_hash_X = _off.get('hash_X').tolist()
        #
        # """ Check duplicate in current offsprings """
        # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, off.get('hash_X').tolist())
        # _off = _off[not_duplicate]
        #
        # if len(_off) > self.n_offsprings - len(off):
        #     I = random.perm(self.n_offsprings - len(off))
        #     _off = _off[I]
        # if len(_off) != 0:
        #     _off_f = self.evaluator.eval(self.problem, _off.get('X'), check=True, algorithm=self)
        #     _off.set('F', _off_f)
        #     # add to the offsprings and increase the mating counter
        # off = off.merge(_off)

        CV = np.zeros((len(off), 1))
        feasible = np.ones((len(off), 1), dtype=np.bool)

        off.set('CV', CV)
        off.set('feasible', feasible)
        return off

    def _local_search_on_x(self, pop, x):
        off_ = pop.new()
        off_ = off_.merge(x)

        x_old_X = x.get('X')
        x_old_hash_X = x.get('hash_X')
        x_old_F = x.get('F')

        non_dominance_X = []
        non_dominance_hash_X = []
        non_dominance_F = []

        if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
            for i in range(len(x_old_X)):
                checked = []
                stop_iter = 14 * 2
                j = 0
                best_X = x_old_X[i].copy()
                best_hash_X = x_old_hash_X[i].copy()
                best_F = x_old_F[i].copy()
                while j < stop_iter:
                    idx = np.random.randint(0, 14)
                    ops = ['I', '1', '2']
                    ops.remove(x_old_X[i][idx])
                    new_op = np.random.choice(ops)
                    if [idx, new_op] not in checked:
                        checked.append([idx, new_op])

                        x_new_X = x_old_X[i].copy()
                        x_new_X[idx] = new_op

                        x_new_hash_X = ''.join(x_new_X.tolist())

                        if x_new_hash_X not in x_old_hash_X:
                            x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                          check=True, algorithm=self)
                            # result = check_better(x_new_F[0], x_old_F[i])
                            result = check_better(x_new_F[0], best_F)
                            if result == 'obj1':
                                best_X = x_new_X
                                best_hash_X = x_new_hash_X
                                best_F = x_new_F[0]
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(x_new_hash_X)
                                non_dominance_F.append(x_new_F[0])
                            elif result == 'none':
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(x_new_hash_X)
                                non_dominance_F.append(x_new_F[0])
                        j += 1
                x_old_X[i] = best_X
                x_old_hash_X[i] = best_hash_X
                x_old_F[i] = best_F
        elif self.problem.problem_name == 'nas101':
            for i in range(len(x_old_X)):
                checked = []
                stop_iter = 5 * 2
                j = 0
                while j < stop_iter:
                    idx = np.random.randint(1, 6)
                    ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                    ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    if [idx, new_op] not in checked:
                        checked.append([idx, new_op])
                        x_new_X = x_old_X[i].copy()
                        x_new_X[-1][idx] = new_op
                        neighbor = api.ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int), ops=x_new_X[-1].tolist())
                        neighbor_hash = self.benchmark.get_module_hash(neighbor)

                        if neighbor_hash not in x_old_hash_X:
                            neighbor_F = self.evaluator.eval(self.problem, np.array([x_new_X]), check=True,
                                                             algorithm=self)
                            result = check_better(neighbor_F[0], x_old_F[i])
                            if result == 'obj1':
                                x_old_X[i] = x_new_X
                                x_old_hash_X[i] = neighbor_hash
                                x_old_F[i] = neighbor_F
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                                break
                            elif result == 'none':
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                        j += 1

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hash_X = np.array(non_dominance_hash_X)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hash_X', x_old_hash_X)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hash_X, non_dominance_F

    def _local_search_on_x_bosman(self, pop, x):
        off_ = pop.new()
        off_ = off_.merge(x)

        x_old_X = x.get('X')
        x_old_hash_X = x.get('hash_X')
        x_old_F = x.get('F')

        non_dominance_X = []
        non_dominance_hash_X = []
        non_dominance_F = []

        if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
            for i in range(len(x_old_X)):
                checked = [''.join(x_old_X[i].copy().tolist())]
                stop_iter = 20
                j = 0
                alpha = np.random.rand(1)
                while j < stop_iter:
                    idx = np.random.randint(0, 14)

                    ops = ['I', '1', '2']
                    ops.remove(x_old_X[i][idx])
                    new_op = np.random.choice(ops)

                    tmp = x_old_X[i].copy().tolist()
                    tmp[idx] = new_op
                    tmp_str = ''.join(tmp)

                    if tmp_str not in checked:
                        checked.append(tmp_str)

                        x_new_X = x_old_X[i].copy()
                        x_new_X[idx] = new_op

                        x_new_hash_X = ''.join(x_new_X.tolist())

                        if x_new_hash_X not in x_old_hash_X:
                            x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                          check=True, algorithm=self)
                            result = check_better(x_new_F[0], x_old_F[i])
                            if result == 'none':
                                # print("Non-dominated solution")
                                x_old_X[i] = x_new_X
                                x_old_hash_X[i] = x_new_hash_X
                                x_old_F[i] = x_new_F[0]
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(x_new_hash_X)
                                non_dominance_F.append(x_new_F[0])
                            else:
                                result_ = check_better_bosman(alpha=alpha, f_obj1=x_new_F[0], f_obj2=x_old_F[i])
                                if result_ == 'obj1':
                                    # print("Improved solution")
                                    x_old_X[i] = x_new_X
                                    x_old_hash_X[i] = x_new_hash_X
                                    x_old_F[i] = x_new_F[0]
                                    non_dominance_X.append(x_new_X)
                                    non_dominance_hash_X.append(x_new_hash_X)
                                    non_dominance_F.append(x_new_F[0])
                    j += 1
        elif self.problem.problem_name == 'nas101':
            for i in range(len(x_old_X)):
                checked = []
                stop_iter = 5 * 2
                j = 0
                while j < stop_iter:
                    idx = np.random.randint(1, 6)
                    ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                    ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    if [idx, new_op] not in checked:
                        checked.append([idx, new_op])
                        x_new_X = x_old_X[i].copy()
                        x_new_X[-1][idx] = new_op
                        neighbor = api.ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int), ops=x_new_X[-1].tolist())
                        neighbor_hash = self.benchmark.get_module_hash(neighbor)

                        if neighbor_hash not in x_old_hash_X:
                            neighbor_F = self.evaluator.eval(self.problem, np.array([x_new_X]), check=True,
                                                             algorithm=self)
                            result = check_better(neighbor_F[0], x_old_F[i])
                            if result == 'obj1':
                                x_old_X[i] = x_new_X
                                x_old_hash_X[i] = neighbor_hash
                                x_old_F[i] = neighbor_F
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                                break
                            elif result == 'none':
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                        j += 1

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hash_X = np.array(non_dominance_hash_X)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hash_X', x_old_hash_X)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hash_X, non_dominance_F

    def _finalize(self):
        plt.plot(self.no_eval, self.dpf)
        plt.grid()
        plt.savefig(f'{self.path}/dpfs_and_no_evaluations')
        plt.clf()
        # pf = np.array(self.elitist_archive_F)
        # pf = np.unique(pf, axis=0)
        # plt.scatter(pf[:, 1], pf[:, 0], c='blue')
        # plt.savefig('xxxxx')
        # plt.clf()


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")
    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]
        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:
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

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)
    return S[:, None].astype(np.int)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, pop, n_survive, D=None, **kwargs):
        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

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


# =========================================================================================================
# Interface
# =========================================================================================================


def nsganet(
        pop_size=100,
        sampling=MySampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=PointCrossover(n_points=2),
        mutation=Mutation(prob=0.05),
        n_offsprings=None,
        local_search_on_pf=False,
        local_search_on_knee=False,
        opt_val_acc_and_training_time=True,
        **kwargs):
    """

    Parameters
    ----------
    pop_size : {pop_size}
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    n_offsprings : {n_offsprings}

    Returns
    -------
    nsganet : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGANet algorithm object.
        :param sampling:
        :param selection:
        :param crossover:
        :param mutation:
        :param n_offsprings:
        :param local_search_on_knee:
        :param opt_val_acc_and_training_time:
        :type pop_size: object
        :param local_search_on_pf:


    """

    return NSGANet(pop_size=pop_size,
                   sampling=sampling,
                   selection=selection,
                   crossover=crossover,
                   mutation=mutation,
                   survival=RankAndCrowdingSurvival(),
                   n_offsprings=n_offsprings,
                   local_search_on_pf=local_search_on_pf,
                   local_search_on_knee=local_search_on_knee,
                   opt_val_acc_and_training_time=opt_val_acc_and_training_time,
                   **kwargs)


parse_doc_string(nsganet)
