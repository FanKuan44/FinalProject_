import numpy as np
import matplotlib.pyplot as plt
import pickle

from nasbench.lib import model_spec

from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival

from pymoo.operators.selection.tournament_selection import compare, TournamentSelection

from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import Dominator
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

from wrap_pymoo.algorithms.genetic_algorithm import GeneticAlgorithm

from wrap_pymoo.operators.crossover.point_crossover import MyPointCrossover as PointCrossover
from wrap_pymoo.operators.mutation.mutation import MyMutation as Mutation

from wrap_pymoo.model.individual import MyIndividual as Individual
from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.model.sampling import MySampling as Sampling

from wrap_pymoo.util.compare import find_better_idv, find_better_idv_bosman_ver
from wrap_pymoo.util.dpfs_calculating import cal_dpfs
from wrap_pymoo.util.elitist_archive import update_elitist_archive
from wrap_pymoo.util.find_knee_solutions import kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3, cal_angle
# =========================================================================================================
# Implementation based on nsga2 from https://github.com/msu-coinlab/pymoo
# =========================================================================================================
from nasbench import wrap_api as api

ModelSpec = model_spec.ModelSpec


# def find_better_idv(f1, f2, position=None):
#     """
#     Kiem tra xem ca the nao tot hon.
#     Neu ca the do nam o vi tri dau hoac cuoi thi chi xet 1 trong 2 fitness value, con lai thi xet binh thuong.
#     =========================================================================================================
#
#     Parameters:
#     ----------
#     :param f1: fitness value of first individual
#     :param f2: fitness value of second individual
#     :param position: position of individual in pareto front or set of knee solutions (first, last, none)
#     =========================================================================================================
#
#     Returns:
#     ------
#     :return:
#     1: individual 1
#     2: individual 2
#     0: non-dominated
#     """
#     if position is None:
#         if (f1[0] <= f2[0] and f1[1] < f2[1]) or (f1[0] < f2[0] and f1[1] <= f2[1]):
#             return 1
#         if (f2[0] <= f1[0] and f2[1] < f1[1]) or (f2[0] < f1[0] and f2[1] <= f1[1]):
#             return 2
#     elif position == 'first':
#         if f1[0] < f2[0]:
#             return 1
#         if f2[0] <= f1[0]:
#             return 2
#     else:
#         if f1[1] < f2[1]:
#             return 1
#         if f2[1] <= f1[1]:
#             return 2
#     return 0
#
#
# def find_better_idv_bosman_ver(alpha, f1, f2):
#     """
#     Kiem tra xem ca the nao tot hon theo paper cua Bosman.
#     Ngau nhien 1 gia tri alpha, tinh lai fitness value cua tung individual theo alpha va so sanh.
#     =========================================================================================================
#
#     Parameters:
#     -----------
#     :param alpha: value of alpha
#     :param f1: the old fitness value of first individual
#     :param f2: the old fitness value of first individual
#     =========================================================================================================
#
#     Returns:
#     :return:
#     1: individual 1
#     2: individual 2
#     """
#     f1_new = alpha * f1[0] + (1 - alpha) * f1[1]
#     f2_new = alpha * f2[0] + (1 - alpha) * f2[1]
#     if f1_new < f2_new:
#         return 1
#     return 2


# def kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3(p1, p2, p3):
#     v_cp = p3 - p2
#     dt = -v_cp[1] * (p1[0] - p2[0]) + v_cp[0] * (p1[1] - p2[1])
#     if dt > 0:
#         return 'tren'
#     return 'duoi'


# def cal_angle(p_middle, p_top, p_bot):
#     x1 = p_top - p_middle
#     x2 = p_bot - p_middle
#     cosine_angle = (x1[0] * x2[0] + x1[1] * x2[1]) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)))
#
#     angle = np.arccos(cosine_angle)
#     return 360 - np.degrees(angle)


# def update_elitist_archive(new_idv_X_lst, new_idv_hashX_lst, new_idv_F_lst,
#                            elitist_archive_X, elitist_archive_hashX, elitist_archive_F, first=False):
#     if first:
#         current_elitist_archive_X = elitist_archive_X.copy()
#         current_elitist_archive_hashX = elitist_archive_hashX.copy()
#         current_elitist_archive_F = elitist_archive_F.copy()
#     else:
#         current_elitist_archive_X = elitist_archive_X.copy().tolist()
#         current_elitist_archive_hashX = elitist_archive_hashX.copy().tolist()
#         current_elitist_archive_F = elitist_archive_F.copy().tolist()
#
#     rank = np.zeros(len(current_elitist_archive_X))
#
#     for i in range(len(new_idv_X_lst)):  # Duyet cac phan tu trong list can check
#         if new_idv_hashX_lst[i] not in current_elitist_archive_hashX:
#             flag = True  # Check xem co bi dominate khong?
#             for j in range(len(current_elitist_archive_X)):  # Duyet cac phan tu trong elitist archive hien tai
#                 better_idv = find_better_idv(new_idv_F_lst[i],
#                                              current_elitist_archive_F[j])  # Kiem tra xem tot hon hay khong?
#                 if better_idv == 1:
#                     rank[j] += 1
#                 elif better_idv == 2:
#                     flag = False
#                     break
#             if flag:
#                 current_elitist_archive_X.append(np.array(new_idv_X_lst[i]))
#                 current_elitist_archive_hashX.append(np.array(new_idv_hashX_lst[i]))
#                 current_elitist_archive_F.append(np.array(new_idv_F_lst[i]))
#                 rank = np.append(rank, 0)
#
#     current_elitist_archive_X = np.array(current_elitist_archive_X)[rank == 0]
#     current_elitist_archive_hashX = np.array(current_elitist_archive_hashX)[rank == 0]
#     current_elitist_archive_F = np.array(current_elitist_archive_F)[rank == 0]
#
#     return current_elitist_archive_X, \
#            current_elitist_archive_hashX, \
#            current_elitist_archive_F


class NSGANet(GeneticAlgorithm):
    def __init__(self, benchmark, local_search_on_pf, local_search_on_knee, followed_bosman_paper, path, n_points=1,
                 **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective

        self.local_search_on_pf = local_search_on_pf
        self.local_search_on_knee = local_search_on_knee
        self.n_points = n_points
        self.followed_bosman_paper = followed_bosman_paper

        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = [], [], []

        self.dpf = []
        self.no_eval = []

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

    # DONE
    def _initialize(self):
        pop = Population(n_individuals=0, individual=self.individual)
        pop = self.sampling.sample(problem=self.problem, pop=pop, n_samples=self.pop_size, algorithm=self)

        if self.survival:
            pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        ''' UPDATE ELITIST ARCHIVE AFTER INITIALIZE POPULATION '''
        pop_X, pop_hashX, pop_F = pop.get('X'), pop.get('hashX'), pop.get('F')

        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(pop_X, pop_hashX, pop_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F,
                                   first=True)
        return pop

    # DONE
    def local_search_on_X(self, pop, X, n_points=2, ls_on_knee_solutions=False):
        off_ = pop.new()
        off_ = off_.merge(X)

        x_old_X, x_old_hashX, x_old_F = off_.get('X'), off_.get('hashX'), off_.get('F')

        non_dominance_X, non_dominance_hashX, non_dominance_F = [], [], []

        # Using for local search on knee solutions
        first, last = 0, 0
        if ls_on_knee_solutions:
            first, last = len(x_old_X) - 2, len(x_old_X) - 1

        if n_points == 1:
            stop_iter = 30

            for i in range(len(x_old_X)):
                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
                        idx = np.random.randint(0, 14)
                        ops = ['I', '1', '2']
                        ops.remove(x_old_X[i][idx])
                    else:
                        idx = np.random.randint(1, 6)
                        ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    x_new_X = x_old_X[i].copy()
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
                        x_new_X[idx] = new_op

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx] = new_op

                        module = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                           ops=x_new_X[-1].tolist())
                        x_new_hashX = self.benchmark_api.get_module_hash(module)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        checked.append(x_new_hashX)
                        j += 1
                        x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                      check=True, algorithm=self)

                        if i == first and ls_on_knee_solutions:
                            better_idv = find_better_idv(x_new_F[0], x_old_F[i], 'first')
                        elif i == last and ls_on_knee_solutions:
                            better_idv = find_better_idv(x_new_F[0], x_old_F[i], 'last')
                        else:
                            better_idv = find_better_idv(x_new_F[0], x_old_F[i])

                        if better_idv == 1:
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F[0]

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F[0])

                        elif better_idv == 0:
                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F[0])

        elif n_points == 2:
            stop_iter = 30

            for i in range(len(x_old_X)):

                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
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
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
                        x_new_X[idx[0]], x_new_X[idx[1]] = new_op1, new_op2

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx[0]], x_new_X[-1][idx[1]] = new_op1, new_op2

                        module = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                           ops=x_new_X[-1].tolist())
                        x_new_hashX = self.benchmark_api.get_module_hash(module)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        j += 1
                        checked.append(x_new_hashX)

                        x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                      check=True, algorithm=self)
                        if i == first and ls_on_knee_solutions:
                            better_idv = find_better_idv(x_new_F[0], x_old_F[i], 'first')
                        elif i == last and ls_on_knee_solutions:
                            better_idv = find_better_idv(x_new_F[0], x_old_F[i], 'last')
                        else:
                            better_idv = find_better_idv(x_new_F[0], x_old_F[i])

                        if better_idv == 1:
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F[0]

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F[0])

                        elif better_idv == 0:
                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F[0])

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hashX = np.array(non_dominance_hashX)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hashX', x_old_hashX)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hashX, non_dominance_F

    # DONE
    def local_search_on_X_bosman(self, pop, X, n_points=2):
        off_ = pop.new()
        off_ = off_.merge(X)

        x_old_X, x_old_hashX, x_old_F = off_.get('X'), off_.get('hashX'), off_.get('F')

        non_dominance_X, non_dominance_hashX, non_dominance_F = [], [], []

        if n_points == 1:
            stop_iter = 30

            for i in range(len(x_old_X)):
                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0
                alpha = np.random.rand()

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
                        idx = np.random.randint(0, 14)
                        ops = ['I', '1', '2']
                        ops.remove(x_old_X[i][idx])
                    else:
                        idx = np.random.randint(1, 6)
                        ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                        ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    x_new_X = x_old_X[i].copy()
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
                        x_new_X[idx] = new_op

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx] = new_op

                        module = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                           ops=x_new_X[-1].tolist())
                        x_new_hashX = self.benchmark_api.get_module_hash(module)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        checked.append(x_new_hashX)

                        x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                      check=True, algorithm=self)
                        better_idv = find_better_idv(f1=x_new_F[0], f2=x_old_F[i])

                        if better_idv == 0:  # Non-dominated solution
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F[0]

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F[0])

                        else:
                            better_idv_ = find_better_idv_bosman_ver(alpha=alpha, f1=x_new_F[0], f2=x_old_F[i])
                            if better_idv_ == 1:  # Improved solution
                                x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F[0]

                                non_dominance_X.append(x_new_X)
                                non_dominance_hashX.append(x_new_hashX)
                                non_dominance_F.append(x_new_F[0])
                        j += 1

        elif n_points == 2:
            stop_iter = 30

            for i in range(len(x_old_X)):
                max_n_searching = 100
                n_searching = 0
                checked = [x_old_hashX[i]]
                j = 0
                alpha = np.random.rand()

                while (j < stop_iter) and (n_searching < max_n_searching):
                    n_searching += 1
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
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
                    if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
                        x_new_X[idx[0]], x_new_X[idx[1]] = new_op1, new_op2

                        x_new_hashX = ''.join(x_new_X.tolist())
                    else:
                        x_new_X[-1][idx[0]], x_new_X[-1][idx[1]] = new_op1, new_op2

                        module = ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int),
                                           ops=x_new_X[-1].tolist())
                        x_new_hashX = self.benchmark_api.get_module_hash(module)

                    if (x_new_hashX not in checked) and (x_new_hashX not in x_old_hashX):
                        checked.append(x_new_hashX)

                        x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                      check=True, algorithm=self)

                        better_idv = find_better_idv(f1=x_new_F[0], f2=x_old_F[i])

                        if better_idv == 0:  # Non-dominated solution
                            x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F[0]

                            non_dominance_X.append(x_new_X)
                            non_dominance_hashX.append(x_new_hashX)
                            non_dominance_F.append(x_new_F[0])

                        else:
                            better_idv_ = find_better_idv_bosman_ver(alpha=alpha, f1=x_new_F[0], f2=x_old_F[i])
                            if better_idv_ == 1:  # Improved solution
                                x_old_X[i], x_old_hashX[i], x_old_F[i] = x_new_X, x_new_hashX, x_new_F[0]

                                non_dominance_X.append(x_new_X)
                                non_dominance_hashX.append(x_new_hashX)
                                non_dominance_F.append(x_new_F[0])
                        j += 1

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hashX = np.array(non_dominance_hashX)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hashX', x_old_hashX)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hashX, non_dominance_F

    # DONE
    def _next(self, pop):
        self.off = self._mating(pop)

        # merge the offsprings with the current population
        pop = pop.merge(self.off)

        # the do survival selection
        pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        ''' Local Search on PF '''
        if self.local_search_on_pf == 1:
            pop_F = pop.get('F')

            front_0 = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size, only_non_dominated_front=True)

            pareto_front = pop[front_0].copy()

            if self.followed_bosman_paper == 1:
                pareto_front, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X_bosman(pop, X=pareto_front, n_points=self.n_points)
            else:
                pareto_front, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X(pop, X=pareto_front, n_points=self.n_points)
            pop[front_0] = pareto_front

            ''' UPDATE ELITIST ARCHIVE AFTER LOCAL SEARCH '''
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

        ''' Local Search on Knee Solutions '''
        if self.local_search_on_knee == 1:
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

            if self.followed_bosman_paper == 1:
                knee_solutions, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X_bosman(pop, X=knee_solutions, n_points=self.n_points)
            else:
                knee_solutions, non_dominance_X, non_dominance_hashX, non_dominance_F = \
                    self.local_search_on_X(pop, X=knee_solutions, n_points=self.n_points, ls_on_knee_solutions=True)

            ''' UPDATE ELITIST ARCHIVE AFTER LOCAL SEARCH '''
            self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hashX, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

            pareto_front[idx_knee_solutions] = knee_solutions

            pop[front_0] = pareto_front

        return pop

    # DONE
    def _mating(self, pop):
        # CROSSOVER
        off = self.crossover.do(problem=self.problem, pop=pop, algorithm=self)
        # print('crossover-done')

        # EVALUATE OFFSPRINGS AFTER CROSSOVER
        off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
        off.set('F', off_F)

        # UPDATE ELITIST ARCHIVE AFTER CROSSOVER
        off_X = off.get('X')
        off_hashX = off.get('hashX')
        off_F = off.get('F')
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(off_X, off_hashX, off_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

        # MUTATION
        off = self.mutation.do(self.problem, off, algorithm=self)
        # print('mutation-done')

        # EVALUATE OFFSPRINGS AFTER MUTATION
        off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
        off.set('F', off_F)

        # UPDATE ELITIST ARCHIVE AFTER MUTATION
        off_X = off.get('X')
        off_hashX = off.get('hashX')
        off_F = off.get('F')
        self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F = \
            update_elitist_archive(off_X, off_hashX, off_F,
                                   self.elitist_archive_X, self.elitist_archive_hashX, self.elitist_archive_F)

        CV = np.zeros((len(off), 1))
        feasible = np.ones((len(off), 1), dtype=np.bool)

        off.set('CV', CV)
        off.set('feasible', feasible)
        return off

    # DONE
    def _solve(self, problem, termination):
        self.n_gen = 0

        ''' INITIALIZE '''
        self.pop = self._initialize()

        ''' CALCULATE DPFS EACH GEN - USING FOR VISUALIZE'''
        dpfs = round(cal_dpfs(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
        self.dpf.append(dpfs)
        self.no_eval.append(self.problem._n_evaluated)

        self._each_iteration(self, first=True)

        # while termination criteria not fulfilled
        while termination.do_continue(self):
            self.n_gen += 1

            self.pop = self._next(self.pop)

            ''' CALCULATE DPFS EACH GEN - USING FOR VISUALIZE'''
            dpf = round(cal_dpfs(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
            self.dpf.append(dpf)
            self.no_eval.append(self.problem._n_evaluated)

            self._each_iteration(self)

        self._finalize()

        return self.pop

    # DONE
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

        pickle.dump([self.no_eval, self.dpf], open(f'{self.path}/no_eval_and_dpfs.p', 'wb'))
        plt.plot(self.no_eval, self.dpf)
        plt.xlabel('No.Evaluations')
        plt.ylabel('DPFS')
        plt.grid()
        plt.savefig(f'{self.path}/dpfs_and_no_evaluations')
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
        sampling=Sampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=PointCrossover(type_crossover='2X'),
        mutation=Mutation(prob=0.05),
        local_search_on_pf=False,
        local_search_on_knee=False,
        followed_bosman_version=False,
        n_points=1,
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
        :param n_points:
        :param followed_bosman_version:
        :param sampling:
        :param selection:
        :param crossover:
        :param mutation:
        :param local_search_on_knee:
        :param local_search_on_pf:
        :type pop_size: object

    """

    return NSGANet(pop_size=pop_size,
                   sampling=sampling,
                   selection=selection,
                   crossover=crossover,
                   mutation=mutation,
                   survival=RankAndCrowdingSurvival(),
                   local_search_on_pf=local_search_on_pf,
                   local_search_on_knee=local_search_on_knee,
                   followed_bosman_version=followed_bosman_version,
                   n_points=n_points,
                   **kwargs)


parse_doc_string(nsganet)
