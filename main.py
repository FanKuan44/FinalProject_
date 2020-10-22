import sys
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

import numpy as np
from search import nsganet as engine

from wrap_pymoo.problem import MyProblem
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.optimize import minimize

from nasbench import wrap_api as api

# NASBENCH_TFRECORD = 'nasbench/nasbench_only108.tfrecord'
# benchmark_api = api.NASBench_(NASBENCH_TFRECORD)

# update your projecty root path before running
sys.path.insert(0, '/path/to/nsga-net')

parser = argparse.ArgumentParser("Multi-Objective Genetic Algorithm for NAS")

# hyper-parameters for problem
parser.add_argument('--benchmark_name', type=str, default='nas101',
                    help='the benchmark is used for optimizing')
parser.add_argument('--n_eval', type=int, default=10000, help='number of max evaluated')

parser.add_argument('--opt_val_acc_and_training_time', type=int, default=1,
                    help='optimize validation accuracy and training time')

# hyper-parameters for main
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--number_of_runs', type=int, default=1, help='number of runs')
parser.add_argument('--save', type=int, default=0, help='save to log file')

# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--local_search_on_pf', type=int, default=0, help='local search on pareto front')
parser.add_argument('--local_search_on_knee', type=int, default=0, help='local search on knee solutions')

args = parser.parse_args()


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(MyProblem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_obj=2, problem_name='nas101', save_dir=None):
        super().__init__(n_obj=n_obj)

        self._save_dir = save_dir
        self._n_evaluated = 0

        self.problem_name = problem_name
        if self.problem_name == 'nas101':
            self.min_max = pickle.load(open('101_benchmark/min_max_NAS101.p', 'rb'))
            self.benchmark_data = pickle.load(open('101_benchmark/nas101.p', 'rb'))
        elif self.problem_name == 'cifar10':
            self.min_max = pickle.load(open('bosman_benchmark/cifar10/min_max_cifar10.p', 'rb'))
            self.benchmark_data = pickle.load(open('bosman_benchmark/cifar10/cifar10.p', 'rb'))
        elif self.problem_name == 'cifar100':
            self.min_max = pickle.load(open('bosman_benchmark/cifar100/min_max_cifar100.p', 'rb'))
            self.benchmark_data = pickle.load(open('bosman_benchmark/cifar100/cifar100.p', 'rb'))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LOAD BENCHMARK DONE! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def _evaluate(self, x, out, check=False, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)

        if self.problem_name == 'nas101':
            benchmark = kwargs['algorithm'].benchmark
            for i in range(x.shape[0]):
                cell = api.ModelSpec(matrix=np.array(x[i][:-1, :], dtype=np.int), ops=x[i][-1, :].tolist())
                # data = benchmark.query(cell)
                module_hash = benchmark.get_module_hash(cell)

                f[i, 0] = 1 - self.benchmark_data[module_hash]['val_acc']
                f[i, 1] = (self.benchmark_data[module_hash]['training_time'] - self.min_max['min_training_time']) / (
                        self.min_max['max_training_time'] - self.min_max['min_training_time'])

                self._n_evaluated += 1

        elif self.problem_name == 'cifar10' or self.problem_name == 'cifar100':
            for i in range(x.shape[0]):
                x_str = ''.join(x[i].tolist())
                f[i, 0] = 1 - self.benchmark_data[x_str]['val_acc']/100
                f[i, 1] = (self.benchmark_data[x_str]['MMACs'] - self.min_max['min_MMACs']) / (
                        self.min_max['max_MMACs'] - self.min_max['min_MMACs'])

                self._n_evaluated += 1
        out["F"] = f
        if check:
            return f


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation                                             |
# ---------------------------------------------------------------------------------------------------------


def cal_euclid_distance(x1, x2):
    e_dis = np.sqrt(np.sum((x1 - x2) ** 2))
    return e_dis


def cal_dpf(pareto_front, pareto_s):
    d = 0
    for solution in pareto_front:
        d_ = np.inf
        for solution_ in pareto_s:
            d_ = min(cal_euclid_distance(solution, solution_), d_)
        d += d_
    return d / len(pareto_front)


def do_every_generations(algorithm):
    gen = algorithm.n_gen
    # print("Gen:", gen)

    if args.save == 1:
        """ Write down on File Log """
        logfile.write('-' * 20)
        logfile.write(f' Gen {algorithm.n_gen} ')
        logfile.write('-' * 20)
        logfile.write('\n')
        logfile.write(f'# No.Evaluated: {algorithm.problem._n_evaluated}\n')

        # pf = algorithm.elitist_archive_F
        # logfile.write(f'# Number of Elitist Archive: {len(pf)}\n')
        # pf = np.unique(pf, axis=0)
        # pf = pf[np.argsort(pf[:, 1])]
        logfile.write(f'# Distance from True Pareto Front to Approximate Pareto Front: {algorithm.dpf[-1]}\n\n')

        # pickle.dump([pf, algorithm.problem._n_evaluated],
        #             open(f'{algorithm.path}/pf_eval/pf_and_evaluated_gen_{gen}.p', 'wb'))

        """ Plot pareto front / elitist archive"""
        # pop = algorithm.pop
        # pop_f = pop.get("F")
        # front = NonDominatedSorting().do(pop_f, n_stop_if_ranked=algorithm.pop_size)
        # pf_ = pop_f[front[0]]
        #
        # plt.scatter(pf_[:, 1], pf_[:, 0], c='blue', s=10, label=f'front0')

        # plt.scatter(pf[:, 1], pf[:, 0], s=20, edgecolors='red', facecolors='none', label=f'elitist archive {dpf}')
        # plt.title(f'Gen {gen}')
        # plt.legend()
        # plt.savefig(f'{algorithm.path}/visualize_pf_each_gen/{gen}')
        # plt.clf()


if __name__ == '__main__':
    now = datetime.now()
    dir_name = now.strftime(f"{args.benchmark_name}_popsize_{args.pop_size}_"
                            f"{bool(args.local_search_on_pf)}_{bool(args.local_search_on_knee)}_%d_%m_%Y_%H_%M_%S")
    path = dir_name

    if args.save == 1:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    problem = NAS(problem_name=args.benchmark_name)

    for run_i in range(args.number_of_runs):
        np.random.seed(args.seed + run_i * 100)
        print(f"---------------------- LAN {run_i} ----------------------")
        if args.save == 1:
            # Name of sub-folder
            path_ = path + f'/{run_i}/'

            # Create sub-folder
            try:
                os.mkdir(path_)
            except FileExistsError:
                pass

            # Open file log
            logfile = open(path_ + 'log_fife.txt', 'w')

            # Create sub-sub-folder (pf_eval)
            try:
                os.mkdir(path_ + 'pf_eval')
            except FileExistsError:
                pass

            # Create sub-sub-folder (visualize_pf_each_gen)
            try:
                os.mkdir(path_ + 'visualize_pf_each_gen')
            except FileExistsError:
                pass

            # -------------------------------------------------------------------------------------------------------------
            # Write down on log file
            # -------------------------------------------------------------------------------------------------------------

            if args.benchmark_name == 'nas101':
                logfile.write(f'*************************************** NSGA-II ON NAS-BENCHMARK-101 '
                              f'***************************************\n\n')
            elif args.benchmark_name == 'cifar10':
                logfile.write(f'*************************************** NSGA-II ON BOSMAN-BENCHMARK-CIFAR10 '
                              f'***************************************\n\n')
            elif args.benchmark_name == 'cifar100':
                logfile.write(f'*************************************** NSGA-II ON BOSMAN-BENCHMARK-CIFAR100 '
                              f'***************************************\n\n')

            logfile.write('--------------------------------------- Hyper-Parameters of Problem '
                          '---------------------------------------\n\n')
            logfile.write(f'# Benchmark Dataset: {args.benchmark_name}\n')
            if args.benchmark_name == 'nas101':
                if args.opt_val_acc_and_training_time == 1:
                    logfile.write(f'# Objectives Optimize: [1 - Validation Accuracy; Training Time (normalize)]\n\n')
                else:
                    logfile.write(f'# Objectives Optimize: [1 - Validation Accuracy; Model Parameters]\n\n')
            elif args.benchmark_name == 'cifar10' or args.benchmark_name == 'cifar100':
                logfile.write(f'# Objectives Optimize: [1 - Validation Accuracy; MMACs (normalize)]\n\n')
            logfile.write('*' * 50)
            logfile.write('\n\n')

            logfile.write('--------------------------------------- Hyper-Parameters of Algorithm '
                          '---------------------------------------\n\n')
            logfile.write(f'# Random seed: {args.seed + run_i * 100}\n')
            logfile.write(f'# Population Size: {args.pop_size}\n')
            logfile.write(f'# Number of Max Evaluate: {args.n_eval}\n')
            logfile.write(f'# Local Search on PF: {bool(args.local_search_on_pf)}\n')
            logfile.write(f'# Local Search on Knee Solutions: {bool(args.local_search_on_knee)}\n\n')
            logfile.write('*' * 50)
            logfile.write('\n\n')
        else:
            path_ = path
        # ==============================================================================================================

        # configure the nsga-net method
        method = engine.nsganet(pop_size=args.pop_size,
                                n_offsprings=args.pop_size,
                                local_search_on_pf=args.local_search_on_pf,
                                local_search_on_knee=args.local_search_on_knee,
                                path=path_)

        res = minimize(problem,
                       method,
                       callback=do_every_generations,
                       termination=('n_eval', args.n_eval))
        problem._n_evaluated = 0
    if args.save == 1:
        print('All files are saved on ' + path)
