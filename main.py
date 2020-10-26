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

# update your projecty root path before running
sys.path.insert(0, '/path/to/nsga-net')

parser = argparse.ArgumentParser("Multi-Objective Genetic Algorithm for NAS")

# hyper-parameters for problem
parser.add_argument('--benchmark_name', type=str, default='nas101',
                    help='the benchmark is used for optimizing')
parser.add_argument('--n_eval', type=int, default=10000, help='number of max evaluated')

# hyper-parameters for main
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--number_of_runs', type=int, default=1, help='number of runs')
parser.add_argument('--save', type=int, default=0, help='save to log file')

# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--local_search_on_pf', type=int, default=0, help='local search on pareto front')
parser.add_argument('--local_search_on_knee', type=int, default=0, help='local search on knee solutions')
parser.add_argument('--n_points', type=int, default=1, help='local search on n-points')
parser.add_argument('--followed_bosman_paper', type=int, default=0,
                    help='local search followed by bosman paper')

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
        F = np.full((x.shape[0], self.n_obj), np.nan)

        if self.problem_name == 'nas101':
            benchmark_api = kwargs['algorithm'].benchmark_api
            for i in range(x.shape[0]):
                cell = api.ModelSpec(matrix=np.array(x[i][:-1, :], dtype=np.int), ops=x[i][-1, :].tolist())
                module_hash = benchmark_api.get_module_hash(cell)

                F[i, 0] = (self.benchmark_data[module_hash]['params'] - self.min_max['min_model_params']) / (
                        self.min_max['max_model_params'] - self.min_max['min_model_params'])
                F[i, 1] = 1 - self.benchmark_data[module_hash]['val_acc']

                self._n_evaluated += 1

        elif self.problem_name == 'cifar10' or self.problem_name == 'cifar100':
            for i in range(x.shape[0]):
                x_str = ''.join(x[i].tolist())

                F[i, 0] = (self.benchmark_data[x_str]['MMACs'] - self.min_max['min_MMACs']) / (
                        self.min_max['max_MMACs'] - self.min_max['min_MMACs'])
                F[i, 1] = 1 - self.benchmark_data[x_str]['val_acc'] / 100

                self._n_evaluated += 1

        out['F'] = F
        if check:
            return F


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation                                             |
# ---------------------------------------------------------------------------------------------------------


def do_every_generations(algorithm):
    gen = algorithm.n_gen
    print('Gen:', gen)

    if args.save == 1:
        ''' Write down on File Log '''
        logfile.write('-' * 20)
        logfile.write(f' Gen {algorithm.n_gen} ')
        logfile.write('-' * 20)
        logfile.write('\n')
        logfile.write(f'# No.Evaluated: {algorithm.problem._n_evaluated}\n')

        pf = algorithm.elitist_archive_F
        pf = pf[np.argsort(pf[:, 0])]
        logfile.write(f'# Distance from True Pareto Front to Approximate Pareto Front: {algorithm.dpf[-1]}\n\n')

        pickle.dump([pf, algorithm.problem._n_evaluated],
                    open(f'{algorithm.path}/pf_eval/pf_and_evaluated_gen_{gen}.p', 'wb'))

        ''' Plot pareto front / elitist archive '''
        # plt.scatter(algorithm.pf_true[:, 0], algorithm.pf_true[:, 1], facecolors='none', edgecolors='red', s=40,
        #             label='True PF')
        # plt.scatter(pf[:, 0], pf[:, 1], c='blue', s=20, label='Elitist Archive')
        # plt.legend()
        # plt.savefig(f'{algorithm.path}/visualize_pf_each_gen/{gen}')
        # plt.clf()


if __name__ == '__main__':
    print('*** Details of Experiment ***')
    print('- Algorithm: NSGAII')
    print('- Benchmark:', args.benchmark_name)
    print('- Population Size:', args.pop_size)
    print('- Max of No.evaluations:', args.n_eval)
    print('- Local Search on PF:', bool(args.local_search_on_pf))
    print('- Local Search on Knee Solutions:', bool(args.local_search_on_knee))
    if bool(args.local_search_on_pf) or bool(args.local_search_on_pf):
        print('- Local Search on n-points:', args.n_points)
    print('- Local Search followed by Bosman ver:', bool(args.followed_bosman_paper))
    print('*' * 40)
    print()

    # Syntax: 'benchmark_name'_'pop_size'_'local_search_on_pf'_'local_search_on_knee'_'followed_bosman_paper'_'n_points'
    now = datetime.now()
    dir_name = now.strftime(f'{args.benchmark_name}_{args.pop_size}_'
                            f'{bool(args.local_search_on_pf)}_{bool(args.local_search_on_knee)}_'
                            f'{bool(args.followed_bosman_paper)}_{args.n_points}point_%d_%m_%H_%M')
    path = dir_name

    if args.save == 1:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    problem = NAS(problem_name=args.benchmark_name)

    for run_i in range(args.number_of_runs):
        seed = args.seed + run_i * 100
        np.random.seed(seed)
        print(f'---------------------- LAN {run_i} ----------------------')
        if args.save == 1:
            # Name of sub-folder
            path_ = path + f'/{run_i}'

            # Create sub-folder
            try:
                os.mkdir(path_)
            except FileExistsError:
                pass

            # Open file log
            logfile = open(path_ + '/log_fife.txt', 'w')

            # Create sub-sub-folder (pf_eval)
            try:
                os.mkdir(path_ + '/pf_eval')
            except FileExistsError:
                pass

            # Create sub-sub-folder (visualize_pf_each_gen)
            try:
                os.mkdir(path_ + '/visualize_pf_each_gen')
            except FileExistsError:
                pass

            # ----------------------------------------------------------------------------------------------------------
            # Write down on log file
            # ----------------------------------------------------------------------------------------------------------

            if args.benchmark_name == 'nas101':
                logfile.write(f'*************************************** NSGA-II ON NASBENCH-101 '
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
                logfile.write(f'# Objectives Optimize: [Model Params (normalize); 1 - Validation Accuracy]\n\n')
            elif args.benchmark_name == 'cifar10' or args.benchmark_name == 'cifar100':
                logfile.write(f'# Objectives Optimize: [MMACs (normalize); 1 - Validation Accuracy]\n\n')
            logfile.write('*' * 50)
            logfile.write('\n\n')

            logfile.write('--------------------------------------- Hyper-Parameters of Algorithm '
                          '---------------------------------------\n\n')
            logfile.write(f'# Random seed: {seed}\n')
            logfile.write(f'# Population Size: {args.pop_size}\n')
            logfile.write(f'# Number of Max Evaluate: {args.n_eval}\n')
            logfile.write(f'# Local Search on PF: {bool(args.local_search_on_pf)}\n')
            logfile.write(f'# Local Search on Knee Solutions: {bool(args.local_search_on_knee)}\n')
            if bool(args.local_search_on_pf) or bool(args.local_search_on_pf):
                logfile.write(f'# Local Search on n-points: {args.n_points}\n')
            logfile.write(f'# Local Search followed by Bosman ver: {bool(args.followed_bosman_paper)}\n\n')
            logfile.write('*' * 50)
            logfile.write('\n\n')
        else:
            path_ = path
        # ==============================================================================================================

        # configure the nsga-net method
        method = engine.nsganet(pop_size=args.pop_size,
                                benchmark=args.benchmark_name,
                                local_search_on_pf=args.local_search_on_pf,
                                local_search_on_knee=args.local_search_on_knee,
                                followed_bosman_paper=args.followed_bosman_paper,
                                n_points=args.n_points,
                                path=path_)

        res = minimize(problem,
                       method,
                       callback=do_every_generations,
                       termination=('n_eval', args.n_eval))
        problem._n_evaluated = 0

    if args.save == 1:
        print('All files are saved on ' + path)
