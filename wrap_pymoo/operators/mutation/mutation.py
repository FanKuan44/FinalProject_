import numpy as np
import copy

from pymoo.rand import random
from nasbench import api


from abc import abstractmethod


class Mutation:
    def __init__(self, eta=3, prob=0.05):
        self.eta = float(eta)
        self.prob = prob
        self.algorithm = None
        self.problem = None

    def do(self, problem, pop, **kwargs):
        """

        Mutate variables in a genetic way.

        Parameters
        ----------
        problem : class
            The problem instance - specific information such as variable bounds might be needed.
        pop : Population
            A population object

        Returns
        -------
        Y : Population
            The mutated population.

        """

        return self._do(problem, pop, **kwargs)

    @abstractmethod
    def _do(self, problem, pop, **kwargs):
        pop_X = pop.get('X')
        pop_hash_X = pop.get('hash_X')

        off_X = []
        off_hash_X = []
        while len(off_X) < len(pop_X):
            if problem.problem_name == 'nas101':
                benchmark = kwargs['algorithm'].benchmark
                for x in pop_X:
                    while True:
                        new_matrix = copy.deepcopy(np.array(x[:-1, :], dtype=np.int))
                        new_ops = copy.deepcopy(x[-1, :])
                        # In expectation, V edges flipped (note that most end up being pruned).
                        edge_mutation_prob = 1 / 7
                        for src in range(0, 7 - 1):
                            for dst in range(src + 1, 7):
                                if random.random() < edge_mutation_prob:
                                    new_matrix[src, dst] = 1 - new_matrix[src, dst]

                        # In expectation, one op is resampled.
                        op_mutation_prob = 1 / 5
                        for ind in range(1, 7 - 1):
                            if random.random() < op_mutation_prob:
                                available = [o for o in benchmark.config['available_ops'] if o != new_ops[ind]]
                                new_ops[ind] = random.choice(available)
                        new_spec = api.ModelSpec(new_matrix, new_ops.tolist())
                        if benchmark.is_valid(new_spec):
                            pop_new_X.append(np.concatenate((new_matrix, np.array([new_ops])), axis=0))
                            pop_new_hash_X.append(benchmark.get_module_hash(new_spec))
                            break
            elif problem.problem_name == 'cifar10' or problem.problem_name == 'cifar100':
                # Co the xay ra truong hop dot bien nhieu vi tri
                idx_mutation = np.random.rand(pop_X.shape[0], pop_X.shape[1])

                for i in range(pop_X.shape[0]):
                    off_old_X = pop_X[i].copy()

                    for j in range(pop_X.shape[1]):
                        if idx_mutation[i][j] <= self.prob:
                            choices = ['I', '1', '2']
                            choices.remove(off_old_X[j])
                            off_old_X[j] = np.random.choice(choices)

                    off_new_hash_X = ''.join(off_old_X)

                    if off_new_hash_X not in off_hash_X and off_new_hash_X not in pop_hash_X:
                        off_new_X = off_old_X.copy()
                        off_X.append(off_new_X)
                        off_hash_X.append(off_new_hash_X)

        off_X = np.array(off_X)[:pop_X.shape[0]]
        off_hash_X = np.array(off_hash_X)[:pop_X.shape[0]]

        # USING FOR CHECKING DUPLICATE
        if np.sum(np.unique(off_hash_X, return_counts=True)[-1]) != pop_X.shape[0]:
            print('Duplicate')

        for hash_X in off_hash_X:
            if hash_X in pop_hash_X:
                print('Duplicate', hash_X)
                break
        # -----------------------------------

        offspring = pop.new(pop_X.shape[0])
        offspring.set('X', off_X)
        offspring.set('hash_X', off_hash_X)
        return offspring


# class MyMutation(Mutation):
#     def __init__(self, eta, prob=None):
#         super().__init__()
#         self.eta = float(eta)
#         if prob is not None:
#             self.prob = float(prob)
#         else:
#             self.prob = None
#
#     def _do(self, problem, pop, **kwargs):
#         pop_X = pop.get("X")
#         pop_new_X = []
#         pop_new_hash_X = []
#         if problem.problem_name == 'nas101':
#             benchmark = kwargs['algorithm'].benchmark
#             for x in pop_X:
#                 while True:
#                     new_matrix = copy.deepcopy(np.array(x[:-1, :], dtype=np.int))
#                     new_ops = copy.deepcopy(x[-1, :])
#                     # In expectation, V edges flipped (note that most end up being pruned).
#                     edge_mutation_prob = 1 / 7
#                     for src in range(0, 7 - 1):
#                         for dst in range(src + 1, 7):
#                             if random.random() < edge_mutation_prob:
#                                 new_matrix[src, dst] = 1 - new_matrix[src, dst]
#
#                     # In expectation, one op is resampled.
#                     op_mutation_prob = 1 / 5
#                     for ind in range(1, 7 - 1):
#                         if random.random() < op_mutation_prob:
#                             available = [o for o in benchmark.config['available_ops'] if o != new_ops[ind]]
#                             new_ops[ind] = random.choice(available)
#                     new_spec = api.ModelSpec(new_matrix, new_ops.tolist())
#                     if benchmark.is_valid(new_spec):
#                         pop_new_X.append(np.concatenate((new_matrix, np.array([new_ops])), axis=0))
#                         pop_new_hash_X.append(benchmark.get_module_hash(new_spec))
#                         break
#         elif problem.problem_name == 'cifar10' or problem.problem_name == 'cifar100':
#             idx_mutation = np.random.rand(pop_X.shape[0], pop_X.shape[1])
#             for i in range(len(pop_X)):
#                 new_X = pop_X[i].copy()
#                 for j in range(len(new_X)):
#                     if idx_mutation[i][j] <= 0.1:
#                         choices = ['I', '1', '2']
#                         choices.remove(new_X[j])
#                         new_X[j] = np.random.choice(choices)
#                 pop_new_X.append(new_X)
#                 pop_new_hash_X.append(''.join(new_X.tolist()))
#
#         pop_new_X = np.array(pop_new_X)
#         pop_new_hash_X = np.array(pop_new_hash_X)
#
#         pop_new = pop.new('X', pop_new_X)
#         pop_new.set('hash_X', pop_new_hash_X)
#         return pop_new
