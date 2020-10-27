import numpy as np
import copy

from nasbench import api


class MyMutation:
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

    def _do(self, problem, pop, **kwargs):
        pop_X = pop.get('X')
        pop_hashX = pop.get('hashX')

        offspring_X = []
        offspring_hashX = []

        while len(offspring_X) < len(pop_X):
            if problem.problem_name == 'nas101':
                benchmark_api = kwargs['algorithm'].benchmark_api

                # mutation_pts = np.random.rand(pop_X.shape[0], 7)
                #
                # for i in range(len(pop_X)):
                #     offspring_old_X = pop_X[i].copy()
                #     for j in range(1, 6):
                #         if mutation_pts[i][j] <= self.prob:
                #             choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                #             choices.remove(offspring_old_X[-1][j])
                #             offspring_old_X[-1][j] = np.random.choice(choices)
                #
                #     new_spec = api.ModelSpec(np.array(offspring_old_X[:-1], dtype=np.int), offspring_old_X[-1].tolist())
                #
                #     if benchmark_api.is_valid(new_spec):
                #         module_hash_spec = benchmark_api.get_module_hash(new_spec)
                #         if module_hash_spec in offspring_hashX:
                #             print('duplicate offspring')
                #         if module_hash_spec in pop_hashX:
                #             print('duplicate pop')
                #         if (module_hash_spec not in offspring_hashX) and \
                #                 (module_hash_spec not in pop_hashX):
                #             offspring_X.append(offspring_old_X)
                #             offspring_hashX.append(module_hash_spec)
                #     else:
                #         print('invalid-mutation')

                for x in pop_X:
                    new_matrix = copy.deepcopy(np.array(x[:-1, :], dtype=np.int))
                    new_ops = copy.deepcopy(x[-1, :])
                    # In expectation, V edges flipped (note that most end up being pruned).
                    edge_mutation_prob = 1 / 7
                    for src in range(0, 7 - 1):
                        for dst in range(src + 1, 7):
                            if np.random.rand() < edge_mutation_prob:
                                new_matrix[src, dst] = 1 - new_matrix[src, dst]

                    # In expectation, one op is resampled.
                    op_mutation_prob = 1 / 5
                    for ind in range(1, 7 - 1):
                        if np.random.rand() < op_mutation_prob:
                            available = [o for o in benchmark_api.config['available_ops'] if o != new_ops[ind]]
                            new_ops[ind] = np.random.choice(available)
                    new_spec = api.ModelSpec(new_matrix, new_ops.tolist())

                    if benchmark_api.is_valid(new_spec):
                        module_hash_spec = benchmark_api.get_module_hash(new_spec)
                        if (module_hash_spec not in offspring_hashX) and \
                                (module_hash_spec not in pop_hashX):
                            offspring_X.append(np.concatenate((new_matrix, np.array([new_ops])), axis=0))
                            offspring_hashX.append(module_hash_spec)

            elif problem.problem_name == 'cifar10' or problem.problem_name == 'cifar100':
                # Co the xay ra dot bien o nhieu vi tri
                mutation_pts = np.random.rand(pop_X.shape[0], pop_X.shape[1])

                for i in range(len(pop_X)):
                    offspring_old_X = pop_X[i].copy()

                    for j in range(pop_X.shape[1]):
                        if mutation_pts[i][j] <= self.prob:
                            choices = ['I', '1', '2']
                            choices.remove(offspring_old_X[j])
                            offspring_old_X[j] = np.random.choice(choices)

                    offspring_new_hashX = ''.join(offspring_old_X)

                    if offspring_new_hashX not in offspring_hashX and offspring_new_hashX not in pop_hashX:
                        offspring_new_X = offspring_old_X.copy()
                        offspring_X.append(offspring_new_X)
                        offspring_hashX.append(offspring_new_hashX)

        offspring_X = np.array(offspring_X)[:len(pop_X)]
        offspring_hashX = np.array(offspring_hashX)[:len(pop_X)]

        # USING FOR CHECKING DUPLICATE
        if np.sum(np.unique(offspring_hashX, return_counts=True)[-1]) != pop_X.shape[0]:
            print('DUPLICATE')

        for hashX in offspring_hashX:
            if hashX in pop_hashX:
                print('DUPLICATE', hashX)
                break
        # -----------------------------------
        offspring = pop.new(len(pop_X))
        offspring.set('X', offspring_X)
        offspring.set('hashX', offspring_hashX)
        return offspring
