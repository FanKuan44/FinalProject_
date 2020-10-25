import numpy as np
from nasbench import api


class MyPointCrossover:
    """
    The crossover combines parents to offsprings. Some crossover are problem specific and use additional information.
    This class must be inherited from to provide a crossover method to an algorithm.
    """

    def __init__(self, n_points, n_parents=2, n_offsprings=2):
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.n_points = n_points

    def do(self, problem, pop, **kwargs):
        """

        This method executes the crossover on the parents. This class wraps the implementation of the class
        that implements the crossover.

        Parameters
        ----------
        problem: class
            The problem to be solved. Provides information such as lower and upper bounds or feasibility
            conditions for custom crossovers.

        pop : Population
            The population as an object

        kwargs : dict
            Any additional data that might be necessary to perform the crossover. E.g. constants of an algorithm.

        Returns
        -------
        offsprings : Population
            The off as a matrix. n_children rows and the number of columns is equal to the variable
            length of the problem.

        """
        off = self._do(problem, pop, **kwargs)
        return off

    def _do(self, problem, pop, **kwargs):
        benchmark_api = kwargs['algorithm'].benchmark_api

        pop_X = pop.get('X')
        pop_hashX = pop.get('hashX')

        offspring_X = []
        offspring_hashX = []

        number_of_crossover = 0
        flag = False

        while len(offspring_X) < len(pop_X):
            if number_of_crossover > 100:
                flag = True
            idx = np.random.choice(len(pop_X), size=(len(pop_X) // 2, 2), replace=False)
            pop_X_ = pop.get('X')[idx]

            if problem.problem_name == 'nas101':
                # 2 points crossover
                for i in range(len(pop_X_)):
                    parent1_matrix, parent2_matrix = pop_X_[i][0][:-1, :].copy(), pop_X_[i][1][:-1, :].copy()
                    parent1_ops, parent2_ops = pop_X_[i][0][-1, :].copy(), pop_X_[i][1][-1, :].copy()

                    points_crossover = np.random.randint(0, 8, (1, self.n_points))
                    while True:
                        if points_crossover[0][0] - points_crossover[0][1] == 0:
                            points_crossover = np.random.randint(0, 9, (1, self.n_points))
                        else:
                            break
                    low = points_crossover[0][0]
                    if low > points_crossover[0][1]:
                        high = low
                        low = points_crossover[0][1]
                    else:
                        high = points_crossover[0][1]

                    parent1_matrix[low:high], parent2_matrix[low:high] = \
                        parent2_matrix[low:high], parent1_matrix[low: high].copy()
                    parent1_ops[low:high], parent2_ops[low:high] = \
                        parent2_ops[low:high], parent1_ops[low:high].copy()

                    idv_check = [[parent1_matrix, parent1_ops], [parent2_matrix, parent2_ops]]
                    for idv in idv_check:
                        spec = api.ModelSpec(matrix=np.array(idv[0], dtype=np.int), ops=idv[1].tolist())
                        if benchmark_api.is_valid(spec):
                            module_hash_spec = benchmark_api.get_module_hash(spec)
                            if not flag:
                                X = np.concatenate((idv[0], np.array([idv[1]])), axis=0)
                                if (module_hash_spec not in offspring_hashX) and (
                                        module_hash_spec not in pop_hashX):
                                    offspring_X.append(X)
                                    offspring_hashX.append(module_hash_spec)
                                else:
                                    offspring_X.append(X)
                                    offspring_hashX.append(module_hash_spec)

            elif problem.problem_name == 'cifar10' or problem.problem_name == 'cifar100':
                # 1 point crossover
                for i in range(len(pop_X_)):
                    offspring1_X, offspring2_X = pop_X_[i][0].copy(), pop_X_[i][1].copy()

                    crossover_pt = np.random.randint(1, len(offspring1_X))

                    offspring1_X[crossover_pt:], offspring2_X[crossover_pt:] = \
                        offspring2_X[crossover_pt:], offspring1_X[crossover_pt:].copy()

                    offspring1_hashX = ''.join(offspring1_X.tolist())
                    offspring2_hashX = ''.join(offspring2_X.tolist())

                    if not flag:
                        if (offspring1_hashX not in offspring_hashX) and (offspring1_hashX not in pop_hashX):
                            offspring_X.append(offspring1_X)
                            offspring_hashX.append(offspring1_hashX)

                        if (offspring2_hashX not in offspring_hashX) and (offspring2_hashX not in pop_hashX):
                            offspring_X.append(offspring2_X)
                            offspring_hashX.append(offspring2_hashX)
                    else:
                        offspring_X.append(offspring1_X)
                        offspring_hashX.append(offspring1_hashX)

                        offspring_X.append(offspring2_X)
                        offspring_hashX.append(offspring2_hashX)
            number_of_crossover += 1
        offspring_X = np.array(offspring_X)[:len(pop_X)]
        offspring_hashX = np.array(offspring_hashX)[:len(pop_X)]
        if flag:
            print('Exist Duplicate')
        ''' USING FOR CHECKING DUPLICATE '''
        # if np.sum(np.unique(offspring_hashX, return_counts=True)[-1]) != pop_X.shape[0]:
        #     print('DUPLICATE')
        #
        # for hashX in offspring_hashX:
        #     if hashX in pop_hashX:
        #         print('DUPLICATE', hashX)
        #         break
        # -----------------------------------
        offspring = pop.new(len(pop_X))
        offspring.set('X', offspring_X)
        offspring.set('hashX', offspring_hashX)
        return offspring
