import numpy as np
from nasbench import api


class PointCrossover:
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
        benchmark = kwargs['algorithm'].benchmark

        pop_X = pop.get('X')
        pop_hash_X = pop.get('hash_X')

        off_X = []
        off_hash_X = []

        while len(off_X) < len(pop_X):
            idx = np.random.choice(len(pop_X), size=(len(pop_X) // 2, 2), replace=False)
            pop_X_ = pop.get('X')[idx]
            if problem.problem_name == 'nas101':
                for i in range(len(pop_X_)):
                    while True:
                        val_ = []
                        hash_val_ = []
                        parent_1_matrix, parent_2_matrix = pop_X_[i][0][:-1, :].copy(), pop_X_[i][1][:-1, :].copy()
                        parent_1_ops, parent_2_ops = pop_X_[i][0][-1, :].copy(), pop_X_[i][1][-1, :].copy()
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

                        parent_1_matrix[low:high], parent_2_matrix[low:high] = \
                            parent_2_matrix[low:high].copy(), parent_1_matrix[low: high].copy()
                        parent_1_ops[low:high], parent_2_ops[low:high] = \
                            parent_2_ops[low:high].copy(), parent_1_ops[low:high].copy()

                        idv_check = [[parent_1_matrix, parent_1_ops], [parent_2_matrix, parent_2_ops]]
                        for idv in idv_check:
                            spec = api.ModelSpec(matrix=np.array(idv[0], dtype=np.int), ops=idv[1].tolist())
                            if benchmark.is_valid(spec):
                                hash_val_.append(benchmark.get_module_hash(spec))
                                val_.append(np.concatenate((idv[0], np.array([idv[1]])), axis=0))
                        if len(val_) == 2:
                            for idv in val_:
                                off_X.append(idv)
                            for idv in hash_val_:
                                off_hash_X.append(idv)
                            break
            elif problem.problem_name == 'cifar10' or problem.problem_name == 'cifar100':
                # 1 point
                for i in range(len(pop_X_)):
                    off1_X, off2_X = pop_X_[i][0].copy(), pop_X_[i][1].copy()

                    points_crossover = np.random.randint(0, len(off1_X))

                    off1_X[:points_crossover], off2_X[:points_crossover] = \
                        off2_X[:points_crossover], off1_X[:points_crossover].copy()

                    off1_hash_X = ''.join(off1_X.tolist())
                    off2_hash_X = ''.join(off2_X.tolist())

                    if off1_hash_X not in off_hash_X and off1_hash_X not in pop_hash_X:
                        off_X.append(off1_X)
                        off_hash_X.append(off1_hash_X)

                    if off2_hash_X not in off_hash_X and off2_hash_X not in pop_hash_X:
                        off_X.append(off2_X)
                        off_hash_X.append(off2_hash_X)

        off_X = np.array(off_X)[:len(pop_X)]
        off_hash_X = np.array(off_hash_X)[:len(pop_X)]

        # USING FOR CHECKING DUPLICATE
        if np.sum(np.unique(off_hash_X, return_counts=True)[-1]) != pop_X.shape[0]:
            print('DUPLICATE')

        for hash_X in off_hash_X:
            if hash_X in pop_hash_X:
                print('DUPLICATE', hash_X)
                break
        # -----------------------------------

        offspring = pop.new(len(pop_X))
        offspring.set('X', off_X)
        offspring.set('hash_X', off_hash_X)
        return offspring

# class MyPointCrossover(Crossover):
#     def __init__(self, n_points):
#         super().__init__(2, 2)
#         self.n_points = n_points
#
#     def _do(self, problem, pop, parents, **kwargs):
#         benchmark = kwargs['algorithm'].benchmark
#
#         pop_X = pop.get('X')
#         pop_hash_X = pop.get('hash_X')
#
#         idx = np.random.choice(len(pop_X), size=(len(pop_X)//2, 2), replace=False)
#         pop_X_ = pop.get('X')[idx]
#
#         off_X = []
#         off_hash_X = []
#
#         while len(off_X) < len(pop_X):
#             if problem.problem_name == 'nas101':
#                 for i in range(len(pop_X_)):
#                     while True:
#                         val_ = []
#                         hash_val_ = []
#                         parent_1_matrix, parent_2_matrix = pop_X_[i][0][:-1, :].copy(), pop_X_[i][1][:-1, :].copy()
#                         parent_1_ops, parent_2_ops = pop_X_[i][0][-1, :].copy(), pop_X_[i][1][-1, :].copy()
#                         points_crossover = np.random.randint(0, 8, (1, self.n_points))
#                         while True:
#                             if points_crossover[0][0] - points_crossover[0][1] == 0:
#                                 points_crossover = np.random.randint(0, 9, (1, self.n_points))
#                             else:
#                                 break
#                         low = points_crossover[0][0]
#                         if low > points_crossover[0][1]:
#                             high = low
#                             low = points_crossover[0][1]
#                         else:
#                             high = points_crossover[0][1]
#
#                         parent_1_matrix[low:high], parent_2_matrix[low:high] = \
#                             parent_2_matrix[low:high].copy(), parent_1_matrix[low: high].copy()
#                         parent_1_ops[low:high], parent_2_ops[low:high] = \
#                             parent_2_ops[low:high].copy(), parent_1_ops[low:high].copy()
#
#                         idv_check = [[parent_1_matrix, parent_1_ops], [parent_2_matrix, parent_2_ops]]
#                         for idv in idv_check:
#                             spec = api.ModelSpec(matrix=np.array(idv[0], dtype=np.int), ops=idv[1].tolist())
#                             if benchmark.is_valid(spec):
#                                 hash_val_.append(benchmark.get_module_hash(spec))
#                                 val_.append(np.concatenate((idv[0], np.array([idv[1]])), axis=0))
#                         if len(val_) == 2:
#                             for idv in val_:
#                                 off_X.append(idv)
#                             for idv in hash_val_:
#                                 off_hash_X.append(idv)
#                             break
#             elif problem.problem_name == 'cifar10' or problem.problem_name == 'cifar100':
#                 # 1 point
#                 for i in range(len(pop_X_)):
#                     off1_X, off2_X = pop_X_[i][0].copy(), pop_X_[i][1].copy()
#
#                     points_crossover = np.random.randint(0, len(off1_X))
#
#                     off1_X[:points_crossover], off2_X[:points_crossover] = \
#                         off2_X[:points_crossover], off1_X[:points_crossover].copy()
#
#                     off1_hash_X = ''.join(off1_X.tolist())
#                     off2_hash_X = ''.join(off2_X.tolist())
#
#                     if off1_hash_X not in off_hash_X and off1_hash_X not in pop_hash_X:
#                         off_X.append(off1_X)
#                         off_hash_X.append(off1_hash_X)
#
#                     if off2_hash_X not in off_hash_X and off2_hash_X not in pop_hash_X:
#                         off_X.append(off2_X)
#                         off_hash_X.append(off2_hash_X)
#
#         off_X = np.array(off_X)[:len(pop_X)]
#         off_hash_X = np.array(off_hash_X)[:len(pop_X)]
#
#         offspring = pop.new('X', off_X)
#         offspring.set('hash_X', off_hash_X)
#         return offspring
