import numpy as np

from nasbench import wrap_api as api
from wrap_pymoo.model.population import MyPopulation as Population
from wrap_pymoo.factory_nasbench import combine_matrix1D_and_opsINT, create_model, decoding_ops, decoding_matrix


def convert_X_to_hashX(x):
    if not isinstance(x, list):
        x = x.tolist()
    hashX = ''.join(x)
    return hashX


class MySampling:
    def __init__(self, benchmark_name):
        self.benchmark_name = benchmark_name
        if benchmark_name == 'nas101':
            self.benchmark_api = api.NASBench_()

    def _sampling(self, n_samples):
        pop = Population(n_samples)
        pop_X, pop_hashX = [], []

        if self.benchmark_name == 'cifar10' or self.benchmark_name == 'cifar100':
            allowed_choices = ['I', '1', '2']

            while len(pop_X) < n_samples:
                new_X = np.random.choice(allowed_choices, 14)
                new_hashX = convert_X_to_hashX(new_X)
                if new_hashX not in pop_hashX:
                    pop_X.append(new_X)
                    pop_hashX.append(new_hashX)

        else:
            while len(pop_X) < n_samples:
                matrix_2D, ops_STRING = create_model()
                modelspec = api.ModelSpec(matrix=matrix_2D, ops=ops_STRING)

                if self.benchmark_api.is_valid(modelspec):
                    hashX = self.benchmark_api.get_module_hash(modelspec)

                    if hashX not in pop_hashX:
                        matrix_1D = decoding_matrix(matrix_2D)
                        ops_INT = decoding_ops(ops_STRING)

                        X = combine_matrix1D_and_opsINT(matrix=matrix_1D, ops=ops_INT)
                        pop_X.append(X)
                        pop_hashX.append(hashX)

        pop.set('X', pop_X)
        pop.set('hashX', pop_hashX)

        return pop
