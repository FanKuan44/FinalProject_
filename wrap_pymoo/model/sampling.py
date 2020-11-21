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
        print(len(pop))
        pop_X, pop_hashX = [], []

        if self.benchmark_name == 'cifar10' or self.benchmark_name == 'cifar100':
            allowed_choices = ['I', '1', '2']

            i = 0
            while i < n_samples:
                new_X = np.random.choice(allowed_choices, 14)
                new_hashX = convert_X_to_hashX(new_X)
                if new_hashX not in pop_hashX:
                    pop[i].set('X', new_X)
                    pop[i].set('hashX', new_hashX)
                    i += 1

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

        return pop


if __name__ == '__main__':
    s = MySampling('cifar10')
    pop = s._sampling(10)
    print(pop[0].get('X'))