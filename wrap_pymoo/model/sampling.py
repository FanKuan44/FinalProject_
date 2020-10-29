from nasbench import wrap_api as api
from pymoo.model.sampling import Sampling
from wrap_pymoo.model.population import MyPopulation as Population
import numpy as np


class MySampling(Sampling):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def sample_custom(problem_name, n_samples, **kwargs):
        pop_X, pop_hashX, pop_F = [], [], []

        if problem_name == 'nas101':
            benchmark_api = kwargs['algorithm'].benchmark_api
            INPUT = 'input'
            OUTPUT = 'output'
            CONV3X3 = 'conv3x3-bn-relu'
            CONV1X1 = 'conv1x1-bn-relu'
            MAXPOOL3X3 = 'maxpool3x3'
            NUM_VERTICES = 7
            ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
            ALLOWED_EDGES = [0, 1]
            for i in range(n_samples):
                while True:
                    matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
                    matrix = np.triu(matrix, 1)
                    ops = np.random.choice(ALLOWED_OPS, size=(1, NUM_VERTICES))
                    ops[0][0] = INPUT
                    ops[0][-1] = OUTPUT
                    spec = api.ModelSpec(matrix=matrix, ops=ops[0].tolist())

                    if benchmark_api.is_valid(spec):
                        module_hash_spec = benchmark_api.get_module_hash(spec)
                        if module_hash_spec not in pop_hashX:
                            X = np.concatenate((matrix, ops), axis=0)
                            F = kwargs['algorithm'].evaluator.eval(
                                problem, np.array([X]), check=True, algorithm=kwargs['algorithm'])[0]
                            pop_F.append(F)
                            pop_X.append(X)
                            pop_hashX.append(module_hash_spec)
                            break

        elif problem_name == 'cifar10' or problem_name == 'cifar100':
            allowed_choices = ['I', '1', '2']
            while len(pop_X) < n_samples:
                new_X = np.random.choice(allowed_choices, size=14)
                new_hashX = ''.join(new_X.tolist())
                if new_hashX not in pop_hashX:
                    # F = kwargs['algorithm'].evaluator.eval(
                    #     problem, X, check=True, algorithm=kwargs['algorithm'])
                    # pop_F.append(F)
                    pop_X.append(new_X)
                    pop_hashX.append(new_hashX)

        CV = np.zeros((n_samples, 1))
        feasible = np.ones((n_samples, 1), dtype=np.bool)

        # pop_X, pop_hashX, pop_F = np.array(pop_X), np.array(pop_hashX), np.array(pop_F)
        pop = Population(n_samples)
        pop.set('X', pop_X)
        pop.set('hashX', pop_hashX)
        # pop.set('F', np.array(pop_F))
        pop.set('CV', CV)
        pop.set('feasible', feasible)
        return pop
