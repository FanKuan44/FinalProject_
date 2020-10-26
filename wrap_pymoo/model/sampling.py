from nasbench import wrap_api as api
from pymoo.model.sampling import Sampling
import numpy as np


class MySampling(Sampling):
    def __init__(self) -> None:
        super().__init__()

    def sample(self, problem, pop, n_samples, **kwargs):
        pop_X, pop_hashX, pop_F = [], [], []

        if problem.problem_name == 'nas101':
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

        elif problem.problem_name == 'cifar10' or problem.problem_name == 'cifar100':
            choices = ['I', '1', '2']
            for i in range(n_samples):
                while True:
                    X = np.random.choice(choices, size=14)
                    hashX = ''.join(X.tolist())
                    if hashX not in pop_hashX:
                        F = kwargs['algorithm'].evaluator.eval(
                            problem, np.array([hashX]), check=True, algorithm=kwargs['algorithm'])
                        pop_F.append(F)
                        pop_X.append(X)
                        pop_hashX.append(hashX)
                        break

        pop_ = pop.new(n_samples)

        CV = np.zeros((n_samples, 1))
        feasible = np.ones((n_samples, 1), dtype=np.bool)

        pop_X, pop_hashX, pop_F = np.array(pop_X), np.array(pop_hashX), np.array(pop_F)

        pop_.set('X', np.array(pop_X))
        pop_.set('hashX', np.array(pop_hashX))
        pop_.set('F', np.array(pop_F))
        pop_.set('CV', CV)
        pop_.set('feasible', feasible)
        return pop_
