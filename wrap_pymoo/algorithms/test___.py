import numpy as np
import pickle
import matplotlib.pyplot as plt


def sampling(problem_name, pop, n_samples):
    pop_X = []
    pop_hash_X = []
    pop_F = []
    if problem_name == 'nas_101':
        benchmark = kwargs['algorithm'].benchmark
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
                # spec = api.ModelSpec(matrix=matrix, ops=ops[0].tolist())

                if benchmark.is_valid(spec):
                    module_hash_spec = benchmark.get_module_hash(spec)
                    if module_hash_spec not in pop_hash_X:
                        X = np.concatenate((matrix, ops), axis=0)
                        pop_F.append(kwargs['algorithm'].evaluator.eval(problem, np.array([X]),
                                                                        check=True, algorithm=kwargs['algorithm'])[0])
                        pop_X.append(X)
                        pop_hash_X.append(module_hash_spec)
                        break
    elif problem_name == 'cifar10':
        # benchmark = pickle.load(open('D:/Desktop/nsga-net-master/bosman_cifar10.p', 'rb'))
        # min_max = pickle.load(open('D:/Desktop/nsga-net-master/min_max_cifar10.p', 'rb'))
        ALLOWED_OPS = ['I', '1', '2']
        for i in range(n_samples):
            while True:
                model = np.random.choice(ALLOWED_OPS, size=14)
                model_str = ''.join(model.tolist())
                if model_str not in pop_hash_X:
                    f_0 = 1 - benchmark[model_str]['val_acc']/100
                    f_1 = (benchmark[model_str]['MMACs'] - min_max['min_MMACs']) / \
                          (min_max['max_MMACs'] - min_max['min_MMACs'])
                    pop_F.append(np.array([f_0, f_1]))
                    pop_X.append(model)
                    pop_hash_X.append(model_str)
                    break
    else:
        pass

    pop_X = np.array(pop_X)
    pop_hash_X = np.array(pop_hash_X)
    pop_F = np.array(pop_F)
    print(pop_X)
    print(pop_hash_X)
    print(pop_F)

    pop_ = pop.new(n_samples)

    CV = np.zeros((n_samples, 1))
    feasible = np.ones((n_samples, 1), dtype=np.bool)

    pop_.set('X', pop_X)
    pop_.set('hash_X', pop_hash_X)
    pop_.set('F', pop_F)
    pop_.set('CV', CV)
    pop_.set('feasible', feasible)
    return pop_


if __name__ == '__main__':
    # sampling('cifar10', 10)
    # f_value = pickle.load(open('D:/Desktop/nsga-net-master/value_3_obj_cifar10.p', 'rb'))
    # print(len(f_value))
    plt.show()