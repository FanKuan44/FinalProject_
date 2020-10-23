from wrap_pymoo.algorithms.genetic_algorithm import GeneticAlgorithm


class NSGANet(GeneticAlgorithm):
    def __init__(self, local_search_on_pf, local_search_on_knee, path, opt_val_acc_and_training_time=1,
                 **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective

        self.local_search_on_pf = local_search_on_pf
        self.local_search_on_knee = local_search_on_knee
        self.opt_val_acc_and_training_time = opt_val_acc_and_training_time

        self.elitist_archive_X = []
        self.elitist_archive_hash_X = []
        self.elitist_archive_F = []

        self.dpf = []
        self.no_eval = []

        self.benchmark = benchmark
        self.path = path

        if benchmark == 'nas101':
            self.pf_true = pickle.load(open('101_benchmark/pf_validation_parameters.p', 'rb'))
        elif benchmark == 'cifar10':
            self.pf_true = pickle.load(open('bosman_benchmark/cifar10/pf_validation_MMACs_cifar10.p', 'rb'))
        elif benchmark == 'cifar100':
            self.pf_true = pickle.load(open('bosman_benchmark/cifar100/pf_validation_MMACs_cifar100.p', 'rb'))

    def _solve(self, problem, termination):
        self.n_gen = 0
        print('Gen:', self.n_gen)

        ''' Initialization '''
        self.pop = self._initialize()

        dpf = round(cal_dpf(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
        self.dpf.append(dpf)
        self.no_eval.append(self.problem._n_evaluated)

        self._each_iteration(self, first=True)

        # while termination criteria not fulfilled
        while termination.do_continue(self):
            self.n_gen += 1
            print('Gen:', self.n_gen)

            # do the next iteration
            self.pop = self._next(self.pop)

            dpf = round(cal_dpf(pareto_s=self.elitist_archive_F, pareto_front=self.pf_true), 5)
            self.dpf.append(dpf)
            self.no_eval.append(self.problem._n_evaluated)

            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()

        return self.pop

    def _initialize(self):
        pop = Population(n_individuals=0, individual=self.individual)
        pop = self.sampling.sample(problem=self.problem, pop=pop, n_samples=self.pop_size, algorithm=self)

        if self.survival:
            pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        ''' Update Elitist Archive '''
        # print('--> UPDATE ELITIST ARCHIVE AFTER INITIALIZE POPULATION')
        pop_X = pop.get('X')
        pop_hash_X = pop.get('hash_X')
        pop_F = pop.get('F')
        self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
            update_elitist_archive(pop_X, pop_hash_X, pop_F,
                                   self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F,
                                   first=True)
        # print('--> UPDATE ELITIST ARCHIVE AFTER INITIALIZE POPULATION - DONE')
        return pop

    def _next(self, pop):
        """ Mating """
        self.off = self._mating(pop)

        # merge the offsprings with the current population
        pop = pop.merge(self.off)

        # the do survival selection
        pop = self.survival.do(self.problem, pop, self.pop_size, algorithm=self)

        """Local Search on PF"""
        if self.local_search_on_pf == 1:
            pop_F = pop.get("F")

            front = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size)

            pareto_front = pop[front[0]].copy()
            f_pareto_front = pop_F[front[0]].copy()

            new_idx = np.argsort(f_pareto_front[:, 1])

            pareto_front = pareto_front[new_idx]

            pareto_front, non_dominance_X, non_dominance_hash_X, non_dominance_F = \
                self._local_search_on_x_bosman(pop, x=pareto_front)
            pop[front[0]] = pareto_front
            # print('non_dominance_F\n', non_dominance_F)
            """ Update Elitist Archive """
            self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hash_X, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)

        """Local Search on Knee Solutions"""
        if self.local_search_on_knee == 1:
            pop_F = pop.get("F")

            front = NonDominatedSorting().do(pop_F, n_stop_if_ranked=self.pop_size)

            pareto_front = pop[front[0]].copy()
            f_pareto_front = pop_F[front[0]].copy()

            new_idx = np.argsort(f_pareto_front[:, 1])

            pareto_front = pareto_front[new_idx]
            f_pareto_front = f_pareto_front[new_idx]

            angle = [np.array([360, 0])]
            for i in range(1, len(f_pareto_front) - 1):
                if kiem_tra_p1_nam_phia_duoi_p2_p3(f_pareto_front[i], f_pareto_front[i - 1],
                                                   f_pareto_front[i + 1]):
                    angle.append(
                        np.array([cal_angle(f_pareto_front[i], f_pareto_front[i - 1],
                                            f_pareto_front[i + 1]), i]))
                else:
                    angle.append(np.array([0, i]))
            angle.append(np.array([360, len(pareto_front) - 1]))
            angle = np.array(angle)
            angle = angle[np.argsort(angle[:, 0], )]
            angle = angle[angle[:, 0] > 210]

            idx_knee_solutions = np.array(angle[:, 1], dtype=np.int)
            knee_solutions = pareto_front[idx_knee_solutions].copy()

            knee_solutions_, non_dominance_X, non_dominance_hash_X, non_dominance_F = \
                self._local_search_on_x_bosman(pop, x=knee_solutions)

            """ Update Elitist Archive """
            self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
                update_elitist_archive(non_dominance_X, non_dominance_hash_X, non_dominance_F,
                                       self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)

            pareto_front[idx_knee_solutions] = knee_solutions_

            # f_knee_solutions_ = knee_solutions_.get("F")
            # # plt.scatter(f_knee_solutions_[:, 1], f_knee_solutions_[:, 0], s=15, c='blue')
            pop[front[0]] = pareto_front
        """----------------------------------------------------------------------------"""
        return pop

    def _mating(self, pop):
        """ CROSSOVER """
        # print("--> CROSSOVER")
        off = self.crossover.do(problem=self.problem, pop=pop, algorithm=self)
        # print("--> CROSSOVER - DONE")

        """ EVALUATE OFFSPRINGS AFTER CROSSOVER (USING FOR UPDATING ELITIST ARCHIVE) """
        # print("--> EVALUATE AFTER CROSSOVER")
        off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
        off.set('F', off_F)
        # print("--> EVALUATE AFTER CROSSOVER - DONE")

        """ UPDATING ELITIST ARCHIVE """
        # print('--> UPDATE ELITIST ARCHIVE AFTER CROSSOVER')
        off_X = off.get('X')
        off_hash_X = off.get('hash_X')
        off_F = off.get('F')
        self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
            update_elitist_archive(off_X, off_hash_X, off_F,
                                   self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)
        # print('--> UPDATE ELITIST ARCHIVE AFTER CROSSOVER - DONE')

        """ MUTATION """
        # print("--> MUTATION")
        off = self.mutation.do(self.problem, off, algorithm=self)
        # print("--> MUTATION - DONE")

        """ EVALUATE OFFSPRINGS AFTER MUTATION (USING FOR UPDATING ELITIST ARCHIVE) """
        # print("--> EVALUATE AFTER MUTATION")
        off_F = self.evaluator.eval(self.problem, off.get('X'), check=True, algorithm=self)
        off.set('F', off_F)
        # print("--> EVALUATE AFTER MUTATION - DONE")

        """ UPDATING ELITIST ARCHIVE """
        # print('--> UPDATE ELITIST ARCHIVE AFTER MUTATION')
        off_X = off.get('X')
        off_hash_X = off.get('hash_X')
        off_F = off.get('F')
        self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F = \
            update_elitist_archive(off_X, off_hash_X, off_F,
                                   self.elitist_archive_X, self.elitist_archive_hash_X, self.elitist_archive_F)
        # print('--> UPDATE ELITIST ARCHIVE AFTER MUTATION - DONE')

        # _off_hash_X = _off.get('hash_X').tolist()
        #
        # """ Check duplicate in pop """
        # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, pop_hash_X)
        #
        # _off = _off[not_duplicate]
        # _off_hash_X = _off.get('hash_X').tolist()
        #
        # """ Check duplicate in new offsprings """
        # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, _off_hash_X, True)
        # _off = _off[not_duplicate]
        #
        # _off_hash_X = _off.get('hash_X').tolist()
        #
        # """ Check duplicate in current offsprings """
        # not_duplicate = is_x1_not_duplicate_x2(_off_hash_X, off.get('hash_X').tolist())
        # _off = _off[not_duplicate]
        #
        # if len(_off) > self.n_offsprings - len(off):
        #     I = random.perm(self.n_offsprings - len(off))
        #     _off = _off[I]
        # if len(_off) != 0:
        #     _off_f = self.evaluator.eval(self.problem, _off.get('X'), check=True, algorithm=self)
        #     _off.set('F', _off_f)
        #     # add to the offsprings and increase the mating counter
        # off = off.merge(_off)

        CV = np.zeros((len(off), 1))
        feasible = np.ones((len(off), 1), dtype=np.bool)

        off.set('CV', CV)
        off.set('feasible', feasible)
        return off

    def _local_search_on_x(self, pop, x):
        off_ = pop.new()
        off_ = off_.merge(x)

        x_old_X = x.get('X')
        x_old_hash_X = x.get('hash_X')
        x_old_F = x.get('F')

        non_dominance_X = []
        non_dominance_hash_X = []
        non_dominance_F = []

        if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
            for i in range(len(x_old_X)):
                checked = []
                stop_iter = 14 * 2
                j = 0
                best_X = x_old_X[i].copy()
                best_hash_X = x_old_hash_X[i].copy()
                best_F = x_old_F[i].copy()
                while j < stop_iter:
                    idx = np.random.randint(0, 14)
                    ops = ['I', '1', '2']
                    ops.remove(x_old_X[i][idx])
                    new_op = np.random.choice(ops)
                    if [idx, new_op] not in checked:
                        checked.append([idx, new_op])

                        x_new_X = x_old_X[i].copy()
                        x_new_X[idx] = new_op

                        x_new_hash_X = ''.join(x_new_X.tolist())

                        if x_new_hash_X not in x_old_hash_X:
                            x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                          check=True, algorithm=self)
                            # result = check_better(x_new_F[0], x_old_F[i])
                            result = check_better(x_new_F[0], best_F)
                            if result == 'obj1':
                                best_X = x_new_X
                                best_hash_X = x_new_hash_X
                                best_F = x_new_F[0]
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(x_new_hash_X)
                                non_dominance_F.append(x_new_F[0])
                            elif result == 'none':
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(x_new_hash_X)
                                non_dominance_F.append(x_new_F[0])
                        j += 1
                x_old_X[i] = best_X
                x_old_hash_X[i] = best_hash_X
                x_old_F[i] = best_F
        elif self.problem.problem_name == 'nas101':
            for i in range(len(x_old_X)):
                checked = []
                stop_iter = 5 * 2
                j = 0
                while j < stop_iter:
                    idx = np.random.randint(1, 6)
                    ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                    ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    if [idx, new_op] not in checked:
                        checked.append([idx, new_op])
                        x_new_X = x_old_X[i].copy()
                        x_new_X[-1][idx] = new_op
                        neighbor = api.ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int), ops=x_new_X[-1].tolist())
                        neighbor_hash = self.benchmark.get_module_hash(neighbor)

                        if neighbor_hash not in x_old_hash_X:
                            neighbor_F = self.evaluator.eval(self.problem, np.array([x_new_X]), check=True,
                                                             algorithm=self)
                            result = check_better(neighbor_F[0], x_old_F[i])
                            if result == 'obj1':
                                x_old_X[i] = x_new_X
                                x_old_hash_X[i] = neighbor_hash
                                x_old_F[i] = neighbor_F
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                                break
                            elif result == 'none':
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                        j += 1

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hash_X = np.array(non_dominance_hash_X)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hash_X', x_old_hash_X)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hash_X, non_dominance_F

    def _local_search_on_x_bosman(self, pop, x):
        off_ = pop.new()
        off_ = off_.merge(x)

        x_old_X = x.get('X')
        x_old_hash_X = x.get('hash_X')
        x_old_F = x.get('F')

        non_dominance_X = []
        non_dominance_hash_X = []
        non_dominance_F = []

        if self.problem.problem_name == 'cifar10' or self.problem.problem_name == 'cifar100':
            for i in range(len(x_old_X)):
                checked = [''.join(x_old_X[i].copy().tolist())]
                stop_iter = 20
                j = 0
                alpha = np.random.rand(1)
                while j < stop_iter:
                    idx = np.random.randint(0, 14)

                    ops = ['I', '1', '2']
                    ops.remove(x_old_X[i][idx])
                    new_op = np.random.choice(ops)

                    tmp = x_old_X[i].copy().tolist()
                    tmp[idx] = new_op
                    tmp_str = ''.join(tmp)

                    if tmp_str not in checked:
                        checked.append(tmp_str)

                        x_new_X = x_old_X[i].copy()
                        x_new_X[idx] = new_op

                        x_new_hash_X = ''.join(x_new_X.tolist())

                        if x_new_hash_X not in x_old_hash_X:
                            x_new_F = self.evaluator.eval(self.problem, np.array([x_new_X]),
                                                          check=True, algorithm=self)
                            result = check_better(x_new_F[0], x_old_F[i])
                            if result == 'none':
                                # print("Non-dominated solution")
                                x_old_X[i] = x_new_X
                                x_old_hash_X[i] = x_new_hash_X
                                x_old_F[i] = x_new_F[0]
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(x_new_hash_X)
                                non_dominance_F.append(x_new_F[0])
                            else:
                                result_ = check_better_bosman(alpha=alpha, f_obj1=x_new_F[0], f_obj2=x_old_F[i])
                                if result_ == 'obj1':
                                    # print("Improved solution")
                                    x_old_X[i] = x_new_X
                                    x_old_hash_X[i] = x_new_hash_X
                                    x_old_F[i] = x_new_F[0]
                                    non_dominance_X.append(x_new_X)
                                    non_dominance_hash_X.append(x_new_hash_X)
                                    non_dominance_F.append(x_new_F[0])
                    j += 1
        elif self.problem.problem_name == 'nas101':
            for i in range(len(x_old_X)):
                checked = []
                stop_iter = 5 * 2
                j = 0
                while j < stop_iter:
                    idx = np.random.randint(1, 6)
                    ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                    ops.remove(x_old_X[i][-1][idx])
                    new_op = np.random.choice(ops)

                    if [idx, new_op] not in checked:
                        checked.append([idx, new_op])
                        x_new_X = x_old_X[i].copy()
                        x_new_X[-1][idx] = new_op
                        neighbor = api.ModelSpec(matrix=np.array(x_new_X[:-1], dtype=np.int), ops=x_new_X[-1].tolist())
                        neighbor_hash = self.benchmark.get_module_hash(neighbor)

                        if neighbor_hash not in x_old_hash_X:
                            neighbor_F = self.evaluator.eval(self.problem, np.array([x_new_X]), check=True,
                                                             algorithm=self)
                            result = check_better(neighbor_F[0], x_old_F[i])
                            if result == 'obj1':
                                x_old_X[i] = x_new_X
                                x_old_hash_X[i] = neighbor_hash
                                x_old_F[i] = neighbor_F
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                                break
                            elif result == 'none':
                                non_dominance_X.append(x_new_X)
                                non_dominance_hash_X.append(neighbor_hash)
                                non_dominance_F.append(neighbor_F[0])
                        j += 1

        non_dominance_X = np.array(non_dominance_X)
        non_dominance_hash_X = np.array(non_dominance_hash_X)
        non_dominance_F = np.array(non_dominance_F)

        off_.set('X', x_old_X)
        off_.set('hash_X', x_old_hash_X)
        off_.set('F', x_old_F)

        return off_, non_dominance_X, non_dominance_hash_X, non_dominance_F

    def _finalize(self):
        plt.plot(self.no_eval, self.dpf)
        plt.grid()
        plt.savefig(f'{self.path}/dpfs_and_no_evaluations')
        plt.clf()
        # pf = np.array(self.elitist_archive_F)
        # pf = np.unique(pf, axis=0)
        # plt.scatter(pf[:, 1], pf[:, 0], c='blue')
        # plt.savefig('xxxxx')
        # plt.clf()
