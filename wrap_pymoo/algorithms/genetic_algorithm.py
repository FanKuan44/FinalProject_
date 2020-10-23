import math

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.rand import random


class GeneticAlgorithm(Algorithm):

    def __init__(self,
                 pop_size,
                 sampling,
                 selection,
                 crossover,
                 mutation,
                 survival,
                 n_offsprings=None,
                 repair=None,
                 individual=Individual(),
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # population size of the genetic algorithm
        self.pop_size = pop_size

        # initial sampling method: object, 2d array, or population (already evaluated)
        self.sampling = sampling

        # the method to be used to select parents for recombination
        self.selection = selection

        # method to do the crossover
        self.crossover = crossover

        # method for doing the mutation
        self.mutation = mutation

        # function to repair an offspring after mutation if necessary
        self.repair = repair

        # survival selection
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # the object to be used to represent an individual - either individual or derived class
        self.individual = individual

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # other run specific data updated whenever solve is called - to share them in all methods
        self.n_gen = None
        self.pop = None
        self.off = None

    def _solve(self, problem, termination):
        self.n_gen = 1

        self.pop = self._initialize()

        self._each_iteration(self, first=True)

        # while termination criteria not fulfilled
        while termination.do_continue(self):
            self.n_gen += 1

            # do the next iteration
            self.pop = self._next(self.pop)

            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()
        return self.pop

    def _initialize(self):
        pass

    def _next(self, pop):
        pass

    def _mating(self, pop):
        pass

    def _finalize(self):
        pass

