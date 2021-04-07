import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import pandas as pd
import timeit
import torch

from acc_predictor.factory import get_acc_predictor
from datetime import datetime

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.util.display import disp_multi_objective
from pymoo.util.non_dominated_sorting import NonDominatedSorting

from nasbench import wrap_api as api

from wrap_pymoo.algorithms.genetic_algorithm import GeneticAlgorithm

from wrap_pymoo.model.individual import MyIndividual as Individual
from wrap_pymoo.model.population import MyPopulation as Population

from wrap_pymoo.util.compare import find_better_idv
from wrap_pymoo.util.IGD_calculating import calc_IGD
from wrap_pymoo.util.find_knee_solutions import cal_angle, kiem_tra_p1_nam_phia_tren_hay_duoi_p2_p3
from wrap_pymoo.util.survial_selection import RankAndCrowdingSurvival
from wrap_pymoo.util.tournament_selection import binary_tournament

from wrap_pymoo.factory_nasbench import combine_matrix1D_and_opsINT, split_to_matrix1D_and_opsINT, create_model
from wrap_pymoo.factory_nasbench import encoding_ops, decoding_ops, encoding_matrix, decoding_matrix

if __name__ == '__main__':
    pass
