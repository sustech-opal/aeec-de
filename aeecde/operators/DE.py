#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differential Evolution operators
'''
import numpy as np

# internal imports


# HB : the following imports are for personal purpose
try:
    import sys, IPython
    sys.excepthook = IPython.core.ultratb.ColorTB()
except:
    pass



#!------------------------------------------------------------------------------
#!                                 Mutation Operators
#!------------------------------------------------------------------------------
# References:
# [1] Comparison of mutation strategies in Differential Evolution-A
# probabilistic perspective
# [2] Differential evolution algorithm with ensemble of parameters and
# mutation strategies

# Validate in Ref[1]
def rand_1(pop, target_idx, F):
    ''' DE/rand/1 mutation operator '''
    ind_r1, ind_r2, ind_r3 = pop._parent_select(target_idx, k=3)
    mutant_vec = (ind_r1.xvalue
                    + F * (ind_r2.xvalue - ind_r3.xvalue))
    return mutant_vec


# Validate in Ref[1]
def rand_2(pop, target_idx, F):
    ''' DE/rand/2 mutation operator '''
    ind_r1, ind_r2, ind_r3, ind_r4, ind_r5 = pop._parent_select(target_idx,
                                                                    k=5)
    mutant_vec = (ind_r1.xvalue
                    + F * (ind_r2.xvalue - ind_r3.xvalue)
                    + F * (ind_r4.xvalue - ind_r5.xvalue))
    return mutant_vec


# Validate in Ref[1]
def best_1(pop, target_idx, F):
    ''' DE/best/2 mutation operator '''
    ind_r1, ind_r2 = pop._parent_select(target_idx, k=2)
    pop.find_best_individual()
    mutant_vec = (pop.X_best
                    + F * (ind_r1.xvalue - ind_r2.xvalue))
    return mutant_vec


# Validate in Ref[1]
def best_2(pop, target_idx, F):
    ''' DE/best/2 mutation operator '''
    ind_r1, ind_r2, ind_r3, ind_r4 = pop._parent_select(target_idx, k=4)
    pop.find_best_individual()
    mutant_vec = (pop.X_best
                    + F * (ind_r1.xvalue - ind_r2.xvalue)
                    + F * (ind_r3.xvalue - ind_r4.xvalue))
    return mutant_vec


# Validate in Ref[1]
def current_1(pop, target_idx, F):
    ''' DE/current/1 mutation operator '''
    ind_r1, ind_r2 = pop._parent_select(target_idx, k=2)
    # F = exp(pop.rng.rand()) # CWH: if do so, no need to input `F`
    mutant_vec = (pop.list_ind[target_idx].xvalue
                    + F * (ind_r1.xvalue - ind_r2.xvalue))
    return mutant_vec


# Validate in Ref[1]
def current_to_rand_1(pop, target_idx, F):
    ''' DE/current-to-rand/1 mutation operator '''
    ind_r1, ind_r2, ind_r3 = pop._parent_select(target_idx, k=3)
    K = pop.rng.rand()
    mutant_vec = (pop.list_ind[target_idx].xvalue
                    + K * (ind_r1.xvalue
                            - pop.list_ind[target_idx].xvalue)
                    + F * (ind_r2.xvalue - ind_r3.xvalue))
    return mutant_vec


# Validate in Ref[1]
def current_to_best_1(pop, target_idx, F):
    ''' DE/current-to-best/1 mutation operator '''
    ind_r1, ind_r2 = pop._parent_select(target_idx, k=2)
    pop.find_best_individual()
    # K = pop.rng.rand()
    mutant_vec = (pop.list_ind[target_idx].xvalue
                    + F * (pop.X_best - pop.list_ind[target_idx].xvalue)
                    + F * (ind_r1.xvalue - ind_r2.xvalue))
    return mutant_vec


# Validate in Ref[2] 'DE/target-to-best/2'
def current_to_best_2(pop, target_idx, F):
    ''' DE/current-to-best/2 mutation operator '''
    ind_r1, ind_r2, ind_r3, ind_r4 = pop._parent_select(target_idx, k=4)
    pop.find_best_individual()
    K = pop.rng.rand()
    mutant_vec = (pop.list_ind[target_idx].xvalue
                    + K * (pop.X_best - pop.list_ind[target_idx].xvalue)
                    + F * (ind_r1.xvalue - ind_r2.xvalue)
                    + F * (ind_r3.xvalue - ind_r4.xvalue))
    return mutant_vec


# Validate in Ref[1]
# def current_to_pbest_1(pop, target_idx, F, p=0.2):
#     ''' DE/current-to-pbest/1 mutation operator '''
#     ind_r1, ind_r2 = pop._parent_select(target_idx, k=2)
#     sorted_idxs = np.argsort(pop.F)
#     top_p = int(np.ceil(pop.size * p))
#     idxs = sorted_idxs[:top_p]
#     pop.rng.shuffle(idxs)
#     K = pop.rng.rand()
#     mutant_vec = ( pop.list_ind[target_idx].xvalue
#                    + K *(pop.list_ind[idxs[0]].xvalue
#                    - pop.list_ind[target_idx].xvalue)
#                    + F * (ind_r1.xvalue - ind_r2.xvalue) )
#     return mutant_vec


def current_to_pbest_1(pop, target_idx, F, **kwargs):
    ''' DE/current-to-pbest/1 mutation operator with optional external archive
    Ref:
        J. Zhang and A. C. Sanderson, "JADE: Adaptive Differential Evolution 
        With Optional External Archive," in IEEE Transactions on 
        Evolutionary Computation, vol. 13, no. 5, pp. 945-958, Oct. 2009.
    '''
    p = kwargs.get("p")
    if p is None:
        p = 0.2
    archive = kwargs.get("archive")
    if archive is None or archive == []:
        ind_r1, ind_r2 = pop._parent_select(target_idx, k=2)
    else:
        assert isinstance(archive, list), "`archive` must be a list of ind"
        union = pop.list_ind + archive
        idxs = list(range(pop.size))
        idxs.remove(target_idx)
        pop.rng.shuffle(idxs)
        r1 = idxs[0]
        ind_r1 = pop.list_ind[r1]
        idxs_union = list(range(len(union)))
        idxs_union.remove(target_idx)
        idxs_union.remove(r1)
        pop.rng.shuffle(idxs_union)
        r2 = idxs_union[0]
        ind_r2 = union[r2]
    # randonly select one ind from the 100p% best inds in pop
    pop.get_F()
    sorted_idxs = np.argsort(pop.F)
    top_p = int(np.ceil(pop.size * p))
    idxs_top = sorted_idxs[:top_p]
    pop.rng.shuffle(idxs_top)
    pbest = idxs_top[0]
    ind_pbest = pop.list_ind[pbest]
    # K = pop.rng.rand()
    mutant_vec = (pop.list_ind[target_idx].xvalue
                    + F * (ind_pbest.xvalue
                            - pop.list_ind[target_idx].xvalue)
                    + F * (ind_r1.xvalue - ind_r2.xvalue))
    return mutant_vec


def rand_to_best_1(pop, target_idx, F):
    ''' DE/rand-to-best/1 mutation operator '''
    ind_r1, ind_r2, ind_r3 = pop._parent_select(target_idx, k=3)
    pop.find_best_individual()
    # K = pop.rng.rand()
    mutant_vec = (ind_r1.xvalue
                    + F * (pop.X_best - ind_r1.xvalue)
                    + F * (ind_r2.xvalue - ind_r3.xvalue))
    return mutant_vec


def rand_to_pbest_1(pop, target_idx, F, **kwargs):
    ''' DE/rand-to-pbest/1 mutation operator '''
    p = kwargs.get("p")
    if p is None:
        p = 0.2
    ind_r1, ind_r2, ind_r3 = pop._parent_select(target_idx, k=3)
    pop.get_F()
    sorted_idxs = np.argsort(np.array(pop.F))
    top_p = int(np.ceil(pop.size * p))
    idxs = sorted_idxs[:top_p]
    pop.rng.shuffle(idxs)
    ind_pbest = pop.list_ind[idxs[0]]
    # K = pop.rng.rand()
    mutant_vec = (ind_r1.xvalue
                    + F * (ind_pbest.xvalue - ind_r1.xvalue)
                    + F * (ind_r2.xvalue - ind_r3.xvalue))
    return mutant_vec



#!------------------------------------------------------------------------------
#!                                Crossover Operators
#!------------------------------------------------------------------------------
# Reference: [Book] Computational Intelligence: An Introduction, page239-240
def bin_crossover(pop, target_idx, mutant_vec, CR):
    ''' Binomial crossover operator '''
    target_vec = pop.list_ind[target_idx].xvalue
    dimension = len(mutant_vec)
    if target_vec.size != mutant_vec.size:
        raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                            " equal")
    # 1. Determine crossover points (indices)
    crossover_points = []
    ensured_idx = pop.rng.randint(0, dimension)
    crossover_points.append(ensured_idx)
    for j in range(dimension):
        if pop.rng.rand() < CR and j != ensured_idx:
            crossover_points.append(j)
    # 2. Do crossover
    trial_vec = np.copy(target_vec)
    for idx in crossover_points:
        trial_vec[idx] = mutant_vec[idx]
    return trial_vec


def exp_crossover(pop, target_idx, mutant_vec, CR):
    ''' Exponential crossover operator '''
    target_vec = pop.list_ind[target_idx].xvalue
    dimension = len(mutant_vec)
    if target_vec.size != mutant_vec.size:
        raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                            " equal")
    # 1. Determine crossover points (indices)
    crossover_points = []
    idx = pop.rng.randint(0, dimension)
    while True:
        crossover_points.append(idx)
        idx = (idx + 1) % dimension
        if (pop.rng.rand() >= CR
                or len(crossover_points) == dimension):
            break
    # 2. Do crossover
    trial_vec = np.copy(target_vec)
    for idx in crossover_points:
        trial_vec[idx] = mutant_vec[idx]
    return trial_vec


def none_crossover(pop, target_idx, mutant_vec, CR):
    ''' No crossover is used, directly use the mutant as trial vector '''
    target_vec = pop.list_ind[target_idx].xvalue
    if target_vec.size != mutant_vec.size:
        raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                            " equal")
    trial_vec = np.copy(mutant_vec)
    return trial_vec


def eig_crossover(pop, target_idx, mutant_vec, CR):
    '''Eigenvector-Based Crossover
    Ref: S. Guo and C. Yang, "Enhancing Differential Evolution Utilizing 
    Eigenvector-Based Crossover Operator," in IEEE Transactions on 
    Evolutionary Computation, vol. 19, no. 1, pp. 31-49, Feb. 2015.'''
    # if eig vector not exit, calculate them, else do eig crossover directly.
    if pop._Q is None or pop._Qct is None:
        pop._calculate_CovMat_of_pop()
    # transform the tatget_vec and mutant_vec into eigenvector basis spacce:
    target_vec = pop.list_ind[target_idx].xvalue
    eig_target_vec = np.dot(pop._Q, target_vec)
    eig_mutant_vec = np.dot(pop._Q, mutant_vec)
    # do bin crossover on eig_target_vec and eig_mutant_vec
    dimension = len(eig_mutant_vec)
    if eig_target_vec.size != eig_mutant_vec.size:
        raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                            " equal")
    # Determine crossover points (indices)
    crossover_points = []
    ensured_idx = pop.rng.randint(0, dimension)
    crossover_points.append(ensured_idx)
    for j in range(dimension):
        if pop.rng.rand() < CR and j != ensured_idx:
            crossover_points.append(j)
    # Do crossover
    eig_trial_vec = np.copy(eig_target_vec)
    for idx in crossover_points:
        eig_trial_vec[idx] = eig_mutant_vec[idx]
    # transform the eig_trial_vec into original space
    trial_vec = np.dot(pop._Qct, eig_trial_vec)
    return np.array(trial_vec)



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    pass


if __name__ == "__main__":
    main()
