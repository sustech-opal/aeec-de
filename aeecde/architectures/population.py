#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Populations related classes and functions
'''
import numpy as np
# internal imports
from .individual import Individual, Particle
from aeecde.operators import DE



#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------
class __base(object):
    ''' :class:`__base` is the base class for defining other population classes.
    '''

    def __init__(self, list_of_individuals, nth_generation,
        random_number_generator):
        '''
        Creates a new :class:`__base` for populations.

        Parameters
        ----------
            To be specified in its successors

        Returns
        -------
        A population object.
        '''
        self.list_ind = list_of_individuals
        self.nth_generation = nth_generation
        self.rng = random_number_generator
        self.state = {"explore":None, "exploit":None, "neutral":None}
        self._sum_distance = None
        self._avg_distance = None

    def __repr__(self):
        line1 = "A population of {} class including {} individuals:\n".format(
            type(self).__name__, self.size)
        line2 = "\n".join([str(ind) for ind in self.list_ind])
        return line1 + line2

    def __len__(self):
        return self._size

    def __add__(self, other):
        '''
        Stack another population to the current one.

        Parameters
        ----------
        other : instance of a population class
            A population object that belongs to the same class of the other one

        Returns
        -------
        output : instance of a population class
            A new population object including all individuals from the two
            previous objects
        '''
        if isinstance(other, type(self)):
            return self.__class__(self.list_ind + other.list_ind, -1)
        else:
            raise TypeError("Cannot stack a {} population to a {} population"
                .format(type(self).__name__, type(other).__name__))

    def __iter__(self):
        ''' Return the iterator object'''
        return iter(self.list_ind)

    @property
    def sum_distance(self):
        return self._sum_distance

    @property
    def avg_distance(self):
        return self._avg_distance

    @property
    def size(self):
        return self._size

    @property
    def list_ind(self):
        return self._list_ind

    @list_ind.setter
    def list_ind(self, value):
        if isinstance(value, list):
            self._list_ind = value
            self._size = len(self._list_ind)
        else:
            raise TypeError("{} object's `list_ind` attribute must be a list,"
                " not a {}".format(type(self).__name__, type(value).__name__))

    @property
    def nth_generation(self):
        return self._nth_generation

    @nth_generation.setter
    def nth_generation(self, value):
        if isinstance(value, (int, float)):
            self._nth_generation = int(value)
        else:
            raise TypeError("{} object's `nth_generation` attribute must be an"
                " integer or a float, not a {}".format(type(self).__name__,
                type(value).__name__))

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, value):
        if value is None or isinstance(value, np.random.RandomState):
            self._rng = value
        else:
            raise TypeError("{} object's `random_number_generator` attribute"
                " must be None or an instance of np.random.RandomState, not an"
                " instance of {}".format(type(self).__name__, type(value)))

    @property
    def F(self):
        return self._F

    @property
    def X(self):
        return self._X

    @property
    def F_mean(self):
        return np.mean(self.F)

    @property
    def X_mean(self):
        return np.mean(self.X, axis=0)

    @property
    def X_best(self):
        return self._X_best

    @property
    def F_best(self):
        return self._F_best

    @property
    def ind_best(self):
        return self._ind_best

    @property
    def list_sorted(self):
        return self._list_sorted

    def find_best_individual(self):
        self._list_sorted = sorted(self.list_ind, key=get_f,
            reverse=False)
        self._ind_best = self._list_sorted[0]
        self._X_best = self._ind_best.xvalue
        self._F_best = self._ind_best.fvalue
        return self._ind_best

    def add(self, individual):
        if isinstance(individual, type(self.list_ind[0])):
            self.list_ind.append(individual)
        else:
            raise TypeError("Cannot add a {} to a population of {} class"
                .format(type(individual), type(self.list_ind[0]).__name__))

    def delete(self, index=None, xvalue=None, fvalue=None):
        '''
        Delete an individual from population through 3 options: index, xvalue or
        fvalue

        Parameters
        ----------
        index : int
            An index used to locate in the list of individuals
        xvalue : 2D array of shape (1,?)
            A optimized variable
        fvalue : float
            A value of objective function
        '''
        if index is not None:
            del self.list_ind[index]
        elif xvalue is not None:
            self.list_ind = list(
                filter(lambda ind: (ind.xvalue != xvalue).any(), self.list_ind))
        elif fvalue is not None:
            self.list_ind = list(
                filter(lambda ind: ind.fvalue != fvalue, self.list_ind))
        else:
            raise ValueError("You must indicate which individual to be removed"
                " by providing its `index`, or `xvalue`, or `fvalue`")

    def get_X(self):
        self._X = np.array([ind.xvalue for ind in self.list_ind])

    def get_F(self):
        self._F = np.array([ind.fvalue for ind in self.list_ind])

    def get_ind_distance(self, calculator):
        return calculator(self)

    def get_sum_distance(self):
        self._sum_distance = np.sum([ind.distance for ind in self.list_ind])

    def get_avg_distance(self):
        self._avg_distance = self.sum_distance / self.size

    def get_state(self):
        N_explore, N_exploit, N_neutral = 0, 0, 0
        #TODO: not finished

class Swarm(__base):
    ''' :class:`Swarm` is the definition of swarm in PSO methods.
    '''

    def __init__(self, list_of_particles=[], nth_generation=0,
        random_number_generator=None):
        '''
        Creates a new :class:`Swarm` of PSO algorithm.

        Parameters
        ----------
        list_of_particles : list
            A list of objects belonging to an individual/particle class.
        nth_generation : int [default=0]
            A integer number to indicate the N-th generation of population. (If
            this value is -1, it represents a temporary population that shoud
            be eliminated after usage)
        '''
        super(Swarm, self).__init__(list_of_particles, nth_generation,
            random_number_generator)
        self.list_par = list_of_particles
        # Available topologies:
        self._ALL_topologies_find = {
            "star": self._find_in_star,
            "ring": self._find_in_ring,
            "von_neumann": self._find_in_von_neumann,
            "random": self._find_in_random}
        self._ALL_topologies_move = {
            "star": self._move_in_star,
            "ring": self._move_in_ring}

    @property
    def list_par(self):
        return self._list_par

    @list_par.setter
    def list_par(self, value):
        self.list_ind = value
        self._list_par = self.list_ind

##   -----------------------------  PSO Operators -----------------------------
    def initialize(self, X0, vinit, vmin=None, vmax=None):
        if None in vinit:
            self.list_ind = [
            Particle(position=x, velocity=self.rng.uniform(low=vmin, high=vmax))
            for x in X0]
        else:
            self.list_ind = [Particle(position=x, velocity=vinit) for x in X0]

    def update_all_particles(self):
        [ind.update_own_best() for ind in self.list_ind]

    def find_best_particle(self, topology, all_time_best, **kwargs):
        func = self._ALL_topologies_find.get(topology.lower())
        if func is None:
            raise NameError("`topology` is invalid, please choose between"
                " 'star' and 'ring'")
        else:
            if all_time_best:
                func("f_best", kwargs)
            else:
                func("fvalue", kwargs)

    def _find_in_star(self, attrib, kwargs):
        self._list_sorted = sorted(self.list_ind,
            key=lambda ind: getattr(ind, attrib), reverse=False)
        self._ind_best = self._list_sorted[0]
        self._X_best = self._ind_best.xvalue
        self._F_best = self._ind_best.fvalue
        return self._ind_best

    def _find_in_ring(self, attrib, kwargs):
        temp_ind_best, temp_X_best, temp_F_best = [], [], []
        n_neighbors = kwargs.get("n_neighbors")
        temp = self.list_ind + self.list_ind[:n_neighbors]
        chunks = [temp[i:i+n_neighbors+1] for i in range(self.size)]
        for i in range(self.size):
            ind_best = sorted(chunks[i],
                key=lambda ind: getattr(ind, attrib), reverse=False)[0]
            temp_ind_best.append(ind_best)
            temp_X_best.append(ind_best.xvalue)
            temp_F_best.append(ind_best.fvalue)
        # save to attribute
        self.X_best_ring = temp_X_best
        # get the best particle in all swarm
        self._F_best = min(temp_F_best)
        index = temp_F_best.index(self._F_best)
        self._X_best = temp_X_best[index]
        self._ind_best = temp_ind_best[index]
        return self._ind_best

    def _find_in_von_neumann(self, attrib):
        pass

    def _find_in_random(self, attrib):
        pass

    def move_particle(self, topology, particle, w, c1, c2, bounds_method, vmin,
        vmax, **kwargs):
        func = self._ALL_topologies_move.get(topology.lower())
        if func is None:
            raise NameError("`topology` is invalid, please choose between"
                            " 'star' and 'ring'")
        else:
            func(particle, w, c1, c2, bounds_method, vmin, vmax)

    def _move_in_star(self, particle, w, c1, c2, bounds_method, vmin, vmax):
        v = particle.update_velocity(w, c1*self.rng.rand(), c2*self.rng.rand(),
            self.X_best)
        particle.velocity = bounds_method(v, vmin, vmax)
        particle.update_position()

    def _move_in_ring(self, particle, w, c1, c2, bounds_method, vmin, vmax):
        index = self.list_ind.index(particle)
        v = particle.update_velocity(w, c1*self.rng.rand(), c2*self.rng.rand(),
            self.X_best_ring[index])
        particle.velocity = bounds_method(v, vmin, vmax)
        particle.update_position()


class DEpopulation(__base):
    ''' :class:`DEpopulation` is the definition of population in DE methods.
    '''

    def __init__(self, list_of_individuals=[], nth_generation=0,
        random_number_generator=None):
        '''
        Creates a new :class:`DEpopulation` of DE algorithm.

        Parameters
        ----------
        list_of_individuals : list
            A list of objects belonging to an individual class.
        nth_generation : int [default=0]
            A integer number to indicate the N-th generation of population. (If
            this value is -1, it represents a temporary population that shoud
            be eliminated after usage)
        random_number_generator : instance of np.random.RandomState
            [default=None]
            A random number generator.
        '''
        super(DEpopulation, self).__init__(list_of_individuals, nth_generation,
            random_number_generator)
        # Available mutation and crossover operators:
        self._ALL_mutation_schemes = {
            "de/rand/1": DE.rand_1,
            "de/rand/2": DE.rand_2,
            "de/best/1": DE.best_1,
            "de/best/2": DE.best_2,
            "de/current/1": DE.current_1,
            "de/current-to-rand/1": DE.current_to_rand_1,
            "de/current-to-best/1": DE.current_to_best_1,
            "de/current-to-best/2": DE.current_to_best_2,
            "de/current-to-pbest/1": DE.current_to_pbest_1,
            "de/rand-to-best/1": DE.rand_to_best_1,
            "de/rand-to-pbest/1": DE.rand_to_pbest_1,}
        # Available crossover operators:
        self._ALL_crossover_schemes = {
            "bin": DE.bin_crossover,
            "exp": DE.exp_crossover,
            "none": DE.none_crossover,
            "eig": DE.eig_crossover }
        # Attributes
        self._gen_hist = {"mutation": [], "crossover": [], "F": [], "CR": []}
        # some new attributes of population for Eigenvector based crossover:
        self._Cov = None # Covariance matrix of current population
        self._Q = None   # Matrix of eigenvectors of Cov, each col is a eigenvec
        self._Qct = None # Conjugate transpose of `self._Q`

    @property
    def nth_generation(self):
        return self._nth_generation

    @nth_generation.setter
    def nth_generation(self, value):
        # When `nth_generation` is updated, reset the Cov and Q as None
        self._Cov = None
        self._Q = None
        self._Qct = None
        if isinstance(value, (int, float)):
            self._nth_generation = int(value)
        else:
            raise TypeError("{} object's `nth_generation` attribute must be an"
                " integer or a float, not a {}".format(type(self).__name__,
                type(value).__name__))

    @property
    def gen_info(self):
        temp = {}
        temp["mutation"] = [ind.mutation for ind in self.list_ind]
        temp["crossover"] = [ind.crossover for ind in self.list_ind]
        temp["F"] = [ind.F for ind in self.list_ind]
        temp["CR"] = [ind.CR for ind in self.list_ind]
        return temp

##   ------------- Three Mainly used methods of DEpopulation class -------------
    def initialize(self, X0):
        self.list_ind = [Individual(solution=x) for x in X0]

    def create_offspring(self, target_idx, mutation, F, crossover, CR, **kwargs):
        mutant_vec = self._mutate(target_idx, mutation, F, **kwargs)
        trial = self._crossover(target_idx, mutant_vec, crossover, CR)
        trial.mutation, trial.F = mutation, F
        trial.crossover, trial.CR = crossover, CR
        return trial

##   -----------------------------  DE Operators -----------------------------
    def _parent_select(self, target_idx, k=5):
        '''
        Select k individuals for mutation from the current population.

        Parameters
        ----------
        target_idx : int within [0, self.size-1]
            Index of target vector (solution).
        k : int within [1, self.size-1]
            How many individuals are selected from the current population.

        Returns
        -------
            Return k selected individuals in a list.
        '''
        idxs = list(range(self.size))
        idxs.remove(target_idx)
        self.rng.shuffle(idxs)
        return [self.list_ind[idx] for idx in idxs[:k]]

    def _mutate(self, target_idx, mutation, F, **kwargs):
        ''' Perform mutation for target vector by using the specified
        mutation and scaling_factor (F)

        Parameters
        ----------
        target_idx : integer, within [0, self.size-1]
            Index of target vector (solution).
        mutation : str, in DEpopulation._available_mutation_operators
            The name of the used mutation operator.
        F : float, within (0.0, 2.0)
            The scaling factor used in mutation

        Returns
        -------
            Return a mutant vector generated by mutation.
        '''
        if F < 0.0 or F > 2.0:
            raise ValueError("The scale factor `F` must be in the interval of"
                " 0.0 <= `F` <= 2.0")
        func = self._ALL_mutation_schemes.get(mutation.lower())
        if func is None:
            raise NameError("`mutation` is invalid, please select a"
                " valid mutation operator")
        if mutation.lower() in ["de/current-to-pbest/1", "de/rand-to-pbest/1"]:
            return func(self, target_idx, F, **kwargs)
        else:
            return func(self, target_idx, F, )

    def _crossover(self, target_idx, mutant_vec, crossover, CR):
        ''' Perform crossover on given <individual> by using the specified
        mutation operator named as `self.crossver_operator`'''
        if CR < 0.0 or CR > 1.0:
            raise ValueError("The crossover rate `CR` must be in the interval"
                " of 0.0 <= `CR` <= 1.0")
        func = self._ALL_crossover_schemes.get(crossover.lower())
        if func is None:
            raise NameError("`crossover` is invalid, please select a"
                " valid crossover operator among 'bin', 'exp' and 'none'")
        trial_vec = func(self, target_idx, mutant_vec, CR,)
        trial_ind = self.list_ind[target_idx].clone()
        trial_ind.xvalue = trial_vec
        trial_ind.fvalue = None
        return trial_ind

    def survival_select(self, parent_ind, trial_ind):
        ''' Deterministic elitist replacement (parent vs. child) '''
        if trial_ind.fvalue <= parent_ind.fvalue:
            survival_ind = trial_ind
        else:
            survival_ind = parent_ind
        return survival_ind

##   ------------------------- DE Mutation Operators --------------------------
    # References:
    # [1] Comparison of mutation strategies in Differential Evolution-A
    # probabilistic perspective
    # [2] Differential evolution algorithm with ensemble of parameters and
    # mutation strategies

    # Validate in Ref[1]
    def __rand_1(self, target_idx, F):
        ''' DE/rand/1 mutation operator '''
        ind_r1, ind_r2, ind_r3 = self._parent_select(target_idx, k=3)
        mutant_vec = ( ind_r1.xvalue
                       + F * (ind_r2.xvalue - ind_r3.xvalue) )
        return mutant_vec

    # Validate in Ref[1]
    def __rand_2(self, target_idx, F):
        ''' DE/rand/2 mutation operator '''
        ind_r1, ind_r2, ind_r3, ind_r4, ind_r5 = self._parent_select(target_idx,
            k=5)
        mutant_vec = ( ind_r1.xvalue
                       + F * (ind_r2.xvalue - ind_r3.xvalue)
                       + F * (ind_r4.xvalue - ind_r5.xvalue) )
        return mutant_vec

    # Validate in Ref[1]
    def __best_1(self, target_idx, F):
        ''' DE/best/2 mutation operator '''
        ind_r1, ind_r2 = self._parent_select(target_idx, k=2)
        self.find_best_individual()
        mutant_vec = ( self.X_best
                       + F * (ind_r1.xvalue - ind_r2.xvalue) )
        return mutant_vec

    # Validate in Ref[1]
    def __best_2(self, target_idx, F):
        ''' DE/best/2 mutation operator '''
        ind_r1, ind_r2, ind_r3, ind_r4 = self._parent_select(target_idx, k=4)
        self.find_best_individual()
        mutant_vec = ( self.X_best
                       + F * (ind_r1.xvalue - ind_r2.xvalue)
                       + F * (ind_r3.xvalue - ind_r4.xvalue) )
        return mutant_vec

    # Validate in Ref[1]
    def __current_to_rand_1(self, target_idx, F):
        ''' DE/current-to-rand/1 mutation operator '''
        ind_r1, ind_r2, ind_r3 = self._parent_select(target_idx, k=3)
        K = self.rng.rand()
        mutant_vec = ( self.list_ind[target_idx].xvalue
                       + K * (ind_r1.xvalue
                       - self.list_ind[target_idx].xvalue)
                       + F * (ind_r2.xvalue - ind_r3.xvalue) )
        return mutant_vec

    # Validate in Ref[1]
    def __current_to_best_1(self, target_idx, F):
        ''' DE/current-to-best/1 mutation operator '''
        ind_r1, ind_r2 = self._parent_select(target_idx, k=2)
        self.find_best_individual()
        # K = self.rng.rand()
        mutant_vec = ( self.list_ind[target_idx].xvalue
                       + F * (self.X_best - self.list_ind[target_idx].xvalue)
                       + F * (ind_r1.xvalue - ind_r2.xvalue) )
        return mutant_vec

     # Validate in Ref[2] 'DE/target-to-best/2'
    def __current_to_best_2(self, target_idx, F):
        ''' DE/current-to-best/2 mutation operator '''
        ind_r1, ind_r2, ind_r3, ind_r4 = self._parent_select(target_idx, k=4)
        self.find_best_individual()
        K = self.rng.rand()
        mutant_vec = ( self.list_ind[target_idx].xvalue
                       + K * (self.X_best - self.list_ind[target_idx].xvalue)
                       + F * (ind_r1.xvalue - ind_r2.xvalue)
                       + F * (ind_r3.xvalue - ind_r4.xvalue) )
        return mutant_vec

    # Validate in Ref[1]
    def __current_1(self, target_idx, F):
        ''' DE/current/1 mutation operator '''
        ind_r1, ind_r2 = self._parent_select(target_idx, k=2)
        mutant_vec = ( self.list_ind[target_idx].xvalue
                       + F * (ind_r1.xvalue - ind_r2.xvalue) )
        return mutant_vec

    # Validate in Ref[1]
    def __current_to_pbest_1(self, target_idx, F, **kwargs):
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
            ind_r1, ind_r2 = self._parent_select(target_idx, k=2)
        else:
            assert isinstance(archive, list), "`archive` must be a list of ind"
            union = self.list_ind + archive
            idxs = list(range(self.size))
            idxs.remove(target_idx)
            self.rng.shuffle(idxs)
            r1 = idxs[0]
            ind_r1 = self.list_ind[r1]
            idxs_union = list(range(len(union)))
            idxs_union.remove(target_idx)
            idxs_union.remove(r1)
            self.rng.shuffle(idxs_union)
            r2 = idxs_union[0]
            ind_r2 = union[r2]
        # randonly select one ind from the p% best inds in pop
        self.get_F()
        sorted_idxs = np.argsort(self.F)
        top_p = int(np.ceil(self.size * p))
        idxs_top = sorted_idxs[:top_p]
        self.rng.shuffle(idxs_top)
        pbest = idxs_top[0]
        ind_pbest = self.list_ind[pbest]
        # K = self.rng.rand()
        mutant_vec = ( self.list_ind[target_idx].xvalue
                       + F *(ind_pbest.xvalue
                       - self.list_ind[target_idx].xvalue)
                       + F * (ind_r1.xvalue - ind_r2.xvalue) )
        return mutant_vec

    def __rand_to_pbest_1(self, target_idx, F, **kwargs):
        ''' DE/rand-to-pbest/1 mutation operator '''
        p = kwargs.get("p")
        if p is None:
            p = 0.2
        ind_r1, ind_r2, ind_r3 = self._parent_select(target_idx, k=3)
        self.get_F()
        sorted_idxs = np.argsort(np.array(self.F))
        top_p = int(np.ceil(self.size * p))
        idxs = sorted_idxs[:top_p]
        self.rng.shuffle(idxs)
        ind_pbest = self.list_ind[idxs[0]]
        # K = self.rng.rand()
        mutant_vec = ( ind_r1.xvalue
                       + F *(ind_pbest.xvalue - ind_r1.xvalue)
                       + F * (ind_r2.xvalue - ind_r3.xvalue) )
        return mutant_vec

    def __rand_to_best_1(self, target_idx, F):
        ''' DE/rand-to-best/1 mutation operator '''
        ind_r1, ind_r2, ind_r3 = self._parent_select(target_idx, k=3)
        self.find_best_individual()
        # K = self.rng.rand()
        mutant_vec = ( ind_r1.xvalue
                       + F * (self.X_best - ind_r1.xvalue)
                       + F * (ind_r2.xvalue - ind_r3.xvalue) )
        return mutant_vec

##   ------------------------- DE Crossover Operators -------------------------
    # Reference: [Book] Computational Intelligence: An Introduction, page239-240
    def __bin_crossover(self, target_idx, mutant_vec, CR):
        ''' Binomial crossover operator '''
        target_vec = self.list_ind[target_idx].xvalue
        dimension = len(mutant_vec)
        if target_vec.size != mutant_vec.size:
            raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                " equal")
        # 1. Determine crossover points (indices)
        crossover_points = []
        ensured_idx = self.rng.randint(0, dimension)
        crossover_points.append(ensured_idx)
        for j in range(dimension):
            if self.rng.rand() < CR and j != ensured_idx:
                crossover_points.append(j)
        # 2. Do crossover
        trial_vec = np.copy(target_vec)
        for idx in crossover_points:
            trial_vec[idx] = mutant_vec[idx]
        return trial_vec

    def __exp_crossover(self, target_idx, mutant_vec, CR):
        ''' Exponential crossover operator '''
        target_vec = self.list_ind[target_idx].xvalue
        dimension = len(mutant_vec)
        if target_vec.size != mutant_vec.size:
            raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                " equal")
        # 1. Determine crossover points (indices)
        crossover_points = []
        idx = self.rng.randint(0, dimension)
        while True:
            crossover_points.append(idx)
            idx = (idx + 1) % dimension
            if (self.rng.rand() >= CR
                or len(crossover_points)==dimension):
                break
        # 2. Do crossover
        trial_vec = np.copy(target_vec)
        for idx in crossover_points:
            trial_vec[idx] = mutant_vec[idx]
        return trial_vec

    def __none_crossover(self, target_idx, mutant_vec, CR):
        ''' No crossover is used, directly use the mutant as trial vector '''
        target_vec = self.list_ind[target_idx].xvalue
        if target_vec.size != mutant_vec.size:
            raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                " equal")
        trial_vec = np.copy(mutant_vec)
        return trial_vec

    def __eig_crossover(self, target_idx, mutant_vec, CR):
        '''Eigenvector-Based Crossover
        Ref: S. Guo and C. Yang, "Enhancing Differential Evolution Utilizing
        Eigenvector-Based Crossover Operator," in IEEE Transactions on
        Evolutionary Computation, vol. 19, no. 1, pp. 31-49, Feb. 2015.'''
        # if eig vector not exit, calculate them, else do eig crossover directly.
        if self._Q is None or self._Qct is None:
            self.__calculate_CovMat_of_pop()
        # transform the tatget_vec and mutant_vec into eigenvector basis spacce:
        target_vec = self.list_ind[target_idx].xvalue
        eig_target_vec = np.dot(self._Q, target_vec)
        eig_mutant_vec = np.dot(self._Q, mutant_vec)
        # do bin crossover on eig_target_vec and eig_mutant_vec
        dimension = len(eig_mutant_vec)
        if eig_target_vec.size != eig_mutant_vec.size:
            raise ValueError("The size of *target_vec* and *mutant_vec* must be"
                " equal")
        # Determine crossover points (indices)
        crossover_points = []
        ensured_idx = self.rng.randint(0, dimension)
        crossover_points.append(ensured_idx)
        for j in range(dimension):
            if self.rng.rand() < CR and j != ensured_idx:
                crossover_points.append(j)
        # Do crossover
        eig_trial_vec = np.copy(eig_target_vec)
        for idx in crossover_points:
            eig_trial_vec[idx] = eig_mutant_vec[idx]
        # transform the eig_trial_vec into original space
        trial_vec = np.dot(self._Qct, eig_trial_vec)
        return np.array(trial_vec)

    def _calculate_CovMat_of_pop(self):
        '''Calculate the Covariance Matrix of current population'''
        self.get_X()
        X = np.copy(self._X) # Each row is the xvalue of an individual, each col is a dimension
        COV = np.cov(X, rowvar=0)
        # Eigen decomposition of COV:
        ## COV is a symmetric matrix, so use eigh() method, which is more quick than eig()
        A, Q = np.linalg.eigh(COV) # A: eigenvalues; Q: matrix, each column is a eigenvector
        Qct =  (np.matrix(Q)).H
        self._Cov = COV
        self._Q = Q
        self._Qct = Qct


class ESpopulation(__base):
    ''' :class:`ESpopulation` is the definition of population in Evolution
    Strategy (ES) methods.
    '''

    def __init__(self, list_of_individuals=[], nth_generation=0,
        random_number_generator=None):
        '''
        Creates a new :class:`ESpopulation` of ES algorithm.

        Parameters
        ----------
        list_of_individuals : list
            A list of objects belonging to an individual class.
        nth_generation : int [default=0]
            A integer number to indicate the N-th generation of population. (If
            this value is -1, it represents a temporary population that shoud
            be eliminated after usage)
        seed : int | instance of np.random.RandomState, optional [default=None]
            Seed used for creating a random number generator. If it is None, the
            RandomState singleton of np.random will be used
        '''
        super(ESpopulation, self).__init__(list_of_individuals, nth_generation,
            random_number_generator)

    def fill_X(self, X):
        self.list_ind = [Individual(solution=vec) for vec in X]

    def select_best_individuals(self, k):
        self.find_best_individual()
        return self.list_sorted[0:k]



#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def get_f(object):
    return object.fvalue



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    case = "ALL"
    # --- Test Swarm
    if case == 1 or "ALL":
        a = Individual(fitness=1, solution=np.array([10., 20., 30.]))
        b = Individual(np.array([10., 20., 30.]), 2)
        c = Individual(np.array([10., 20., 30.]), 3)
        pop1 = Swarm([b, a, c])

        A = Individual(fitness=0.1, solution=np.array([100, 200, 300]))
        B = Individual(np.array([100, 200, 300]), 0.2)
        C = Individual(np.array([100, 200, 300]), 0.3)
        D = Individual(np.array([10, 200, 300]) , 0.4)
        E = Individual(np.array([100, 20, 300]) , 0.4)
        F = Individual(np.array([100, 200, 30]) , 0.4)
        pop2 = Swarm([B, A, C, D, E, F])

        newpop = (pop1 + pop2)
        print("[OK] add population (pop1 + pop2):\n", newpop)
        newpop.find_best_individual()
        print("[OK] newpop.ind_best:\n", newpop.ind_best)

        newpop.delete(index=1)
        print("[OK] delete an index=1:\n", newpop)
        newpop.delete(fvalue=2.0)
        print("[OK] delete a single fvalue=2.0:\n", newpop)
        newpop.delete(xvalue=np.array([[100, 200, 300]]))
        print("[OK] delete multiple xvalue=[100, 200, 300]:\n", newpop)
        newpop.delete(fvalue=0.4)
        print("[OK] delete multiple fvalue=0.4:\n", newpop)

        aa = Individual(np.array([-10, -20, -30]), -1)
        newpop.add(aa)
        print("[OK] add individual newpop.add(aa):\n", newpop)

        try:
            bb = Particle()
            newpop.add(bb)
        except Exception as e:
            print("[OK] #1 Error message:", e)

if __name__ == '__main__':
        main()
