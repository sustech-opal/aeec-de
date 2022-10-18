#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Optimization algorithms
'''
import numpy as np
import matplotlib.pyplot as plt
from concurrent import futures
from copy import deepcopy
# internal imports
import utils
import EAs.tools as tools
import EAs.parameterize as para
from EAs.problems.problem import Benchmark
from EAs.individual import Individual
from EAs.population import DEpopulation, ESpopulation, Swarm
# HB : the following imports are for personal purpose
try:
    import sys, IPython
    sys.excepthook = IPython.core.ultratb.ColorTB()
except:
    pass

# import warnings
# np.seterr(all="raise")


#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------
class __base(object):
    ''' :class:`__base` is the base class for defining an optimization method.
    '''
    OUTPUT_ITEMS = ["nth", "FES", "x_best", "f_best", "f_mean", "x_opt", 
        "f_opt"]

    def __init__(self, opt_problem, algo_parameters, stop_conditions):
        '''
        Creates a new :class:`__base` for optimization methods.

        Parameters
        ----------
        opt_problem : instance of problem.SingleObject
            The optimization problem that will be solved by the current 
            algorithm
        algo_parameters : dict | instance of option.
            The parameter setting/configuration of the current algorithm
        stop_conditions : dict | instance of option.StopCondition
            Stopping conditions which determine when the algorithm terminates.
        '''
        self.pb = opt_problem
        self.para = algo_parameters 
        self.stop = stop_conditions
        if (self.pb.f_opt_theory and self.stop.delta_ftarget) is not None:
            self.stop.ftarget = self.pb.f_opt_theory + self.stop.delta_ftarget
        self._ALL_initial_schemes = {
            "latin_hypercube": tools.latin_hypercube_sampling,
            "random": tools.uniform_random_sampling}

#   ----------------------------- Setter/Getter -----------------------------
    @property
    def pb(self):
        return self._pb

    @pb.setter
    def pb(self, problem):
        if problem is not None:
            self._pb = problem
        else:
            raise TypeError("The optimization problem must be an instance of"
                " problem.SingleObject, not an instance of {}".format(
                type(problem).__name__))

    @property
    def para(self):
        return self._para

    @para.setter
    def para(self, hyperparameters):
        if isinstance(hyperparameters, self.CONFIG_TYPE):
            self._para = hyperparameters
        else:
            raise TypeError("The {} algorithm's configuration (i.e."
                " hyperparameters) must be an instance of {}, not an instance"
                " of {}".format(type(self).__name__, self.CONFIG_TYPE,
                type(hyperparameters)))

    @property
    def stop(self):
        return self._stop

    @stop.setter
    def stop(self, stop_condition):
        if isinstance(stop_condition, para.StopCondition):
            self._stop = stop_condition
        else:
            raise TypeError("The algorithm's stopping condition must be an"
                " instance of {}, not an instance of {}".format(
                para.StopCondition, type(stop_condition)))

#   ----------------------------- Public Methods -----------------------------
    #TODO HB: get criterion, iterations from config object
    def initialize_X0(self, initial_scheme, pop_size, criterion="classic", 
        iterations=5):
        ''' Initialize population with random canidate solutions. '''
        func = self._ALL_initial_schemes.get(initial_scheme)
        if func is None:
            raise ValueError("The `initial_scheme` must be one of"
                " 'latin_hypercube' or 'random', not {}".format(initial_scheme))
        else:
            X0 = func(pop_size, self.pb.D,
                self.pb.init_lower_bound, 
                self.pb.init_upper_bound,
                self.para.rng,
                criterion, iterations)
        return X0

    def terminate(self):
        ''' Check whether termination conditions are satisfied or not. 
        
        Returns
        -------
        termination : dict
            The satisfied termination conditions. Generally speaking it has a
            form of ``{'termination_reason':value, ...}``, for example
            ``{'ftarget':1e-8}``, or the an empty dictionary in case of
            non satisfaction.
        '''
        termination = {}
        if self.stop.max_FES is not None:
            if self.pb.num_calls >= self.stop.max_FES:
                termination['maxFE'] = self.stop.max_FES
        if self.stop.max_iter is not None:
            if self.pop.nth_generation >= self.stop.max_iter:
                termination['max_iter'] = self.stop.max_iter
        if self.stop.ftarget is not None:
            if self.pop.F_best <= self.stop.ftarget:
                termination['ftarget'] = self.stop.ftarget       
        return termination
    
    def reset(self):
        '''Reset the attributes of the optimizer.

        All variables/atributes will be re-initialized when this method is 
        called.

        '''
        # Initialize history lists
        self.history = {key: [] for key in self.OUTPUT_ITEMS}
        # Reset some counters used in optimization process
        self.pb.num_calls = 0
        self.pop.nth_generation = 0
        self.pb.x_opt = None
        self.pb.f_opt = None

    def save(self, filestring="", replace=True):
        if ".json" in filestring:
            utils.save_json(self.data, filestring, replace)
            return
        elif ".rng" in filestring:
            utils.save_binary(self.para.rng, filestring, replace)
            return
        elif filestring == "":
            filename1 = self.pb.name + "_" + self.ALGO_NAME + ".json"
            filename2 = self.pb.name + "_" + self.ALGO_NAME + ".rng"
        else:
            filename1 = str(filestring) + ".json"
            filename2 = str(filestring) + ".rng"
        utils.save_json(self.data, filename1, replace)
        utils.save_binary(self.para.rng, filename2, replace)

    def display_result(self, flag):
        if flag:
            print("Generation {0:5d}: f(x) = {1:.20f}, objective function has"
                  " been called {2} times".format(self.pop.nth_generation, 
                  self.pop.F_best, self.pb.num_calls))

    def plot(self, x, y, linestyle="-", label="", xlabel="", ylabel="", 
        title="", ax=None, legend=False, show=True):
        tools.plot(x=x, y=y, linestyle=linestyle, label=label, xlabel=xlabel, 
            ylabel=ylabel, title=title, ax=ax, legend=legend, show=show)

#   ----------------------------- Private Methods -----------------------------
    def _write_to_history(self, keys_list):
        available_items = {"nth": self.pop.nth_generation,
                           "FES": self.pb.num_calls,
                           "x_best": self.pop.X_best,
                           "f_best": self.pop.F_best,
                           "f_mean": self.pop.F_mean,
                           "x_opt": self.pb.x_opt,
                           "f_opt": self.pb.f_opt,
                           "pop_state": deepcopy(self.pop.state),
                           "pop_sum_dist": self.pop.sum_distance,
                           "pop_avg_dist": self.pop.avg_distance}
        for key in keys_list:
            self.history[key].append(available_items.get(key))

    def _collect_result(self):
        result = {}
        result["problem"] = self.pb.name
        result["config"] = self.para.current()
        result["stop_conditions"] = self.stop.current()
        result["stop_status"] = self.terminate()
        result["f_opt_theory"] = self.pb.f_opt_theory
        result["x_best"] = self.pop.X_best
        result["f_best"] = self.pop.F_best
        result["x_opt"] = self.pb.x_opt
        result["f_opt"] = self.pb.f_opt
        result["nth_hist"] = self.history.get("nth")
        result["FES_hist"] = self.history.get("FES")
        result["x_best_hist"] = self.history.get("x_best")
        result["f_best_hist"] = self.history.get("f_best")
        result["f_mean_hist"] = self.history.get("f_mean")
        result["x_opt_hist"] = self.history.get("x_opt")
        result["f_opt_hist"] = self.history.get("f_opt")
        if self.pb.f_opt_theory is not None:
            result["ferror_min"] = result["f_opt"] - self.pb.f_opt_theory
            result["ferror_hist"] = (np.array(result["f_opt_hist"])
                                     - self.pb.f_opt_theory)
        else:
            result["ferror_min"] = None
            result["ferror_hist"] = None
        return result


#   --------------------------- Evolution Strategy ---------------------------
class CMAES(__base):
    ''' :class:`CMAES` implements the Covariance Matrix Adaptation Evolution 
    Strategy (CMA-ES) algorithm
    '''
    ALGO_NAME = "CMAES"
    CONFIG_TYPE = para.CMAES

    def __init__(self, opt_problem, algo_parameters=None,
                 stop_conditions=para.StopCondition()):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.pop = ESpopulation()

    def solve(self, init_xmean=None, init_sigma=None, disp=True, plot=True):
        """ 
        Run CMA-ES algorithm to solve the given problem. 
        
        Parameters
        ----------- 
        init_xmean: float, array
            the starting point of CMA-ES, i.e., the initial mean of sampling.
            
        init_sigma: float
            the intial value of sigma (step-size).
            
        maxFE: int, default=1e4
            the maximum number of fitness function evaluations during the run.
            
        delta_ftarget: float, default=1e-8
            the precision to reach, that is, the difference to the optimum fitness function value(fopt).
            ftarget = delta_ftarget + fpara. If delta_ftarget = None, only use maxFE as the stop condition.
            
        Returns
        -------
        result_data: dict, save all the required data of this run.   
            the result_data is collected by method postprocess().

        Notes
        -----  
        After instantiate the CMAES, it can be run multiple times. At the 
        beginning of each run, all the info/data related to current run will be 
        initialized.
        """
        # Set initial start point and initial step-size:
        if init_xmean is None:
            init_xmean = (self.pb.init_lower_bound + self.para.rng.rand(self.pb.D)
                * (self.pb.init_upper_bound - self.pb.init_lower_bound))
        if init_sigma is None:
            init_sigma = 0.3 * (self.pb.init_upper_bound - self.pb.init_lower_bound).max()

        # 1. Initialization -----------------------:
        self._initialize(init_xmean, init_sigma) 
        self._write_to_trace()
        self.reset() 

        # 2. Main Loop of evolution ---------------:
        while True:
            if (self.stop.max_FES - self.pb.num_calls) >= self.para.N:
                self._evolve_a_generation() # evolve one generation
                self.pop.find_best_individual()
                self.pb.update_opt(self.pop.X_best, self.pop.F_best)
                self._write_to_history(self.OUTPUT_ITEMS)
                self._write_to_trace()
                self.display_result(disp)
                self.pop.nth_generation += 1
            else:
                # sample new population
                self.pop.fill_X(X=self.__sample_population(
                    pop_size = int(self.stop.max_FES - self.pb.num_calls)))  
                # evalaute each individual in population
                for ind in self.pop:
                    ind.xvalue, ind.fvalue = self.pb.evaluate(ind.xvalue)
                self.pop.find_best_individual()
                self.pb.update_opt(self.pop.X_best, self.pop.F_best)
                self._write_to_history(self.OUTPUT_ITEMS)
                self.display_result(disp)
                self.pop.nth_generation += 1
            # Update the population's attributes
            self.pop.get_F()
            self.pop.get_X()
            if self.terminate() or (self.condition_number > 
                self.stop.condition_limit):
                break
        self.data = self._collect_result()
        return self.pb.x_opt, self.pb.f_opt

    def _initialize(self, init_xmean, init_sigma):
        """ Initialize dynamic state variables, counters, & traces. """
        # Initialize dynamice strategy parameters:
        self.xmean = init_xmean
        self.sigma = init_sigma
        self.ps = np.zeros(self.pb.D, dtype=np.float)
        self.pc = np.zeros(self.pb.D, dtype=np.float)
        self.B = np.eye(self.pb.D, self.pb.D, dtype=np.float)
        self.D = np.eye(self.pb.D, self.pb.D, dtype=np.float)
        self.C = self.B @ self.D @ (self.B @ self.D).T
        self.invsqrtC = self.B @ (np.linalg.inv(self.D)) @ self.B.T
        self.condition_number = 1.0

        # Initialize traces:
        self.trace = {}  # traces to record info about Covariance Matrix of each generation
        self.trace['xmean'] = []
        self.trace['sigma'] = []
        self.trace['ps'] = []
        self.trace['pc'] = []
        self.trace['B'] = []
        self.trace['D'] = []
        self.trace['C'] = []
        self.trace['invsqrtC'] = []
        self.trace['condition_number'] = []

        # Initialize counters:
        self.EIG_counter = 0    # number of eigendecomposition of C

        # Update the population's attributes
        self.pop.get_F()
        self.pop.get_X()

    def _evolve_a_generation(self):
        """ Evolve one generation, which involves:
               - Sample new population (Mutation)
               - Evalaute the population
               - Selection and recombination
               - Covariance Matrix Adaptation
               - Step-size Conotrol. 
               
            *** ALL dynamic parameters/varibales are updated during evolve. ***
             """
        self.pop.fill_X(X=self.__sample_population(pop_size=self.para.N))  # sample new population
        # evalaute each individual in population
        for ind in self.pop:
            ind.xvalue, ind.fvalue = self.pb.evaluate(ind.xvalue)
        xold = self.xmean
        parents = self.pop.select_best_individuals(k=self.para.mu) # Selection
        self.xmean = self.__recombine(parents)  # Recombination

        # Cumulation: Update evolution paths
        self.ps = ( (1 - self.para.cs) * self.ps + np.sqrt(self.para.cs
            * (2 - self.para.cs) * self.para.MU_EFF) * self.invsqrtC
            @ ((self.xmean - xold) / self.sigma) )
        
        cond1 = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.para.cs)
                ** (2 * (self.pop.nth_generation + 1)))
        cond2 = (1.4 + 2 / (self.pb.D + 1)) * self.para.CHI_N
        #// hsig = ( np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.para.cs)
        #//     ** (2 * (self.pop.nth_generation + 1))) / self.para.CHI_N < 1.4
        #//     + 2 / (self.pb.D + 1) )
        if cond1 < cond2:
            hsig = 1
        else:
            hsig = 0

        self.pc = ( (1 - self.para.cc) * self.pc + hsig * np.sqrt(self.para.cc
            * (2 - self.para.cc) * self.para.MU_EFF)
            * ((self.xmean - xold) / self.sigma) )

        # Covariance Matrix Adaptation
        artmp = ( (1 / self.sigma)
                   * (np.array([parent.xvalue for parent in parents]) - xold) )
        self.C =( (1 - self.para.c1 - self.para.cmu) * self.C + self.para.c1
            * (self.pc[:, None] @ self.pc[None, :] + (1 - hsig) * self.para.cc 
            * (2 - self.para.cc) * self.C) + self.para.cmu
            * artmp.T @ np.diag(self.para.weights) @ artmp )

        # Step-size Control
        self.sigma = ( self.sigma * np.exp((self.para.cs / self.para.damps)
            * (np.linalg.norm(self.ps) / self.para.CHI_N - 1)) )

        # Decomposition of C into B*diag(D**2)*B.T (diagonalization)
        cond1 = self.pb.num_calls - self.EIG_counter
        cond2 = self.para.N / (self.para.c1+self.para.cmu) / self.pb.D / 10

        if cond1 > cond2:
            self.EIG_counter = self.pb.num_calls
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            d, self.B = np.linalg.eigh(self.C)
            #//d = d.real
            #//self.B = self.B.real
            self.D = np.diag(np.sqrt(d))
            self.invsqrtC = self.B @ (np.linalg.inv(self.D)) @ self.B.T
            self.condition_number = np.max(d) / np.min(d)
    

    def _write_to_trace(self):
        """ Records data in 'self.traces', which is a dict. """
        self.trace['xmean'].append(self.xmean)
        self.trace['sigma'].append(self.sigma)
        self.trace['ps'].append(self.ps)
        self.trace['pc'].append(self.pc)
        self.trace['B'].append(self.B)
        self.trace['D'].append(self.D)
        self.trace['C'].append(self.C)
        self.trace['invsqrtC'].append(self.invsqrtC)
        self.trace['condition_number'].append(self.condition_number)


    def plot_sigma(self):
        """ Plot some curves to illustrate the evolutionary search process.
        """
        # plot the evolution curve of step-size (sigma)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.trace['sigma'])
        ax.set(xlabel='Number of Generations', ylabel='$\sigma$ (Step-Size)',
               title='Step-Size Adapatation')
        plt.show()

    # --------------- Priviate Methods for Evolutionary Search ----------------
    def __sample_population(self, pop_size):
        """ Create population vectors using multivariate normal sampling. """
        # list_ind = []
        X = []
        for k in range(pop_size):
            xvalue = (self.xmean + self.sigma
                * self.B @ self.D @ self.para.rng.randn(self.pb.D))
            X.append(xvalue)
        return X

    def __recombine(self, parents):
        """ Recombinnation
        """
        parents_vec = [parent.xvalue for parent in parents]
        new_xmean = self.para.weights @ (np.array(parents_vec))
        return new_xmean


#TODO CWH: inspect the PSO algo
#   ----------------------------- Particle Swarm -----------------------------
class PSO(__base):
    ''' :class:`PSO` implements the Particle Swarm Optimization (PSO) 
    optimization method '''
    ALGO_NAME = "PSO"
    CONFIG_TYPE = para.PSO
    OUTPUT_ITEMS = ["nth", "FES", "x_best", "f_best", "f_mean", "x_opt", 
        "f_opt", "w", "c1", "c2"]

    def __init__(self, opt_problem, algo_parameters=para.PSO(),
                 stop_conditions=para.StopCondition()):
        super(PSO, self).__init__(opt_problem, algo_parameters, stop_conditions)
        self.pop = Swarm(random_number_generator=self.para.rng)
        # construct the initial, min, max velocity vectors
        self.V_init = np.array([self.para.v_init] * self.pb.D)
        self.V_min = np.array([self.para.v_min] * self.pb.D)
        self.V_max = np.array([self.para.v_max] * self.pb.D)



    def solve(self, disp=True, plot=True):
        self._initiate_generation_0(disp)
        # Main loop of evolution -----------------------------------------------
        while self.terminate() == {}:
            self._evolve_a_generation()
            self.pop.nth_generation += 1
            # Update the population's attributes
            self.pop.get_F()
            self.pop.get_X()
            # Update history
            self.pb.update_opt(self.pop.X_best, self.pop.F_best)
            self._write_to_history(self.OUTPUT_ITEMS)
            self.display_result(disp)
        # Collect result data for output ---------------------------------------
        self.data = self._collect_result()
        return self.pop.X_best, self.pop.F_best

    def _initiate_generation_0(self, disp):
        self.reset()
        X0 = self.initialize_X0(initial_scheme=self.para.initial_scheme,
                                pop_size=self.para.N)
        self.pop.initialize(X0, self.V_init, self.V_min, 
            self.V_max)
        for ind in self.pop:
            ind.xvalue, ind.fvalue = self.pb.evaluate(ind.xvalue)
            ind.update_own_best()
        self.pop.find_best_particle(topology=self.para.topology,
            all_time_best=self.para.all_time_best,
            n_neighbors=self.para.n_neighbors)
        self.pb.update_opt(self.pop.X_best, self.pop.F_best)
        # Update the population's attributes
        self.pop.get_F()
        self.pop.get_X()
        self._write_to_history(self.OUTPUT_ITEMS)
        self.display_result(disp)

    def _evolve_a_generation(self):
        if self.para.update_scheme.lower() == "immediate":
            for ind in self.pop:
                # check the comsumed FES:
                if self.pb.num_calls >= self.stop.max_FES:
                    break
                # move the particle:
                self.pop.move_particle(topology=self.para.topology,
                                       particle=ind,
                                       w=self.para.w,
                                       c1=self.para.c1,
                                       c2=self.para.c2,
                                       bounds_method=self.pb.bounds_method,
                                       vmin=self.V_min,
                                       vmax=self.V_max)
                ind.xvalue, ind.fvalue = self.pb.evaluate(ind.xvalue)
                # update the particle's best:
                ind.update_own_best()
                # update IMMEDIATELY the new best position:
                self.pop.find_best_particle(topology=self.para.topology,
                    all_time_best=self.para.all_time_best,
                    n_neighbors=self.para.n_neighbors)
        elif self.para.update_scheme.lower() == "deferred":
            for ind in self.pop:
                if self.pb.num_calls >= self.stop.max_FES:
                    break
                self.pop.move_particle(topology=self.para.topology,
                                       particle=ind,
                                       w=self.para.w,
                                       c1=self.para.c1,
                                       c2=self.para.c2,
                                       bounds_method=self.pb.bounds_method,
                                       vmin=self.V_min,
                                       vmax=self.V_max)
                ind.xvalue, ind.fvalue = self.pb.evaluate(ind.xvalue)
                ind.update_own_best()
            # update ONLY ONCE per iteration:
            self.pop.find_best_particle(topology=self.para.topology,
                all_time_best=self.para.all_time_best,
                n_neighbors=self.para.n_neighbors)
        else:
            raise NameError("`update_scheme` is invalid, please choose"
                " between 'immediate' and 'deferred'")

    def _write_to_history(self, keys_list):
        available_items = {"nth": self.pop.nth_generation,
                           "FES": self.pb.num_calls,
                           "x_best": self.pop.X_best,
                           "f_best": self.pop.F_best,
                           "f_mean": self.pop.F_mean,
                           "x_opt": self.pb.x_opt,
                           "f_opt": self.pb.f_opt,
                           "w": self.para.w,
                           "c1": self.para.c1,
                           "c2": self.para.c2}
        for key in keys_list:
            self.history[key].append(available_items.get(key))

    def _collect_result(self):
        result = super(PSO, self)._collect_result()
        result["w"] = (self.history.get("w"))
        result["c1"] = (self.history.get("c1"))
        result["c2"] = (self.history.get("c2"))
        return result



#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
@utils.timer
def main():
    case = "PSO"
    
    # 1. Problem Configuration
    D = 2
    problem = Benchmark(benchmark_set="cec2017", D=D, funID=5, instanceID=1)

    # 2. Algorithm Configuration
    # --- [Option 1] by using the default (optimized) hyperparameters
    config1  = para.DE()
    # --- [Option 2] by using generalized <method set> (process-oriented)
    config2 = para.DE(seed=1)
    config2.set("N", 100)
    config2.set("F", 0.8)
    config2.set("CR", 0.9)
    config2.set("mutation", "de/current/1")
    config2.set("crossover", "exp")
    config2.set("initial_scheme", "latin_hypercube")
    config2.set("update_scheme", "immediate")
    # --- [Option 3] by assigning directly the value to each hyperparameter 
    # (object-oriented)
    config3 = para.DE()
    config3.N = 100
    config3.F = 0.8
    config3.CR = 0.5
    config3.mutation = "de/current-to-rand/1"
    config3.crossover = "exp"
    config3.initial_scheme = "random"
    config3.update_scheme = "immediate"
    # --- [Option 4] by defining when instantiating the object
    config4 = para.DE(seed = 1,
                      N = 30,
                      F = 1.7,
                      CR = 0.1,
                      mutation_scheme = "de/current-to-rand/1",
                      crossover_scheme = "exp",
                      initial_scheme = "latin_hypercube",
                      update_scheme = "deferred")
    # --- Defining stop conditions
    stop = para.StopCondition(max_FES=1e5*D, max_iter=None,
                              delta_ftarget=1e-8)

    # if case == "DE" or case == 1 or case == "ALL":
    #     stop.max_FES = np.inf
    #     stop.max_iter = None
    #     stop.delta_ftarget = 1e-8

    #     # 3. Aglorithm Selection: using DE basic algorithm
    #     optimizer = DE(opt_problem=problem,
    #                    algo_parameters=config2, stop_conditions=stop)

    #     # 4. Problem Solving
    #     results = optimizer.solve(disp=True, plot=True)
    
    # if case == "DERandPop" or case == 2 or case == "ALL":
    #     # 3. Aglorithm Selection: using DE Random Population algorithm
    #     optimizer = DERandomPopulation(opt_problem=problem,
    #         algo_parameters = config2, stop_conditions = stop,
    #         random_para_names = ["F","mutation","CR","crossover"])

    #     # 4. Problem Solving
    #     results = optimizer.solve(disp=True, plot=True)
    #     optimizer.plot_hist(bins=20, density=False, cumulative=False)
    
    # if case == "DERandInd" or case == 3 or case == "ALL":
    #     # 3. Aglorithm Selection: using DE Random Individual algorithm
    #     optimizer = DERandomIndividual(opt_problem=problem,
    #         algo_parameters=config3, stop_conditions=stop,
    #         random_para_names=["crossover", "CR", "F"])

    #     # 4. Problem Solving
    #     results = optimizer.solve(disp=True, plot=False)
    
    # if case == "DEComposite" or case == 4 or case == "ALL":
    #     # 2. Algorithm Configuration
    #     generation_strategy = [{"mutation": "de/rand/1", "crossover": "bin"},
    #                 {"mutation": "de/rand/2", "crossover": "exp"},
    #                 {"mutation": "de/current-to-rand/1", "crossover": "none"}]
    #     F_CR_pair = [{"F": 1.0, "CR": 0.1}, 
    #                  {"F": 1.0, "CR": 0.9},
    #                  {"F": 0.8, "CR": 0.2}]
    #     config4.generation_strategy = generation_strategy
    #     config4.F_CR_pair = F_CR_pair
        
    #     # 3. Aglorithm Selection: using DE Random Population algorithm
    #     optimizer = DEComposite(opt_problem=problem,
    #         algo_parameters=config4, stop_conditions=stop)

    #     # 4. Problem Solving
    #     results = optimizer.solve(disp=0, plot=1)

    if case == "CMAES" or case == 5 or case == "ALL":
        # 2. Algorithm Configuration
        config5 = para.CMAES(1, D)
        stop.delta_ftarget=None

        # 3. Aglorithm Selection: using DE Random Population algorithm
        optimizer = CMAES(opt_problem=problem,
            algo_parameters=config5, stop_conditions=stop)

        # 4. Problem Solving
        results = optimizer.solve(disp=True, plot=True)
        optimizer.plot_sigma()
    
    if case == "PSO" or case == 6 or case == "ALL":
        # 2. Algorithm Configuration
        config6 = para.PSO(seed=1, w=0.7, c1=0.5, c2=0.5, v_min=-1e4, v_max=1e4,
            topology="ring", n_neighbors=2)

        # 3. Aglorithm Selection: using DE basic algorithm
        optimizer = PSO(opt_problem=problem,
                        algo_parameters=config6, stop_conditions=stop)

        # 4. Problem Solving
        results = optimizer.solve(disp=True, plot=True)

    # 5. Post-processing
    output = optimizer.data
    # print("Initial f_best:", output.get("f_best_hist")[0])
    print("Calculated results:", results)
    print("Theoretical optimal value:", problem.f_opt_theory)
    # return
    optimizer.save() # save to file
    # [option 1] by using external module (process-oriented)
    tools.plot(x=output.get("FES_hist"),
            y=np.log10(output.get("ferror_hist")),
            xlabel="FES", ylabel="$log_{10} (f-f^*)$",
            # title="{}/{}-{} on {}".format(
            #     output.get("config")["mutation"],
            #     output.get("config")["crossover"],
            #     output.get("config")["update_scheme"],
            #     output.get("problem"),)
                )
    if case in (2,3,4):
        # [option 2] by using internal method (object-oriented)
        optimizer.plot_hist(bins=20, density=False, cumulative=False)
        
    return output


if __name__ == "__main__":
    data = main()

