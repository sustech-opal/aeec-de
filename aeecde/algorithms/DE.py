#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Differential Evolution
'''
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import cauchy
from scipy.stats import norm
# internal imports
from aeecde.publics import tools
from aeecde.publics import parameterize as para
from aeecde.architectures.population import DEpopulation



#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------
class __base(object):
    ''' :class:`__base` is the base class for defining an optimization method.
    '''
    HIST_ITEMS = ["nth", "FES", "x_best", "f_best", "f_mean", "x_opt", "f_opt"]

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

    #----------------------------- Setter/Getter ------------------------------
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

    #----------------------------- Public Methods -----------------------------
    def initialize_X0(self, initial_scheme, pop_size, criterion="classic",
        iterations=5):
        ''' Initialize population with random canidate solutions. '''
        func = self._ALL_initial_schemes.get(initial_scheme)
        if func is None:
            raise ValueError("The `initial_scheme` must be one of"
                " 'latin_hypercube' or 'random', not {}".format(initial_scheme))
        else:
            X0 = func(pop_size, self.pb.D,
                      self.pb.init_lower_bound, self.pb.init_upper_bound,
                      self.para.rng, criterion, iterations)
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

        All variables/atributes will be re-initialized when this method is called.

        '''
        # Initialize history lists
        self.history = {key: [] for key in self.HIST_ITEMS}
        # Reset some counters used in optimization process
        self.pb.num_calls = 0
        self.pop.nth_generation = 0
        self.pb.x_opt = None
        self.pb.f_opt = None

    def save(self, filestring="", replace=True):
        if ".json" in filestring:
            tools.to_json(self.data, filestring, replace)
            return
        elif filestring == "":
            filename1 = self.pb.name + "_" + self.ALGO_NAME + ".json"
            filename2 = self.pb.name + "_" + self.ALGO_NAME + ".rng"
        else:
            filename1 = str(filestring) + ".json"
            filename2 = str(filestring) + ".rng"
        tools.to_json(self.data, filename1, replace)

    def display_result(self, flag):
        if flag:
            print("Generation {0:5d}: f(x) = {1:.20f}, objective function has"
                  " been called {2} times".format(self.pop.nth_generation,
                  self.pop.F_best, self.pb.num_calls))

    def plot_convergence_curve(self, algo_name="auto_get"):
        if algo_name == "auto_get":
            algo_name = self.ALGO_NAME
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.data["FES_hist"], np.log10(self.data["ferror_hist"]))
        ax.set(xlabel="Number of Fitness Function Evaluations (FES)",
               ylabel="$log_{10} (f-f^*)$",
               xlim=(0, self.pb.num_calls),
               title="{} on {}".format(algo_name, self.data.get("problem")))
        ax.grid(color="gray", linestyle="--")
        plt.show()

    #----------------------------- Private Methods ----------------------------
    def _update_pop_attribute(self):
        # Update the population's attributes
        self.pop.get_F()
        self.pop.get_X()
        # Update best solution found so far
        self.pop.find_best_individual()
        self.pb.update_opt(self.pop.X_best, self.pop.F_best)

    def _write_to_history(self, keys_list):
        available_items = {"nth": self.pop.nth_generation,
                           "FES": self.pb.num_calls,
                           "x_best": self.pop.X_best.tolist(),
                           "f_best": self.pop.F_best,
                           "f_mean": self.pop.F_mean,
                           "x_opt": self.pb.x_opt.tolist(),
                           "f_opt": self.pb.f_opt,
                           "pop_state": deepcopy(self.pop.state),
                           "pop_sum_dist": self.pop.sum_distance,
                           "pop_avg_dist": self.pop.avg_distance,}
        for key in keys_list:
            self.history[key].append(available_items.get(key))

    def _collect_result(self):
        result = {}
        result["problem"] = self.pb.name
        result["config"] = self.para.current()
        result["stop_conditions"] = self.stop.current()
        result["stop_status"] = self.terminate()
        result["f_opt_theory"] = self.pb.f_opt_theory
        result["x_best"] = self.pop.X_best.tolist()
        result["f_best"] = self.pop.F_best
        result["x_opt"] = self.pb.x_opt.tolist()
        result["f_opt"] = self.pb.f_opt
        result["consumed_FES"] =  self.pb.num_calls
        if self.terminate().get("ftarget") is not None:
            result["success"] = True
        else:
            result["success"] = False
        result["nth_hist"] = self.history.get("nth")
        result["FES_hist"] = self.history.get("FES")
        result["x_best_hist"] = self.history.get("x_best")
        result["f_best_hist"] = self.history.get("f_best")
        result["f_mean_hist"] = self.history.get("f_mean")
        result["x_opt_hist"] = self.history.get("x_opt")
        result["f_opt_hist"] = self.history.get("f_opt")
        if self.pb.f_opt_theory is not None:
            result["ferror_min"] = result["f_opt"] - self.pb.f_opt_theory
            result["ferror_hist"] = list((np.array(result["f_opt_hist"])
                                     - self.pb.f_opt_theory))
        else:
            result["ferror_min"] = None
            result["ferror_hist"] = None
        return result


#*  ------------------------- Differential Evolution -------------------------
class DE(__base):
    ''' :class:`DE` implements the basic/standard Differential Evolution (DE)
    algorithm '''
    ALGO_NAME = "DE Basic"
    CONFIG_TYPE = para.DE

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                       stop_conditions=para.StopCondition()):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.pop = DEpopulation(random_number_generator=self.para.rng)

    def solve(self, disp=False, plot=False):
        '''Solve the given problem instance'''
        self._initiate_generation_0()
        self._update_pop_attribute()
        self._write_to_history(self.HIST_ITEMS)
        self.display_result(disp)
        # Main loop of evolution
        while self.terminate() == {}:
            self._evolve_a_generation()
            self._update_pop_attribute()
            self._write_to_history(self.HIST_ITEMS)
            self.display_result(disp)
        # Collect result data for output
        self.data = self._collect_result()
        return self.pb.x_opt, self.pb.f_opt

    def _initiate_generation_0(self):
        # Initialize the algorithm
        self.reset()
        # Initialize the population
        X0 = self.initialize_X0(initial_scheme=self.para.initial_scheme,
                                pop_size=self.para.N)
        self.pop.initialize(X0)
        # Evaluate the population
        for ind in self.pop:
            self.pb.evaluate_ind(ind)

    def _evolve_a_generation(self):
        list_next_gen_ind = []
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int(min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        for idx in range(n_trials):
            # Generate a trial individual
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR)
            # Evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            # Survival selection
            target_ind = self.pop.list_ind[idx]
            list_next_gen_ind.append(self.pop.survival_select(target_ind, trial_ind))
        # Update pop individuals and Incease the generation count
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1

    #------------- Some Methods for DE Operator Name Formatting---------------
    def _upper_de_in_mutation(self, mutation_str):
        '''Make the two letters 'de' in mutation operator into upper case. '''
        str_list = mutation_str.split("/")
        str_list[0] = str_list[0].upper()
        Standard_mutation_name = "/".join(str_list)
        return Standard_mutation_name

    def _get_STname_from_config(self, config_dict):
        '''Get strategy name from a configuration. '''
        strategy_dict = {"mutation":  config_dict["mutation"],
                         "crossover": config_dict["crossover"]}
        strategy_name = self._STdict_to_STname(strategy_dict)
        return strategy_name

    def _STdict_to_STname(self, strategy_dict):
        '''Transform the strategy given by a dict into a conventional name in string,
        which is usually used in literature. For instance,
        `{"mutation": "de/rand/1", "crossover": "bin"}` --> "DE/rand/1/bin" '''
        if isinstance(strategy_dict, dict):
            mut_name = self._upper_de_in_mutation(strategy_dict.get("mutation"))
            cx_name = strategy_dict.get("crossover")
            strategy_name = "{}/{}".format(mut_name, cx_name)
        else:
            raise TypeError("`strategy_dict` must be a dict with keys of"
                            "'mutation' and 'crossover' ")
        return strategy_name

    def _STname_to_STdict(self, strategy_name):
        '''Transform the name of strategy into strategy dict, i.e.,
        the reverse operation of `self.__STdict_to_STname()`.'''
        if isinstance(strategy_name, str):
            str_list = (strategy_name.lower()).split("/")
            mutation = "/".join(str_list[:3])
            crossover = str_list[-1]
            strategy_dict = {"mutation": mutation, "crossover": crossover}
        else:
            TypeError("`strategy_name` must be a string. ")
        return strategy_dict


class _DEVariant(DE):
    ''' :class:`_DEVariant` is an internal-used-only class for defining a
    variant of DE algorithm.'''
    HIST_ITEMS = DE.HIST_ITEMS + ["configs_assignment",
                                  "mutation", "crossover", "F", "CR"]

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                       stop_conditions=para.StopCondition()):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.configs_assignment = None  # Assignment of configs for a generation
        self._mutation_pop_record = []  # record mutation operators used in current generation
        self._crossover_pop_record = [] # record crossover operators used in current generation
        self._F_pop_record = []         # record F values used in current generation
        self._CR_pop_record = []        # record CR values used in current generation

    def _record_hyperparameter(self):
        self._mutation_pop_record.append(self.para.mutation)
        self._crossover_pop_record.append(self.para.crossover)
        self._F_pop_record.append(self.para.F)
        self._CR_pop_record.append(self.para.CR)

    def _write_to_history(self, keys_list):
        available_items = {"nth": self.pop.nth_generation,
                           "FES": self.pb.num_calls,
                           "X": self.pop.X,
                           "x_best": self.pop.X_best.tolist(),
                           "f_best": self.pop.F_best,
                           "f_mean": self.pop.F_mean,
                           "x_opt": self.pb.x_opt.tolist(),
                           "f_opt": self.pb.f_opt,
                           "pop_state": deepcopy(self.pop.state),
                           "pop_sum_dist": self.pop.sum_distance,
                           "pop_avg_dist": self.pop.avg_distance,
                           "configs_assignment": self.configs_assignment,
                           "mutation": self._mutation_pop_record,
                           "crossover": self._crossover_pop_record,
                           "F": self._F_pop_record,
                           "CR": self._CR_pop_record }
        for key in keys_list:
            if key in ("mutation", "crossover", "F", "CR"):
                self.history[key].extend(available_items.get(key))
                setattr(self, "_"+key+"_pop_record", [])
            else:
                self.history[key].append(available_items.get(key))

    def _collect_result(self):
        result = super()._collect_result()
        result["configs_assignment_hist"] = self.history.get("configs_assignment")
        result["mutation_hist"] = self.history.get("mutation")
        result["crossover_hist"] = self.history.get("crossover")
        result["F_hist"] = self.history.get("F")
        result["CR_hist"] = self.history.get("CR")
        return result

    def plot_para_histogram(self, bins=20, save=False, show=False):
        '''Plot the histogram of each parameter independently. In other words,
        mutation, crossover, F, CR are considered as independently,
        and an histogram is plot seperately for each parameter.

        For the two categorical parameters [mutation, crossover], use plt.bar()
        For the two continuous parameters [F, CR], use pl.hist(). When F and CR
        are selected from a set of discreted values, use plt.bar().
        '''
        # Get raw data
        mut_hist = self.data.get("mutation_hist")
        cx_hist =  self.data.get("crossover_hist")
        F_hist =   self.data.get("F_hist")
        CR_hist =  self.data.get("CR_hist")
        # Get the histogram of mutation
        if hasattr(self, "ST_POOL"):
            mut_names = [st["mutation"] for st in self.ST_POOL]
        elif hasattr(self, "MUT_POOL"):
            mut_names = self.MUT_POOL
        else:
            mut_names = set(mut_hist)
        mut_histgram = {key: mut_hist.count(key) for key in mut_names}
        # Because mutation operator names are very long, so it not convinent to
        # show them on the figure, we use abbreviated names as the ticks of axis x.
        mut_hist_values = [mut_histgram[key] for key in mut_names]
        mut_hist_ticks = ["MUT{}".format(i) for i in range(1, 1+len(mut_names))]
        str_list = ["{}: {}".format(tick, self._upper_de_in_mutation(name))
                    for tick, name in zip(mut_hist_ticks, mut_names)]
        mut_anotation = "\n".join(str_list) # Illustration of the tick_label.
        # Get the histogram of  crossvoer
        if hasattr(self, "ST_POOL"):
            cx_names = list(set([st["crossover"] for st in self.ST_POOL]))
        elif hasattr(self, "CX_POOL"):
            cx_names = self.CX_POOL
        else:
            cx_names = list(set(cx_hist))
        cx_histgram = {key: cx_hist.count(key) for key in cx_names}
        cx_hist_values = [cx_histgram[key] for key in cx_names]
        cx_hist_ticks = cx_names

        # Create figure object with 4 subplots
        fig = plt.figure(figsize=(10,8))
        axs = [fig.add_subplot(2,2,i) for i in range(1,5)] # create 4 subplots
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # Plot histogram of mut_hist and cx_hist:
        axs[0].bar(x=range(len(mut_names)), height=mut_hist_values,
                   tick_label=mut_hist_ticks)
        axs[0].set(xlabel="Mutation Operator",
                   ylabel="Number of Individuals",
                   title="Histogram of Mutation Operator")
        axs[0].text(-0.25,max(mut_hist_values)/1.5, mut_anotation)

        axs[1].bar(x=range(len(cx_names)), height=cx_hist_values,
                   tick_label=cx_hist_ticks)
        axs[1].set(xlabel="Crossover Operator",
                   ylabel="Number of Individuals",
                   title="Histogram of Crossover Operator")
        # Plot histogram of F and CR:
        if len(set(F_hist)) < bins:
            # Use bar plot
            F_values = sorted(list(set(F_hist)))
            F_histgram = [F_hist.count(F) for F in F_values]
            axs[2].bar(x=range(len(F_values)), height=F_histgram,
                       tick_label=F_values)
        else:
            # Use hist plot
            axs[2].hist(x=F_hist, bins=bins)
        axs[2].set(xlabel="Scale Factor (F)",
                   ylabel="Number of Individuals",
                   title="Histogram of Scale Factor (F)")

        if len(set(CR_hist)) < bins:
            # Use bar plot
            CR_values = sorted(list(set(CR_hist)))
            CR_histgram = [CR_hist.count(CR) for CR in CR_values]
            axs[3].bar(x=range(len(CR_values)), height=CR_histgram,
                       tick_label=CR_values)
        else:
            # Use hist plot
            axs[3].hist(x=CR_hist, bins=bins)
        axs[3].set(xlabel="Crossover Rate (CR)",
                   ylabel="Number of Individuals",
                   title="Histogram of Crossover Rate (CR)")
        if save == True:
            plt.savefig("{}_{}_2.pdf".format(self.pb.name, self.ALGO_NAME))
        if show == True:
            plt.show()
            plt.close()


class RandomDE(_DEVariant):
    '''Randomly generate hyperparameters for each Population or Individual.

    For "population_level": at each generation, new parameter setting is sampled
        for generating new population.
    For "individual_level": new parameter setting is sampled when generating each
        new individual.'''

    ALGO_NAME = "Random DE"

    MUT_POOL = ["de/rand/1", "de/rand/2", "de/best/1", "de/rand-to-best/1"]
    CX_POOL =  ["bin", "exp", "eig", "none"]
    F_RANGE =  (0.0, 2.0)
    CR_RANGE = (0.0, 1.0)

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                       stop_conditions=para.StopCondition(),
                       random_level="individual_level"):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.random_level = random_level

    @property
    def random_level(self):
        return self._random_level

    @random_level.setter
    def random_level(self, level):
        allowable_levels = ("individual_level", "population_level")
        if isinstance(level, str):
            if level not in allowable_levels:
                raise NameError("`random_level` get and unexpected level name "
                                "'{}'. The allowable level names are: "
                                "'individual_level', 'population_level'".format(level))
            self._random_level = level
        else:
            raise TypeError("The level, at which hyperparameters are randomly assigned,"
                            "must be a string, not a {}".format(type(level).__name__))

    def _initiate_generation_0(self):
        '''Initialize a population, reset the algorithm, and assign configs for
        initial generaiton/population. '''
        super()._initiate_generation_0()
        # Assign configs for generation 0:
        self._assign_configs_randomly()

    def _evolve_a_generation(self):
        '''Evolve one generation '''
        list_next_gen_ind = [] # to save the selected individuals for next generation
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int( min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # Perform mutation, crossover and selection
        for idx in range(n_trials):
            # Get a config from the configs assignment for current target ind:
            self._get_config_from_assignment(idx)
            # Record the hyperparameters to history
            self._record_hyperparameter()
            # Generate a trial individual
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR)
            # Evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            # Survival selection
            target_ind = self.pop.list_ind[idx]
            list_next_gen_ind.append(self.pop.survival_select(target_ind, trial_ind))
        # Update pop individuals and Incease the generation count
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Assign configs for next generation
        self._assign_configs_randomly()

    def _assign_configs_randomly(self):
        '''Randomly select or sample configs from the given POOL or within the RANGE.'''
        if self.random_level == "population_level":
            config = {}
            config["mutation"] = self.para.rng.choice(self.MUT_POOL)
            config["crossover"]  = self.para.rng.choice(self.CX_POOL)
            config["F"]  = self.para.rng.uniform(self.F_RANGE[0], self.F_RANGE[1])
            config["CR"]  = self.para.rng.uniform(self.CR_RANGE[0], self.CR_RANGE[1])
            list_configs = [config] * self.para.N # each ind is assgined the same config
        elif self.random_level == "individual_level":
            # Implementation using list comprehension -------------------------
            list_MUT =list(self.para.rng.choice(self.MUT_POOL,size=self.para.N, replace=True))
            list_CX = list(self.para.rng.choice(self.CX_POOL,size=self.para.N, replace=True))
            list_F =  list(self.para.rng.uniform(self.F_RANGE[0],self.F_RANGE[1], self.para.N))
            list_CR = list(self.para.rng.uniform(self.CR_RANGE[0],self.CR_RANGE[1], self.para.N))
            list_configs = [{"mutation": mut, "crossover": cx, "F": F, "CR": CR}
                    for mut, cx, F, CR in zip(list_MUT, list_CX, list_F, list_CR)]
        self.configs_assignment = list_configs

    def _get_config_from_assignment(self, idx):
        '''Get the config from `self.configs_assignment` according to the index
        of target individual in the population. '''
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]


class CoDE(_DEVariant):
    ''' :class:`CoDE` implements the Differential Evolution with
    Composite Trial Vector Generation Strategies and Control Parameters algorithm.
    Ref:
        Y. Wang, Z. Cai and Q. Zhang, "Differential Evolution With Composite
        Trial Vector Generation Strategies and Control Parameters,"
        in IEEE Transactions on Evolutionary Computation,
        vol. 15, no. 1, pp. 55-66, Feb. 2011.'''
    ALGO_NAME = "CoDE"

    ST_POOL = [{"mutation": "de/rand/1", "crossover": "bin"},
               {"mutation": "de/rand/2", "crossover": "bin"},
               {"mutation": "de/current-to-rand/1", "crossover": "none"}]
    F_CR_POOL = [{"F": 1.0, "CR": 0.1},
                 {"F": 1.0, "CR": 0.9},
                 {"F": 0.8, "CR": 0.2}]

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                       stop_conditions=para.StopCondition()):
        super().__init__(opt_problem, algo_parameters, stop_conditions)

    def _initiate_generation_0(self):
        '''Initialize a population, reset the algorithm, and assign configs for
        initial generaiton. '''
        super()._initiate_generation_0()
        # Assign configs for generation 0
        self._assign_configs()

    def _evolve_a_generation(self):
        '''Evolve one generation'''
        list_next_gen_ind = []
        for idx in range(self.pop.size):
            list_temp_trial = []
            for idST in range(len(self.ST_POOL)):
                # Check the comsumed FES:
                if self.pb.num_calls >= self.stop.max_FES:
                    break
                # Get a config from assignment:
                self._get_config_from_assignment(idx, idST)
                # Record the used hyperparameter:
                self._record_hyperparameter()
                # Create a trial individual:
                trial_ind = self.pop.create_offspring(target_idx=idx,
                            mutation=self.para.mutation,
                            F=self.para.F,
                            crossover=self.para.crossover,
                            CR=self.para.CR)
                # Evaluate the trial individual:
                self.pb.evaluate_ind(trial_ind)
                list_temp_trial.append(trial_ind)
            if len(list_temp_trial) == 0:
                break
            else:
                trial_best = sorted(list_temp_trial, key=lambda ind: ind.fvalue,
                                    reverse=False)[0]
            # Survival selection
            target_ind = self.pop.list_ind[idx]
            list_next_gen_ind.append(self.pop.survival_select(target_ind, trial_best))
        # Update pop individuals and Incease the generation count
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Assign configs for nrxt generation
        self._assign_configs()

    def _assign_configs(self):
        '''In CoDE, for each target vector, three strategies in the ST_POOL with
        parameters randomly selected from the F_CR_POOL are used to create three
        trial vectors.'''
        list_configs = []
        for i in range(self.para.N):
            configs = []
            for strategy in self.ST_POOL:
                config = {}
                # get mutation and crossover operator:
                config["mutation"] = strategy.get("mutation")
                config["crossover"] = strategy.get("crossover")
                # randomly select control parameter from the pool:
                F_CR_pair = self.para.rng.choice(self.F_CR_POOL)
                config["F"] = F_CR_pair.get("F")
                config["CR"] = F_CR_pair.get("CR")
                configs.append(config)
            list_configs.append(configs)
        self.configs_assignment = list_configs

    def _get_config_from_assignment(self, idx, idST):
        config = self.configs_assignment[idx][idST]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]


class SaDE(_DEVariant):
    ''':class:`SaDE` implements the Self-Adaptive DE.
    Ref:
        A. K. Qin, V. L. Huang and P. N. Suganthan, "Differential Evolution
        Algorithm With Strategy Adaptation for Global Numerical Optimization,"
        in IEEE Transactions on Evolutionary Computation,
        vol. 13, no. 2, pp. 398-417, April 2009.'''

    ALGO_NAME = "SaDE"

    ST_POOL = [{"mutation": "de/rand/1", "crossover": "bin"},
               {"mutation": "de/current-to-best/2", "crossover": "bin"},
               {"mutation": "de/rand/2", "crossover": "bin"},
               {"mutation": "de/current-to-rand/1", "crossover": "none"} ]
    F_RANGE  = (0.0, 2.0)
    CR_RANGE = (0.0, 1.0)

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                       stop_conditions=para.StopCondition(),
                       learning_period=50):
        super().__init__(opt_problem, algo_parameters,
                                             stop_conditions)
        self.LP = learning_period  # The learning period
        self._ST_names = [self._STdict_to_STname(st) for st in self.ST_POOL] #strategy names
        # The strategy probability and CR means are used in adaptation
        self.ST_probability = {key: 1.0/len(self._ST_names) for key in self._ST_names}
        self.CR_mean = {key: 0.5 for key in self._ST_names}
        # Success and Failure memory of applying each strategy:
        self._success_memory = {key: [] for  key in self._ST_names}
        self._failure_memory = {key: [] for  key in self._ST_names}
        # Memory of success CR values with respect to each strategy:
        self._CR_memory = {key: [] for  key in self._ST_names}
        # Some additional histories for SaDE:
        self._ST_prob_hist = {key: [] for  key in self._ST_names}
        self._ST_n_hist = {key: [] for  key in self._ST_names}
        self._CR_mean_hist = {key: [] for  key in self._ST_names}
        self._CR_of_ST_hist = {key: [] for  key in self._ST_names}

    def _initiate_generation_0(self):
        super()._initiate_generation_0()
        # Assign configs for generation 0
        self._assign_configs()
        self._update_SaDE_histories()

    def _evolve_a_generation(self):
        # Initialize some dicts to save the memories of current generation
        self.__gen_success_memory = {key: 0 for key in self._ST_names}
        self.__gen_failure_memory = {key: 0 for key in self._ST_names}
        self.__gen_CR_memory = {key: [] for key in self._ST_names}
        # Initialize two list to save trial ind and ind selected to enter next gen:
        list_trial_ind = []
        list_next_gen_ind = []
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int( min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # -------------------- Evolution of Population ------------------------
        # perform evolution using the assigned config for each target vector
        for idx in range(n_trials):
            # Generation Step:create a trial individual by mutation and crossover
            self._get_config_from_assignment(idx)
            # record the changed hyperparameters to history
            self._record_hyperparameter()
            ST_name = self._STdict_to_STname({"mutation": self.para.mutation,
                                              "crossover": self.para.crossover})
            # generate a trial individual
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR)
            # Evaluation Step: evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            list_trial_ind.append(trial_ind)
            # Selection Step: survival selection
            target_ind = self.pop.list_ind[idx]
            if trial_ind.fvalue <= target_ind.fvalue:
                list_next_gen_ind.append(trial_ind)
                self.__gen_CR_memory[ST_name].append(self.para.CR)
                self.__gen_success_memory[ST_name] += 1
            else:
                list_next_gen_ind.append(target_ind)
                self.__gen_failure_memory[ST_name] += 1
        #----------------------------------------------------------------------
        # Update the memories
        self._update_SaDE_memories()
        # Update pop individuals and Incease the generation count
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Update Strategy probability and CR means
        if self.pop.nth_generation >= self.LP:
            self._update_ST_probability()
            self._update_CR_mean()
        # Assigne configs based on currrent ST_probability and CR_means for next gen:
        self._assign_configs()
        self._update_SaDE_histories()

    def _collect_result(self):
        result = super()._collect_result()
        result["ST_prob_hist"] = self._ST_prob_hist
        result["CR_mean_hist"] = self._CR_mean_hist
        result["ST_n_hist"] = self._ST_n_hist
        result["CR_of_ST_hist"] = self._CR_of_ST_hist
        return result

    # ------------- Some new priviate methods designed for SaDE ---------------
    def _update_SaDE_memories(self):
        for key in self._ST_names:
            # keep the success and failure memories with length = self.LP:
            if len(self._success_memory[key]) == self.LP:
                self._success_memory[key].pop(0)
            if len(self._failure_memory[key]) == self.LP:
                self._failure_memory[key].pop(0)
            # collect new data in success and failure memories:
            self._success_memory[key].append(self.__gen_success_memory[key])
            self._failure_memory[key].append(self.__gen_failure_memory[key])
            # Update CR memory only when `self.__gen_CR_memory[key]` is not Empty:
            if self.__gen_CR_memory[key]:
                if len(self._CR_memory[key]) == self.LP:
                    self._CR_memory[key].pop(0) # keep memory length
                self._CR_memory[key].append(self.__gen_CR_memory[key])

    def _update_SaDE_histories(self):
        ST_n_gen =  {key: 0 for key in self._ST_names}
        CR_of_ST_gen = {key: [] for key in self._ST_names}
        for config in self.configs_assignment:
            strategy = {"mutation": config["mutation"], "crossover": config["crossover"]}
            ST_name = self._STdict_to_STname(strategy)
            ST_n_gen[ST_name] += 1
            CR_of_ST_gen[ST_name].append(config["CR"])
        # Update the additional histories for SaDE ------------------------
        for key in self._ST_names:
            self._ST_n_hist[key].append(ST_n_gen[key])
            self._ST_prob_hist[key].append(self.ST_probability[key])
            self._CR_mean_hist[key].append(self.CR_mean[key])
            self._CR_of_ST_hist[key].extend(CR_of_ST_gen[key])

    def _update_ST_probability(self):
        '''Calculate strategy probability using the Success and Failure Memory'''
        EPSILON = 0.01 # small constant to avoid the possible null success rate
        # Firstly, calculate the success rate for each strategy:
        ST_success_rates = {}
        ns = {key: sum(self._success_memory[key]) for key in self._ST_names}
        nf = {key: sum(self._failure_memory[key]) for key in self._ST_names}
        n_total = {key: ns[key]+nf[key] for key in self._ST_names}
        for key in self._ST_names:
            if n_total[key] > 0:
                ST_success_rates[key] = (ns[key] / n_total[key]) + EPSILON
            else:
                ST_success_rates[key] = EPSILON
        # Then, calculate the new probability
        sum_success_rates = sum(list(ST_success_rates.values()))
        for key in self._ST_names:
            self.ST_probability[key] = (ST_success_rates[key]/sum_success_rates)

    def _update_CR_mean(self):
        '''update the means of CR for each strategy'''
        success_CR = {}
        for key in self._ST_names:
            raw_data = self._CR_memory[key]
            sum_data = []
            for elem in raw_data:
                sum_data = sum_data + elem
            success_CR[key] = sum_data
            self.CR_mean[key] = np.median(success_CR[key])

    def _assign_configs(self):
        '''Assign a configuration for each target vector'''
        list_STname = self.__assign_STname()
        list_ST = [self._STname_to_STdict(name) for name in list_STname]
        list_F = self.__assign_F()
        dict_CR = self.__assign_CR()
        list_CR = [dict_CR[key][i] for i, key in enumerate(list_STname)]
        configs = [{"mutation": ST["mutation"], "crossover": ST["crossover"], "F": F, "CR": CR}
                   for ST, F, CR in zip(list_ST, list_F, list_CR)]
        self.configs_assignment = configs

    def _get_config_from_assignment(self, idx):
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]

    def __assign_STname(self):
        '''Use Stochastic Universal Sampling (SUS) to select one generation strategy
        for each target vector.
        Ref of SUS: Book-Computational Itelligence: An Introduction. p.p.137'''
        spacing = 1.0/self.para.N
        r = self.para.rng.rand()/self.para.N
        prob_sum = 0.0
        list_STname = [] # save the names of strategies
        for ST_name in self._ST_names:
            prob_sum += self.ST_probability[ST_name]
            while r < prob_sum:
                list_STname.append(ST_name)
                r += spacing
        # randomly shuffle the list of strategy names
        self.para.rng.shuffle(list_STname)
        return list_STname

    def __assign_F(self):
        '''The control parameter F is randomly sampled from N(0.5, 0.3) for
        each target vector.
        If F <= 0, regenerate; if F >=2.0, truncated to be F=2.
        '''
        list_F = []
        for i in range(self.para.N):
            F = self.para.rng.normal(loc=0.5, scale=0.3)
            while F <= self.F_RANGE[0]:
                F = self.para.rng.normal(loc=0.5, scale=0.3)
            if F >= self.F_RANGE[1]:
                F = self.F_RANGE[1]
            list_F.append(F)
        return list_F

    def __assign_CR(self):
        '''The control parameter CR with respect to each strategy is randomly
        sampled from N(CR_mean, 0.1) for each target vector
        While CR <0 or CR > 1: regenerate CR.'''
        # generate CR values according to CR_mean and distribution
        dict_CR = {key: [] for key in self._ST_names}
        for key in self._ST_names:
            for i in range(self.para.N):
                CR = self.para.rng.normal(loc=self.CR_mean[key], scale=0.1)
                while CR < self.CR_RANGE[0] or CR > self.CR_RANGE[1]:
                    CR = self.para.rng.normal(loc=self.CR_mean[key], scale=0.1)
                dict_CR[key].append(CR)
        return dict_CR

    def plot_ST_and_CRofST_histogram(self, bins=20):
        '''Plot the histogram of strategy, and the CR values with respect to
        strategy.
        Since DE/current-to-rand/1/none does not use crossover, histogram of
        CR with respect to this strategy is not plotted.
        '''
        fig = plt.figure(figsize=(10,8))
        axs = [fig.add_subplot(2,2,i) for i in range(1,5)] # create 4 subplots
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # assign color for each strategy:
        color_pool = {key: self.para.rng.rand(3) for key in self._ST_names}
        # Bar plot for generation strategies ----------------------------------
        bar_data = {key: sum(self.data["ST_n_hist"][key]) for key in self._ST_names}
        values = [bar_data[key] for key in self._ST_names]
        colors = [color_pool[key] for key in self._ST_names]
        labels = ["S{}".format(i) for i in range(1, 1+len(values))]
        axs[0].bar(x=range(len(values)), height=values, color=colors, tick_label=labels)
        axs[0].set(xlabel="Trial Vector Generation Strategies",
                   ylabel="The Number of Usage Times ",
                   title="Histogram of Generation Strategies")
        str_list = ["{}: {}".format(labels[i], self._ST_names[i])
                    for i in range(len(self._ST_names))]
        anotation = "\n".join(str_list)
        axs[0].text(-0.5,max(values)/1.3, anotation)
        # Hist plot for CR vlaues respect to each strategies ------------------
        hist_data = self.data["CR_of_ST_hist"]
        for i, key in enumerate(self._ST_names):
            if self._STname_to_STdict(key)["crossover"] !="none":
                axs[i+1].hist(x=hist_data[key], color=color_pool[key], bins=bins)
                axs[i+1].set(xlabel= "CR Values with Respect to {}".format(key),
                             ylabel="Count",
                             title="Histogram of CR for {}".format(key),
                             xlim=(0, 1))
        plt.show

    def plot_adaptation_hist(self, skip=10, aggregate_method="average"):
        '''Plot the adaptation history of Strategy Probability and CR Mean.
        Since DE/current-to-rand/1/none does not use crossover, CR mean
        with respect to this strategy is not plotted.
        '''
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])

        x = self.data["nth_hist"]
        x_skip = skip_and_aggregate_of_list_data(x, skip, aggregate_method="none")
        # Plot strategy probability --------------
        probs_data = self.data["ST_prob_hist"]
        for key in self._ST_names:
            y = probs_data[key]
            y_skip = skip_and_aggregate_of_list_data(y, skip, aggregate_method)
            ax1.plot(x_skip, y_skip, label=key)
        ax1.set(xlabel="Generation", ylabel="$p_k$",
                xlim=(0, self.pop.nth_generation),
                title="Adaptation of $p_k$ and $CR_m$ for each Strategy")
        ax1.legend(loc="upper left", frameon=False, ncol=len(self._ST_names))
        # plot CR means --------------------------
        CR_data = self.data["CR_mean_hist"]
        for key in self._ST_names:
            if self._STname_to_STdict(key)["crossover"] !="none":
                y = CR_data[key]
                y_skip = skip_and_aggregate_of_list_data(y, skip, aggregate_method)
                ax2.plot(x_skip, y_skip, label=key)
        ax2.set(xlabel="Generation", ylabel="$CR_m$",
                xlim=(0, self.pop.nth_generation))
        ax2.legend(loc="upper left", frameon=False, ncol=len(self._ST_names))
        plt.show()


class EPSDE(_DEVariant):
    ''':class:`EPSDE` implements the Differential Evolution with
     Ensemble of mutation strategies and control parameters.

     Ref:
         Mallipeddi, R., Suganthan, P. N., Pan, Q. K., & Tasgetiren, M. F. (2011).
         Differential evolution algorithm with ensemble of parameters
         and mutation strategies. Applied soft computing, 11(2), 1679-1696.'''

    ALGO_NAME = "EPSDE"

    # ST is the abbreviation of Strategy for generating traial vector
    ST_POOL = [{"mutation": "de/best/2", "crossover": "bin"},
               {"mutation": "de/rand/1", "crossover": "bin"},
               {"mutation": "de/current-to-rand/1", "crossover": "none"}]
    F_POOL =  [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    CR_POOL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                 stop_conditions=para.StopCondition()):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        # Memory of the successful combinations of strategy and parameter
        self.successful_configs = []
        # Usage number of each element in each pool used in current generation
        self.ST_keys = [self._STdict_to_STname(st) for st in self.ST_POOL]
        self.F_keys =  [str(k) for k in self.F_POOL]
        self.CR_keys = [str(k) for k in self.CR_POOL]
        # history of the usage frequency of each element in each Pool in each generation:
        self.ST_n_hist = {key: [] for  key in self.ST_keys}
        self.F_n_hist =  {key: [] for  key in self.F_keys}
        self.CR_n_hist = {key: [] for  key in self.CR_keys}

    def _initiate_generation_0(self):
        super()._initiate_generation_0()
        # Assign configs for generation 0
        self._assign_configs_for_init_gen()
        # Record configs/para used in current generation:
        self._collect_para_frequency()

    def _evolve_a_generation(self):
        # Initialize two list to save trial ind and ind selected to enter next gen:
        list_trial_ind = []
        list_next_gen_ind = []
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int(min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # -------------------- Evolution of Population ------------------------
        # perform evolution using the assigned config for each target vector
        for idx in range(n_trials):
            # Generation Step:create a trial individual by mutation and crossover
            self._get_config_from_assignment(idx)
            # record the changed hyperparameters to history
            self._record_hyperparameter()
            # Generation Step:create a trial individual by mutation and crossover
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR)
            # Evaluation Step: evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            list_trial_ind.append(trial_ind)
            # Selection Step: survival selection
            target_ind = self.pop.list_ind[idx]
            if trial_ind.fvalue <= target_ind.fvalue:
                list_next_gen_ind.append(trial_ind)
                # Store the successful config:
                config = {"mutation": self.para.mutation,
                          "crossover": self.para.crossover,
                          "F": self.para.F,
                          "CR": self.para.CR}
                self.successful_configs.append(config)
            else:
                list_next_gen_ind.append(target_ind)
        # Update the configs assignment for next generation pop:
        self._update_configs_assignment(list_trial_ind, self.pop.list_ind)
        self._collect_para_frequency()
        # Incease the generation count and update the population:
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1

    def _assign_configs_for_init_gen(self):
        '''Each individual in the initial population '''
        list_ST = list(self.para.rng.choice(self.ST_POOL,size=self.para.N, replace=True))
        list_F =  list(self.para.rng.choice(self.F_POOL, size=self.para.N, replace=True))
        list_CR = list(self.para.rng.choice(self.CR_POOL,size=self.para.N, replace=True))
        configs = [{"mutation": ST["mutation"], "crossover": ST["crossover"], "F": F, "CR": CR}
                    for ST, F, CR in zip(list_ST, list_F, list_CR)]
        # configs = [self.__randomly_choose_config_from_pool() for _ in range(self.para.N)]
        self.configs_assignment = configs

    def _update_configs_assignment(self, list_trial_ind, list_target_ind):
        rand = self.para.rng.rand(self.para.N)
        for i, (trial_ind, target_ind) in enumerate(zip(list_trial_ind, list_target_ind)):
            if trial_ind.fvalue > target_ind.fvalue:
                if rand[i] > 0.5 and len(self.successful_configs) >= 1:
                    self.configs_assignment[i] = self.para.rng.choice(self.successful_configs)
                else:
                    self.configs_assignment[i] = self.__randomly_choose_config_from_pool()

    def _get_config_from_assignment(self, idx):
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]

    def __randomly_choose_config_from_pool(self):
        '''Randomly choose a parameter setting from the pool, that is,
        randonly choose a strategy and associated parameter values from
        the respecive pools.'''
        config = self.para.rng.choice(self.ST_POOL)
        config["F"] = self.para.rng.choice(self.F_POOL)
        config["CR"] = self.para.rng.choice(self.CR_POOL)
        return config

    def _collect_para_frequency(self):
        '''Collect the frequency of each parameter used in current population/generation`'''
        ST_n_pop = {key: 0 for key in self.ST_keys}
        F_n_pop =  {key: 0 for key in self.F_keys}
        CR_n_pop = {key: 0 for key in self.CR_keys}
        for config in self.configs_assignment:
            strategy = {"mutation": config["mutation"], "crossover": config["crossover"]}
            ST_key = self._STdict_to_STname(strategy)
            F = config["F"]
            CR = config["CR"]
            ST_n_pop[ST_key]  += 1
            F_n_pop[str(F)]   += 1
            CR_n_pop[str(CR)] += 1
        # collect parameter frequency to histories:
        for ST_key in self.ST_keys:
            self.ST_n_hist[ST_key].append(ST_n_pop[ST_key])
        for F_key in self.F_keys:
            self.F_n_hist[F_key].append(F_n_pop[F_key])
        for CR_key in self.CR_keys:
            self.CR_n_hist[CR_key].append(CR_n_pop[CR_key])

    def _collect_result(self):
        result = super()._collect_result()
        result["ST_n_hist"] = self.ST_n_hist
        result["F_n_hist"] = self.F_n_hist
        result["CR_n_hist"] = self.CR_n_hist
        result["success_configs"] = self.successful_configs
        result["ST_prob"] = {key: (np.array(result["ST_n_hist"][key])/self.pop.size).tolist() for key in result["ST_n_hist"].keys()}
        result["F_prob"] = {key: (np.array(result["F_n_hist"][key])/self.pop.size).tolist() for key in result["F_n_hist"].keys()}
        result["CR_prob"] = {key: (np.array(result["CR_n_hist"][key])/self.pop.size).tolist() for key in result["CR_n_hist"].keys()}
        return result

    def plot_para_prob(self, k=20):
        # data preparation:
        gens = self.data["nth_hist"]
        data1 = {key: np.array(self.data["ST_n_hist"][key])/self.para.N for key in self.data["ST_n_hist"].keys()}
        data2 = {key: np.array(self.data["F_n_hist"][key])/self.para.N for key in self.data["F_n_hist"].keys()}
        data3 = {key: np.array(self.data["CR_n_hist"][key])/self.para.N for key in self.data["CR_n_hist"].keys()}
        # data points are averaged every k generations
        x = skip_and_aggregate_of_list_data(gens, skip=k, aggregate_method="none")
        Ds = [data1, data2, data3]
        Ys = []
        for data in Ds:
            Ys.append({key: skip_and_aggregate_of_list_data(list(data[key]),
                                        skip=k, aggregate_method="average")
                       for key in data.keys()} )

        fig = plt.figure(figsize=(10,20))
        ax1 = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)
        fig.subplots_adjust(wspace=0.2)

        # data = [data1, data2, data3]
        axs = [ax1, ax2, ax3]
        titles = ["Adaptation of Strategy", "Adaptation of F", "Adaptation of CR"]

        for i in range(3):
            for key in Ys[i].keys():
                y = Ys[i][key]
                axs[i].plot(x, y, label=key)
            axs[i].set(xlabel="Generation", ylabel="Probability", title=titles[i],
                       xlim=(0, self.pop.nth_generation))
            axs[i].legend(loc="upper left", frameon=False, ncol=len(Ys[i].keys()))
        plt.show()


class SAKPDE(_DEVariant):
    ''':class:`SAKPDE` implements the DE with Strategy Adaptation and
    Knowledgebased control Parameters.

    Ref:
        Fan, Q., Wang, W., & Yan, X. (2019). Differential evolution algorithm
        with strategy adaptation and knowledge-based control parameters.
        Artificial Intelligence Review, 51(2), 219-253.'''

    ALGO_NAME = "SAKPDE"

    # Mutation (MUT) and Crossover (CX) pools for SAKPDE:
    MUT_POOL = ["de/rand/1", "de/rand/2", "de/best/2",
                "de/current-to-best/1",
                "de/current-to-best/2" ]
    CX_POOL = ["bin", "exp", "eig"]
    F_RANGE  = (0.4, 1.0)
    CR_RANGE = (0.3, 0.9)

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                 stop_conditions=para.StopCondition(),
                 forgetting_factor=0.7, opposition_learning_ratio=0.8,
                 para_range_handle="regenerate"):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.forgetting_factor = forgetting_factor
        self.op_learning_ratio = opposition_learning_ratio
        if para_range_handle in ("regenerate", "truncate"):
            self.para_range_handle = para_range_handle
        else:
            raise ValueError("`para_range_handle` should be one of 'regenerate' and 'truncate' ")
        if self.stop.max_iter != None:
            self.Gmax = int(self.stop.max_iter)
        else:
            self.Gmax = int(self.stop.max_FES/self.para.N)
        self.Gs = int(0.3 * self.Gmax)
        self.MUT_prob = {key: 1.0/len(self.MUT_POOL) for key in self.MUT_POOL}
        self.CX_prob = {key: 1.0/len(self.CX_POOL) for key in self.CX_POOL}
        self.MUT_n_hist = {key: [] for key in self.MUT_POOL}
        self.CX_n_hist = {key: [] for key in self.CX_POOL}
        self._MUT_prob_hist = {key: [] for key in self.MUT_POOL}
        self._CX_prob_hist = {key: [] for key in self.CX_POOL}

    def _initiate_generation_0(self):
        super()._initiate_generation_0()
        # Assign configs for generation 0
        self._assign_configs()
        self._collect_MUT_CX_frequency()

    def _evolve_a_generation(self):
        # Initialize two list to save trial ind and ind selected to enter next gen:
        list_trial_ind = []
        list_next_gen_ind = []
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int(min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # -------------------- Evolution of Population ------------------------
        for idx in range(n_trials):
            # Generation Step:create a trial individual by mutation and crossover
            self._get_config_from_assignment(idx)
            # record the changed hyperparameters to history
            self._record_hyperparameter()
            # Generation Step:create a trial individual by mutation and crossover
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR)
            # Evaluation Step: evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            list_trial_ind.append(trial_ind)
            # Selection Step: survival selection
            target_ind = self.pop.list_ind[idx]
            if trial_ind.fvalue <= target_ind.fvalue:
                list_next_gen_ind.append(trial_ind)
            else:
                list_next_gen_ind.append(target_ind)
        # ---------------------------------------------------------------------
        # Incease the generation count and update the population:
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Assign configs for next generation
        if self.pop.nth_generation >= self.Gs + 1 :
            self._update_MUT_CX_prob(list_trial_ind)
        self._assign_configs()
        self._collect_MUT_CX_frequency()

    def _collect_result(self):
        result = super()._collect_result()
        result["MUT_n_hist"] = self.MUT_n_hist
        result["CX_n_hist"] = self.CX_n_hist
        result["MUT_prob_hist"] = self._MUT_prob_hist
        result["CX_prob_hist"] = self._CX_prob_hist
        return result

    def _assign_configs(self):
        '''Assign a configuration for each target vector'''
        list_MUT = self.__assign_mutaion()
        list_CX =  self.__assign_crossover()
        list_F = self.__assign_F()
        list_CR = self.__assign_CR()
        configs = [{"mutation": MUT, "crossover": CX, "F": F, "CR": CR}
                   for MUT, CX, F, CR in zip(list_MUT, list_CX, list_F, list_CR)]
        self.configs_assignment = configs

    def _get_config_from_assignment(self, idx):
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]

    def _collect_MUT_CX_frequency(self):
        '''Collect the frequency of each MUT and CX operator used in current generation'''
        MUT_n_gen = {key: 0 for key in self.MUT_POOL}
        CX_n_gen =  {key: 0 for key in self.CX_POOL}
        for config in self.configs_assignment:
            MUT_name = config["mutation"]
            CX_name = config["crossover"]
            MUT_n_gen[MUT_name] += 1
            CX_n_gen[CX_name] += 1
        # collect parameter frequency to histories:
        for MUT_name in self.MUT_POOL:
            self.MUT_n_hist[MUT_name].append(MUT_n_gen[MUT_name])
            self._MUT_prob_hist[MUT_name].append(self.MUT_prob[MUT_name])
        for CX_name in self.CX_POOL:
            self.CX_n_hist[CX_name].append(CX_n_gen[CX_name])
            self._CX_prob_hist[CX_name].append(self.CX_prob[CX_name])

    def _update_MUT_CX_prob(self, list_trial_ind):
        MUT_diff, CX_diff = self.__calculate_fitness_difference(list_trial_ind)
        # update mut prob
        max_mut= max(MUT_diff, key=lambda mut: MUT_diff[mut])
        MUT_diff[max_mut] = MUT_diff[max_mut] * self.forgetting_factor
        sum_MUT_diff = sum(list(MUT_diff.values()))
        for mut in self.MUT_POOL:
            self.MUT_prob[mut] = MUT_diff[mut]/sum_MUT_diff
        # update cx prob
        max_cx= max(CX_diff, key=lambda mut: CX_diff[mut])
        CX_diff[max_cx] = CX_diff[max_cx] * self.forgetting_factor
        sum_CX_diff = sum(list(CX_diff.values()))
        for cx in self.CX_POOL:
            self.CX_prob[cx] = CX_diff[cx]/sum_CX_diff

    def __calculate_fitness_difference(self, list_trial_ind):
        f_max = max([ind.fvalue for ind in list_trial_ind])
        F_diff = [f_max-ind.fvalue for ind in list_trial_ind]
        MUT_diff = {key: 0.0 for key in self.MUT_POOL}
        CX_diff = {key: 0.0 for key in self.CX_POOL}
        for i, diff in enumerate(F_diff):
            mut = self.configs_assignment[i]["mutation"]
            cx = self.configs_assignment[i]["crossover"]
            MUT_diff[mut] += diff
            CX_diff[cx] += diff
        return MUT_diff, CX_diff

    def __assign_mutaion(self):
        if self.pop.nth_generation < self.Gs:
            list_MUT = ["de/rand/1" for i in range(self.para.N)]
        else:
            # [Option 1] Roulette selection
            # list_MUT = [self.__roulette_wheel_selection(self.MUT_POOL,
            #   self.MUT_prob) for i in range(self.para.N)]
            # [Option 2] Uniform random selection
            list_MUT = self.__stochastic_universal_selection(self.MUT_POOL,
                self.MUT_prob)
        return list_MUT

    def __assign_crossover(self):
        if self.pop.nth_generation < self.Gs:
            list_CX = ["bin" for i in range(self.para.N)]
        else:
            # [Option 1] Roulette selection
            # list_CX = [self.__roulette_wheel_selection(self.CX_POOL,
            #   self.CX_prob) for i in range(self.para.N)]
            # [Option 2] Uniform random selection
            list_CX = self.__stochastic_universal_selection(self.CX_POOL,
                self.CX_prob)
        return list_CX

    def __sample_F_Cauchy(self, loc_para, scale_para):
        F = abs(cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng))
        if self.para_range_handle == "regenerate":
            while F <= self.F_RANGE[0] or F >= self.F_RANGE[1]:
                F = abs(cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng))
        elif  self.para_range_handle == "truncate":
            if F <= self.F_RANGE[0]:
                F = self.F_RANGE[0]
            if F >=  self.F_RANGE[1]:
                F = self.F_RANGE[1]
        return F

    def __sample_F_opp_learning(self, loc_para, scale_para):
        F = abs(1-cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng))
        if self.para_range_handle == "regenerate":
            while F <= self.F_RANGE[0] or F >= self.F_RANGE[1]:
                F = abs(1-cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng))
        elif self.para_range_handle == "truncate":
            if F <=  self.F_RANGE[0]:
                F = self.F_RANGE[0]
            if F >=  self.F_RANGE[1]:
                F = self.F_RANGE[1]
        return F

    def __assign_F(self):
        loc_para = 1.0 - 0.6*self.pop.nth_generation/self.Gmax
        scale_para = 0.8 - 0.6*(1.0 - (self.pop.nth_generation/self.Gmax)**2)
        r = self.para.rng.rand()
        if r > self.op_learning_ratio:
            list_F = [self.__sample_F_opp_learning(loc_para, scale_para)
                      for i in range(self.para.N)]
        else:
            list_F = [self.__sample_F_Cauchy(loc_para, scale_para)
                      for i in range(self.para.N)]
        return list_F

    def __sample_CR_normal(self, local_para, scale_para):
        CR = norm.rvs(loc=local_para, scale=scale_para, random_state=self.para.rng)
        if self.para_range_handle == "regenerate":
            while CR <= self.CR_RANGE[0] or CR >= self.CR_RANGE[1]:
                CR = norm.rvs(loc=local_para, scale=scale_para, random_state=self.para.rng)
        elif self.para_range_handle == "truncate":
            if CR <= self.CR_RANGE[0]:
                CR = self.CR_RANGE[0]
            if CR >= self.CR_RANGE[1]:
                CR = self.CR_RANGE[1]
        return CR

    def __sample_CR_opp_learning(self, local_para, scale_para):
        CR = 1.0 - norm.rvs(loc=local_para, scale=scale_para, random_state=self.para.rng)
        if self.para_range_handle == "regenerate":
            while CR <= self.CR_RANGE[0] or CR >= self.CR_RANGE[1]:
                CR = 1.0 - norm.rvs(loc=local_para, scale=scale_para, random_state=self.para.rng)
        elif self.para_range_handle == "truncate":
            if CR <= self.CR_RANGE[0]:
                CR = self.CR_RANGE[0]
            if CR >= self.CR_RANGE[1]:
                CR = self.CR_RANGE[1]
        return CR

    def __assign_CR(self):
        local_para = 1.0 - 0.7*(1.0 - self.pop.nth_generation/self.Gmax)
        scale_para = 0.8 - 0.6*(1.0 - (self.pop.nth_generation/self.Gmax)**2)
        r = self.para.rng.rand()
        if r > self.op_learning_ratio:
            list_CR = [self.__sample_CR_opp_learning(local_para, scale_para)
                       for i in range(self.para.N)]
        else:
            list_CR = [self.__sample_CR_normal(local_para, scale_para)
                      for i in range(self.para.N)]
        return list_CR

    def __roulette_wheel_selection(self, items_pool, items_probability):
        '''Roulette wheel selection scheme to select an item from the pool
        according to probability of items.

        Parameters
        ----------
        items_pool: list, a pool of items from which an item is selected
        items_probability: list of floats, each element is the
                selection probability of the respect item. '''
        if len(items_pool) != len(items_probability):
            raise ValueError("`items_pool` and `items_probability` msut have "
                             "the same length.")
        i = 0
        cumulative_prob = items_probability[items_pool[i]]
        r = self.para.rng.rand()
        while cumulative_prob < r:
            i += 1
            cumulative_prob += items_probability[items_pool[i]]
        return items_pool[i]

    def __stochastic_universal_selection(self, items_pool, items_probability):
        '''Use Stochastic Universal Sampling (SUS) to select one generation strategy
        for each target vector.
        Ref of SUS: Book-Computational Itelligence: An Introduction. p.p.137'''
        spacing = 1.0/self.para.N
        r = self.para.rng.rand()/self.para.N
        prob_sum = 0.0
        list_name = [] # save the names of items
        for item in items_pool:
            prob_sum += items_probability[item]
            while r < prob_sum:
                list_name.append(item)
                r += spacing
        # randomly shuffle the list of strategy names
        self.para.rng.shuffle(list_name)
        return list_name

    def plot_MUT_CX_n_hist(self, skip=1, aggregate_method="average"):
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        fig.subplots_adjust(hspace=0.2)
        x = self.data["nth_hist"]
        x_skip = skip_and_aggregate_of_list_data(x, skip, aggregate_method="none")
        # plot MUT_n_hist --------------
        mut_n_hist = self.data["MUT_n_hist"]
        for mut in self.MUT_POOL:
            y = mut_n_hist[mut]
            y_skip = skip_and_aggregate_of_list_data(y, skip, aggregate_method)
            ax1.plot(x_skip, y_skip, label=self._upper_de_in_mutation(mut))
        ax1.set(xlabel="Generation", ylabel="Number of Individuals",
                xlim=(0, self.pop.nth_generation),
                title="Evolution of Mutation Operator")
        ax1.legend(loc="best", frameon=False)
        # plot plot CX_n_hist  --------------------------
        cx_n_hist = self.data["CX_n_hist"]
        for cx in self.CX_POOL:
            y = cx_n_hist[cx]
            y_skip = skip_and_aggregate_of_list_data(y, skip, aggregate_method)
            ax2.plot(x_skip, y_skip, label=cx)
        ax2.set(xlabel="Generation", ylabel="$Number of Individuals$",
                xlim=(0, self.pop.nth_generation),
                title="Evolution of Crossover Operator")
        ax2.legend(loc="best", frameon=False)
        plt.show()

    def plot_F_CR_mean_std_hist(self, skip=20, aggregate_method="average"):
        raw_data = self.data.get("configs_assignment_hist")
        F_mean_hist = []
        F_std_hist = []
        CR_mean_hist = []
        CR_std_hist = []
        for pop_configs in raw_data:
            F_pop = [config["F"] for config in pop_configs]
            CR_pop = [config["CR"] for config in pop_configs]
            F_mean_hist.append(np.mean(F_pop))
            F_std_hist.append(np.std(F_pop, ddof=1))
            CR_mean_hist.append(np.mean(CR_pop))
            CR_std_hist.append(np.std(CR_pop, ddof=1))
        # plot
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        fig.subplots_adjust(hspace=0.2)
        x = self.data["nth_hist"]
        x_skip = skip_and_aggregate_of_list_data(x, skip, aggregate_method="none")
        # plot F_hist --------------
        Theory_Fmean = [1.0 - 0.6*i/self.Gmax for i in x]
        Theory_CRmean = [1.0 - 0.7*(1.0 - i/self.Gmax) for i in x]
        Theory_std = [ 0.8 - 0.6*(1.0-(i/self.Gmax)**2) for i in x]
        F_mean_skip = skip_and_aggregate_of_list_data(F_mean_hist, skip, aggregate_method)
        F_std_skip = skip_and_aggregate_of_list_data(F_std_hist, skip, aggregate_method)
        ax1.plot(x_skip, F_mean_skip, label="Fmean")
        ax1.plot(x_skip, F_std_skip, label="Fstd")
        ax1.plot(x, Theory_Fmean, label="Theory_Fmean", linestyle="--")
        ax1.plot(x, Theory_std, label="Theory_Fstd", linestyle="--")
        ax1.set(xlabel="Generation", ylabel="F",
                xlim=(0, self.pop.nth_generation),
                title="Evolution of F")
        ax1.legend(loc="best", frameon=False)
        # plot CR  --------------------------
        CR_mean_skip = skip_and_aggregate_of_list_data(CR_mean_hist, skip, aggregate_method)
        CR_std_skip = skip_and_aggregate_of_list_data(CR_std_hist, skip, aggregate_method)
        ax2.plot(x_skip, CR_mean_skip, label="CRmean")
        ax2.plot(x_skip, CR_std_skip, label="CRstd")
        ax2.plot(x, Theory_CRmean, label="Theory_CRmean", linestyle="--")
        ax2.plot(x, Theory_std, label="Theory_CRstd", linestyle="--")
        ax2.set(xlabel="Generation", ylabel="CR",
                xlim=(0, self.pop.nth_generation),
                title="Evolution of CR")
        ax2.legend(loc="best", frameon=False)

        plt.show()


class JADE(_DEVariant):
    ''':class:`JADE` implements the Adaptive Differential Evolution with
    Optional External Archive.

    Ref:
        J. Zhang and A. C. Sanderson, "JADE: Adaptive Differential Evolution
        With Optional External Archive," in IEEE Transactions on
        Evolutionary Computation, vol. 13, no. 5, pp. 945-958, Oct. 2009.'''

    ALGO_NAME = "JADE"

    F_RANGE  = (0.0, 1.0)
    CR_RANGE = (0.0, 1.0)

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                 stop_conditions=para.StopCondition(),
                 constant_c=0.1, probability_p=0.05,
                 with_external_archive=True):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.c = constant_c
        self.p = probability_p
        self.with_external_archive = with_external_archive
        self.archive = []
        self.F_mean = 0.5
        self.CR_mean = 0.5
        self._F_mean_hist = []
        self._CR_mean_hist = []

    def _initiate_generation_0(self):
        super()._initiate_generation_0()
        # Assign configs for generation 0
        self._assign_configs()
        self._F_mean_hist.append(self.F_mean)
        self._CR_mean_hist.append(self.CR_mean)

    def _evolve_a_generation(self):
        self.__gen_success_F = []
        self.__gen_success_CR = []
        # Initialize two list to save trial ind and ind selected to enter next gen:
        list_trial_ind = []
        list_next_gen_ind = []
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int(min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # -------------------- Evolution of Population ------------------------
        # perform evolution using the assigned config for each target vector
        for idx in range(n_trials):
            # Generation Step:create a trial individual by mutation and crossover
            self._get_config_from_assignment(idx)
            # record the changed hyperparameters to history
            self._record_hyperparameter()
            # Generation Step:create a trial individual by mutation and crossover
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR,
                                                  p=self.p, archive=self.archive)
            # Evaluation Step: evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            list_trial_ind.append(trial_ind)
            # Selection Step: survival selection
            target_ind = self.pop.list_ind[idx]
            if trial_ind.fvalue < target_ind.fvalue:
                list_next_gen_ind.append(trial_ind)
                self.__gen_success_F.append(self.para.F)
                self.__gen_success_CR.append(self.para.CR)
                if self.with_external_archive:
                    self.archive.append(target_ind)
            else:
                list_next_gen_ind.append(target_ind)
        # Incease the generation count and update the population:
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Randomly remove some inds from archive so that len(archive) <= self.para.N
        while len(self.archive) > self.para.N:
            idxs_A = list(range(len(self.archive)))
            self.para.rng.shuffle(idxs_A)
            self.archive.pop(idxs_A[0])
        # update F_mean and CX_mean
        self._update_F_mean()
        self._update_CR_mean()
        # Assign configs for next generation
        self._assign_configs()
        self._F_mean_hist.append(self.F_mean)
        self._CR_mean_hist.append(self.CR_mean)

    def _collect_result(self):
        result = super()._collect_result()
        result["F_mean_hist"] = self._F_mean_hist
        result["CR_mean_hist"] = self._CR_mean_hist
        return result

    def _assign_configs(self):
        '''Assign a configuration for each target vector'''
        list_F = self.__assign_F()
        list_CR = self.__assign_CR()
        configs = [{"mutation": "de/current-to-pbest/1",
                    "crossover": "bin",
                    "F": F, "CR": CR}
                   for F, CR in zip(list_F, list_CR)]
        self.configs_assignment = configs

    def _get_config_from_assignment(self, idx):
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]

    def _update_F_mean(self):
        '''Update the mean of F'''
        if self.__gen_success_F != []:
            SF = np.array(self.__gen_success_F)
            SF_L_mean = sum(SF**2) / sum(SF)
            self.F_mean = (1.0- self.c) * self.F_mean + self.c * SF_L_mean

    def _update_CR_mean(self):
        if self.__gen_success_CR != []:
            SCR_A_mean = sum(self.__gen_success_CR)/ len(self.__gen_success_CR)
            self.CR_mean = (1.0- self.c) * self.CR_mean + self.c * SCR_A_mean

    def __assign_F(self):
        list_F = [self.__sample_F_Cauchy(loc_para=self.F_mean, scale_para=0.1)
                  for i in range(self.para.N)]
        return list_F

    def __sample_F_Cauchy(self, loc_para, scale_para):
        F = cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng)
        while F <= self.F_RANGE[0]:
            F = cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng)
        if F >= self.F_RANGE[1]:
            F = self.F_RANGE[1]
        return F

    def __assign_CR(self):
        list_CR = [self.__sample_CR_normal(local_para=self.CR_mean, scale_para=0.1)
                   for i in range(self.para.N)]
        return list_CR

    def __sample_CR_normal(self, local_para, scale_para):
        CR = norm.rvs(loc=local_para, scale=scale_para, random_state=self.para.rng)
        if CR <= self.CR_RANGE[0]:
            CR = self.CR_RANGE[0]
        if CR >= self.CR_RANGE[1]:
            CR = self.CR_RANGE[1]
        return CR

    def plot_adaptation_hist(self, skip=1, aggregate_method="average"):
        '''Plot the adaptation history of Strategy Probability and CR Mean.
        Since DE/current-to-rand/1/none does not use crossover, CR mean
        with respect to this strategy is not plotted.
        '''
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()

        x = self.data["nth_hist"]
        x_skip = skip_and_aggregate_of_list_data(x, skip, aggregate_method="none")
        # Plot F_mean_hist and CR_mean_hist --------------
        F_mean_hist = self.data["F_mean_hist"]
        CR_mean_hist = self.data["CR_mean_hist"]
        F_mean_skip = skip_and_aggregate_of_list_data(F_mean_hist, skip, aggregate_method)
        CR_mean_skip = skip_and_aggregate_of_list_data(CR_mean_hist, skip, aggregate_method)

        ax.plot(x_skip, F_mean_skip, label="$\mu_F$")
        ax.plot(x_skip, CR_mean_skip, label="$\mu_{CR}$")
        ax.set(xlabel="Generation", ylabel="$Mean of F and CR$",
               xlim=(0, self.pop.nth_generation),
               title="Adaptation of $\mu_F$ and $\mu_{CR}$ of JADE")
        ax.legend(loc="best", frameon=False)
        plt.show()


class SHADE(_DEVariant):
    ''':class:`SHADE` implements the Success-History based Adaptive DE, which is
    an enhancement to JADE.

    Ref:
        R. Tanabe and A. Fukunaga, "Success-history based parameter adaptation
        for Differential Evolution," 2013 IEEE Congress on Evolutionary Computation,
        Cancun, 2013, pp. 71-78.'''

    ALGO_NAME = "SHADE"

    F_RANGE  = (0.0, 1.0)
    CR_RANGE = (0.0, 1.0)

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                 stop_conditions=para.StopCondition(),
                 memory_size="pop_size",
                 p_min="2/pop_size", p_max=0.2,
                 with_external_archive=True):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        if memory_size == "pop_size":
            self.memory_size = self.para.N  # H in Ref
        elif isinstance(memory_size, int) and memory_size > 0:
            self.memory_size = memory_size
        else:
            raise ValueError("`memory_size` must be positive integer or a string of 'pop_size'. ")
        if p_min == "2/pop_size":
            self.p_min = 2/self.para.N
        elif isinstance(p_min, float) and (0.0<p_min<1.0):
            self.p_min = p_min
        else:
            raise ValueError("`p_min` must be float within (0, 1) or a string of '2/pop_size'. ")
        if isinstance(p_max, float) and p_max > self.p_min:
            self.p_max = p_max
        else:
            raise ValueError("`p_max` must be float > `self.p_min`")
        self.with_external_archive = with_external_archive
        self.archive = []
        self.p_assignment = None
        self.F_mean = [0.5] * self.memory_size
        self.CR_mean= [0.5] * self.memory_size
        self._gen_F_mean = 0.5
        self._gen_CR_mean = 0.5
        self._F_mean_hist = []
        self._CR_mean_hist = []
        self.__index_counter = 0 # k in Ref

    def _initiate_generation_0(self):
        super(SHADE, self)._initiate_generation_0()
        # Assign configs for generation 0
        self._assign_configs()
        self._F_mean_hist.append(self._gen_F_mean)
        self._CR_mean_hist.append(self._gen_CR_mean)

    def _evolve_a_generation(self):
        self.__gen_success_F = []
        self.__gen_success_CR = []
        self.__gen_f_improvement = []
        # Initialize two list to save trial ind and ind selected to enter next gen:
        list_trial_ind = []
        list_next_gen_ind = []
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int(min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # -------------------- Evolution of Population ------------------------
        # perform evolution using the assigned config for each target vector
        for idx in range(n_trials):
            # Generation Step:create a trial individual by mutation and crossover
            self._get_config_from_assignment(idx)
            # record the changed hyperparameters to history
            self._record_hyperparameter()
            # Generation Step:create a trial individual by mutation and crossover
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR,
                                                  p=self.p_assignment[idx],
                                                  archive=self.archive)
            # Evaluation Step: evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            list_trial_ind.append(trial_ind)
            # Selection Step: survival selection
            target_ind = self.pop.list_ind[idx]
            if trial_ind.fvalue < target_ind.fvalue:
                list_next_gen_ind.append(trial_ind)
                self.__gen_success_F.append(self.para.F)
                self.__gen_success_CR.append(self.para.CR)
                self.__gen_f_improvement.append(target_ind.fvalue-trial_ind.fvalue)
                if self.with_external_archive:
                    self.archive.append(target_ind)
            else:
                list_next_gen_ind.append(target_ind)
        # Incease the generation count and update the population:
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Randomly remove some inds from archive so that len(archive) <= self.para.N
        while len(self.archive) > self.para.N:
            idxs_A = list(range(len(self.archive)))
            self.para.rng.shuffle(idxs_A)
            self.archive.pop(idxs_A[0])
        # update F_mean and CX_mean
        self._update_F_mean_and_CR_mean()
        # Assign configs for next generation
        self._assign_configs()
        self._F_mean_hist.append(self._gen_F_mean)
        self._CR_mean_hist.append(self._gen_CR_mean)

    def _collect_result(self):
        result = super(SHADE, self)._collect_result()
        result["F_mean_hist"] = self._F_mean_hist
        result["CR_mean_hist"] = self._CR_mean_hist
        return result

    def _assign_configs(self):
        '''Assign a configuration for each target vector'''
        list_F = self.__assign_F()
        list_CR = self.__assign_CR()
        configs = [{"mutation": "de/current-to-pbest/1",
                    "crossover": "bin",
                    "F": F, "CR": CR}
                   for F, CR in zip(list_F, list_CR)]
        self.configs_assignment = configs
        self.p_assignment = [self.para.rng.uniform(low=self.p_min, high=self.p_max)
                             for _ in range(self.para.N)]

    def _get_config_from_assignment(self, idx):
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]

    def _update_F_mean_and_CR_mean(self):
        '''Update the mean of F'''
        if self.__gen_success_F != []:
            weights = np.array(self.__gen_f_improvement) / sum(self.__gen_f_improvement)
            SF = np.array(self.__gen_success_F)
            SF_WL_mean = sum(weights* (SF**2)) / sum(weights*SF)
            SCR_WA_mean = sum(weights* np.array(self.__gen_success_CR))
            self._gen_F_mean = SF_WL_mean
            self._gen_CR_mean = SCR_WA_mean
            self.F_mean[self.__index_counter] = SF_WL_mean
            self.CR_mean[self.__index_counter] = SCR_WA_mean
            self.__index_counter += 1
            if self.__index_counter >= self.memory_size:
                self.__index_counter = 0

    def __assign_F(self):
        r_idxs = self.para.rng.randint(low=0, high=self.memory_size, size=self.para.N)
        list_F = [self.__sample_F_Cauchy(loc_para=self.F_mean[i], scale_para=0.1)
                  for i in r_idxs]
        return list_F

    def __sample_F_Cauchy(self, loc_para, scale_para):
        F = cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng)
        while F <= self.F_RANGE[0]:
            F = cauchy.rvs(loc=loc_para, scale=scale_para, random_state=self.para.rng)
        if F >= self.F_RANGE[1]:
            F = self.F_RANGE[1]
        return F

    def __assign_CR(self):
        r_idxs = self.para.rng.randint(low=0, high=self.memory_size, size=self.para.N)
        list_CR = [self.__sample_CR_normal(local_para=self.CR_mean[i], scale_para=0.1)
                   for i in r_idxs]
        return list_CR

    def __sample_CR_normal(self, local_para, scale_para):
        CR = norm.rvs(loc=local_para, scale=scale_para, random_state=self.para.rng)
        if CR <= self.CR_RANGE[0]:
            CR = self.CR_RANGE[0]
        if CR >= self.CR_RANGE[1]:
            CR = self.CR_RANGE[1]
        return CR

    def plot_adaptation_hist(self, skip=1, aggregate_method="average"):
        '''Plot the adaptation history of Strategy Probability and CR Mean.
        Since DE/current-to-rand/1/none does not use crossover, CR mean
        with respect to this strategy is not plotted.
        '''
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()

        x = self.data["nth_hist"]
        x_skip = skip_and_aggregate_of_list_data(x, skip, aggregate_method="none")
        # Plot F_mean_hist and CR_mean_hist --------------
        F_mean_hist = self.data["F_mean_hist"]
        CR_mean_hist = self.data["CR_mean_hist"]
        F_mean_skip = skip_and_aggregate_of_list_data(F_mean_hist, skip, aggregate_method)
        CR_mean_skip = skip_and_aggregate_of_list_data(CR_mean_hist, skip, aggregate_method)

        ax.plot(x_skip, F_mean_skip, label="$\mu_F$")
        ax.plot(x_skip, CR_mean_skip, label="$\mu_{CR}$")
        ax.set(xlabel="Generation", ylabel="$Mean of F and CR$",
               xlim=(0, self.pop.nth_generation),
               title="Adaptation of $\mu_F$ and $\mu_{CR}$ of JADE")
        ax.legend(loc="best", frameon=False)
        plt.show()


class jDE(_DEVariant):
    ''':class:`jDE` DE with self-adaptive parameters.
    Ref:
        J. Brest, S. Greiner, B. Boskovic, M. Mernik and V. Zumer,
        "Self-Adapting Control Parameters in Differential Evolution:
            A Comparative Study on Numerical Benchmark Problems,"
            in IEEE Transactions on Evolutionary Computation,
            vol. 10, no. 6, pp. 646-657, Dec. 2006.'''

    ALGO_NAME = "jDE"

    F_RANGE = (0.1, 1.0)
    CR_RANGE =(0.0, 1.0)

    def __init__(self, opt_problem, algo_parameters=para.DE(),
                       stop_conditions=para.StopCondition(),
                       F_lower=0.1, F_upper=0.9,
                       tau_F=0.1,   tau_CR=0.1):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.F_l = F_lower
        self.F_u = F_upper
        self.tau_F = tau_F
        self.tau_CR = tau_CR

    def _initiate_generation_0(self):
        '''Initialize a population, reset the algorithm, and assign configs for
        initial generaiton/population. '''
        super()._initiate_generation_0()
        # Assign configs for generation 0:
        self._assign_configs_for_init_gen()

    def _evolve_a_generation(self):
        '''Evolve one generation '''
        list_next_gen_ind = [] # to save the selected individuals for next generation
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int( min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # perform mutation, crossover and selection
        for idx in range(n_trials):
            # get a config from the configs assignment for current target ind:
            self._get_config_from_assignment(idx)
            # record the hyperparameters to history
            self._record_hyperparameter()
            # generate a trial individual
            trial_ind = self.pop.create_offspring(target_idx=idx,
                                                  mutation=self.para.mutation,
                                                  F=self.para.F,
                                                  crossover=self.para.crossover,
                                                  CR=self.para.CR)
            # evaluate the trial individual
            self.pb.evaluate_ind(trial_ind)
            # survival selection
            target_ind = self.pop.list_ind[idx]
            list_next_gen_ind.append(self.pop.survival_select(target_ind, trial_ind))
        # Update pop individuals and Incease the generation count
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Assign configs for next generation:
        self._adapt_config_assignment()

    def _assign_configs_for_init_gen(self):
        '''Each individual in the initial population '''
        init_config = {"mutation": "de/rand/1",
                       "crossover": "bin",
                       "F": 0.5,
                       "CR": 0.9}
        self.configs_assignment = [init_config for _ in range(self.para.N)]

    def _adapt_config_assignment(self):
        for i in range(self.para.N):
            r1, r2, r3, r4 = self.para.rng.rand(4)
            if r2 < self.tau_F:
                self.configs_assignment[i]["F"] = self.F_l + r1*self.F_u
            if r4 < self.tau_CR:
                self.configs_assignment[i]["CR"] = r3

    def _get_config_from_assignment(self, idx):
        '''Get the config from `self.configs_assignment` according to the index
        of target individual in the population. '''
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]

#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------

def skip_and_aggregate_of_list_data(list_data, skip, aggregate_method="none"):
    '''This function process the data list by skipping at given frenquencey
    and/or aggregate the data in every skip-length.

    Parameters:
    ----------
    list_data: list, data list
    skip: int, > 1,
    aggregate_method: str, "none", "average", "median", "max", "min", "standard_deviation"

    Return:
    new_list_data: list
    '''
    assert isinstance(list_data, list)
    assert isinstance(skip, int)
    assert isinstance(aggregate_method, str)

    if skip == 1:
        new_list_data = deepcopy(list_data)
    elif skip > 1:
        point_idxs = list(range(0, len(list_data), skip))
        if aggregate_method == "none":
            new_list_data = [list_data[i] for i in point_idxs]
        elif aggregate_method == "average":
            new_list_data = [sum(list_data[point_idxs[i]:point_idxs[i+1]])/skip
                             for i in range(len(point_idxs)-1)]
            new_list_data = [list_data[0]] + new_list_data
        elif aggregate_method == "median":
            new_list_data = [np.median(list_data[point_idxs[i]:point_idxs[i+1]])
                             for i in range(len(point_idxs)-1)]
            new_list_data = [list_data[0]] + new_list_data
        elif aggregate_method == "max":
            new_list_data = [max(list_data[point_idxs[i]:point_idxs[i+1]])
                             for i in range(len(point_idxs)-1)]
            new_list_data = [list_data[0]] + new_list_data
        elif aggregate_method == "min":
            new_list_data = [min(list_data[point_idxs[i]:point_idxs[i+1]])
                             for i in range(len(point_idxs)-1)]
            new_list_data = [list_data[0]] + new_list_data
        elif aggregate_method == "standard_deviation":
            new_list_data = [np.std(list_data[point_idxs[i]:point_idxs[i+1]], ddof=1)
                             for i in range(len(point_idxs)-1)]
            new_list_data = [list_data[0]] + new_list_data
        else:
            ValueError("`aggregate_method:` {} is not supported !".format(aggregate_method))
    else:
        raise ValueError("`skip` must be an integer not smaller than 1.")
    return new_list_data



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    pass


if __name__ == "__main__":
    data = main()

