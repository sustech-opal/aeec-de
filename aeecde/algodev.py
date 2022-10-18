#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Algorithms in developping
'''
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

# internal imports
import aeecde
import tools
# import parameterize as para
from tuner import *
from algorithms.DE import _DEVariant
from algorithms.DE import skip_and_aggregate_of_list_data

# HB : the following imports are for personal purpose
try:
    import sys, IPython
    sys.excepthook = IPython.core.ultratb.ColorTB()
except:
    pass



#!-----------------------------------------------------------------------------
#!                                     CLASSES
#!-----------------------------------------------------------------------------
class AOSPCDE(_DEVariant):
    ''' AOSPCDE (Adaptive Operator Selection and Parameter Control DE)
    :class:`AOSPCDE` implements the DE considering the search state of each
    individual during the search process.
    Key features:
        - 3 groups of model are created: 1 for explorating, 1 for exploiting
          and 1 for measuring performance
        - each group includes 1 MAB bandit model and 2 KDE models for F and CR
          separativly
    '''

    ALGO_NAME = "AOSPCDE"

    _DEVariant.HIST_ITEMS.extend(["pop_state", "pop_sum_dist", "pop_avg_dist",
        "X"])

    MUT_POOL = [
                "de/rand/1", # CoDE, SaDE, EPSDE, SAKPDE, jDE
                "de/rand/2", # CoDE, SaDE, SAKPDE
                # "de/best/1",
                # "de/best/2", # EPSDE, SAKPDE
                # "de/current/1",
                # "de/rand-to-best/1",
                # "de/current-to-best/1", # SAKPDE
                # "de/current-to-best/2", # SaDE, SAKPDE
                "de/current-to-rand/1", # CoDE, SaDE, EPSDE
                "de/current-to-pbest/1", # JADE, SHADE
                # "de/rand-to-pbest/1"
                ]

    CX_POOL = ["bin", "exp",
            #    "eig",
            #    "none",
               ]

    F_RANGE  = (0.0, 1.0) #! to conform with the reference DEs
    CR_RANGE = (0.0, 1.0)

    def __init__(self, opt_problem, algo_parameters, stop_conditions,
                 learning_gen, state_threshold, stagnation_gen,
                 model_update_frequency, model_utilization_strategy,
                 KDE_kernel, state_KDE_max_size,
                 para_KDE_width_F, para_KDE_width_CR, para_KDE_max_size,
                 bandit_algo, bandit_value_method,
                 **kw_bandit_args):
        super().__init__(opt_problem, algo_parameters, stop_conditions)
        self.learning_gen = learning_gen
        self.state_threshold = state_threshold # "median" or "quantile"
        self.stagnation_gen = stagnation_gen #  number of gen for which the best solution does not improved
        self.update_frequency = model_update_frequency # "each_ind" or "each_gen"
        self.model_utilization_strategy = model_utilization_strategy # "strategy1", "strategy2"
        # Each arm is a combination of mutation and crossover operator, i.e., a generation strategy
        generation_strategies = []
        for mut in self.MUT_POOL:
            for cx in self.CX_POOL:
                if mut == "de/current-to-rand/1":
                    cx = "none"
                generation_strategies.append((mut, cx))
        self.generation_strategies = generation_strategies
        arms = sorted(set(generation_strategies),key=generation_strategies.index)
        # A KDE model used to delimitate the state of individuals
        self.state_kde = KDE(kernel=KDE_kernel,
                             bandwidth=None,
                             value_range=None,
                             features_name=None,
                             max_training_size=state_KDE_max_size,
                             random_state=self.para.rng)
        # self.state_kde.set_params(rtol=1e-3, atol=1e-8)
        # self.state_kde.set_params(rtol=1e-3, atol=1e-4)
        # self.state_kde.set_params(atol=1e-8)
        self.state_kde.set_params(rtol=1e-4, atol=1e-8)
        # ----------------------------------------------------------------------
        #* A bandit model which focuses on the exploration ability of each arm
        self.explore_bandit = multi_armed_bandit(bandit_algo=bandit_algo,
                                                 list_arm_names=arms,
                                                 random_state=self.para.rng,
                                                 value_estimation_method=bandit_value_method,
                                                 **kw_bandit_args)
        F1 = {arm: KDE(kernel=KDE_kernel,
                       bandwidth=para_KDE_width_F,
                       value_range=self.F_RANGE,
                       features_name = ["F", ],
                       max_training_size=para_KDE_max_size,
                       random_state=self.para.rng)
              for arm in arms}
        CR1 = {arm: KDE(kernel=KDE_kernel,
                       bandwidth=para_KDE_width_CR,
                       value_range=self.CR_RANGE,
                       features_name = ["CR", ],
                       max_training_size=para_KDE_max_size,
                       random_state=self.para.rng)
               for arm in arms}
        self.explore_kde = {"F":F1, "CR":CR1}
        #* A bandit model which focuses on the exploitation ability of each arm
        self.exploit_bandit = multi_armed_bandit(bandit_algo=bandit_algo,
                                                 list_arm_names=arms,
                                                 random_state=self.para.rng,
                                                 value_estimation_method=bandit_value_method,
                                                 **kw_bandit_args)
        F2 = {arm: KDE(kernel=KDE_kernel,
                       bandwidth=para_KDE_width_F,
                       value_range=self.F_RANGE,
                       features_name=["F", ],
                       max_training_size=para_KDE_max_size,
                       random_state=self.para.rng)
              for arm in arms}
        CR2 = {arm: KDE(kernel=KDE_kernel,
                        bandwidth=para_KDE_width_CR,
                        value_range=self.CR_RANGE,
                        features_name=["CR", ],
                        max_training_size=para_KDE_max_size,
                        random_state=self.para.rng)
               for arm in arms}
        self.exploit_kde = {"F": F2, "CR": CR2}
        #* A bandit model which focuses on the performance of each arm
        self.success_bandit = multi_armed_bandit(bandit_algo=bandit_algo,
                                                 list_arm_names=arms,
                                                 random_state=self.para.rng,
                                                 value_estimation_method=bandit_value_method,
                                                 **kw_bandit_args)
        F3 = {arm: KDE(kernel=KDE_kernel,
                       bandwidth=para_KDE_width_F,
                       value_range=self.F_RANGE,
                       features_name=["F", ],
                       max_training_size=para_KDE_max_size,
                       random_state=self.para.rng)
              for arm in arms}
        CR3 = {arm: KDE(kernel=KDE_kernel,
                        bandwidth=para_KDE_width_CR,
                        value_range=self.CR_RANGE,
                        features_name=["CR", ],
                        max_training_size=para_KDE_max_size,
                        random_state=self.para.rng)
               for arm in arms}
        self.success_kde = {"F": F3, "CR": CR3}
        # ----------------------------------------------------------------------
        self.archive = []
        self.gen_effective = 0
        self.gen_ineffective = 0
        self.state_threshold_dens = None  # threshold of density to delimitate exploration and exploitation
        self.pop_states = {"explore": [], "exploit": [], "neutral": []}
        self.explore_rates = {arm: [] for arm in arms}
        self.exploit_rates = {arm: [] for arm in arms}
        self.success_rates = {arm: [] for arm in arms}
        self.config_strategy = []
        self.improve_history = []
        self.improve_moving_average = 1/self.stop.delta_ftarget
        self.x_opt_history = []
        self.f_opt_history = []

    def _initiate_generation_0(self):
        '''Initialize a population, reset the algorithm, and assign configs for
        initial generaiton/population. '''
        super()._initiate_generation_0()
        self.pop.get_X()
        self.state_kde.add_batch_data((self.pop.X).tolist())

    def _update_pop_attribute(self):
        # Update the population's attributes
        self.pop.get_F()
        self.pop.get_X()
        # self.pop.find_best_individual()
        best_ind = self.pop.find_best_individual()
        # mut = best_ind.mutation
        # cx = best_ind.crossover
        # F = best_ind.F
        # CR = best_ind.CR
        if self.pop.nth_generation > 0:
            if self.pop.F_best < self.pb.f_opt:
                self.improve_history.append(abs(self.pop.F_best-self.pb.f_opt))
                self.improve_moving_average = sum(self.improve_history[-10:])/10
                self.gen_effective += 1
                self.gen_ineffective = 0
                # if mut is not None:
                #     self.best_bandit.update(chosen_arm=(mut, cx), reward=1.0)
                #     self.best_kde[(mut, cx)].add_data([F, CR])
            else:
                self.gen_ineffective += 1
                self.gen_effective = 0
                # if mut is not None:
                #     self.best_bandit.update(chosen_arm=(mut, cx), reward=0.0)
        # Update best solution found so far
        self.pb.update_opt(self.pop.X_best, self.pop.F_best)
        # calcualte pop distance
        ind_dist = self.pop.get_ind_distance(tools.scn2)  # * distance measure
        self.pop.get_sum_distance()
        self.pop.get_avg_distance()
        self.pop.min_distance = min(ind_dist)
        self.pop.max_distance = max(ind_dist)
        self.pop.median_distance = np.median(ind_dist)

    def _evolve_a_generation(self):
        '''Evolve one generation '''
        # reset the pop state:
        self.pop.state["explore"] = 0
        self.pop.state["exploit"] = 0
        self.pop.state["neutral"] = 0

        list_trial_ind = []
        list_next_gen_ind = [] # to save the selected individuals for next generation
        # Determine how many trial individuals can be generated in this generation:
        n_trials = int( min(self.pop.size, self.stop.max_FES - self.pb.num_calls))
        # Perform mutation, crossover and selection
        if self.update_frequency == "each_ind":
            self._update_state_threshold()
            self.configs_assignment = []
            for idx in range(n_trials):
                # self._update_state_threshold()
                # Generate a config for current target ind:
                config = self._generate_configs(n_configs=1)
                self.configs_assignment.append(config)
                # print(self.configs_assignment)
                # Get a config from the configs assignment for current target ind:
                self._get_config_from_assignment(idx)
                # Record the hyperparameters to history
                self._record_hyperparameter()
                # Generate a trial individual
                trial_ind = self.pop.create_offspring(target_idx=idx,
                    mutation=self.para.mutation,
                    F=self.para.F,
                    crossover=self.para.crossover,
                    CR=self.para.CR,
                    p=0.1, archive=self.archive)
                # Evaluate the trial individual
                self.pb.evaluate_ind(trial_ind)
                list_trial_ind.append(trial_ind)
                # update search state and bandit/kde data
                survival_ind = self._update_model_each_ind(trial_ind, self.pop.list_ind[idx])
                list_next_gen_ind.append(survival_ind)
        elif self.update_frequency == "each_gen":
            self._update_state_threshold()
            # generate configs for current generation
            configs = self._generate_configs(n_configs=n_trials)
            # print(configs)
            self.configs_assignment = configs
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
                                                      CR=self.para.CR,
                                                      p=0.1, archive=self.archive)
                # Evaluate the trial individual
                self.pb.evaluate_ind(trial_ind)
                list_trial_ind.append(trial_ind)
            # update search state and bandit/kde data
            list_next_gen_ind = self._update_model_each_gen(list_trial_ind, self.pop.list_ind)
        else:
            raise ValueError("`model_update_frequency` must be 'each_ind' or 'each_gen'.")
        # collect pop state and bandits values
        self._collect_pop_states_and_bandit_values()
        # Update pop individuals and Incease the generation count
        self.pop.list_ind = list_next_gen_ind
        self.pop.nth_generation += 1
        # Randomly remove some inds from archive so that len(archive) <= self.para.N
        while len(self.archive) > self.para.N:
            idxs_A = list(range(len(self.archive)))
            self.para.rng.shuffle(idxs_A)
            self.archive.pop(idxs_A[0])
        # print("Archive size: ", len(self.archive))

    def _update_state_threshold(self):
        #! Set the bandwidth of state kde as the avg_distance of current pop
        # self.state_kde.bandwidth = self.pop.min_distance
        # self.state_kde.bandwidth = self.pop.max_distance
        # self.state_kde.bandwidth = self.pop.avg_distance
        self.state_kde.bandwidth = self.pop.median_distance
        # if self.pop.nth_generation % 5 == 0:
        #     self.state_kde.fit()
        self.state_kde.fit()
        log_dens = self.state_kde.score_samples(self.state_kde.training_data)
        if self.state_threshold == "median":
            self.state_threshold_dens = [np.quantile(log_dens, 0.5)] * 2
        elif self.state_threshold == "quantile_4":
            self.state_threshold_dens =  np.quantile(log_dens, [0.25, 0.75])
        elif self.state_threshold == "quantile_3":
            self.state_threshold_dens =  np.quantile(log_dens, [1/3, 2/3])
        else:
            raise ValueError("`state_threshold` is not supported, it must be"
                " 'median', 'quantile_4' or 'quantile_3', not '{}'".format(
                self.state_threshold))

    def _update_model_each_ind(self, trial_ind, target_ind):
        # 1. check state of the trial_ind & update the explore/exploit bandit and kde models
        log_dens = self.state_kde.get_log_density([trial_ind.xvalue,])
        arm = (trial_ind.mutation, trial_ind.crossover)
        F, CR = trial_ind.F, trial_ind.CR
        if log_dens > self.state_threshold_dens[1]:
            self.pop.state["exploit"] += 1
            self.exploit_bandit.update(chosen_arm=arm, reward=1.0)
            self.exploit_kde["F"][arm].add_data([F,])
            self.exploit_kde["CR"][arm].add_data([CR,])
            self.explore_bandit.update(chosen_arm=arm, reward=0.0)
        elif log_dens < self.state_threshold_dens[0]:
            self.pop.state["explore"] += 1
            self.explore_bandit.update(chosen_arm=arm, reward=1.0)
            self.explore_kde["F"][arm].add_data([F,])
            self.explore_kde["CR"][arm].add_data([CR,])
            self.exploit_bandit.update(chosen_arm=arm, reward=0.0)
        else:
            self.pop.state["neutral"] += 1
        # 3. Survival selection and update success bandit and kde
        if trial_ind.fvalue < target_ind.fvalue:
            survival_ind = deepcopy(trial_ind)
            self.success_bandit.update(chosen_arm=arm, reward=1.0)
            self.success_kde["F"][arm].add_data([F,])
            self.success_kde["CR"][arm].add_data([CR,])
        else:
            survival_ind = deepcopy(target_ind)
            self.success_bandit.update(chosen_arm=arm, reward=0.0)
        # 2. add trial_ind into state kde:
        self.state_kde.add_data(list(trial_ind.xvalue))
        return survival_ind

    def _update_model_each_gen(self, list_trial_ind, list_target_ind):
        list_survival_ind = [self._update_model_each_ind(trial_ind, target_ind)
                             for trial_ind, target_ind in zip(list_trial_ind, list_target_ind)]
        return list_survival_ind

    def _collect_pop_states_and_bandit_values(self):
        for key in self.pop.state:
            self.pop_states[key].append(self.pop.state[key])
        for arm in self.explore_bandit.arms:
            self.explore_rates[arm].append(self.explore_bandit.values[arm])
            self.exploit_rates[arm].append(self.exploit_bandit.values[arm])
            self.success_rates[arm].append(self.success_bandit.values[arm])

    def _generate_configs(self, n_configs):
        if self.model_utilization_strategy == "uniform":
            configs = self.__generate_uniform_configs(n_configs=n_configs)
            self.config_strategy.append("random_select")
            return configs

        elif self.model_utilization_strategy == "random":
            configs = self.__generate_random_configs(n_configs=n_configs)
            self.config_strategy.append("random_select")
            return configs

        elif self.model_utilization_strategy == "strategy_1":
            if self.pop.nth_generation < self.learning_gen:
                configs = self.__generate_uniform_configs(n_configs=n_configs)
                self.config_strategy.append("uniform_select")
                # print("generation {}: uniform configs".format(self.pop.nth_generation))
            else:
                cond1 = (self.gen_ineffective >= self.stagnation_gen)
                cond2 = (self.gen_effective >= self.stagnation_gen)
                cond3 = (self.improve_moving_average < self.stop.delta_ftarget*1000)

                if cond1:
                    if cond3:
                        # print("generation {}: cond1 and cond3 avg={}".format(
                        #     self.pop.nth_generation, self.improve_moving_average))
                        #* Re-initialize the model
                        for kde_dict in (self.explore_kde, self.exploit_kde):
                            for v in kde_dict["F"].values():
                                v.reset()
                            for v in kde_dict["CR"].values():
                                v.reset()
                        configs = self.__generate_random_configs(
                            n_configs=n_configs)
                        self.config_strategy.append("random_select")
                    else:
                        # if self.pb.num_calls <= self.stop.max_FES * 2/3:
                        configs1 = self.__generate_explore_configs(
                            n_configs=int(n_configs/4))
                        configs2 = self.__generate_exploit_configs(
                            n_configs=int(n_configs/4))
                        configs3 = self.__generate_uniform_configs(
                            n_configs=int(n_configs/4))
                        configs4 = self.__generate_random_configs(
                            n_configs=int(n_configs/4))
                        configs = configs1+configs2+configs3+configs4
                        self.config_strategy.append("other")
                        # print("generation {}: 4 configs".format(self.pop.nth_generation))
                        # else:
                        #     configs = self.__generate_exploit_configs(n_configs=n_configs)
                        #     self.config_strategy.append("exploit_bandit_kde")
                        #     print("generation {}: exploitation configs".format(self.pop.nth_generation))
                elif cond2 and cond3:
                        # print("generation {}: cond2 and cond3 avg={}".format(self.pop.nth_generation, self.improve_moving_average))
                        self.x_opt_history.append(self.pb.x_opt)
                        self.f_opt_history.append(self.pb.f_opt)
                        #* Re-initialize the population
                        X0 = self.initialize_X0(
                            initial_scheme=self.para.initial_scheme,
                            pop_size=self.para.N,
                            criterion="classic",
                        )
                        self.pop.initialize(X0)
                        # Evaluate the population
                        for ind in self.pop:
                            self.pb.evaluate_ind(ind)
                        self._update_pop_attribute()
                        # Reset other parameters
                        self.gen_effective, self.gen_ineffective = 0, 0
                        self.improve_history = []
                        self.improve_moving_average = 1/self.stop.delta_ftarget
                        #* generate configurations
                        configs = self.__generate_uniform_configs(
                            n_configs=n_configs)
                        self.config_strategy.append("reset")
                        # print("generation {}: RE-INITIALIZE".format(
                        #     self.pop.nth_generation))
                else:
                    configs = self.__generate_success_configs(n_configs=n_configs)
                    self.config_strategy.append("success_bandit_kde")
                    # print("generation {}: success configs".format(self.pop.nth_generation))
            return configs

        elif self.model_utilization_strategy == "strategy_2":
            if self.pop.nth_generation < self.learning_gen:
                configs = self.__generate_uniform_configs(n_configs=n_configs)
                self.config_strategy.append("random_select")
                # print("generation {}: uniform configs".format(self.pop.nth_generation))
            else:
                if self.gen_ineffective >= self.stagnation_gen:
                    if self.pb.num_calls <= self.stop.max_FES * 3/4:
                        if self.gen_ineffective >= 2*self.stagnation_gen:
                            configs = self.__generate_random_configs(n_configs=n_configs)
                            configs = self.__generate_uniform_configs(n_configs=n_configs)
                            self.config_strategy.append("random_select")
                        else:
                            configs = self.__generate_explore_configs(n_configs=n_configs)
                            self.config_strategy.append("explore_bandit_kde")
                        # print("generation {}: exploration configs".format(self.pop.nth_generation))
                    else:
                        configs = self.__generate_exploit_configs(n_configs=n_configs)
                        self.config_strategy.append("exploit_bandit_kde")
                        # print("generation {}: exploitation configs".format(self.pop.nth_generation))
                else:
                    configs = self.__generate_success_configs(n_configs=n_configs)
                    self.config_strategy.append("success_bandit_kde")
                    # print("generation {}: success configs".format(self.pop.nth_generation))
            return configs

        elif self.model_utilization_strategy == "strategy_3":
            if self.pop.nth_generation < self.learning_gen:
                configs = self.__generate_uniform_configs(n_configs=n_configs)
                self.config_strategy.append("uniform_select")
                # print("generation {}: random configs".format(self.pop.nth_generation))
            else:
                if self.gen_ineffective >= self.stagnation_gen:
                    configs = self.__generate_explore_configs(n_configs=n_configs)
                    self.config_strategy.append("explore_bandit_kde")
                    # print("generation {}: exploration configs".format(self.pop.nth_generation))
                else:
                    configs = self.__generate_exploit_configs(n_configs=n_configs)
                    self.config_strategy.append("exploit_bandit_kde")
                    # print("generation {}: exploit configs".format(self.pop.nth_generation))
            return configs

        elif self.model_utilization_strategy == "strategy_4":
            if self.pop.nth_generation < self.learning_gen:
                configs = self.__generate_uniform_configs(n_configs=n_configs)
                self.config_strategy.append("random_select")
                # print("generation {}: uniform configs".format(self.pop.nth_generation))
            else:
                configs = self.__generate_success_configs(n_configs=n_configs)
                self.config_strategy.append("success_bandit_kde")
            return configs

        elif self.model_utilization_strategy == "strategy_5":
            if self.pop.nth_generation <= self.learning_gen:
                configs = self.__generate_uniform_configs(n_configs=n_configs)
                self.config_strategy.append("uniform_select")
                # print("generation {}: random configs".format(self.pop.nth_generation))
            else:
                max_gen = round(self.stop.max_FES/self.para.N)
                r = self.para.rng.rand()
                cond1 = r > self.pop.nth_generation/max_gen
                cond2 = self.gen_ineffective >= self.stagnation_gen
                if cond1 or cond2:
                    configs = self.__generate_explore_configs(n_configs=n_configs)
                    self.config_strategy.append("explore_bandit_kde")
                    # print("generation {}: exploration configs".format(self.pop.nth_generation))
                else:
                    configs = self.__generate_exploit_configs(n_configs=n_configs)
                    self.config_strategy.append("exploit_bandit_kde")
                    # print("generation {}: exploit configs".format(self.pop.nth_generation))
            return configs

        else:
            raise ValueError("`model_utilization_strategy` is not supported.")

    def __generate_random_configs(self, n_configs):
        if n_configs == 1:
            F  = self.para.rng.uniform(self.F_RANGE[0], self.F_RANGE[1])
            CR = self.para.rng.uniform(self.CR_RANGE[0], self.CR_RANGE[1])
            mut= self.para.rng.choice(self.MUT_POOL)
            cx = self.para.rng.choice(self.CX_POOL)
            if mut == "de/current-to-rand/1":
                cx = "none"
            config = {"mutation": mut, "crossover": cx, "F": F, "CR": CR}
            return config
        elif n_configs > 1:
            list_F  = self.para.rng.uniform(self.F_RANGE[0], self.F_RANGE[1], n_configs).tolist()
            list_CR = self.para.rng.uniform(self.CR_RANGE[0], self.CR_RANGE[1], n_configs).tolist()
            list_mut= self.para.rng.choice(self.MUT_POOL, n_configs, replace=True).tolist()
            list_cx = self.para.rng.choice(self.CX_POOL, n_configs, replace=True).tolist()
            for i, mut in enumerate(list_mut):
                if mut == "de/current-to-rand/1":
                    list_cx[i] = "none"
            list_configs = [{"mutation": mut, "crossover": cx, "F": F, "CR": CR}
                        for mut, cx, F, CR in zip(list_mut, list_cx, list_F, list_CR)]
            return list_configs
        else:
            raise ValueError("`n_configs` must be integer not smaller than 1.")

    def __generate_uniform_configs(self, n_configs):
        if n_configs == 1:
            F  = self.para.rng.uniform(self.F_RANGE[0], self.F_RANGE[1])
            CR = self.para.rng.uniform(self.CR_RANGE[0], self.CR_RANGE[1])
            mut= self.para.rng.choice(self.MUT_POOL)
            cx = self.para.rng.choice(self.CX_POOL)
            if mut == "de/current-to-rand/1":
                cx = "none"
            config = {"mutation": mut, "crossover": cx, "F": F, "CR": CR}
            return config
        elif n_configs > 1:
            spacing = 1.0/n_configs
            r = self.para.rng.rand()/n_configs
            cum_prob = 0.0
            selected_strategies = []
            for strategy in self.generation_strategies:
                cum_prob += 1.0/len(self.generation_strategies)
                while r < cum_prob:
                    selected_strategies.append(strategy)
                    r += spacing
            self.para.rng.shuffle(selected_strategies)
            list_mut = [strategy[0] for strategy in selected_strategies]
            list_cx = [strategy[1] for strategy in selected_strategies]
            list_F  = self.para.rng.uniform(self.F_RANGE[0], self.F_RANGE[1], int(n_configs)).tolist()
            list_CR = self.para.rng.uniform(self.CR_RANGE[0], self.CR_RANGE[1], int(n_configs)).tolist()
            list_configs = [{"mutation": mut, "crossover": cx, "F": F, "CR": CR}
                        for mut, cx, F, CR in zip(list_mut, list_cx, list_F, list_CR)]
            return list_configs
        else:
            raise ValueError("`n_configs` must be integer not smaller than 1.")

    def __generate_explore_configs(self, n_configs):
        if n_configs == 1:
            (mut, cx) = self.explore_bandit.select_arm()
            F = self.explore_kde["F"][(mut, cx)].sample(n_samples=1)[0]
            CR = self.explore_kde["CR"][(mut, cx)].sample(n_samples=1)[0]
            config = {"mutation": mut, "crossover": cx, "F": float(F), "CR": float(CR)}
            return config
        elif n_configs > 1:
            list_mut_cx = self.explore_bandit.select_multiple_arms(k=n_configs)
            list_F = [self.explore_kde["F"][(mut, cx)].sample(n_samples=1)[0]
                      for (mut, cx) in list_mut_cx]
            list_CR = [self.explore_kde["CR"][(mut, cx)].sample(n_samples=1)[0]
                       for (mut, cx) in list_mut_cx]
            list_mut = [mut_cx[0] for mut_cx in list_mut_cx]
            list_cx = [mut_cx[1] for mut_cx in list_mut_cx]
            list_configs = [{"mutation": mut, "crossover": cx, "F": float(F), "CR": float(CR)}
                for mut, cx, F, CR in zip(list_mut, list_cx, list_F, list_CR)]
            return list_configs
        else:
            raise ValueError("`n_configs` must be integer not smaller than 1.")

    def __generate_exploit_configs(self, n_configs):
        if n_configs == 1:
            (mut, cx) = self.exploit_bandit.select_arm()
            F = self.exploit_kde["F"][(mut, cx)].sample(n_samples=1)[0]
            CR = self.exploit_kde["CR"][(mut, cx)].sample(n_samples=1)[0]
            config = {"mutation": mut, "crossover": cx, "F": float(F), "CR": float(CR)}
            return config
        elif n_configs > 1:
            list_mut_cx = self.exploit_bandit.select_multiple_arms(k=n_configs)
            list_F = [self.exploit_kde["F"][(mut, cx)].sample(n_samples=1)[0]
                      for (mut, cx) in list_mut_cx]
            list_CR = [self.exploit_kde["CR"][(mut, cx)].sample(n_samples=1)[0]
                       for (mut, cx) in list_mut_cx]
            list_mut = [mut_cx[0] for mut_cx in list_mut_cx]
            list_cx = [mut_cx[1] for mut_cx in list_mut_cx]
            list_configs = [{"mutation": mut, "crossover": cx, "F": float(F), "CR": float(CR)}
                for mut, cx, F, CR in zip(list_mut, list_cx, list_F, list_CR)]
            return list_configs
        else:
            raise ValueError("`n_configs` must be integer not smaller than 1.")

    def __generate_success_configs(self, n_configs):
        if n_configs == 1:
            (mut, cx) = self.success_bandit.select_arm()
            F = self.success_kde["F"][(mut, cx)].sample(n_samples=1)[0]
            CR = self.success_kde["CR"][(mut, cx)].sample(n_samples=1)[0]
            config = {"mutation": mut, "crossover": cx, "F": float(F), "CR": float(CR)}
            return config
        elif n_configs > 1:
            list_mut_cx = self.success_bandit.select_multiple_arms(k=n_configs)
            list_F = [self.success_kde["F"][(mut, cx)].sample(n_samples=1)[0]
                      for (mut, cx) in list_mut_cx]
            list_CR = [self.success_kde["CR"][(mut, cx)].sample(n_samples=1)[0]
                       for (mut, cx) in list_mut_cx]
            list_mut = [mut_cx[0] for mut_cx in list_mut_cx]
            list_cx = [mut_cx[1] for mut_cx in list_mut_cx]
            list_configs = [{"mutation": mut, "crossover": cx, "F": float(F), "CR":float(CR)}
                for mut, cx, F, CR in zip(list_mut, list_cx, list_F, list_CR)]
            return list_configs
        else:
            raise ValueError("`n_configs` must be integer not smaller than 1.")

    def __generate_best_configs(self, n_configs):
        if n_configs == 1:
            (mut, cx) = self.best_bandit.select_arm()
            (F, CR) = self.best_kde[(mut, cx)].sample(n_samples=1)[0]
            config = {"mutation": mut, "crossover": cx, "F": F, "CR": CR}
            return config
        elif n_configs > 1:
            list_mut_cx = self.best_bandit.select_multiple_arms(k=n_configs)
            list_F_CR = [self.best_kde[(mut, cx)].sample(n_samples=1)[0]
                         for (mut, cx) in list_mut_cx]
            list_F = [F_CR[0] for F_CR in list_F_CR]
            list_CR = [F_CR[1] for F_CR in list_F_CR]
            list_mut = [mut_cx[0] for mut_cx in list_mut_cx]
            list_cx = [mut_cx[1] for mut_cx in list_mut_cx]
            list_configs = [{"mutation": mut, "crossover": cx, "F": F, "CR": CR}
                        for mut, cx, F, CR in zip(list_mut, list_cx, list_F, list_CR)]
            return list_configs
        else:
            raise ValueError("`n_configs` must be integer not smaller than 1.")

    def _get_config_from_assignment(self, idx):
        '''Get the config from `self.configs_assignment` according to the index
        of target individual in the population. '''
        config = self.configs_assignment[idx]
        self.para.mutation = config["mutation"]
        self.para.crossover = config["crossover"]
        self.para.F = config["F"]
        self.para.CR = config["CR"]

    def _collect_result(self):
        result = super()._collect_result()
        result["pop_X_hist"] = self.history.get("X")
        result["pop_state"] = self.history.get("pop_state")
        result["pop_sum_dist"] = self.history.get("pop_sum_dist")
        result["pop_avg_dist"] = self.history.get("pop_avg_dist")
        result["trial_pop_state"] = self.pop_states
        result["explore_rates"] = {
            str(k):v for k, v in self.explore_rates.items()}
        result["exploit_rates"] = {
            str(k): v for k, v in self.exploit_rates.items()}
        result["success_rates"] = {
            str(k): v for k, v in self.success_rates.items()}
        result["config_strategy"] = self.config_strategy
        return result

    def show_evolution(self, skip=5, aggregate_method="average", save=False,
        show=False):
        fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace":0})
        ax1, ax2 = axes[0], axes[1]

        x = self.data["nth_hist"][1:]
        x_skip = skip_and_aggregate_of_list_data(x, skip,
            aggregate_method="none")
        # Plot fitness convergence --------------
        algo_name = self.ALGO_NAME
        ax1.plot(self.data["nth_hist"], np.log10(self.data["ferror_hist"]))
        ax1.set(ylabel="$log_{10} (f-f^*)$",
                title="{} on {}".format(algo_name, self.data.get("problem")),)
        ax1.grid(color="gray", linestyle="--")
        # Plot the config generation strategy of each generation
        ymin = int(np.log10(self.data["ferror_hist"][-1])) - 1
        strategies = self.data["config_strategy"]
        if self.update_frequency == "each_ind":
            strategies = strategies[::self.para.N]
        st_names = ["random_select", "uniform_select", "success_bandit_kde",
                    "explore_bandit_kde", "exploit_bandit_kde", "other",
                    "reset"]
        labels = ["Random Select", "Uniform Select", "Success Bandit & KDE",
                  "Explore Bandit & KDE", "Exploit Bandit & KDE", "other",
                  "reset"]
        st_data = {name: {"x": [], "y": []} for name in st_names}
        # colors = ["C9", "C8", "C7", "C6", "C5", "C4", "C3", "C2", "C1", "C0"]
        colors = ["C9", "C8", "C7", "blue", "red", "C4", "C3", "C2", "C1", "C0"]
        markers = ["o", "o", "s", "s", "s", "D", "X", "o", "o", "o"]
        for i, st in enumerate(strategies):
            st_data[st]["x"].append(i)
            if st == "other":
                st_data[st]["y"].append(ymin+0.1)
            elif st == "reset":
                st_data[st]["y"].append(ymin-0.1)
            else:
                st_data[st]["y"].append(ymin)
        for i, st in enumerate(st_names):
            ax1.scatter(x=st_data[st]["x"], y=st_data[st]["y"], c=colors[i],
                marker=markers[i], label=labels[i])
        ax1.legend(loc="center left", frameon=False)

        # Plot pop state --------------
        state_data = self.data["trial_pop_state"]
        plot_states = ["explore", "exploit"]
        y_explore = skip_and_aggregate_of_list_data(state_data["explore"], skip, aggregate_method)
        y_exploit = skip_and_aggregate_of_list_data(state_data["exploit"], skip, aggregate_method)
        ax2.plot(x_skip, y_explore, label="Explore", color="blue")
        ax2.plot(x_skip, y_exploit, label="Exploit", color="red")
        ax2.axvspan(0, self.learning_gen, label="Learning period",
            hatch="xxx", fill=False, color="silver")
        ax2.set(xlabel="Generation", ylabel="Number of Trial Vectors",
                xlim=(0, self.pop.nth_generation),
                # xticks=range(0, self.pop.nth_generation+100, 100),
                )
        ax2.grid(color="gray", linestyle="--", axis="x")
        ax2.legend(loc="upper left", frameon=False, ncol=len(plot_states)+1)
        # ax2.set_xticks(range(0, self.pop.nth_generation+101, 100))

        if save == True:
            print("saving evolution...")
            plt.savefig("{}_{}_1.pdf".format(self.pb.name, self.ALGO_NAME))
        if show == True:
            plt.show()
            plt.close()



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():

    # TODO: Test algorithm with different configs on
    #                BBOB: F1, F6, F9, F10, | F15, F19, F20, F24
    # and determine the suitable parameter setting for our algorithm.

    case = 0

    if case == 0:
        seed = 4092220473
        data = []
        benchamrk = "bbob2015"
        D = 2

        for F in [1,]:
            pop_size = 50
            problem = aeecde.problem.Benchmark(benchmark_set=benchamrk,
                                    D=D, funID=F, instanceID=1)
            config = aeecde.parameterize.DE(seed=seed, N=pop_size)

            stop = aeecde.parameterize.StopCondition(max_FES=1e4 * D,
                                          max_iter=None, delta_ftarget=1e-8)
            problem = aeecde.problem.Benchmark(benchmark_set=benchamrk,
                                    D=D, funID=F, instanceID=1)
            config = aeecde.parameterize.DE(seed=seed, N=pop_size)

            optimizer = AOSPCDE(opt_problem=problem,
                                algo_parameters=config,
                                stop_conditions=stop,
                                learning_gen=20,
                                state_threshold="median",
                                stagnation_gen=10,
                                model_update_frequency="each_gen",
                                model_utilization_strategy="strategy_5",
                                KDE_kernel="gaussian",
                                para_KDE_width_F=0.2,
                                para_KDE_width_CR=0.2,
                                para_KDE_max_size=50,
                                state_KDE_max_size=50*50,
                                bandit_algo="UCB",
                                bandit_value_method="sliding_window_average",
                                epsilon=0.2,
                                temperature=None,
                                sliding_window_size=50
                               )
            print("\n====={}-F{}-D{}: {} using {} with {} value =====".format(
                        benchamrk.upper(), F, D,
                        optimizer.ALGO_NAME, "UCB", "sliding_window_average"))
            results = optimizer.solve(disp=False, plot=True)
            output = optimizer.data
            data.append(output)

            print("Initial f_best:", output.get("f_best_hist")[0])
            print("Calculated results:", results)
            print("Theoretical optimal value:", problem.f_opt_theory)
            print("Evolved Generations:", output.get("nth_hist")[-1])

            optimizer.show_evolution()

            optimizer.plot_para_histogram()
            plt.show()
            plt.close()

        return data

if __name__ == "__main__":
    data = main()
