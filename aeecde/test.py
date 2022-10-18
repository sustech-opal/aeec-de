#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Validation of modules and functions
'''
import time
import aeecde
from matplotlib import pyplot as plt


tik = time.time()
#! -----------------------------------------------------------------------------
seed = 0 #! Please don't change this value

## 1. Problem Configuration
print("--- Step 1")
print("Let's start by selecting a testing function from bbob benchmark, "
    "you'll get an object of `<class Benchmark>`:")
benchmark = "bbob2015"
D = 2
funID = 7 # 8 9 Rosenbrock , 15 Rastrigin
problem = aeecde.problem.Benchmark(
    benchmark_set=benchmark,
    D=D,
    funID=funID,
    instanceID=1)
print(problem)


## 2. Algorithm Configuration
print("\n--- Step 2")
print("Next, we need to initialize the DE's hyperparameters:")
NP = 24
config = aeecde.publics.parameterize.DE(
    seed = seed,
    N = NP,
    )
print(config)
print("As well as stop conditions:")
stop = aeecde.publics.parameterize.StopCondition(
    max_FES=1000.*NP,
    max_iter=None,
    delta_ftarget=1e-8)
print(stop)


## 3. Aglorithm Execution: using OAC-DE algorithm
print("\n--- Step 3")
print("Now, you can personalize the OAC-DE's parameters such as the "
    "mutation operators, the crossover operators, ...")
optimizer = aeecde.AEECDE(opt_problem=problem,
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
                          sliding_window_size=50, # required for
                                          # "sliding_window_average"
                         )
print("You can solve the optimization problem by running `solve()` method. "
    "If you want to follow the result step by step, you can set "
    "`solve(disp=True)`:")
results = optimizer.solve(disp=False, plot=True)
#! -----------------------------------------------------------------------------
tok = time.time()

## 4. (Optional) Save results to disk
print("\n--- Step 4 (Optional)")
print("If you want to save the results permanently, you can use `save()` "
    "method. It will save the configurations, the stop conditions, "
    "the iteration history, and the final optimum to a json file.")
optimizer.save()

## 5. (Optional) Post-processing
print("\n--- Step 5 (Optional)")
print("Or you can just view the results on your screen like:")
print("\tElapsed Time: {:.2f}s".format(tok-tik))
print("\tCalculated results:", results)
print("\tTheoretical optimal value:", problem.f_opt_theory)
print("\tRelative error:", results[1]-problem.f_opt_theory)
print("\tEvolved Generations:", optimizer.data.get("nth_hist")[-1])

print(optimizer.history)
optimizer.show_evolution()

optimizer.plot_para_histogram()
plt.show()