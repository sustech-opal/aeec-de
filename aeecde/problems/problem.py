#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Optimization problem definitions
'''
import numpy as np
# internal imports
from aeecde import algorithms as alg
from aeecde.publics import parameterize as para
from aeecde.publics.tools import ensure_bounds_clip, ensure_bounds_mirror



#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------
class __base(object):
    ''' :class:`__base` is the base class for defining the optimization problem
    (everything else except what has been defined in optimization algorithms).
    '''
    def __init__(self, fobj, D, lower_bound, upper_bound, bounds_method,
                 name, **kwargs):
        '''
        Parameters
        ----------
            To be specified in its successors

        Attributes
        ----------
        x_opt : int | float | ndarray [default=None]
            The global optimal solution
        f_opt : float [default=None]
            The global optimal value of objective function
        num_calls : int [default=0]
            Counts of calling objective function
        '''
        # Attributes
        self.x_opt = None # The optimal (best) solution founded by optimizer
        self.f_opt = None # The optimal (best) objective value founded by optimizer
        self.f_opt_theory = None # The theoretical optimal objective value the problem.
        self.num_calls = 0
        self._ALL_algorithms = {"DE":self.__DE, "PSO":self.__PSO}
        self._ALL_ensure_bounds_methods = {
            "clip": ensure_bounds_clip,
            "mirror": ensure_bounds_mirror}
        # Parameters
        self.fobj, self.D = fobj, D
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.bounds_method, self.name = bounds_method, name
        self.init_lower_bound = kwargs.get("init_lower_bound")
        self.init_upper_bound = kwargs.get("init_upper_bound")

#   ----------------------------- Setter/Getter ------------------------------
    @property
    def fobj(self):
        return self._fobj

    @fobj.setter
    def fobj(self, func):
        if callable(func):
            self._fobj = func
        else:
            raise TypeError("The objective function must be callable (e.g."
                " function, method, ...), not an instance of {}".format(
                type(func).__name__))

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, value):
        if isinstance(value, int):
            self._D = value
        elif value is None:
            raise ValueError("The dimension of decision variable `D` must be"
                " provided")
        else:
            raise TypeError("The dimension of decision variable must be an int,"
                " not a {}".format(type(value).__name__))

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, bound):
        if isinstance(bound, (int, float)):
            new_bound = np.array([bound] * self.D)
        elif isinstance(bound, list):
            if np.isreal(bound).all() and not np.isnan(bound).any():
                new_bound = np.array(bound)
            else:
                raise ValueError("The lower bound of decision variable must be"
                    " a list only including real numbers, not {}".format(bound))
        else:
            raise TypeError("The lower bound of decision variable must be a"
                " real number or a list of real numbers, not {}".format(
                type(bound).__name__))
        self._lower_bound = new_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, bound):
        if isinstance(bound, (int, float)):
            new_bound = np.array([bound] * self.D)
        elif isinstance(bound, list):
            if np.isreal(bound).all() and not np.isnan(bound).any():
                new_bound = np.array(bound)
            else:
                raise ValueError("The upper bound of decision variable must be"
                    " a list only including real numbers, not {}".format(bound))
        else:
            raise TypeError("The upper bound of decision variable must be a"
                " real number or a list of real numbers, not {}".format(
                type(bound).__name__))
        self._upper_bound = new_bound

    @property
    def bounds_method(self):
        return self._bounds_method

    @bounds_method.setter
    def bounds_method(self, name):
        if isinstance(name, str):
            func = self._ALL_ensure_bounds_methods.get(name.lower())
            if func is None:
                raise NameError("To ensure the decision variable within the"
                    " feasible region, method name must be 'clip' or 'mirror'"
                    " not {}".format(name))
            else:
                self._bounds_method = func
        else:
            raise TypeError("The method used for ensuring feasible region"
                " (search domain) must be str, not {}".format(
                type(name).__name__))

    @property
    def init_lower_bound(self):
        return self._init_lower_bound

    @init_lower_bound.setter
    def init_lower_bound(self, bound):
        if bound is None:
            if not np.isinf(self.lower_bound).any():
                new_bound = self.lower_bound
            else:
                raise ValueError("When some of the `lower_bound` is `-np.inf`,"
                    " the `init_lower_bound` must be provided.")
        else:
            if isinstance(bound, (int, float)):
                new_bound = np.array([bound] * self.D)
            elif isinstance(bound, list):
                if np.isreal(bound).all() and not np.isnan(bound).any():
                    new_bound = np.array(bound)
                else:
                    raise ValueError("The lower bound for initialization must"
                        " be a list only including real numbers, not {}"
                        .format(bound))
            else:
                raise TypeError("The lower bound for initialization must be a"
                    " real number or a list of real numbers, not {}".format(
                    type(bound).__name__))
        self._init_lower_bound = new_bound

    @property
    def init_upper_bound(self):
        return self._init_upper_bound

    @init_upper_bound.setter
    def init_upper_bound(self, bound):
        if bound is None:
            if not np.isinf(self.upper_bound).any():
                new_bound = self.upper_bound
            else:
                raise ValueError("When some of the `upper_bound` is `-np.inf`,"
                    " the `init_upper_bound` must be provided.")
        else:
            if isinstance(bound, (int, float)):
                new_bound = np.array([bound] * self.D)
            elif isinstance(bound, list):
                if np.isreal(bound).all() and not np.isnan(bound).any():
                    new_bound = np.array(bound)
                else:
                    raise ValueError("The lower bound for initialization must"
                        " be a list only including real numbers, not {}"
                        .format(bound))
            else:
                raise TypeError("The lower bound for initialization must be a"
                    " real number or a list of real numbers, not {}".format(
                    type(bound).__name__))
        self._init_upper_bound = new_bound

    @property
    def x_opt(self):
        return self._x_opt

    @x_opt.setter
    def x_opt(self, array):
        if array is None or isinstance(array, (np.ndarray, int, float)):
            self._x_opt = array
        else:
            raise TypeError("The optimal solution must be None, or a real"
                " number or a ndarray, not a {}".format(type(array).__name__))

    @property
    def f_opt(self):
        return self._f_opt

    @f_opt.setter
    def f_opt(self, value):
        if value is None or isinstance(value, (int, float)):
            self._f_opt = value
        else:
            raise TypeError("The optimal value of objective function must be"
                " None or a real number, not {}".format(type(value).__name__))

    @property
    def f_opt_theory(self):
        return self._f_opt_theory

    @f_opt_theory.setter
    def f_opt_theory(self, value):
        if value is None or isinstance(value, (int, float)):
            self._f_opt_theory = value
        else:
            raise TypeError("The theoretical optimal value of objective"
                " function must be None or a real number, not {}".format(
                type(value).__name__))

    @property
    def algo_name(self):
        return self._algo_name

    @algo_name.setter
    def algo_name(self, name):
        if isinstance(name, str):
            method = self._ALL_algorithms.get(name.upper())
            if method is None:
                raise NotImplementedError("The optimizing algorithm '{}' is not"
                    " implemented, it should be one of {}".format(name,
                    self._ALL_algorithms.keys()))
            else:
                self._algo_name = name.upper()
                self.algo = method
        else:
            raise TypeError("The optimizing algorithm must be str not {}"
                .format(type(name).__name__))

    @property
    def algo_config(self):
        return self._algo_config

    @algo_config.setter
    def algo_config(self, config):
        if isinstance(config, (para.DE)):
            self._algo_config = config
        else:
            raise TypeError("The optimizing algorithm's hyperparameters must be"
                " an instance of para.DE, not {}".format(
                type(config).__name__))

    @property
    def stop_condition(self):
        return self._stop_condition

    @stop_condition.setter
    def stop_condition(self, cond):
        if isinstance(cond, (para.StopCondition)):
            self._stop_condition = cond
        else:
            raise TypeError("The stop conditions must be an instance of"
                " option.StopCondition, not {}".format(type(cond).__name__))

#   ----------------------------- EA algorithms -----------------------------
    def __DE(self, algo_config, stop_condition, seed, disp, plot):
        optimizer = alg.DE(opt_problem=self, algo_parameters=algo_config,
                           stop_conditions=stop_condition, seed=seed)
        return optimizer.solve(disp=disp, plot=plot)

    def __PSO(self, algo_config, stop_condition, seed):
        return

#   --------------------------- End-user interface ---------------------------
    def evaluate(self, x): #TODO HB: add optional arguments of function
        x = self.bounds_method(x, self.lower_bound, self.upper_bound)
        fx = self.fobj(x)
        self.num_calls = self.num_calls + 1
        return x, fx

    def evaluate_ind(self, ind):  # Add by CWH, to make the evaluation simpler.
        ind.xvalue = self.bounds_method(ind.xvalue, self.lower_bound, self.upper_bound)
        ind.fvalue = self.fobj(ind.xvalue)
        self.num_calls = self.num_calls + 1

    def update_opt(self, new_xvalue, new_fvalue):
        if (self.f_opt is None) or (new_fvalue < self.f_opt):
            self.f_opt = new_fvalue
            self.x_opt = new_xvalue

    def select_algo(self, algo_name, auto_select=False):
        self.algo_name = algo_name

    def set_algo(self, algo_name, configs, auto_set=False):
        if self.algo_name == algo_name.upper():
            self.algo_config = configs
        else:
            raise ValueError("Cannot set {}'s hyperparameters to {} algorithm"
                .format(algo_name, self.algo_name))

    def solve(self, stop_condition=None, seed=None, disp=True, plot=True,
        parallel=False):
        self.stop_condition = stop_condition
        # execution
        self.algo(self.algo_config,self.stop_condition, seed, disp, plot)
        return self.x_opt, self.f_opt


class SingleObject(__base):
    ''' :class:`SingleObject` is the class for defining a single objective
    optimization problem without constraints
    '''

    def __init__(self, fobj, D=None, lower_bound=-np.inf, upper_bound=np.inf,
                 bounds_method="mirror",
                 name="Single Objective Optimization Problem",
                 init_lower_bound=None, init_upper_bound=None):
        '''
        Creates a new :class:`SingleObject` single objective optimization
        problem

        Parameters
        ----------
        fobj : callable instance
            The objective function to be minimized. It can be a function, or a
            method, or any other callable instance.
        D : int
            The Dimensionality of decision variables
        lower_bound : int | float | list, optional [default=-np.inf]
            Lower boundary of decision variables
        upper_bound : int | float | list, optional [default=np.inf]
            Upper boundary of decision variables
        bounds_method : str, optional [default="mirror"]
            Method used for ensuring boundary conditions, available options are
            "mirror" and "clip"
        name : str, optional [default="Single Object Optimization Problem"]
            Name the current optimization problem
        init_lower_bound : str, optional [default=None]
            If the lower boundary contain infinity, the initial lower boundary
            must be given to specify the search domain for initialization
        init_upper_bound : str, optional [default=None]
            If the upper boundary contain infinity, the initial upper boundary
            must be given to specify the search domain for initialization
        '''
        super().__init__(fobj, D, lower_bound, upper_bound, bounds_method,
            name, init_lower_bound=init_lower_bound,
            init_upper_bound=init_upper_bound)


class Benchmark(SingleObject):
    ''' :class:`Benchmark` is the class for defining optimization problem
    benchmarks
    '''

    def __init__(self, benchmark_set, D, funID, instanceID, **kwargs):
        '''
        Creates a new :class:`Benchmark` for CEC2005 and BBOB2015 benchmarks

        Parameters
        ----------
        benchmark_set : str
            The benchmark set of test functions that will be used, choose
            between 'cec2005' and 'bbob2015'.
        D : int
            The dimensionality of test function.
        funID : int
            Function ID to specify the function of benchmark set.
        instanceID : int, optional (only valid for 'bbob2015')
            Instance ID to specify the instance of selected function.
        '''
        try:
            from . import bbobbenchmarks as bbob2015
        except Exception as e:
            print("Cannot import benchmark sets, please check that you have"
                  " them installed !")
            raise e
        init_lower_bound, init_upper_bound = None, None

        #* BBOB2015
        if benchmark_set.lower() == "bbob2015":
            if funID not in range(1, 25):
                raise ValueError("The `funID` for BBOB2015 benchmark set should"
                    " be integers within [1, 24]")
            objective_function, fopt = bbob2015.instantiate(funID, instanceID)
            lower_bound = [-5.0] * D
            upper_bound = [5.0] * D
            name = "BBOB2015-F{}(I{}), D={}".format(
                objective_function.funId, objective_function.iinstance, D)
        else:
            raise NameError("`benchmark_set` get an unsupported name, please"
                            " use 'bbob2015'")
        # HB : adapt to our algo
        if lower_bound is None:
            lower_bound = [-np.inf] * D
        if upper_bound is None:
            upper_bound = [np.inf] * D
        # initiate the instance
        super().__init__(objective_function, D, lower_bound, upper_bound,
                 name=name, init_lower_bound=init_lower_bound,
                 init_upper_bound=init_upper_bound)
        self.f_opt_theory = fopt



#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def x2(x):
    '''
    Benchmark function .. math :: f(x) = x^2
    Has a global minimum at :code:`0` and with a search domain of :code:`[-inf,
        inf]`

    Parameters
    ----------
    x : float
        Set of inputs

    Returns
    -------
    float
    '''
    return float(x**2)

def x2_add_y2(x):
    '''
    Benchmark function .. math :: f(x) = x^2 + y^2
    Has a global minimum at :code:`(0, 0)` and with a search domain of :code:`
    [-inf, inf]`

    Parameters
    ----------
    x : list of length=2
        Set of inputs

    Returns
    -------
    float
    '''
    return x[0]**2 + x[1]**2

def Ax2_add_By2(x, a, b, **kwargs):
    '''
    Benchmark function .. math :: f(x) = ax^2 + by^2
    If a and b are positive, it has a global minimum at :code:`(0, 0)` and with
    a search domain of :code:`[-inf, inf]`

    Parameters
    ----------
    x : list of length=2
        Set of inputs
    a : float
    b : float

    Returns
    -------
    float
    '''
    for key, value in kwargs:
        if key == "a":
            a = value
        else:
            a = a
        if key == "b":
            b = value
        else:
            b = b
    return a * x[0]**2 + b * x[1]**2



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    case = 1
#*  --- case 1
    if case == 1:
        # [Step 1] Problem configuration
        # pb = SingleObject(fobj=x2, D=1, init_lower_bound=-10, init_upper_bound=20)
        # pb = Benchmark(benchmark_set="bbob2015", D=2, funID=1, instanceID=0)
        # print(pb.name, "D=", pb.D, "\n")
        # print('init_LB: {}, \ninit_UB: {}'.format(pb.init_lower_bound,
        #     pb.init_upper_bound))

        # pb = Benchmark(benchmark_set="bbob2015", D=10, funID=5, instanceID=1)
        # print(pb.name)
        # print('LB: {}, \nUB: {}'.format(pb.lower_bound, pb.upper_bound))

        pb = Benchmark(benchmark_set="cec2005",  D=2, funID=1, instanceID=0)
        print(pb.name)
        print('LB: {}, \nUB: {}'.format(pb.lower_bound, pb.upper_bound))
        print("F_opt: {}".format(pb.f_opt))

        # print("x0:", pb.x0)

        # [Step 2] Algorithm selection
        pb.select_algo(algo_name="De")

        # [Step 3] Algorithm configuration
        config1 = para.DE(N=20, F=0.75, CR=0.35,
                               mutation_scheme="DE/rand/1",
                               crossover_scheme="exp",
                               initial_scheme="latin_hypercube",
                               update_scheme="immediate")
        pb.set_algo("DE", config1)

        # [Step 4] Problem solving
        sc1 = para.StopCondition(maxFES=500, maxG=None, delta_ftarget=None)
        pb.solve(stop_condition=sc1, seed=1)

        # [Step 5] Postprocess
        print("x_opt = {}, f_opt = {}".format(pb.x_opt, pb.f_opt))

#*  --- case 2
    if case == 2:
        MUTs = ["DE/rand/1",
                "DE/rand/2",
                "DE/best/1",
                "DE/best/2",
                "DE/rand-to-best/1",
                "DE/current-to-best/1",
                "DE/current-to-rand/1",
                "DE/current-to-best/2",
                "DE/current/1",
                "DE/current-to-pbest/1",
                "DE/rand-to-pbest/1"]

        CXs = ["bin", "exp", "none"]

        UDs = ["deferred", "immediate"]

        INs = ["latin_hypercube", "random"]

        sc2 = para.StopCondition(max_FES=1000, max_iter=None, delta_ftarget=1e-8)

        pb = Benchmark(benchmark_set="bbob2015",  D=2, funID=1, instanceID=0)
        pb.select_algo(algo_name="De")

        # Test each combination of (mutation, crossover, update) schemes:
        for mutation in MUTs:
            for crossover in CXs:
                for update in UDs:
                    print("Mut: {}, Cx: {}, Ud: {}".format(mutation, crossover,
                        update))
                    config = para.DE(N=20, F=0.75, CR=0.35,
                                          mutation_scheme=mutation,
                                          crossover_scheme=crossover,
                                          update_scheme=update)
                    pb.set_algo("DE", config)
                    pb.solve(stop_condition=sc2, seed=1, disp=False, plot=False)

if __name__ == "__main__":
    main()
