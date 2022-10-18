#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' New module for options (algorithm configurations, hyperparameters, problem
settings)
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
#!                                     CLASSES
#!------------------------------------------------------------------------------
class __base(object):
    ''' :class:`__base` is the base class for anything that participates in the 
    definition of optimization problem.
    '''
    def __init__(self, seed):
        '''
        Creates a new :class:`__base`.

        Parameters
        ----------
        seed : None | int | instance of RandomState
            - If it is None, return the RandomState singleton used by np.random.
            - If it is an int, return a new RandomState instance seeded with seed.
            - If it is already a RandomState instance, return it.        
        '''
        self.rng = seed

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, seed):
        if seed is None or seed is np.random:
            self._rng = np.random.mtrand._rand
        elif isinstance(seed, (int, np.int32, np.int64)):
            self._rng = np.random.RandomState(seed)
            self.seed = int(seed)
        elif isinstance(seed, np.random.RandomState):
            self._rng = seed
        else:
            raise TypeError("The random seed must be None (or np.random), or"
                " int, or instance of np.random.RandomState, not {}".format(
                type(seed).__name__))

    def current(self):
        return {k: str(v) for k, v in vars(self).items() if k != "_rng"}


#*  ----------------------------- Basic Data Types -----------------------------
class _Hyperparameter(__base):
    def __init__(self, value, default, min, max, *args):
        self.min = min
        self.max = max
        self.admission = args[0].get("admission") # allowable items
        self.name = args[0].get("name")
        self.remark = args[0].get("remark")
        # self.type = None # value type: int, float, list, dict, str
        # self.form = None # constant | variable
        self.distribution = None # name of distribution: uniform, normal
        self._history = []
        self._value, self._values = None, None #TODO HB: to rewrite
        self.value = value
        self.default = default
        super(_Hyperparameter, self).__init__(args[0].get("seed"))

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, input):
        if input is None:
            self._value = input
            return
        elif isinstance(input, self.TYPE): # assign value
            self._value = input
        elif isinstance(input, (list, tuple, np.ndarray)):
            for elem in input:
                if not isinstance(elem, self.TYPE):
                    raise TypeError("Parameter '{}' must be a sequence of {},"
                        " it cannot contain a value of {}".format(self.name, 
                    self.TYPE.__name__, type(elem).__name__))
            self._values = np.array(input)
        else:
            raise TypeError("Parameter '{}' must be {}, not {}".format(
                self.name, self.TYPE.__name__, type(input).__name__))
        self._check_both_values() # check both `self.value` and `self.values`

    @property
    def values(self):
        return self._values
    
    #? HB: this may be useless
    @values.setter
    def values(self, inputs):
        self._values = inputs

    @property
    def default(self):
        return self._default
    
    @default.setter
    def default(self, value):
        self._default = value
        if self.value is None:
            self.value = self.default

    @property
    def history(self):
        return self._history

    def _check_both_values(self):
        if self.value is not None:
            self._inspect(self.value)
            # self.history.append(self.value)
        else:
            [self._inspect(v) for v in self.values]
            # self.history.append(self.values)

    def _inspect(self, value):
        if (self.min is not None) and (value < self.min):
            raise ValueError("Parameter '{}' gets a value {} smaller"
                " than the lower bound {}".format(self.name, value, self.min))
        elif (self.max is not None) and (value > self.max):
            raise ValueError("Parameter '{}' gets a value {} larger"
                " than the upper bound {}".format(self.name, value, self.max))
        elif (self.admission is not None) and (not value in self.admission):
            raise ValueError("Parameter '{}' gets a value '{}' not"
                " allowed in {}".format(self.name, value, self.admission))
        else:
            return value

#   ----------------------------- Magic Methods -----------------------------
    def __repr__(self):
        line = []
        if self.name is not None:
            line.append(self.name)
        line.append("Type: {}".format(type(self).__name__))
        line.append("Value: {}".format(self.value))
        if self.values is not None:
            line.append("Values: {}".format(self.values))
        if self.min is not None or self.max is not None:
            line.append("Range: [{}, {})".format(self.min, self.max))
        if self.admission is not None:
            line.append("Allowable options: {}".format(self.admission))
        if self.default is not None:
            line.append("Default: {}".format(self.default))
        if self.distribution is not None:
            line.append("Distribution: {}".format(self.distribution))
        if self.history != []:
            line.append("Used: {}".format(self.history))
        if self.remark is not None:
            line.append("Remark: {}".format(self.remark))
        return "; ".join(line)

    def __format__(self, formatstr):
        return self.__repr__().replace("; ", "\n\t")

    def __cmp_check(self, other, operator):
        if not isinstance(other, type(self)):
            raise TypeError("'{}' not supported between instances of '{}' and"
                " '{}'".format(operator, type(self).__name__,
                type(other).__name__))
        else:
            value = getattr(self.value, operator)(other.value)
            if operator == "__add__":
                return type(self)(value, name="{} + {}".format(self.name, 
                    other.name))
            elif operator == "__sub__":
                return type(self)(value, name="{} - {}".format(self.name,
                    other.name))
            elif operator == "__mul__":
                return type(self)(value, name="{} * {}".format(self.name,
                    other.name))
            elif operator == "__truediv__":
                return Float(value, name="{} / {}".format(self.name,
                    other.name))
            elif operator == "__floordiv__":
                return type(self)(value, name="{} // {}".format(self.name,
                    other.name))
            elif operator == "__mod__":
                return type(self)(value, name="{} % {}".format(self.name,
                    other.name))
            elif operator == "__pow__":
                return type(self)(value, name="{} ** {}".format(self.name,
                    other.name))
            else:
                return value
            return 

    def __lt__(self, other): # <
        return self.__cmp_check(other, "__lt__")

    def __le__(self, other): # <=
        return self.__cmp_check(other, "__le__")

    def __eq__(self, other): # ==
        return self.__cmp_check(other, "__eq__")

    def __ne__(self, other): # !=
        return self.__cmp_check(other, "__ne__")

    def __ge__(self, other): # >=
        return self.__cmp_check(other, "__ge__")

    def __gt__(self, other): # >
        return self.__cmp_check(other, "__gt__")

    def __add__(self, other): # +
        return self.__cmp_check(other, "__add__")
    
    def __sub__(self, other): # -
        return self.__cmp_check(other, "__sub__")
    
    def __mul__(self, other): # *
        return self.__cmp_check(other, "__mul__")
    
    def __truediv__(self, other): # /
        return self.__cmp_check(other, "__truediv__")
    
    def __floordiv__(self, other): # //
        return self.__cmp_check(other, "__floordiv__")
    
    def __mod__(self, other): # %
        return self.__cmp_check(other, "__mod__")
    
    def __pow__(self, other): # **
        return self.__cmp_check(other, "__pow__")

    def __abs__(self):  # abs()
        value = getattr(self.value, "__abs__")()
        return type(self)(value, name="abs({})".format(self.name))

    #   --------------------------- Public Methods ---------------------------
    def choice(self, from_where, prob=None):
        ''' Generates a uniform random sample from a given 1-D array-like'''
        self.history.append(self.value)
        self.value = self.rng.choice(from_where, p=prob)
        return self.value


class Integer(_Hyperparameter):
    TYPE = int

    def __init__(self, value, default=None, min=-2**63, max=2**63, **kwargs):
        super(Integer, self).__init__(value, default, min, max, kwargs)

    def random(self, dist, **kwargs):
        ''' Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high). If high is None (the default), then results are from [0, low).
        '''
        self.history.append(self.value)
        if dist.lower() == "uniform":
            self.value = self.rng.randint(low=self.min, high=self.max)
        elif dist.lower() == "normal": # ? HB: this maneuver needs be justified
            mean = kwargs.get("mean")
            stdev = kwargs.get("stdev")
            self.value = int(self.rng.normal(mean, stdev))
        else:
            raise NotImplementedError("The random distribution '{}' is invalid"
                .format(dist))
        self.distribution = dist.lower()


class Float(_Hyperparameter):
    TYPE = float

    def __init__(self, value, default=None, min=np.NINF, max=np.PINF, **kwargs):
        super(Float, self).__init__(value, default, min, max, kwargs)

    def random(self, dist, **kwargs):
        ''' Draw samples from a uniform distribution. Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high).
        '''
        self.history.append(self.value)
        if dist.lower() == "uniform":
            self.value = self.rng.uniform(low=self.min, high=self.max)
        elif dist.lower() == "normal":
            mean = kwargs.get("mean")
            stdev = kwargs.get("stdev")
            self.value = self.rng.normal(mean, stdev)
        else:
            raise NotImplementedError("The random distribution '{}' is invalid"
                .format(dist))
        self.distribution = dist.lower()


class String(_Hyperparameter):
    TYPE = str

    def __init__(self, value, default=None, **kwargs):
        super(String, self).__init__(value, default, None, None, kwargs)


class Dict(_Hyperparameter):
    TYPE = dict

    def __init__(self, value, default=None, min=None, max=None, **kwargs):
        super(Dict, self).__init__(value, default, min, max, kwargs)

    def _check_both_values(self):
        if self.value is not None:
            [self._inspect(v) for k, v in self.value.items()]
            # self.history.append(self.value)
        else:
            [self._inspect(v) for elem in self.values for k, v in elem.items()]
            # self.history.append(self.values)


class Boolean(_Hyperparameter):
    TYPE = bool

    def __init__(self, value, default=None, **kwargs):
        super(Boolean, self).__init__(bool(value), default, None, None, kwargs)


#*  -------------------------- Convergence Criteria --------------------------
class StopCondition(__base):
    
    def __init__(self, **kwargs):
        self._max_FES = Float(
            value=kwargs.get("max_FES"),
            default=np.PINF, min=1.,
            name="Maximum Function Evaluations",
            remark="The maximum number of calling objective function.")
        self._max_iter = Float(
            value=kwargs.get("max_iter"),
            default=None, min=1.,
            name="Maximum Iterations",
            remark="The maximum number of the principal iteration in"
                   " algorithm.")
        self._delta_ftarget = Float(
            value=kwargs.get("delta_ftarget"),
            default=np.finfo(float).tiny, min=0.,
            name="Difference between the target and the theoretical optimum",
            remark="The tolerance to accept a value as the optumum.")
        self._ftarget = Float(
            value=kwargs.get("ftarget"),
            default=None,
            name="Target value of the objective function",
            remark="Stop if the value of objective function is less than this"
                   " threshold.")
        self._condition_limit = Float(
            value=kwargs.get("condition_limit"),
            default=1e7, min=0., max=np.PINF,
            name="Target value of the objective function",
            remark="Stop if the condition number of the covariance matrix"
                " exceeds this limit (only valid in CMAES algorithm).")

    @property
    def max_FES(self):
        return self._max_FES.value

    @max_FES.setter
    def max_FES(self, value):
        self._max_FES.value = value

    @property
    def max_iter(self):
        return self._max_iter.value

    @max_iter.setter
    def max_iter(self, value):
        self._max_iter.value = value

    @property
    def delta_ftarget(self):
        return self._delta_ftarget.value

    @delta_ftarget.setter
    def delta_ftarget(self, value):
        self._delta_ftarget.value = value

    @property
    def ftarget(self):
        return self._ftarget.value

    @ftarget.setter
    def ftarget(self, value):
        self._ftarget.value = value

    @property
    def condition_limit(self):
        return self._condition_limit.value

    @condition_limit.setter
    def condition_limit(self, value):
        self._condition_limit.value = value

    def __repr__(self):
        return "\n".join(["`{}`: {}".format(k,v)for k,v in vars(self).items()])


#*  ------------------------ Algorithm Configurations ------------------------
class _ConfigureAlgorithm(__base):
    
    def __init__(self, seed, N, initial_scheme, update_scheme):
        super(_ConfigureAlgorithm, self).__init__(seed)
        self._N = Integer(
            seed=seed,
            value=N,
            default=None, min=0,
            name="Number of individuals",
            remark="The number of individuals in the population, e.g."
                   " particles in PSO, individuals in DE ...")
        self._initial_scheme = String(
            seed=seed,
            value=initial_scheme.lower(),
            default="latin_hypercube",
            admission=("latin_hypercube", "random"),
            name="Initialization scheme",
            remark="The method to be used in generating the initial sample."
                   " The allowable options are 'latin_hypercube' and 'random'.")
        self._update_scheme = String(
            seed=seed,
            value=update_scheme.lower(),
            default="immediate",
            admission=("immediate", "deferred"),
            name="Update scheme",
            remark="The method to be used in updating the generation."
                   " The allowable options are 'immediate' and 'deferred'.")
    
    @property
    def N(self):
        return self._N.value
    
    @N.setter
    def N(self, value):
        self._N.value = value

    @property
    def initial_scheme(self):
        return self._initial_scheme.value
    
    @initial_scheme.setter
    def initial_scheme(self, value):
        self._initial_scheme.value = value
    
    @property
    def update_scheme(self):
        return self._update_scheme.value
    
    @update_scheme.setter
    def update_scheme(self, value):
        self._update_scheme.value = value

    def __repr__(self):
        return "\n".join(["`{}`: {}".format(k,v)for k,v in vars(self).items()])
        
    
    def set(self, attr, value, **kwargs):
        setattr(self, attr, value)


class DE(_ConfigureAlgorithm):
    CONFIG_NAME = "DE related configuration"

    def __init__(self, seed=None, N=20, F=0.75, CR=0.7,
                 mutation="de/best/1", crossover="bin", 
                 initial_scheme="latin_hypercube", update_scheme="immediate",
                 generation_strategy=[{}], F_CR_pair=[{}], **kwargs):
        super(DE, self).__init__(seed, N, initial_scheme, update_scheme)
        self._F = Float(
            seed=seed,
            value=F,
            default=0.75, min=0., max=2.,
            name="Scaling factor",
            remark="The factor to be used in mutation operation."
                   " The allowable range is 0.0 <= `F` < 2.0.")
        self._CR = Float(
            seed=seed,
            value=CR,
            default=0.7, min=0., max=1.,
            name="Crossover rate",
            remark="The rate to be used in crossover operation."
                   " The allowable range is 0.0 <= `CR` < 1.0.")
        self._mutation = String(
            seed=seed,
            value=mutation.lower(),
            default="de/best/1",
            admission=("de/rand/1", "de/rand/2", "de/best/1", "de/best/2",
                    "de/current/1", "de/rand-to-best/1", "de/current-to-best/1",
                    "de/current-to-rand/1", "de/current-to-best/2",
                    "de/current-to-pbest/1", "de/rand-to-pbest/1"),
            name="Mutation scheme",
            remark="The method to be used in mutation operation.")
        self._crossover = String(
            seed=seed,
            value=crossover.lower(),
            default="bin",
            admission=("bin", "exp", "eig", "none"),
            name="Crossover scheme",
            remark="The method to be used in crossover operation.")
        self._generation_strategy = Dict(
            seed=seed,
            value=generation_strategy,
            admission=("de/rand/1", "de/rand/2", "de/best/1", "de/best/2",
                    "de/current/1", "de/rand-to-best/1", "de/current-to-best/1",
                    "de/current-to-rand/1", "de/current-to-best/2",
                    "de/current-to-pbest/1", "de/rand-to-pbest/1", "bin", "exp",
                    "eig","none"),
            name="Generation strategy",
            remark="A compositie of mutation scheme and crossover scheme. Each"
                " composite should be saved in a dict containing 2 keys:"
                " `mutation` and `crossover`.")
        self._F_CR_pair = Dict(
            seed=seed,
            value=F_CR_pair,
            min=0., max=2.,
            name="Pair of `F` and `CR`",
            remark="A composite of scaling factor `F` and crossover rate `CR`."
                " Each composite should be saved in a dict containing 2 keys:"
                " `F` and `CR`.")
    
    #   --------------------------- Setter/Getter ---------------------------
    @property
    def F(self):
        return self._F.value
    
    @F.setter
    def F(self, value):
        self._F.value = value
    
    @property
    def CR(self):
        return self._CR.value
    
    @CR.setter
    def CR(self, value):
        self._CR.value = value

    @property
    def mutation(self):
        return self._mutation.value
    
    @mutation.setter
    def mutation(self, value):
        self._mutation.value = value
    
    @property
    def crossover(self):
        return self._crossover.value
    
    @crossover.setter
    def crossover(self, value):
        self._crossover.value = value
    
    @property
    def generation_strategy(self):
        return self._generation_strategy.values
    
    @generation_strategy.setter
    def generation_strategy(self, value):
        self._generation_strategy.value = value
    
    @property
    def F_CR_pair(self):
        return self._F_CR_pair.values
    
    @F_CR_pair.setter
    def F_CR_pair(self, value):
        self._F_CR_pair.value = value

#* alias for other DE related algorithms
DERandomPopulation = DE
DERandomIndividual = DE
DEComposite = DE


class CMAES(_ConfigureAlgorithm):
    CONFIG_NAME = "CMAES related configuration"

    def __init__(self, seed=None, D=None, **kwargs):
        if kwargs.get("lambd") is None: # check the nicknames
            super().__init__(seed, kwargs.get("N"), "random", "immediate")
        else:
            super().__init__(seed, kwargs.get("lambd"), "random", "immediate")
        del self._initial_scheme, self._update_scheme
        self._D = Integer(
            seed=seed,
            value=D,
            default=None, min=0,
            name="Dimension",
            remark="The Dimensionality of the decision variables.")
        self._mu = Integer(
            seed=seed,
            value=kwargs.get("mu"),
            min=2, max=self.N,
            name="???mu???",
            remark="这啥玩意儿")
        self._mu_ratio = Float(
            seed=seed,
            value=kwargs.get("mu_ratio"),
            min=0., max=1.,
            name="???mu_ratio???",
            remark="小哥儿 嫩啥累")
        self._weights = Float(
            seed=seed,
            value=kwargs.get("weights"),
            min=0., max=1.,
            name="???weights???",
            remark="苍天啊 大地啊")
        self._cs = Float(
            seed=seed,
            value=kwargs.get("cs"),
            min=0., max=1.,
            name="???cs???",
            remark="写不动了……")
        self._damps = Float(
            seed=seed,
            value=kwargs.get("damps"),
            min=0., max=10.,
            name="???damps???",
            remark="我选择狗带")
        self._cc = Float(
            seed=seed,
            value=kwargs.get("cc"),
            min=0., max=1.,
            name="???cc???",
            remark="苍茫的天涯我的爱")
        self._c1 = Float(
            seed=seed,
            value=kwargs.get("c1"),
            min=0., max=1.,
            name="???c1???",
            remark="长的丑 活得久，长得帅 老得快")
        self._cmu = Float(
            seed=seed,
            value=kwargs.get("cmu"),
            min=0., max=1.,
            name="???cmu???",
            remark="左手一只鸡，右手一只鸭")
        self.__check_static_hyperparameters()
#   --------------------------- Setter/Getter ---------------------------
    @property
    def D(self):
        return self._D.value

    @D.setter
    def D(self, value):
        self._D.value = value
    
    @property
    def mu(self):
        return self._mu.value

    @mu.setter
    def mu(self, value):
        self._mu.value = value
    
    @property
    def mu_ratio(self):
        return self._mu_ratio.value

    @mu_ratio.setter
    def mu_ratio(self, value):
        self._mu_ratio.value = value

    @property
    def weights(self):
        return self._weights.values

    @weights.setter
    def weights(self, value):
        self._weights.values = value

    @property
    def cs(self):
        return self._cs.value

    @cs.setter
    def cs(self, value):
        self._cs.value = value

    @property
    def damps(self):
        return self._damps.value

    @damps.setter
    def damps(self, value):
        self._damps.value = value

    @property
    def cc(self):
        return self._cc.value

    @cc.setter
    def cc(self, value):
        self._cc.value = value

    @property
    def c1(self):
        return self._c1.value

    @c1.setter
    def c1(self, value):
        self._c1.value = value

    @property
    def cmu(self):
        return self._cmu.value

    @cmu.setter
    def cmu(self, value):
        self._cmu.value = value

    @property
    def cmu(self):
        return self._cmu.value
    
    @property
    def CHI_N(self):
        return self._CHI_N.value
    
    @property
    def MU_EFF(self):
        return self._MU_EFF.value

#   --------------------------- Private Methods ---------------------------
    def __check_static_hyperparameters(self):
        # --- add a constant
        self._CHI_N = Float(
            value=self.D ** 0.5 * (1 - 1/(4*self.D) + 1 / (21*self.D ** 2)),
            default=None,
            name="???\u03C7_N???",
            remark="这可以作为一个 关于常数的正确写法 的例子")
        # --- set default values
        # N
        self._N.default = int(4 + np.floor(3 * np.log(self.D)))
        self._N.min, self._N.max = 2, 300
        # mu
        if self.mu_ratio is None:
            self._mu.default = int(np.floor(self.N / 2))
        else:
            self._mu.default = int(np.floor(self.N * self.mu_ratio))
        # weights
        weights = (np.log(self.N/2 + 0.5)
                   - np.log(np.arange(1, self.mu + 1, dtype=np.float)))
        self._weights.default = weights / np.sum(weights)        
        # --- add a constant
        self._MU_EFF = Float(
            value=np.sum(self.weights) ** 2 / np.sum(self.weights ** 2),
            name="???\u03BC_EFF???",
            remark="这可以作为一个 关于常数的正确写法 的例子")
        # --- set default values
        # Step-size Adaptation
        self._cs.default = (self.MU_EFF + 2) / (self.D + self.MU_EFF + 5)
        self._damps.default = (1 
            + 2 * max(0, ((self.MU_EFF - 1)/(self.D + 1)) ** 0.5 - 1) + self.cs)
        # Covariance Matrix Adaptation
        self._cc.default = ((4 + self.MU_EFF / self.D)
                            / (self.D + 4 + 2 * self.MU_EFF / self.D))
        self._c1.default = 2 / ((self.D + 1.3) ** 2 + self.MU_EFF)
        self._cmu.default = min(1 - self.c1,
                                2 * (self.MU_EFF - 2 + 1 / self.MU_EFF)
                                / ((self.D + 2) ** 2 + self.MU_EFF))


class PSO(_ConfigureAlgorithm):
    CONFIG_NAME = "PSO related configuration"
    
    def __init__(self, seed=None, N=20, w=0.7, c1=0.5, c2=0.5,
                 all_time_best=True, topology="star",
                 initial_scheme="latin_hypercube", update_scheme="immediate",
                 **kwargs):
        super(PSO, self).__init__(seed, N, initial_scheme, update_scheme)
        self._w = Float(
            seed=seed,
            value=w,
            default=0.7, min=0., max=1.,
            name="Inertia weight",
            remark="This parameter balances the exploration–exploitation"
                   " trade-off. It serves as a memory of the previous flight"
                   " direction.")
        self._c1 = Float(
            seed=seed,
            value=c1,
            default=0.5, min=0.,
            name="Cognitive coefficient",
            remark="It is referred to as trust parameter. It expresses how much"
                   " confidence a particle has in itself. It also quantifies"
                   " the performance of particle i relative to past"
                   " performances.")
        self._c2 = Float(
            seed=seed,
            value=c2,
            default=0.5, min=0.,
            name="Social coefficient",
            remark="It is referred to as trust parameter. It expresses how much"
                   " confidence a particle has its neighbors. It also"
                   " quantifies the performance of particle i relative to a"
                   " group of particles, or neighbors.")
        self._all_time_best = Boolean(
            seed=seed,
            value=all_time_best,
            default=True,
            admission=(True, False),
            name="How to choose the best position in swarm",
            remark="This choice determinates whether the best positions (gbest"
                   " or lbest) are selected from the particle positions of the"
                   " current iterations (False), or from the personal best"
                   " positions of all particles (True).")
        self._v_init = Float(
            seed=seed,
            value=kwargs.get("v_init"),
            default=0.,
            name="Initial velocity",
            remark="The initial velocities of all particles in swarm. It can be"
                   " initialized to zero (recommended). If, however, velocities"
                   " are also randomly initialized, such velocities should not"
                   " be too large.")
        self._v_min = Float(
            seed=seed,
            value=kwargs.get("v_min"),
            default=np.NINF,
            name="Minimum velocity allowed for particles",
            remark="This value defines the lower bound of particle's velocity.")
        self._v_max = Float(
            seed=seed,
            value=kwargs.get("v_max"),
            default=np.PINF,
            name="Maximum velocity allowed for particles",
            remark="This value defines the upper bound of particle's velocity.")
        self._topology = String(
            seed=seed,
            value=topology.lower(),
            default="star",
            admission=("star", "ring"),
            name="Topology of swarm",
            remark="The topology (social network structure) for PSO is"
                   " determined by the formation of overlapping neighborhoods,"
                   " where particles within a neighborhood influence one"
                   " another.")
        self._n_neighbors = Integer(
            seed=seed,
            value=kwargs.get("n_neighbors"),
            min=1, max=self.N,
            name="Neighborhood size",
            remark="The neighborhood size defines the extent of social"
                   " interaction within the swarm. Starting the search with" 
                   " small neighborhoods and increasing the neighborhood size"
                   " proportionally to the increase in number of iterations is"
                   " a good approach.")
        
    #   --------------------------- Setter/Getter ---------------------------
    @property
    def w(self):
        return self._w.value

    @w.setter
    def w(self, value):
        self._w.value = value

    @property
    def c1(self):
        return self._c1.value
    
    @c1.setter
    def c1(self, value):
        self._c1.value = value

    @property
    def c2(self):
        return self._c2.value

    @c2.setter
    def c2(self, value):
        self._c2.value = value

    @property
    def all_time_best(self):
        return self._all_time_best.value

    @all_time_best.setter
    def all_time_best(self, value):
        self._all_time_best.value = value

    @property
    def v_init(self):
        return self._v_init.value
    
    @v_init.setter
    def v_init(self, value):
        self._v_init.value = value

    @property
    def v_min(self):
        return self._v_min.value

    @v_min.setter
    def v_min(self, value):
        self._v_min.value = value

    @property
    def v_max(self):
        return self._v_max.value

    @v_max.setter
    def v_max(self, value):
        self._v_max.value = value

    @property
    def topology(self):
        return self._topology.value
    
    @topology.setter
    def topology(self, value):
        self._topology.value = value

    @property
    def n_neighbors(self):
        return self._n_neighbors.value
    
    @n_neighbors.setter
    def n_neighbors(self, value):
        self._n_neighbors.value = value



#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    case = 6

# ----- Integer
    if case == 1:
        a = Integer(-1, 0, name="a",)
        print("a:", a)
        b = Integer(2, default=0, min=-1, max=3, remark="This is variable `b`")
        print("b:", b)
        c = Integer([-9,0,9], name="List of integers")
        print("c:", c)

        print("a>b:", a > b)
        print("a>=b", a >= b)
        print("a==b:", a == b)
        print("a!=b:", a != b)
        print("a<=b:",a <= b)
        print("a<b:", a < b)

        print("a+b:", a + b)
        print("a-b:", a - b)
        print("a*b:", a * b)
        print("a/b:", a / b)
        print("a//b:", a // b)
        print("a%b:", a % b)
        print("a**b:", a ** b)
        print("abs(a):", abs(a))

# ----- Float
    if case == 2:
        a = Float(-1.6, 0, name="a",)
        print("a:", a)
        b = Float(2.0, default=0, min=-1, max=3, remark="This is variable `b`")
        print("b:", b)
        c = Float([-9., 0., 9., np.inf, np.nan], name="List of floats")
        print("c:", c)

        print("a>b:", a > b)
        print("a>=b", a >= b)
        print("a==b:", a == b)
        print("a!=b:", a != b)
        print("a<=b:", a <= b)
        print("a<b:", a < b)

        print("a+b:", a + b)
        print("a-b:", a - b)
        print("a*b:", a * b)
        print("a/b:", a / b)
        print("a//b:", a // b)
        print("a%b:", a % b)
        #* Be careful when the base is negative and the exponent is decimal 
        #* (e.g. (-0.1)**3.2), the result is complex
        print("a**b:", a ** b)
        print("abs(a):", abs(a))

# ----- String
    if case == 3:
        a = String("A", name="a",)
        print("a:", a)
        b = String("b", default="B", remark="This is variable `b`")
        print("b:", b)
        c = String(("o", "P", "qR"), name="Tuple of strings")
        print("c:", c)
        print("a+b:", a + b)

# ----- Algorithm configuration DE
    if case == 4:
        config = DE()
        print(config._N.seed)
        exit()
        config.set("CR", 0.5)
        config.set("crossover", "bin")
        print("after:", config)
        try:
            config.set("F", 3.5)
        except Exception as e:
            print("#1 error message:", e)
        try:
            config.set("initial_scheme", 2)
        except Exception as e:
            print("#2 error message:", e)
        try:
            config.set("CR", [2.0, "a", 4])
        except Exception as e:
            print("#3 error message:", e)
        try:
            config.set("mutation", "AAAA")
        except Exception as e:
            print("#4 error message:", e)

# ----- Algorithm configuration CMAES
    if case == 5:
        config = CMAES(D=5)
        print(config)

# ----- Algorithm configuration PSO
    if case == 6:
        config = PSO()
        print(config)

if __name__ == '__main__':
        main()
