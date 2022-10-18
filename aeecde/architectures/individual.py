#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Individuals related classes and functions
'''
import numpy as np
from copy import deepcopy



#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------
class __base(object):
    ''' :class:`__base` is the base class for defining other individual classes.
    '''
    def __init__(self):
        '''
        Creates a new :class:`__base` for individuals.

        Attributes
        ----------
        xvalue : 1D array of shape (1,?)
            The variable to be optimized
        fvalue : float
            The value of objective function for xvalue.

        '''
        self._distance_hist = []
        self._state_hist = []
        self._xvalue_hist = []
        self._fvalue_hist = []

    # Returns the representation of the instance
    def __repr__(self):
        return "{}( x={}, f(x)={} )".format(type(self).__name__, self.xvalue,
            self.fvalue)

    # Rich comparison methods with respect to the `fvalue` attribute of objects
    def __cmp_check(self, other, operator):
        if not isinstance(other, type(self)):
            raise TypeError('Cannot compare between {} object and {} object'
                .format(type(self).__name__, type(other).__name__))
        elif (self.fvalue is None) or (other.fvalue is None):
            raise AttributeError('Cannot compare unevaluated value')
        else:
            return getattr(self.fvalue, operator)(other.fvalue)

    def __lt__(self, other):
        return self.__cmp_check(other, "__lt__")

    def __le__(self, other):
        return self.__cmp_check(other, "__le__")

    def __eq__(self, other):
        return self.__cmp_check(other, "__eq__")

    def __ne__(self, other):
        return self.__cmp_check(other, "__ne__")

    def __ge__(self, other):
        return self.__cmp_check(other, "__ge__")

    def __gt__(self, other):
        return self.__cmp_check(other, "__gt__")

    @property  # getter
    def xvalue(self):
        return self._xvalue

    @xvalue.setter  # setter
    def xvalue(self, value):
        if value is not None:
            if isinstance(value, (int, float)):
                value = np.array([value])
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:  # convert scalar to 1D array
                    value.shape = (1, )
                elif value.ndim == 1:
                    pass
                elif (value.ndim == 2 and
                    (value.shape[0] == 1 or value.shape[1] ==1)):
                    value = value.flatten()
                else:
                    raise ValueError("`xvalue` attribute must has a shape of"
                        " (1, ) (i.e. a 1D array) not a shape of {}".format(
                        value.shape))
            else:
                raise TypeError("`xvalue` attribute must be None or a ndarray,"
                    " not a {} of {}".format(type(value).__name__, value))
        self._xvalue = value
        self._xvalue_hist.append(self._xvalue)

    @property
    def xvalue_hist(self):
        return self._xvalue_hist

    @property
    def fvalue(self):
        return self._fvalue

    @fvalue.setter
    def fvalue(self, value):
        if value is None:
            self._fvalue = value
        elif isinstance(value, (int, float,)):
            self._fvalue = float(value)
        else:
            raise TypeError("`fvalue` attribute must be None or a scalar, not a"
                " {} of {}".format(type(value).__name__, value))
        self._fvalue_hist.append(self._fvalue)

    @property
    def fvalue_hist(self):
        return self._fvalue_hist

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, string):
        if string.lower() == "explore":
            self._state = "explore"
        elif string.lower() == "exploit":
            self._state = "exploit"
        elif string.lower() == "neutral":
            self._state = "neutral"
        else:
            raise ValueError("`state` attribute must be on of 'explore',"
                " 'exploit' or 'neutral' ")
        self._state_hist.append(self._state)

    @property
    def state_hist(self):
        return self._state_hist

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = value
        self._distance_hist.append(self._distance)

    @property
    def distance_hist(self):
        return self._distance_hist

    def clone(self):
        ''' Return a copy of this instance. '''
        return deepcopy(self)


class Particle(__base):
    ''' :class:`Particle` is the definition of particles in PSO methods.
    '''
    def __init__(self, position=None, fvalue=None, velocity=None,
                 p_best=None, f_best=None):
        '''
        Creates a new :class:`Particle` used in PSO methods.

        Parameters
        ----------
        position : array of shape (1,?)
            The variable to be optimized, i.e., a candidate solution.
        fvalue : float
            The value of objective function
        velocity : array of shape (1,?)
            The speed of current particle.
        p_best : array of shape (1,?), optional
            The best position found so far of the particle
        f_best : float, optional
            The objective function value at `p_best`.
        '''
        super().__init__()
        self.position = position
        self.fvalue = fvalue
        self.velocity = velocity
        self.p_best = p_best
        self.f_best = f_best

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self.xvalue = value
        self._position = self.xvalue

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if value is not None:
            if isinstance(value, (int, float)):
                value = np.array([value])
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    value.shape = (1, )
                elif value.ndim == 1:
                    pass
                elif (value.ndim == 2 and
                      (value.shape[0] == 1 or value.shape[1] == 1)):
                    value = value.flatten()
                else:
                    raise ValueError("`velocity` attribute must has a shape of"
                        " (1, ) (i.e. a 1D array) not a shape of {}".format(
                        value.shape))
            else:
                raise TypeError("`velocity` attribute must be None or a ndarray"
                    ", not a {} of {}".format(type(value).__name__, value))
        self._velocity = value

    @property
    def p_best(self):
        return self._p_best

    # Same reason as f_best setter
    @p_best.setter
    def p_best(self, value):
        self._p_best = value

    @property
    def f_best(self):
        return self._f_best

    # Since f_best is calculated by the algorithm, there's no need for
    # rechecking the type of this attribute
    @f_best.setter
    def f_best(self, value):
        self._f_best = value

    def update_own_best(self):
        if (self.f_best is None) or (self.fvalue < self.f_best):
            self.f_best = self.fvalue
            self.p_best = self.xvalue

    def update_velocity(self, w, c1, c2, X_best):
        return (w * self.velocity + c1 * (self.p_best-self.xvalue)
                + c2 * (X_best-self.xvalue))

    def update_position(self):
        self.xvalue = self.xvalue + self.velocity


class Individual(__base):
    ''' :class:`Individual` is the definition of individuals by inheriting from
    `__base`. This Individual class is used in many Evolutionary Algorithms,
    such as ES, DE, GA(floating-point representation).
    '''
    def __init__(self, solution=None, fitness=None):
        '''
        Creates a new :class:`Individual` used in EA methods.

        Parameters
        ----------
        solution : array of shape (1,?)
            The candidate solution (a vector of decision variables).
        fitness : float
            The fitness value of the candidate solution.
        '''
        super().__init__()
        self.solution = solution
        self.fitness = fitness
        self._F = None
        self._CR = None
        self._mutation = None
        self._crossover = None
        self._mutation_hist = []
        self._crossover_hist = []
        self._F_hist = []
        self._CR_hist = []

    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self, value):
        self.xvalue = value
        self._solution = self.xvalue

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self.fvalue = value
        self._fitness = self.fvalue

    @property
    def mutation(self):
        return self._mutation

    @mutation.setter
    def mutation(self, operator):
        self._mutation = operator
        self._mutation_hist.append(self._mutation)

    @property
    def mutation_hist(self):
        return self._mutation_hist

    @property
    def crossover(self):
        return self._crossover

    @crossover.setter
    def crossover(self, operator):
        self._crossover = operator
        self._crossover_hist.append(self._crossover)

    @property
    def crossover_hist(self):
        return self._crossover_hist

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value
        self._F_hist.append(self._F)

    @property
    def F_hist(self):
        return self._F_hist

    @property
    def CR(self):
        return self._CR

    @CR.setter
    def CR(self, value):
        self._CR = value
        self._CR_hist.append(self._CR)

    @property
    def CR_hist(self):
        return self._CR_hist



#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    case = "ALL"
    # ----- class Particle()
    # attribute tests
    if case == 1 or case == "ALL":
        # Default instance
        par = Particle()
        print("[OK] Default instance: par = Particle()")

        # Explicit instance
        par = Particle(fvalue=1.0, position=np.array([1,2,3]),
            velocity=np.array([0.1,0.2,0.3]), f_best=10.,
            p_best=np.array([1.1, 2.2, 3.3]))
        print("[OK] Explicit instance: par = Particle(...)")

        # Auto-corrections
        par = Particle(position=np.array(10))
        print("[OK] Scalar: par = Particle(position=np.array(10))")
        par = Particle(position=np.array([20]))
        print("[OK] Scalar: par = Particle(position=np.array([20]))")
        par = Particle(position=np.array([[30]]))
        print("[OK] Scalar: par = Particle(position=np.array([[30]]))")
        par = Particle(position=np.array([1., 2., 3., 4., 5.]))
        print("[OK] 1D array: Particle(position=np.array([1.,2.,3.,4.,5.]))")
        par = Particle(position=np.array([[1., 2., 3., 4., 5.]]))
        print("[OK] 2D array: Particle(position=np.array([[1.,2.,3.,4.,5.]]))")
        par.xvalue
        print("[OK] getter: par.xvalue")
        # par = Particle(xvalue=np.array([1, 2, 3]))
        # print("[OK] **kwargs: Particle(xvalue=np.array([1, 2, 3]))")

        # Error messages
        try:
            par = Particle(fvalue="test")
        except Exception as e:
            print("[OK] #1 Error message:", e)

        try:
            par = Particle(position="test")
        except Exception as e:
            print("[OK] #2 Error message:", e)

        try:
            par = Particle(position=np.array([[[1, 2, 3]]]))
        except Exception as e:
            print("[OK] #3 Error message:", e)

        try:
            par = Particle(position=np.array([1, 2, 3]),
                velocity=np.array([0.1, 0.2]))
        except Exception as e:
            print("[OK] #4 Error message:", e)

    # method tests
    if case == 2 or case == "ALL":
        a = Particle(fvalue=1, position=np.array(10))
        b = Particle(fvalue=2, position=np.array(20))
        print("\ta.fvalue = {}, b.fvalue = {}".format(a.fvalue, b.fvalue))
        print("[OK] a < b:", a < b)
        print("[OK] a <= b:", a <= b)
        print("[OK] a == b:", a == b)
        print("[OK] a != b:", a != b)
        print("[OK] a >= b:", a >= b)
        print("[OK] a > b:", a > b)
        import operator
        print("[OK] operator.lt(a, b):", operator.lt(a, b))
        try:
            c = Individual(fitness=3, solution=np.array(30))
            operator.ge(a, c)
        except Exception as e:
            print("[OK] #5 Error message:", e)

    # ----- class Individual()
    if case == 3 or case == "ALL":
        try:
            a = Individual()
            b = Particle()
            print(a<b)
        except Exception as e:
            print("[OK] #6 Error message:", e)


    # ------ evaluate Individual()
    if case == 4 or case == "ALL":
        # define evaluation function:
        #![***] ensure bounds is performed during evaluate an individual's fvalue.
        # [TO DO]: `xvalue` of individual is directly changed within bounds here,
        # and return the `fvalue` of the changed `xvalue`.
        # It maybe better to not change `xvalue` directly, but return a new individual
        # whose `xvalue` is in bounds, and `fvalue` is the calculated fvalue.
        def evaluate(x):
            lower_bound = [-5] * len(x)
            upper_bound = [5] * len(x)
            # boundary handlinig:
            l_mask = np.where(x <= np.array(lower_bound))
            x[l_mask] = np.array(lower_bound)[l_mask]
            u_mask = np.where(x >= np.array(upper_bound))
            x[u_mask] = np.array(upper_bound)[u_mask]
            fitness = np.sum(x)
            return fitness

        # test evaluation function
        ind1 = Individual(solution=np.array([1,1,1,1], dtype=float))
        ind2 = Individual(solution=np.array([-6, 4, 6, -3], dtype=float))
        print("\tUnevaluated ind1: {}".format(ind1))
        print("\tUnevaluated ind2: {}".format(ind2))
        ind1.fitness = evaluate(ind1.solution)
        ind2.fitness = evaluate(ind2.solution)
        print("\tEvaluated ind1: {}".format(ind1))
        print("\tEvaluated ind2: {}".format(ind2))
        print("[OK] Boundary handling & fitness evaluation are performed at the"
              " same time.")


if __name__ == '__main__':
        main()
