#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Basic component and operator for tuning hyperparameters
TODO
    - Contextual bandit strategies
    - online linear classfier: LinUCB (Upper Confidence Bound) algorithm
    - online non-linear classifier: UCBogram algorithm
'''
import math
import numpy as np
from sklearn.neighbors import KernelDensity
from copy import deepcopy
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
    ''' :class:`__base` is the base class for defining a tuner.'''
    TUNER_NAME = "No name"


#*  ------------------------------ Basic Tuners ------------------------------
class KDE(__base):
    '''Kernel Density Estimation (KDE) tuner'''
    TUNER_NAME = "Kernel Density Estimation tuner"

    def __init__(self, kernel, bandwidth, value_range, features_name,
                 max_training_size, random_state):
        self.kde_model = KernelDensity()
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.value_range = value_range
        self.features_name = features_name
        self.max_training_size = max_training_size
        self.rng = random_state
        self.data = []
        # indicate the KDE model for current data is fitted or not
        self.model_is_fitted = False

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, name):
        self._kernel = name
        self.kde_model.set_params(kernel=self._kernel)

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        if value is None:
            value = 1.0  # default value from scikit-learn
        self._bandwidth = value
        self.kde_model.set_params(bandwidth=self._bandwidth)

    @property
    def value_range(self):
        return self._value_range

    @value_range.setter
    def value_range(self, boundary):
        if boundary is None:
            self._value_range = None
        else:
            if isinstance(boundary[0], (list, tuple)):
                self._value_range = boundary
            else:
                self._value_range = ((boundary[0],), (boundary[1],))

    @property
    def features_name(self):
        return self._features_name
    
    @features_name.setter
    def features_name(self, name):
        if name is None:
            self._features_name = None
        elif isinstance(name, str):
            self._features_name = [name, ]
        elif isinstance(name, (tuple, list)):
            self._features_name = name
        else:
            raise TypeError("`features_name` must be None, str, or a list of"
                " str, not {}".format(type(name)))

    @property
    def data_size(self):
        return len(self.data)

    def reset(self):
        self.data = []
        self.model_is_fitted = False

    def set_params(self, **kwargs):
        self.kde_model.set_params(**kwargs)

    def get_params(self, deep=True):
        return self.kde_model.get_params(deep)

    def get_log_density(self, X):
        return self.kde_model.score_samples(X)

    def add_data(self, new_sample):
        '''update the training data set by a batch'''
        self.data.append(new_sample)
        self.model_is_fitted = False

    def add_batch_data(self, new_batch_samples):
        '''update the training data set by a batch'''
        self.data.extend(new_batch_samples)
        self.model_is_fitted = False

    def sample(self, n_samples=1):
        if self.data_size >= 1:
            if self.model_is_fitted is False:
                self.fit()
            samples = []
            for i in range(int(n_samples)):
                x = self.kde_model.sample(
                    n_samples=1, random_state=self.rng)[0]
                while (x < self.value_range[0]).any() or (x > self.value_range[-1]).any():
                    x = self.kde_model.sample(
                        n_samples=1, random_state=self.rng)[0]
                samples.append(x)

                # x = self.kde_model.sample(n_samples=1, random_state=self.rng)[0]
                # if "F" in self.features_name:
                #     F_index = self.features_name.index("F")
                #     while x[F_index] <= self.value_range[0][F_index]:
                #         x = self.kde_model.sample(n_samples=1, random_state=self.rng)[0]
                # x = np.clip(x, self.value_range[0], self.value_range[1])
                # samples.append(x.tolist())

        else:
            X = self.rng.uniform(low=self.value_range[0],
                                 high=self.value_range[-1],
                                 size=(n_samples, len(self.value_range[0])))
            samples = X.tolist()
        return samples

    def score_samples(self, X):
        return self.kde_model.score_samples(X)

    def fit(self):
        if self.data_size >= 1:
            if self.max_training_size is not None:
                if self.data_size > self.max_training_size:
                    training_data = self.data[-self.max_training_size:]
                else:
                    training_data = self.data
            else:
                training_data = self.data
            self.kde_model.fit(X=np.array(training_data))
            self.training_data = training_data
            self.model_is_fitted = True
        else:
            raise ValueError("The training data set must have at least one data"
                             " point")

#*  ------------------------ Multi-Armed Bandit Tuners ------------------------
class __baseMAB(__base):
    ''' :class:`__baseMAB` is the base class for Multi-Armed Bandit tuner.'''

    def __init__(self, list_arm_names, random_state, value_estimation_method, 
        **kwargs):
        '''
        Creates a new :class:`__baseMAB` for Multi-Armed Bandit.

        Parameters
        ----------
        list_arm_names : list
            The collections of arms' name
        random_state: rng of np.random
            Random state instance of np
        value_estimation_method: str
            Either be 'sliding_window_average' or 'sample_average'. The first
            calculates the moving average in contrast to the global arithmetic
            average
        sliding_window_size: int (optional)
            If 'sliding_window_average' is enabled in `value_estimation_method`,
            this value defines the rolling size of data-set
        '''
        self.arms = list_arm_names
        self.value_estimation_method = value_estimation_method
        self.sw_size = kwargs.get("sliding_window_size")
        self.counts = {arm: 0 for arm in self.arms}
        self.values = {arm: 0.0 for arm in self.arms}
        self.rng = random_state

    @property
    def value_estimation_method(self):
        return self._value_estimation_method
    
    @value_estimation_method.setter
    def value_estimation_method(self, method):
        if isinstance(method, str):
            self._value_estimation_method = method.lower()
            if self._value_estimation_method == "sliding_window_average":
                self.rewards = {arm: [] for arm in self.arms}
        else:
            raise TypeError("`value_estimation_method` must be a str but {} is" 
                " given".format(type(method)))

    @property
    def sw_size(self):
        return self._sw_size

    @sw_size.setter
    def sw_size(self, size):
        if self.value_estimation_method == "sliding_window_average":
            if isinstance(size, (int, float)):
                self._sw_size = int(size)
            else:
                raise TypeError("`sliding_window_size` must be a int but {} is"
                    " given".format(type(method)))
        else:
            self._sw_size = size

    def reset(self):
        if self.value_estimation_method == "sliding_window_average":
            self.rewards = {arm: [] for arm in self.arms}
        self.counts = {arm: 0 for arm in self.arms}
        self.values = {arm: 0.0 for arm in self.arms}

    def select_arm(self):
        raise NotImplementedError("method `select_arm` has not been implemented"
            " yet.")

    def select_multiple_arms(self, k):
        selected_arms = [self.select_arm() for i in range(int(k))]
        self.rng.shuffle(selected_arms)
        return selected_arms

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        if self.value_estimation_method == "sample_average":
            self.values[chosen_arm] = (self.values[chosen_arm]
                                       + (reward - self.values[chosen_arm])
                                       / self.counts[chosen_arm])
        elif self.value_estimation_method == "sliding_window_average":
            self.rewards[chosen_arm].append(reward)
            if self.counts[chosen_arm] > self.sw_size:
                self.values[chosen_arm] = sum(
                    self.rewards[chosen_arm][-self.sw_size:])/self.sw_size
            else:
                self.values[chosen_arm] = sum(
                    self.rewards[chosen_arm])/self.counts[chosen_arm]
        else:
            raise ValueError("`value_estimation_method` must be either"
                             " 'sample_average' or 'sliding_window_average'")

    def update_batch(self, chosen_arms, rewards):
        for chosen_arm, reward in zip(chosen_arms, rewards):
            self.update(chosen_arm, reward)


class EpsilonGreedy(__baseMAB):
    '''Epsilon greedy multi-armed bandit'''
    TUNER_NAME = "Epsilon-Greedy"

    def __init__(self, list_arm_names, random_state,
                 value_estimation_method, **kwargs):
        super().__init__(list_arm_names, random_state,
                         value_estimation_method, **kwargs)
        epsilon = kwargs.get("epsilon")
        if epsilon is None:
            raise ValueError(
                "`epsilon` is required for EpsilonGreedy algorithm.")
        self.epsilon = kwargs.get("epsilon")

    def select_arm(self):
        if self.rng.rand() > self.epsilon:
            # Select the arm with maximum value
            selected_arm = max(self.values, key=self.values.get)
        else:
            # Randomly select an arm
            i_arm = self.rng.randint(0, len(self.arms))
            selected_arm = self.arms[i_arm]
        return selected_arm


class Softmax(__baseMAB):
    '''Softmax multi-armed bandit'''
    TUNER_NAME = "Softmax"

    def __init__(self, list_arm_names, random_state,
                 value_estimation_method, **kwargs):
        super().__init__(list_arm_names, random_state,
                         value_estimation_method, **kwargs)
        self.tau = kwargs.get("temperature")

    def select_arm(self):
        if self.tau is None:
            z = sum([math.exp(self.values[arm]) for arm in self.arms])
            probs = {arm: math.exp(self.values[arm]) / z for arm in self.arms}
        else:
            z = sum([math.exp(self.values[arm] / self.tau)
                     for arm in self.arms])
            probs = {arm: math.exp(
                self.values[arm] / self.tau) / z for arm in self.arms}
        # select one arm using Roulette selection
        r = self.rng.rand()
        cum_prob = 0.0
        for arm in self.arms:
            cum_prob += probs[arm]
            if cum_prob >= r:
                selected_arm = arm
                break
        return selected_arm


class SUS_Softmax(Softmax):
    '''Stochastic Universal Selection (SUS) Softmax multi-armed bandit'''
    TUNER_NAME = "SUS Softmax"

    def __init__(self, list_arm_names, random_state,
                 value_estimation_method, **kwargs):
        super().__init__(list_arm_names, random_state,
                         value_estimation_method, **kwargs)

    def select_multiple_arms(self, k):
        if self.tau is None:
            z = sum([math.exp(self.values[arm]) for arm in self.arms])
            probs = {arm: math.exp(self.values[arm]) / z for arm in self.arms}
        else:
            z = sum([math.exp(self.values[arm] / self.tau)
                     for arm in self.arms])
            probs = {arm: math.exp(
                self.values[arm] / self.tau) / z for arm in self.arms}
        spacing = 1.0/k
        r = self.rng.rand()/k
        cum_prob = 0.0
        selected_arms = []
        for arm in self.arms:
            cum_prob += probs[arm]
            while r < cum_prob:
                selected_arms.append(arm)
                r += spacing
        # randomly shuffle the list of strategy names
        self.rng.shuffle(selected_arms)
        return selected_arms


class UCB(__baseMAB):
    '''Upper Confidence Bound (UCB) multi-armed bandit'''
    TUNER_NAME = "UCB"

    def __init__(self, list_arm_names, random_state,
                 value_estimation_method, **kwargs):
        super().__init__(list_arm_names, random_state,
                         value_estimation_method, **kwargs)

    def select_arm(self):
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm
        ucb_values = {arm: 0.0 for arm in self.arms}
        total_counts = sum(list(self.counts.values()))
        for arm in self.arms:
            bonus = math.sqrt(2 * math.log(total_counts) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        selected_arm = max(ucb_values, key=ucb_values.get)
        return selected_arm

    def select_multiple_arms(self, k):
        counts = deepcopy(self.counts)
        selected_arms = []
        for i in range(int(k)):
            ucb_values = {arm: 0.0 for arm in self.arms}
            total_counts = sum(list(counts.values()))
            for arm in self.arms:
                if counts[arm] == 0:
                    ucb_values[arm] = np.Inf
                else:
                    bonus = math.sqrt(2 * math.log(total_counts) / counts[arm])
                    ucb_values[arm] = self.values[arm] + bonus
            selected_arm = max(ucb_values, key=ucb_values.get)
            counts[selected_arm] = counts[selected_arm] + 1
            selected_arms.append(selected_arm)
        self.rng.shuffle(selected_arms)
        return selected_arms



#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def multi_armed_bandit(bandit_algo, list_arm_names, random_state,
                       value_estimation_method, **kwargs):
    if bandit_algo == "EpsilonGreedy":
        return EpsilonGreedy(list_arm_names, random_state,
                             value_estimation_method, **kwargs)
    elif bandit_algo == "Softmax":
        return Softmax(list_arm_names, random_state,
                       value_estimation_method, **kwargs)
    elif bandit_algo == "SUS_Softmax":
        return SUS_Softmax(list_arm_names, random_state,
                           value_estimation_method, **kwargs)
    elif bandit_algo == "UCB":
        return UCB(list_arm_names, random_state,
                   value_estimation_method, **kwargs)
    else:
        raise ValueError(
            "`bandit_algo` must be one of 'EpsilonGreedy', 'Softmax' or 'UCB1'")



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    case = 35
    pass


if __name__ == "__main__":
    main()    
