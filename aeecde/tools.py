#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Some utilities and useful toolbox
'''
import json
import datetime
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt



#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------



#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def plot(x, y, linestyle="-", label="", xlabel="", ylabel="", title="",ax=None,
    legend=False, show=True, **kwargs):
        ''' Plot result curve '''
        print("[INFO] Plotting ...")
        if x is None:
            raise ValueError("Cannot plot a figure when `x`is None")
        else:
            print("    `x` has a length of:",len(x))
        if y is None:
            raise ValueError("Cannot plot a figure when `y`is None")
        else:
            print("    `y` has a length of:", len(x))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, linestyle, label=label, markersize=kwargs.get("markersize"))
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if label != "" or legend:
            ax.legend()
        if show:
            plt.show()


def plot_hist(x, bins, density, cumulative, label="", xlabel="", ylabel="",
    title="", ax=None, legend=False, show=True):
        ''' Plot result curve '''
        print("[INFO] Plotting ...")
        if x is None:
            raise ValueError("Cannot plot a figure when `x`is None")
        else:
            print("    `x` has a length of:", len(x))
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(x, bins, density=density, cumulative=cumulative, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if label != "" or legend:
            ax.legend()
        if show:
            plt.show()


def plot_bar_stacked(data, x_label=None, y_label=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    test = data.copy()
    # get data length
    temp = data.popitem()
    length = len(temp[1])
    data.update({temp[0]:temp[1]})
    x = range(length)
    cumul = [0] * length
    # personalize the colormap
    my_cmap = plt.cm.get_cmap("YlGn")
    i = 0
    for key, value in data.items(): #! HB: NOT ordered
        ax.bar(x, value, bottom=cumul, label=key, color=my_cmap(i*20))
        cumul = np.add(cumul, value).tolist()
        i += 1
    ax.legend()
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)


def plot_heatmap(data, x_labels=None, y_labels=None, x_label=None, y_label=None,
    ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    '''
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    y_labels
        A list or array of length N with the labels for the rows.
    x_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    '''

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if x_labels is not None:
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticklabels(y_labels)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def check_random_state(seed):
    '''
    Turn seed into a np.random.RandomState object

    Parameters
    ----------
    seed : None | int | instance of RandomState
        - If it is None, return the RandomState singleton used by np.random.
        - If it is an int, return a new RandomState instance seeded with seed.
        - If it is already a RandomState instance, return it.

    Returns
    -------
        An object of np.random.RandomState serving as a random number generator
    '''
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed

def to_json(data, path, replace=True):
    '''
    Convert a Python object to a JSON file

    Note NaN and None will be converted to null and datetime objects will be
    converted to UNIX timestamps.

    Parameters
    ----------
    data : Python data types (dict, list, tuple, str, int, float, bool, None)
        Object to be dumped
    path : str or Path
        File path where the JSON file will be stored
    replace : bool, optional
        Whether to replace the existing file, by default True
    '''
    # convert unserializable objects
    if isinstance(data, dict):
        for key, value in data.items():
            # convert value to be serializable
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, np.ndarray):
                        value[i] = v.tolist()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        value[k] = v.tolist()
            # convert key to string
            if not isinstance(key, str):
                data[str(key)] = data.pop(key)
    # write data by using json format
    encode = json.dumps(data, indent=4)
    # check if the file already exists
    if not isinstance(path, Path):
        path = Path(path)
    if path.exists() and not replace:
        oldname, extension = path.stem, path.suffix
        suffix = str(datetime.datetime.now())
        path = path.with_name(oldname+"_"+suffix+extension)
    # write to file
    with path.open("w", encoding="utf-8") as f:
        f.write(encode)

#   ------------------ Measure the diversity/similarity ------------------
def euclidian_distance(x, y):
    return np.linalg.norm(x-y)

def scn1(IND_new, IND_parent):
    ''' Similarity to its parent(s)

    Parameters
    ----------
    IND_new : list | np.array
        The list of child individuals
    IND_parent : list | np.array
        The list of parent individuals

    Returns
    -------
        The Euclidian distance individuals

    References
    ----------
        M. Crepinsek, et al., 2013, Exploration and exploitation in evolutionary algorithms: A survey, page 35:12, equation (3)
    '''
    X_new, X_parent = IND_new.X, IND_parent.X
    return np.linalg.norm(np.array(X_new)-np.array(X_parent), ord=2, axis=1)

def scn2(POP):
    ''' Similarity to the most similar individual within the whole population P

    Parameters
    ----------
    POP : list | np.array
        The list of all individuals in population P

    Returns
    -------
        The Euclidian distance individuals

    References
    ----------
        M. Crepinsek, et al., 2013, Exploration and exploitation in evolutionary algorithms: A survey, page 35:12, equation (5)
    '''
    temp = POP.X
    ones = np.ones((temp.shape[0]-1, 1))
    dist = []
    for i, x in enumerate(temp): #TODO HB: this loop can still be improved
        d = np.linalg.norm(ones*x - np.delete(temp, i, axis=0), ord=2, axis=1)
        POP.list_ind[i].distance = d.min()
        dist.append(d.min())
    return np.array(dist)

def scn3(IND):
    ''' Similarity to the most similar individual in the subpopulation P'

    Parameters
    ----------
    IND : list | np.array
        The list of all individuals in population P

    Returns
    -------
        The Euclidian distance individuals

    References
    ----------
        M. Crepinsek, et al., 2013, Exploration and exploitation in evolutionary algorithms: A survey, page 35:12, equation (6)
    '''
    pass

def scn4(IND):
    ''' As a similarity to the most similar individual throughout the history of populations

    Parameters
    ----------
    IND : list | np.array
        The list of all individuals in population P

    Returns
    -------
        The Euclidian distance individuals

    References
    ----------
        M. Crepinsek, et al., 2013, Exploration and exploitation in evolutionary algorithms: A survey, page 35:12, equation (6)
    '''
    pass

#   --------------- Make elements out of bounds into the bounds ---------------
def ensure_bounds_clip(vector, lower_bound, upper_bound):
    ''' Make the generated vector within the search domain by clipping
    elements which are out of the bounds '''
    new_vector = np.copy(vector)
    # if lower_bound:
    l_mask = np.where(vector <= np.array(lower_bound))
    new_vector[l_mask] = np.array(lower_bound)[l_mask]
    # if upper_bound:
    u_mask = np.where(vector >= np.array(upper_bound))
    new_vector[u_mask] = np.array(upper_bound)[u_mask]
    return new_vector

def ensure_bounds_mirror(vector, lower_bound, upper_bound):
    ''' Make the generated vector within the search domain '''
    new_vector = np.copy(vector)
    for i in range(len(vector)):
        if vector[i] <= lower_bound[i]:
            new_vector[i] = min(
                [upper_bound[i], 2*lower_bound[i]-vector[i]])
        if vector[i] >= upper_bound[i]:
            new_vector[i] = max(
                [lower_bound[i], 2*upper_bound[i]-vector[i]])
    return new_vector


#   ------------------------ Initial sampling schemes ------------------------
def latin_hypercube_sampling(N, D, lb, ub, rng, criterion="cls", iterations=5):
    '''
    Generate `N` samples in `D` dimensional space using Latain Hypercube
    Sampling (LHS).

    Parameters
    ----------
    N : int
        The number of individuals.
    D : int
        The Dimensionality of decision variables.
    lb : ndarray
        The lower bound of the initial sampling space.
    ub : ndarray
        The upper bound of the initial sampling space.
    rng : instance of np.random.RandomState
        A random number generator.
    criterion : str, optional [default="classic"]
        Choose a method to be used in Latain Hypercube Sampling (LHS), the
        allowable options are: "classic" (or "cls"), "center" (or "c"),
        "maximin" (or "m"), "centermaximin" (or "cm"), "correlation" (or "corr")
    iterations : int, optional [default=5]
        The number of iterations used in the "maximin" and "correlation"
        methods of Latain Hypercube Sampling (LHS).

    Returns
    -------
    LHS : 2D array of shape (N, D)
        A N-by-D array, where each row represents a sample.
    '''
    _ALL_LHS_criteria = {
        "classic": __lhsclassic, "cls": __lhsclassic,
        "center": __lhscenter, "c": __lhscenter,
        "maximin": __lhsmaxmin, "m": __lhsmaxmin,
        "centermaximin": __lhscentermaxmin, "cm": __lhscentermaxmin,
        "correlation": __lhscorrelate, "corr":__lhscorrelate}
    method = _ALL_LHS_criteria.get(criterion)  # get LHS method
    if method is None:
        raise NameError("The criterion for Latain Hypercube Sampling (LHS)"
            " cannot be '{}', it must be one of the following: 'classic'"
            " (or 'cls'), 'center' (or 'c'), 'maximin' (or 'm'),"
            " 'centermaximin' (or 'cm'), 'correlation' (or 'corr')"
            .format(criterion))
    else:
        LHS = method(N, D, rng, iterations)  # generate sample
    # adapt to feasible region or initial region
    LHS = (LHS * (ub - lb)[np.newaxis, :] + lb[np.newaxis, :])
    return LHS

def uniform_random_sampling(N, D, lb, ub, rng, *args, **kwargs):
    '''
    Generate `N` samples in `D` dimensional space using Uniform Random
    Sampling (URS).

    Parameters
    ----------
    N : int
        The number of individuals.
    D : int
        The Dimensionality of decision variables.
    lb : ndarray
        The lower bound of the initial sampling space.
    ub : ndarray
        The upper bound of the initial sampling space.
    rng : instance of np.random.RandomState
        A random number generator.

    Returns
    -------
    URS : 2D array of shape (N, D)
        A N-by-D array, where each row represents a sample.
    '''
    URS = rng.random_sample((N, D))
    URS = (URS * (ub - lb)[np.newaxis, :] + lb[np.newaxis, :])
    return URS

def __lhsclassic(num_samples, dimension, rng, iterations="unused"):
    ''' Random Latin Hypercube Sampling within in [0, 1]'''
    # Generate the intervals
    cut = np.linspace(0, 1, num_samples + 1)
    # Fill points uniformly in each interval
    u = rng.random_sample((num_samples, dimension))
    a = cut[:num_samples]
    b = cut[1:num_samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(dimension):
        rdpoints[:, j] = u[:, j]*(b-a) + a
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(dimension):
        order = rng.permutation(range(num_samples))
        H[:, j] = rdpoints[order, j]
    return H

def __lhscenter(num_samples, dimension, rng, iterations="unused"):
    ''' Center the points within the sampling intervals'''
    # Generate the intervals
    cut = np.linspace(0, 1, num_samples + 1)
    # Fill points uniformly in each interval
    u = rng.random_sample((num_samples, dimension))
    a = cut[:num_samples]
    b = cut[1:num_samples + 1]
    _center = (a + b)/2
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(dimension):
        H[:, j] = rng.permutation(_center)
    return H

def __lhsmaxmin(num_samples, dimension, rng, iterations):
    ''' LHS using maximin metric: maximize the minimal distance between
    all possible pairs of points'''
    maxdist = 0
    # Maximize the minimum distance between points
    for i in range(iterations):
        Hcandidate = __lhsclassic(num_samples, dimension, rng)
        d = __pdist(Hcandidate)
        if maxdist < np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    return H

def __lhscentermaxmin(num_samples, dimension, rng, iterations):
    ''' LHS using center maximin metric: maximize the minimal distance
    between all possible pairs of points'''
    maxdist = 0
    # Maximize the minimum distance between points
    for i in range(iterations):
        Hcandidate = __lhscenter(num_samples, dimension, rng)
        d = __pdist(Hcandidate)
        if maxdist < np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    return H

def __lhscorrelate(num_samples, dimension, rng, iterations):
    ''' LHS using metric of Minimizing the components correlation
    coefficients'''
    mincorr = np.inf
    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        Hcandidate = __lhsclassic(num_samples, dimension, rng)
        R = np.corrcoef(Hcandidate)
        if np.max(np.abs(R[R != 1])) < mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            # print("Info: Latain Hypercube Sampling (LHS) find new candidate"
            #         " solution with max, abs corrcoef = {}".format(mincorr))
            H = Hcandidate.copy()
    return H

def __pdist(x, p=2):
    '''
    Calculate the pair-wise point distances of a matrix

    Parameters
    ----------
    x : 2d-array
        An M-by-N array of scalars, where there are M points in N dimensions

    p : 2, 1, or np.Inf, optional
        p-norm distance measure. If p = 1, Manhattan distance is used. If
        p = 2, Euclidean distance is used.

    Returns
    -------
    d : ndarray
        A 1-by-b array of scalars, where b = M*(M - 1)/2. This array
        contains all the pair-wise point distances, arranged in the order
        (1, 0), (2, 0), ..., (M-1, 0), (2, 1), ..., (M-1, 1), ..., (M-1,
        M-2).
    '''
    x = np.atleast_2d(x)
    if x.ndim != 2:
        raise ValueError("In order to calculate the pair-wise point"
            " distances, the input array must be 2D dimensional, not {}D"
            .format(x.ndim))
    m, n = x.shape
    if m < 2:
        return []
    d = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            d.append(np.linalg.norm(x[j, :] - x[i, :], ord=p))
    return np.array(d)



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    case = 3

    # case 1: test `chech_random_state`
    if case == 1:
        seed = check_random_state(seed=1)
        print("[OK] The return is np.RandomState: {}".format(isinstance(seed,
            np.random.RandomState)))

    # case 2: test ensure_bounds
    if case == 2:
        D = 3
        lb = [-5]*D; ub = [5]*D
        x = np.random.rand(D) * 8
        print("The original x is out of the boundary: {}".format((np.any(x<lb)
            or np.any(x>ub))))
        clipped_x = ensure_bounds_clip(vector=x, lower_bound=lb, upper_bound=ub)
        clip_flag = np.any(clipped_x<lb) or np.any(clipped_x>ub)
        print("The clipped x is out of the boundary: {}".format(clip_flag))
        mirrored_x = ensure_bounds_mirror(vector=x, lower_bound=lb,
            upper_bound=ub)
        mirror_flag = np.any(mirrored_x<lb) or np.any(mirrored_x>ub)
        print("The mirrored x is out of the boundary: {}".format(mirror_flag))
        print("Origin x:\t{}".format(x))
        print("Clipped x:\t{}".format(clipped_x))
        print("Mirrored x:\t{}".format(mirrored_x))
        if not clip_flag and not mirror_flag:
            print("[OK] The clip & mirror ensure bounds methods.")

    # case 3: test sampling methods
    if case == 3:
        D = 3; N = 5
        lb = [-5]*D; ub = [5]*D
        lb = np.array(lb); ub = np.array(ub)
        CRITERIONS = ["classic",  "cls","center", "c", "maximin", "m",
         "centermaximin", "cm",  "correlation", "corr"]
        iters = 10
        random_state = check_random_state(seed=1)
        urs = uniform_random_sampling(N, D, lb, ub, seed=random_state)
        if urs.shape == (N, D):
            print("[OK] random sampling")
        for cr in CRITERIONS:
            lhs = latin_hypercube_sampling(N, D, lb, ub, seed=random_state,
                criterion=cr, iterations=iters)
            if lhs.shape == (N, D):
                print("[OK] Latin Hypercube samping with criterion `{}`."
                    .format(cr))

if __name__ == "__main__":
    main()
