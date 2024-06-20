import json
import numbers
import re
from ast import literal_eval

import numpy as np
from skopt.space import space as skspace
from skopt.space.space import check_dimension

__all__ = [
    "is_pareto_efficient",
    "all_pareto_fronts",
    "check_random_state",
    "get_random_state",
    "make_numeric",
    "parse_ranges",
    "is_score_prediction",
    "tensor_to_numpy",
]


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    costs = -costs
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def all_pareto_fronts(X):
    """Generate the pareto fronts for a set of point clouds.

    It is assumed that each attribute should be maximized.

    Parameters
    ----------
    X : array, float, shape (n_instances, n_points, n_attributes)
        Input array

    Returns
    -------
    Y : array, bool, shape (n_instances, n_points)
        Pareto front is represented by a boolean choice from the set of points
    """
    n_instances, n_points, n_attributes = X.shape
    result = np.empty((n_instances, n_points), dtype=bool)
    for i in range(n_instances):
        result[i] = is_pareto_efficient(X[i])
    return result


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is a SeedSequence, return a Generator.
        If seed is an int, construct a new SeedSequence and return a new Generator instance seeded with it.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        ss = np.random.SeedSequence(np.random.mtrand._rand.randint(0, 2 ** 32 - 1))
        return np.random.default_rng(ss)
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    if isinstance(seed, numbers.Integral):
        ss = np.random.SeedSequence(seed)
        return np.random.default_rng(ss)
    if isinstance(seed, np.random.RandomState):
        ss = np.random.SeedSequence(seed.randint(0, 2 ** 32 - 1))
        return np.random.default_rng(ss)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def get_random_state(seed_sequence):
    """Converts a seed sequence to a RandomState for backwards compatibility.

    This seed sequence is assumed to be the child of another seed sequence.
    If you expect a new stream of random numbers, call seed_sequence.spawn first."""
    return np.random.RandomState(np.random.MT19937(seed_sequence))


def make_numeric(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def parse_ranges(s):
    if isinstance(s, str):
        j = json.loads(s)
    else:
        j = s

    dimensions = []
    for i, s in enumerate(j.values()):
        # First check, if the string is a list/tuple or a function call:
        param_str = re.findall(r"(\w+)\(", s)
        if len(param_str) > 0:  # Function
            args, kwargs = [], dict()
            # TODO: this split does not always work (example Categorical(["a", "b", "c"]))
            prior_param_strings = re.findall(r"\((.*?)\)", s)[0].split(",")
            for arg_string in prior_param_strings:
                # Check if arg or kwarg:
                if "=" in arg_string:  # kwarg:
                    # trim all remaining whitespace:
                    arg_string = "".join(arg_string.split())

                    key, val = arg_string.split("=")
                    kwargs[key] = make_numeric(val)
                elif "[" in arg_string or "(" in arg_string:
                    args.append(literal_eval(arg_string))
                else:  # args:
                    val = make_numeric(arg_string)
                    args.append(val)
            if hasattr(skspace, param_str[0]):
                dim = getattr(skspace, param_str[0])(*args, **kwargs)
            else:
                raise ValueError("Dimension {} does not exist.".format(param_str))
            dimensions.append(dim)
        else:  # Tuple or list
            # We assume that the contents of the collection should be used as is and construct a python list/tuple
            # skopt.space.check_dimension will be used for validation
            parsed = literal_eval(s)
            if isinstance(parsed, (tuple, list)):
                dimensions.append(check_dimension(parsed))
            else:
                raise ValueError(
                    "Dimension {} is not valid. Make sure you pass a Dimension, tuple or list.".format(
                        param_str
                    )
                )

    return dict(zip(j.keys(), dimensions))


def is_score_prediction(arr):
    if arr.dtype == np.bool_:
        return False
    if len(np.unique(arr)) == 2:
        return False
    return True


def tensor_to_numpy(tensor, copy=True):
    if tensor.is_cuda:
        array = tensor.cpu().numpy()
    else:
        array = tensor.numpy()
    if copy:
        return np.copy(array)
    return array
