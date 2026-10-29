#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import boolforge as bf
import os
import random
from joblib import dump, load
from collections import defaultdict
import utils_function as utils

OTHER_OUTPUT_FOLDER = '../other/'

def get_function_metrics(functions, precision=7):
    """
    Compute canalization metrics (canalizing strength and input redundancy)
    for a collection of Boolean functions.

    Parameters
    ----------
    functions : list of bf.BooleanFunction
        Boolean functions for which metrics will be computed.
    precision : int, default=7
        Number of decimal places used when rounding metric values.

    Returns
    -------
    dict
        Dictionary mapping metric names to NumPy arrays. The returned
        metrics include:
            - 'canalizing strength'
            - 'input redundancy'
            - 'effective connectivity'
    """
    function_metric_labels = [
        'canalizing strength',
        'input redundancy',
        'effective connectivity',
    ]

    n_functions = len(functions)

    function_metrics = {metric: np.zeros(n_functions) 
        for metric in function_metric_labels}

    for i, f in enumerate(functions):
        if i % 1000 == 999:
            print(f"Generating function metrics. {i} of {n_functions} functions complete.")
        
        function_metrics['canalizing strength'][i] = round(f.get_canalizing_strength(), precision)
        function_metrics['input redundancy'][i] = round(f.get_input_redundancy(), precision)
        function_metrics['effective connectivity'][i] = round(f.get_effective_degree()/4, precision)

    return function_metrics


def get_network_metrics(networks, equiv_classes, nsim):
    """
    Compute dynamical metrics for a collection of Boolean networks.

    Networks are assumed to be ordered by equivalence class, with each
    equivalence class containing ``nsim`` simulated networks. The returned
    metrics are grouped by equivalence class, allowing statistics to be
    computed across network realizations with the update functions from the
    same class.

    Parameters
    ----------
    networks : list of bf.BooleanNetwork
        Boolean networks for which dynamical metrics are computed.
    equiv_classes : dict
        Dictionary mapping each equivalence class, represented by a
        tuple of canalization metrics ``(canalizing strength, input redundancy)``,
        to a representative bf.BooleanFunction. Networks are assumed
        to be ordered so that each group of nsim consecutive networks
        corresponds to one equivalence class.
    nsim : int
        Number of simulated networks in each equivalence class.

    Returns
    -------
    dict
        Nested dictionary of network metrics. The outer dictionary maps
        metric names to equivalence class defined by canalization metrics
        ``(canalizing strength, input redundancy)``, where each value is a
        NumPy array of length ``nsim``. The computed metrics are:
            - 'number of attractors'
            - 'average attractor length'
            - 'basin entropy'
            - 'coherence'
    """
    network_metric_labels = [
        'number of attractors',
        'average attractor length',
        'basin entropy',
        'coherence',
    ]

    n_networks = len(networks)

    network_metrics = {
        metric: {
            equiv_class: np.zeros(nsim)
            for equiv_class in equiv_classes
        }
        for metric in network_metric_labels
    }
    
    print(f"Generating network metrics. 0 of {n_networks} networks complete.")
    for j, bn in enumerate(networks):
        equiv_class = equiv_classes[j // 1000]
        network_idx = j % 1000

        if j % 1000 == 999:
            print(f"Generating network metrics. {j} of {n_networks} networks complete.")
        
        #Compute exact dynamics
        results = (bn.get_attractors_and_robustness_synchronous_exact())

        #Record several metrics
        lengths = [len(el) for el in results['Attractors']]
        network_metrics['number of attractors'][equiv_class][network_idx] = results['NumberOfAttractors']
        network_metrics['average attractor length'][equiv_class][network_idx] = np.mean(lengths)
        network_metrics['coherence'][equiv_class][network_idx] = results['Coherence']
        network_metrics['basin entropy'][equiv_class][network_idx] = bf.get_entropy_of_basin_size_distribution(results['BasinSizes'])

    return network_metrics


def get_equivalent_functions_from_examples(sampled_df, redundancy_metrics):
    """
    Generate equivalent Boolean functions for each equivalence class.

    For each equivalence class in the input DataFrame, the representative
    Boolean function in the ``example`` column is used to generate all
    equivalent functions. The generated functions are stored in a dictionary
    indexed by the equivalence class label defined by the tuple:
    ``(canalizing strength, input redundancy)``.

    Parameters
    ----------
    sampled_df : pandas.DataFrame
        DataFrame containing sampled equivalence classes. Required columns:
            - 'example': representative Boolean function for the class
            - 'canalizing strength': canalizing strength of the class
            - 'input redundancy': input redundancy of the class
            - 'effective connectivity': effective connectivity of the class

    redundancy_metrics : str
        Either 'input redundancy' or 'effective connectivity'

    Returns
    -------
    dict
        Dictionary mapping each equivalence class label
        ``(canalizing strength, input redundancy)`` to a list of equivalent
        Boolean functions.
    """

    dict_functions = {}
    n_functions = len(sampled_df)

    for row in range(n_functions):
        functions = utils.get_equivalent_functions(sampled_df.iloc[row, sampled_df.columns.get_loc('example')])

        # Define equivalence class label using canalization metrics.
        key = (
            np.float64(sampled_df.iloc[row, sampled_df.columns.get_loc('canalizing strength')]),
            np.float64(sampled_df.iloc[row, sampled_df.columns.get_loc(redundancy_metrics)])
        )

        dict_functions[key] = functions

        if (row + 1) % 10 == 0:
            print(f"Generating functions from {row + 1}/{n_functions} equivalence classes")

    return dict_functions


def simulate_network_dynamics(dict_functions, N, nsim, seed=2026):
    """
    Generate and simulate Boolean networks from groups of equivalent Boolean
    functions.

    For each equivalence class of Boolean functions, ``nsim`` Boolean
    networks are generated by randomly sampling update functions from the
    corresponding function set and assigning a randomly generated wiring
    diagram. The same wiring diagrams are reproduced across repeated calls
    when the same seed is provided.

    Parameters
    ----------
    dict_functions : dict
        Dictionary mapping equivalence class labels to lists of allowable
        Boolean functions. Each key represents an equivalence class defined
        by canalization metrics ``(canalizing strength, input redundancy)``,
        and all functions across classes must have the same number of inputs.
    N : int
        Number of nodes in each Boolean network.
    nsim : int
        Number of network realizations simulated for each equivalence class.
    seed : int, default=2026
        Random seed used for reproducible network generation.

    Returns
    -------
    dict
        Nested dictionary containing network dynamical metrics computed from
        the simulated Boolean networks using the function get_network_metrics.
        Metrics are grouped by equivalence class and include:
            - number of attractors
            - average attractor length
            - basin entropy
            - coherence
    """
    random.seed(seed)

    allowable_functions = []
    for key in dict_functions.keys():
        allowable_functions += dict_functions[key]

    assert np.all([allowable_functions[0].n == f.n for f in allowable_functions]), 'all functions must have the same number of inputs'
    n = allowable_functions[0].n

    bns = []
    for key in dict_functions.keys():
        for j in range(nsim):
            # Using BoolForge's rng, create the same wiring diagram for different calls of
            # simulate_network_dynamics, as long as the same seed is passed 
            WiringDiagram = bf.random_wiring_diagram(N, n, rng = seed+j) 
            # Randomly assign update functions from the current equivalence class,
            # sampling with replacement.
            functions = random.choices(dict_functions[key], k=N)
            bns.append( bf.BooleanNetwork(F=functions, I=WiringDiagram) )

    equiv_classes = list(dict_functions.keys())

    return get_network_metrics(networks=bns, equiv_classes=equiv_classes, nsim=nsim)


def get_indices_of_array_equidistant_by_value(arr, n):
    """
    Select values from an array that are approximately equally spaced by value.

    The function generates ``n`` target values evenly spaced between the
    first and last elements of the input array, then selects the closest
    values in the array to those targets. This allows selecting elements
    that are approximately equidistant in value rather than by index.

    Parameters
    ----------
    arr : numpy.ndarray
        Sorted array of values from which elements are selected.
    n : int
        Number of values to select.

    Returns
    -------
    numpy.ndarray
        Array containing ``n`` selected values that are approximately
        equally spaced across the range of ``arr``.
    """
    targets = np.linspace(arr[0], arr[-1], n)

    idx = np.searchsorted(arr, targets)
    idx = np.clip(idx, 1, len(arr) - 1)

    left = arr[idx - 1]
    right = arr[idx]

    indices = np.where(targets - left <= right - targets, idx - 1, idx)
    selected = arr[indices]
    return selected


def simulate_random_network_dynamics(
        output_folder = OTHER_OUTPUT_FOLDER,
        nsim=1000,
        N=12,
        ks=np.arange(1, 5),
        absolute_biases=np.linspace(0, 0.9, 10)
    ):
    filename = os.path.join(output_folder, f'random_networks_dynamics.joblib')
    
    if not os.path.isfile(filename):
        metrics = [
            'Derrida value',
            'number of attractors',
            'average attractor length',
            'basin entropy',
            'coherence'
        ]

        n_ks = len(ks)
        n_abs_biases = len(absolute_biases)

        res = {metric: np.zeros((n_ks, n_abs_biases, nsim))
            for metric in metrics}

        for i, k in enumerate(ks):
            for j, a in enumerate(absolute_biases):
                for ii in range(nsim):

                    bn = bf.random_network(
                        N,
                        k,
                        absolute_bias=a,
                        use_absolute_bias=True,
                        allow_degenerate_functions=True
                    )

                    info = bn.get_attractors_and_robustness_synchronous_exact()

                    res['Derrida value'][i, j, ii] = (bn.get_derrida_value(exact=True))
                    res['number of attractors'][i, j, ii] = (info['NumberOfAttractors'])
                    lengths = [len(el) for el in info['Attractors']]
                    res['average attractor length'][i, j, ii] = np.mean(lengths)
                    res['basin entropy'][i, j, ii] = (bf.get_entropy_of_basin_size_distribution(info['BasinSizes']))
                    res['coherence'][i, j, ii] = info['Coherence']

                print(f"Finished {nsim} simulations: "
                      f"k={k}, absolute bias={a:.2f} "
                      f"({i * len(absolute_biases) + j + 1}/"
                      f"{len(ks) * len(absolute_biases)})")

        results = {'random_networks': res}
        dump(results, filename)

        return res

    else:
        results = load(filename)
        random_networks = results['random_networks']
        
        return random_networks


def get_network_level_results(
        n, 
        redundancy_metric = 'input redundancy',
        output_folder = OTHER_OUTPUT_FOLDER, 
        allow_degeneracy = False,
        min_group_size = 5, 
        max_n_groups = 9,
        max_vals_per_group = 15,
        random_state = 10 
    ):
    """
    Generate or load Boolean network dynamical results grouped by function
    equivalence classes (canalizing strength, input redundancy).

    For small input sizes (n <= 4), all Boolean functions are generated and
    grouped by their canalization metrics. For larger input sizes, a subset
    of functions is sampled from (n-1)-input Boolean function. The selected
    Boolean functions are used to generate and simulate Boolean networks,
    and dynamical metrics are computed for each equivalence class.

    Results are saved as a Joblib file. If a saved result file already exists,
    the results are loaded directly.

    Parameters
    ----------
    n : int
        Number of inputs of the Boolean functions.
    redundancy_metric : str
        Either 'input redundancy' or 'effective connectivity'
    output_folder : str
        Directory where computed results are saved and loaded.
    allow_degeneracy : bool, default=False
        Whether degenerate Boolean functions are included.
    min_group_size : int, default=5
        Minimum number of equivalence classes required when grouping by
        either canalizing strength or input redundancy.
    max_n_groups : int, default=9
        Maximum number of equivalence classes sampled for large function
        spaces.
    max_vals_per_group : int, default=15
        Maximum number of equivalence classes sampled within each group
        when grouped by either canalizing strength or input redundancy.
    random_state : int, default=10
        Random seed used for reproducible sampling.

    Returns
    -------
    network_metrics : dict
        Nested dictionary containing Boolean network dynamical metrics
        grouped by equivalence class. Metrics include:
            - number of attractors
            - avereage attractor length
            - basin entropy
            - coherence

    dict_x_groups : list of dict
        Dictionaries used for grouping equivalence classes by canalizing
        strength and input redundancy. Each dictionary maps a metric value
        to the corresponding equivalence class tuples.
    """
    if n == 3:
        print("Setting the minimum number of equivalence classes required when grouping to 2 due to small number of functions.")
        min_group_size = 2
        filename = os.path.join(output_folder, f"dynamics_n{n}_strength_{redundancy_metric.replace(' ', '_')}_ad{allow_degeneracy}_mgs{min_group_size}.joblib")

    elif n == 4:
        filename = os.path.join(output_folder, f"dynamics_n{n}_strength_{redundancy_metric.replace(' ', '_')}_ad{allow_degeneracy}_mgs{min_group_size}.joblib")

    else:
        filename = os.path.join(output_folder, f"dynamics_n{n}_strength_{redundancy_metric.replace(' ', '_')}_ad{allow_degeneracy}_mgs{min_group_size}_mng{max_n_groups}_mvpg{max_vals_per_group}.joblib")

    if not os.path.isfile(filename):
        if n <= 4:
            functions = get_small_input_boolean_functions(n, allow_degeneracy)
            function_metrics = get_function_metrics(functions)
            strength = function_metrics['canalizing strength']
            redundancy = function_metrics[redundancy_metric]

            dict_functions = defaultdict(list)
            for f, s, r in zip(functions, strength, redundancy):
                dict_functions[(s, r)].append(f)

            dict_functions = dict(sorted(dict_functions.items()))
            sampled_df = pd.DataFrame(list(dict_functions.keys()), columns=['canalizing strength', redundancy_metric])

        else:
            df = utils.get_metrics_for_some_functions(n, output_folder)
            if not allow_degeneracy:
                df = df[df['mean degenerate'] == 0].copy()

            df = df[['canalizing strength', redundancy_metric, 'example']]
            valid_vals = init_valid_counts_metrics(df, redundancy_metric, min_group_size)
            sampled_df = sampling_functions(df, redundancy_metric, valid_vals, max_n_groups, max_vals_per_group, random_state)
            dict_functions = get_equivalent_functions_from_examples(sampled_df, redundancy_metric)

        network_metrics = simulate_network_dynamics(dict_functions, N=12, nsim=1000)
        dict_strength_groups = group_by_metric(sampled_df, 'canalizing strength', redundancy_metric, min_group_size)
        dict_redundancy_groups = group_by_metric(sampled_df, redundancy_metric, 'canalizing strength', min_group_size)
        dict_x_groups = [dict_strength_groups, dict_redundancy_groups]

        results = {'network_metrics': network_metrics, 'dict_x_groups': dict_x_groups}
        dump(results, filename)
    
    else:
        results = load(filename)
        network_metrics = results['network_metrics']
        dict_x_groups = results['dict_x_groups']
        
    return network_metrics, dict_x_groups
    

def get_small_input_boolean_functions(n, allow_degeneracy=False):
    """
    Generate all Boolean functions with a small number of inputs (n <= 4).

    The function generates Boolean functions from all possible truth tables
    with ``n`` inputs. Degenerate functions can optionally be excluded based
    on the value of ``allow_degeneracy``.

    Parameters
    ----------
    n : int
        Number of inputs of the Boolean functions. Must be <= 4 because the
        number of possible Boolean functions grows exponentially.
    allow_degeneracy : bool, default=False
        Whether to include degenerate Boolean functions. If False,
        functions that do not depend on all inputs are excluded.

    Returns
    -------
    list of bf.BooleanFunction
        List of generated Boolean functions satisfying the degeneracy
        criterion.
    """
    assert n <= 4, "Input size too large. Exhaustive generation is only supported for n <= 4."

    str_functions = bf.get_left_side_of_truth_table(2**n)
    functions = []
    
    for function in str_functions:
        f = bf.BooleanFunction(function)

        if allow_degeneracy or not f.is_degenerate():
            functions.append(f)
    
    return functions


def init_valid_counts_metrics(df, redundancy_metric, min_group_size):
    """
    Identify valid canalization metric values based on minimum group size.

    The function counts the number of equivalence classes associated with
    each canalizing strength and input redundancy value, then returns the
    metric values that have at least ``min_group_size`` corresponding
    entries. These valid values can be used for downstream sampling or
    grouping of equivalence classes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing equivalence class information. Required columns:
            - 'canalizing strength': canalizing strength values
            - 'input redundancy': input redundancy values
            - 'effective connectivity': effective connectivity values
    redundancy_metric : str
        Either 'input redundancy' or 'effective connectivity'
    min_group_size : int
        Minimum number of equivalence classes required for a metric value
        to be considered valid.

    Returns
    -------
    dict
        Dictionary containing valid metric values:
            - 'canalizing strength': values of canalizing strength with sufficient
              group size
            - 'input redundancy': values of input redundancy with sufficient
              group size
    """
    strength_counts = (df.groupby('canalizing strength').size())
    redundancy_counts = (df.groupby(redundancy_metric).size())
    valid_strengths = strength_counts[strength_counts >= min_group_size].index
    valid_redundancies = redundancy_counts[redundancy_counts >= min_group_size].index

    return {'canalizing strength': valid_strengths, redundancy_metric: valid_redundancies}


def sampling_functions(df, redundancy_metrics, valid_vals, max_n_groups, max_vals_per_group, random_state):
    """
    Sample Boolean function equivalence classes based on canalization metrics.

    The function selects approximately equidistant metric values for both
    canalizing strength and input redundancy. For each selected value,
    equivalence classes are randomly sampled up to a specified maximum
    number of classes. Duplicate Boolean function examples are removed from
    the final sampled DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Boolean function equivalence classes. Required
        columns:
            - 'canalizing strength': canalizing strength values
            - 'input redundancy': input redundancy values
            - 'effective connectivity': effective connectivity values
            - 'example': representative Boolean functions

    redundancy_metrics : str
        Either 'input redundancy' or 'effective connectivity'

    valid_vals : dict
        Dictionary containing valid metric values that satisfy the minimum
        group size requirement. Required keys:
            - 'canalizing strength'
            - 'input redundancy'

    max_n_groups : int
        Number of metric values sampled for each grouping metric.

    max_vals_per_group : int
        Maximum number of equivalence classes sampled for each selected
        metric value.

    random_state : int
        Random seed used for reproducible sampling.

    Returns
    -------
    pandas.DataFrame
        Sampled equivalence classes containing unique Boolean function
        examples.
    """
    sampled_dfs = []
    selected_strengths = get_indices_of_array_equidistant_by_value(valid_vals['canalizing strength'], max_n_groups)
    selected_redundancies = get_indices_of_array_equidistant_by_value(valid_vals[redundancy_metrics], max_n_groups)
    
    # Sample equivalence classes from both strength and redundancy groups.
    for index in range(max_n_groups):
        group = df[df['canalizing strength'] == selected_strengths[index]]
        sampled_dfs.append(
            group.sample(n=min(max_vals_per_group, len(group)), random_state=random_state)
        )

        group = df[df[redundancy_metrics] == selected_redundancies[index]]
        sampled_dfs.append(
            group.sample(n=min(max_vals_per_group, len(group)), random_state=random_state)
        )

        sampled_df = (pd.concat(sampled_dfs).drop_duplicates(subset='example'))

    return sampled_df


def group_by_metric(df, group_by_metric, varied_metric, min_group_size):
    """
    Group by one metric while requiring a minimum number of unique values
    in the varied metric.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'canalizing strength', 'input redundancy', and 'effective connectivity' columns.
    group_by_metric : str
        Column to group by ('canalizing strength', 'input redundancy' or 'effective connectivity').
    varied_metric : str
        Column whose unique values are enumerated within each group.
    min_group_size : int
        Minimum number of unique values required in `varied_metric`.

    Returns
    -------
    dict
        {
            group_value: [
                (canalizing_strength_value, varied_metric_value),
                ...
            ]
        }
        Tuples are always ordered (canalizing strength value, other metric value),
        regardless of which argument is `group_by_metric`.
    """
    result = {}

    for value, group in df.groupby(group_by_metric):
        varied_values = sorted(group[varied_metric].unique())

        if len(varied_values) >= min_group_size:
            if group_by_metric == 'canalizing strength':
                result[value] = [
                    (np.float64(value), np.float64(varied))
                    for varied in varied_values
                ]
            elif varied_metric == 'canalizing strength':
                result[value] = [
                    (np.float64(varied), np.float64(value))
                    for varied in varied_values
                ]
            else:
                # Neither metric is 'canalizing strength'; preserve (group, varied) order.
                result[value] = [
                    (np.float64(value), np.float64(varied))
                    for varied in varied_values
                ]

    return result


