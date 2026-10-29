#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import boolforge as bf
from itertools import permutations, product
import math
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
#from sklearn.metrics import roc_auc_score
import utils_network as utils

OTHER_OUTPUT_FOLDER = '../other/'
TABLE_FOLDER = '../tables/'


def get_function_level_results(n, precision=7, ):
    """
    Compute function-level metrics for all or a sample of Boolean functions.

    Parameters
    ----------
    n : int
        Number of inputs to the Boolean functions.
    precision : int, optional
        Decimal precision used for metric calculations.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing function-level metrics.
    """
    if n <= 4:
        df = get_metrics_for_all_functions(n, precision=precision)

    else:
        df = get_metrics_for_some_functions(n, precision=precision)

    return df


def run_canalization_classifiers(df, n, n_splits = 5, n_repeats=100, seed = 50011, table_folder=TABLE_FOLDER):
    """
    Run classification models for pairs of canalization-related features
    using repeated stratified cross-validation.

    Models:
        - Logistic regression
        - Decision tree
        - SVC
        - KNN (3 and 5 neighbors)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the predictor features and target.

    n : int
        Number of inputs, used in the output filename.
        
    n_splits: int
        Number of splits in cross-validation.

    n_repeats: int
        Number of independent repeats of cross-validation.
    
    seed: int
        Seed for random number generator for reproducible cross-validation results

    table_folder : str
        Directory where results CSV will be saved.

    Returns
    -------
    classifiers_auc_df : pandas.DataFrame
        Cross-validation results for all feature pairs and models.
    """
    filename = os.path.join(
        table_folder,
        f"avg_sens_canalizing_classifiers_n{n}_{n_splits}fold_cv_{n_repeats}repeats_seed{seed}.csv"
    )

    if not os.path.isfile(filename):
        x1 = df['canalizing strength']
        x2 = df['input redundancy']
        x0 = df['mean average sensitivity']
        y_target = df['mean canalizing'].astype(int)

        df_all = pd.DataFrame({
            'canalizing strength': x1,
            'input redundancy': x2,
            'average sensitivity': x0,
        })

        feature_pairs = [
            ('canalizing strength', 'average sensitivity'),
            ('input redundancy', 'average sensitivity')
        ]

        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=50011
        )

        results = []

        for idx, (feat_x, feat_y) in enumerate(feature_pairs):
            X = df_all[[feat_x, feat_y]]

            models = [
                (
                    "logistic regression",
                    None,
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(class_weight="balanced")
                    )
                ),
                (
                    "decision tree",
                    None,
                    DecisionTreeClassifier(class_weight="balanced")
                ),
                (
                    "svc",
                    None,
                    make_pipeline(
                        StandardScaler(),
                        SVC(class_weight="balanced")
                    )
                )
            ]

            for n_neighbors in [3,5]:
                models.append((
                    "knn",
                    n_neighbors,
                    make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier(n_neighbors=n_neighbors)
                    )
                ))

            for model_name, param, model in models:
                cv_auc = cross_val_score(
                    model,
                    X,
                    y_target,
                    cv=cv,
                    scoring="roc_auc"
                )

                results.append({
                    "pair_id": idx,
                    "model": model_name,
                    "feat_x": feat_x,
                    "feat_y": feat_y,
                    "param": param,
                    "auc_cv_mean": cv_auc.mean(),
                    "auc_cv_std": cv_auc.std()
                })

        classifiers_auc_df = pd.DataFrame(results)
        classifiers_auc_df.to_csv(filename, index=False)

    else:
        classifiers_auc_df = pd.read_csv(filename)

    return classifiers_auc_df


def get_metrics_for_all_functions(n, precision = 7):
    """
    Compute metrics for all Boolean functions, using cached results if available.

    Parameters
    ----------
    n : int
        Number of inputs to the Boolean functions.
    precision : int, optional
        Decimal precision used for metric calculations.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing function metrics and equivalence classes.
    """
    filename = os.path.join(OTHER_OUTPUT_FOLDER, f'unique_strengths_and_input_redundancies_n{n}.csv')

    if os.path.isfile(filename):

        return pd.read_csv(filename)
    
    else:
        all_functions = [bf.BooleanFunction(f) for f in bf.get_left_side_of_truth_table(2**n)]

        return _compute_metrics_and_equivalence_classes(functions=all_functions, 
                                                        filename=filename, 
                                                        precision=precision)


def get_metrics_for_some_functions(n, precision = 7):
    """
    Compute metrics for a sampled subset of Boolean functions, using cached results if available.

    Parameters
    ----------
    n : int
        Number of inputs to the Boolean functions.
    precision : int, optional
        Decimal precision used for metric calculations.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing metrics and equivalence classes for the sampled functions.
    """
    filename = os.path.join(OTHER_OUTPUT_FOLDER, f'sampling_unique_strengths_and_input_redundancies_n{n}.csv')

    if os.path.isfile(filename):

        return pd.read_csv(filename)
    
    else:
        some_canalizing_and_degenerate_functions= get_some_canalizing_and_degenerate_functions(n, precision=precision)
        some_functions = get_some_non_canalizing_and_non_degenerate_functions(n, precision=precision)
        all_sampled_functions = some_canalizing_and_degenerate_functions | some_functions
        
        boolean_all_sampled_functions = [bf.BooleanFunction(
            np.array(tuple(int(bit) for bit in f"{f:0{2**n}b}"))
            ) 
            for f in all_sampled_functions]

        return _compute_metrics_and_equivalence_classes(
            functions=boolean_all_sampled_functions, 
            filename=filename,
            precision=precision,
        )
    

def _compute_metrics_and_equivalence_classes(functions, filename = None, precision = 7):
    """
    Compute function metrics, metric equivalence classes, and network dynamics.

    Parameters
    ----------
    functions : iterable
        Boolean functions to analyze.
    filename : str, optional
        Path to save the resulting DataFrame as a CSV file.
    precision : int, optional
        Decimal precision used to define metric equivalence classes.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing equivalence class metrics and dynamics.
    """
    metrics = get_function_metrics(
        functions=functions
    )
    (
        unique_metric_values, 
        functions_per_class, 
        class_sizes, 
    ) = get_metric_equivalence_classes(
        metrics=metrics, 
        functions=functions, 
        precision = precision
    )
    df = compute_equivalence_class_dynamics(
        unique_metric_values=unique_metric_values, 
        functions_per_class=functions_per_class, 
        class_sizes=class_sizes
    )

    if filename is not None:
        df.to_csv(filename)

    return df


def get_some_canalizing_and_degenerate_functions(n, precision = 7):
    """
    Generate a subset of n-input canalizing and/or degenerate Boolean functions
    from representative (n-1)-input Boolean functions.

    For each example (n-1)-input function in the input file, three n-input
    functions are constructed by:

    1. Concatenating a (n-1)-input function with a (n-1)-input function
    fixed to 0 to create a new n-input canalizing function.
    2. Concatenating a (n-1)-input function with a (n-1)-input function
    fixed to 1 to create a new n-input canalizing function.
    3. Concatenating a (n-1)-input function with itself to create a new degenerate
    n-input functions.

    Duplicate Boolean functions are removed before returning.

    Parameters
    ----------
    n : int
        Number of inputs of the generated Boolean functions.
    filename_for_n_minus_1 : str
        CSV file containing representative (n-1)-input Boolean functions in
        the 'example' column.

    Returns
    -------
    list of bf.BooleanFunction
        A list containing unique canalizing and degenerate n-input Boolean
        functions generated from the (n-1)-input examples.
    """
    df = get_function_level_results(n-1, precision=precision)
    functions = set()
    n_states = 2**(n-1)
    can1_first_variable = (2**n_states - 1) * (1 << n_states)
    all_lower_dim_functions = [int(''.join(map(str, np.fromstring(x.strip('[]'), sep=' ', dtype=int))), 2) 
                               for x in df['example']]

    for f_int in all_lower_dim_functions:
        f_can0 = f_int
        f_can1 = f_int + can1_first_variable
        f_deg = f_int + (f_int << n_states)

        functions.add(f_can0)
        functions.add(f_can1)
        functions.add(f_deg)

    return functions


def get_some_non_canalizing_and_non_degenerate_functions(n, precision = 7):
    """
    Generate a subset of n-input Boolean functions that are neither
    canalizing nor degenerate by combining pairs of representative 
    (n-1)-input Boolean functions.

    Let f and g be two (n-1)-input Boolean functions. For each unordered
    pair (including a function paired with itself), two n-input functions
    are constructed by (f+g) and (f+¬g).

    Parameters
    ----------
    n : int
        Number of inputs of the generated Boolean functions.
    filename_for_n_minus_1 : str
        CSV file containing representative (n-1)-input Boolean functions in
        the 'example' column.

    Returns
    -------
    list of bf.BooleanFunction
        A list of generated n-input Boolean functions.
    """
    df = get_function_level_results(n-1, precision=precision)
    n_states = 2**(n-1)
    invert_mask = (1 << n_states) - 1
    all_lower_dim_functions = [int(''.join(map(str, np.fromstring(x.strip('[]'), sep=' ', dtype=int))), 2)
                               for x in df['example']]
    some_functions = set()

    for i, f in enumerate(all_lower_dim_functions):
        for j, g in enumerate(all_lower_dim_functions):
            if j < i:
                continue

            some_functions.add(f | (g << n_states))
            some_functions.add(f | (((~g) & invert_mask) << n_states))

    return some_functions


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
    """
    function_metric_labels = [
        'canalizing strength',
        'input redundancy',
    ]

    n_functions = len(functions)

    function_metrics = {metric: np.zeros(n_functions) 
        for metric in function_metric_labels}

    for i, f in enumerate(functions):
        if i % 1000 == 999:
            print(f"Generating function metrics. {i} of {n_functions} functions complete.")
        
        function_metrics['canalizing strength'][i] = round(f.get_canalizing_strength(), precision)
        function_metrics['input redundancy'][i] = round(f.get_input_redundancy(), precision)

    return function_metrics


def get_metric_equivalence_classes(metrics, functions, precision=7):
    """
    Group Boolean functions into equivalence classes based on their
    canalization metrics (canalizing strength and input redundancy).

    Each Boolean function is assigned to an equivalence class according to
    the tuple of rounded metric values. Functions with identical metric
    tuples belong to the same equivalence class. The equivalence classes
    are returned in sorted order of their metric tuples.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric names to arrays of metric values. Each
        array must have the same length as functions, with the i-th element
        corresponding to the i-th bf.BooleanFunction.
    functions : list of bf.BooleanFunction
        Boolean functions to be grouped into equivalence classes.
    precision : int, default=7
        Number of decimal places used when rounding metric values before
        grouping.

    Returns
    -------
    unique_metric_values : list of tuple
        Sorted list of unique metric tuples defining the equivalence
        classes.
    functions_per_class : list of list of bf.BooleanFunction
        Representative grouping of Boolean functions, where the i-th list
        contains all functions belonging to the i-th equivalence class.
    class_sizes : list of int
        Number of Boolean functions in each equivalence class.
    """
    rounded = {k: np.round(np.asarray(vals), precision) for k, vals in metrics.items()}
    keys = list(rounded.keys())
    n = len(rounded[keys[0]])

    key_to_id = {}
    example_paired = []
    count_paired = []
    indices = []

    for i in range(n):
        key = tuple(rounded[k][i] for k in keys)

        if key not in key_to_id:
            class_id = len(key_to_id)
            key_to_id[key] = class_id
            example_paired.append([functions[i]])
            count_paired.append(1)
            indices.append([i])

        else:
            class_id = key_to_id[key]
            example_paired[class_id].append(functions[i])
            count_paired[class_id] += 1
            indices[class_id].append(i)

    indices = [np.array(idx) for idx in indices]
    unique_metric_values = sorted(key_to_id.keys())
    old_to_new = {k: i for i, k in enumerate(unique_metric_values)}
    functions_per_class = [[] for _ in range(len(unique_metric_values))]
    class_sizes = [0] * len(unique_metric_values)
    indices_sorted = [[] for _ in range(len(unique_metric_values))]

    for key, old_id in key_to_id.items():
        new_id = old_to_new[key]
        functions_per_class[new_id] = example_paired[old_id]
        class_sizes[new_id] = count_paired[old_id]
        indices_sorted[new_id] = indices[old_id]

    return unique_metric_values, functions_per_class, class_sizes


def compute_equivalence_class_dynamics(unique_metric_values, functions_per_class, class_sizes, precision=7):
    """
    Compute summary statistics for each equivalence class of Boolean
    functions.

    For each equivalence class, the mean and standard deviation of the dynamical and
    structural properties are computed across all Boolean functions in the class.
    The resulting summary statistics, together with the canalization metrics defining
    the equivalence class (canalizing strength and input redundancy) and the class size,
    are returned as a pandas DataFrame.

    Parameters
    ----------
    unique_metric_values : list of tuple
        Sorted list of equivalence class labels, where each label is a
        tuple of canalization metrics (canalizing strength, input
        redundancy).
    functions_per_class : list of list of bf.BooleanFunction
        Boolean functions grouped by equivalence class. The first function
        in each group is saved as the representative.
    class_sizes : list of int
        Number of Boolean functions in each equivalence class.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per equivalence class with the
        following columns:
            - canalizing strength
            - input redundancy
            - effective connectivity
            - count
            - example
            - mean/std excess canalization
            - mean/std canalizing depth
            - mean/std absolute bias
            - mean/std average sensitivity
            - mean/std degenerate
            - mean/std canalizing
    """
    res = []
    n_unique = len(unique_metric_values)
    effective_connectivity = []

    for i in range(n_unique):
        if i % 1000 == 0:
            print(f"Computing functional dynamics of {i}/{n_unique} equivalence classes")

        all_bfs = [functions_per_class[i][0]]
        effective_connectivity.append(round(all_bfs[0].get_effective_degree()/4, precision))
        excess_canalization = [round(sum(bfs.get_edge_effectiveness()) - bfs.get_average_sensitivity(exact=True, normalized=False), precision) for bfs in all_bfs]
        all_canalizing_depths = [bf.get_canalizing_depth() for bf in all_bfs]
        all_abs_bias = [bf.get_absolute_bias() for bf in all_bfs]
        all_average_sensitivities = [bf.get_average_sensitivity(exact=True) for bf in all_bfs]
        degenerate = [f.is_degenerate() for f in all_bfs]
        canalizing = [f.is_canalizing() for f in all_bfs]

        res.append([])

        for metric in [
            excess_canalization,
            all_canalizing_depths,
            all_abs_bias,
            all_average_sensitivities,
            degenerate,
            canalizing
        ]:
            res[-1].append(np.mean(metric))
            res[-1].append(np.std(metric))

    res = np.asarray(res)
    example = [functions[0] for functions in functions_per_class]
    canalizing_strengths = [values[0] for values in unique_metric_values]
    input_redundancies = [values[1] for values in unique_metric_values]
    df = pd.DataFrame({'canalizing strength': canalizing_strengths,
                       'input redundancy': input_redundancies,
                       'effective connectivity': effective_connectivity,
                       'count': class_sizes,
                       'example': example,})

    # Add metric columns
    metric_names = [
        'excess canalization',
        'canalizing depth',
        'absolute bias',
        'average sensitivity',
        'degenerate',
        'canalizing',
    ]

    for i, name in enumerate(metric_names):
        df[f'mean {name}'] = res[:, 2 * i]
        df[f'std {name}'] = res[:, 2 * i + 1]

    return df


def get_metrics_per_variable(n, allow_degeneracy = False, keeping_unique_values = True):
    """Compute canalization and input-redundancy metrics for individual variables.

    For each Boolean function with ``n`` inputs, computes variable-level
    measures of edge effectiveness, input redundancy, and canalizing strength,
    together with information about degeneracy and canalizing
    layer structure. For each variable, the function also records its position
    within the canalizing hierarchy, the number of more-dominant canalizing
    variables, and the size of its canalizing layer.

    For ``n <= 4``, metrics are computed exhaustively over the Boolean function
    space. For ``n > 4``, metrics are computed for a sample of
    the larger function space. By default, degenerate functions are excluded.

    Parameters
    ----------
    n : int
        Number of input variables of the Boolean functions.
    allow_degeneracy : bool, default=False
        If ``False``, exclude Boolean functions containing non-essential
        variables. If ``True``, include both degenerate and non-degenerate
        functions.
    keeping_unique_values : bool, default=True
        If ``True``, remove duplicate rows based on variable-level input
        redundancy, canalizing strength, canalizing layer position, and
        canalizing layer size. If ``False``, retain all variable-level results.

    Returns
    -------
    pandas.DataFrame
        A data frame with one row per function-variable pair (before removal
        of duplicates). Columns include:

        - ``function`` : Boolean function.
        - ``variable`` : Index of the input variable.
        - ``edge effectiveness`` : Edge effectiveness of the variable.
        - ``input redundancy`` : Input redundancy of the variable.
        - ``canalizing strength`` : Canalizing strength of the variable.
        - ``number of essential variables`` : Number of essential inputs in the
          function.
        - ``degenerate`` : Whether the variable is non-essential.
        - ``canalizing layer position`` : Position of the variable's canalizing
          layer, with ``0`` for non-canalizing variables.
        - ``number of  dominant variables `` : Number of canalizing variables
          in more dominant layers, with ``-1`` for non-canalizing variables.
        - ``canalizing layer size`` : Number of variables in the same
          canalizing layer, with ``-1`` for non-canalizing variables.
    """
    if n < 5:
        filename = os.path.join(OTHER_OUTPUT_FOLDER, f'strengths_and_input_redundancies_n{n}_per_var_degenerate{bool(allow_degeneracy)}_unique{bool(keeping_unique_values)}.csv')
    else:
        filename = os.path.join(OTHER_OUTPUT_FOLDER, f'sampling_strengths_and_input_redundancies_n{n}_per_var_degenerate{bool(allow_degeneracy)}_unique{bool(keeping_unique_values)}.csv')
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)

    if n <= 4:
        functions = utils.get_small_input_boolean_functions(n, allow_degeneracy)
    else:
        functions_df = get_function_level_results(n)
        if not allow_degeneracy:
            functions_df = functions_df[functions_df['mean degenerate'] == 0]
        functions = [
            bf.BooleanFunction(np.array(f.strip('[]').split(), dtype=int))
            for f in functions_df['example']
    ]

    n_functions = len(functions)

    edge_effectiveness_per_var = []
    input_redundancy_per_var = []
    can_strength_per_var = []
    essential_vars = []
    degenerate_vars = np.zeros((len(functions), n))
    can_layer_pos = np.zeros((len(functions), n))
    n_dominant_vars = -np.ones((len(functions), n)) # non-canalizing variables at -1
    can_layer_size = -np.ones((len(functions), n))  # non-canalizing variables at -1

    for i, f in enumerate(functions):
        if i % 1000 == 999:
            print(f"Generating variable metrics for each funtion. {i} of {n_functions} functions complete.")

        ee = np.array(f.get_edge_effectiveness())
        edge_effectiveness_per_var.append(ee)
        input_redundancy_per_var.append(1 - ee)
        can_strength_per_var.append(np.array(f.get_canalizing_strength_of_variables()))
        essential_vars.append(np.repeat(f.get_number_of_essential_variables(), n))
        degenerate_vars[i,:] = f.get_type_of_inputs()=='non-essential'
        layers = f.get_layer_structure()
        n_more_dominant_variables = np.cumsum(np.append(0,layers['LayerStructure']))

        count=0
        for j in range(layers['NumberOfLayers']):
            for _ in range(layers['LayerStructure'][j]):
                can_layer_pos [i][layers['OrderOfCanalizingVariables'][count]] = j+1
                n_dominant_vars[i][layers['OrderOfCanalizingVariables'][count]] = n_more_dominant_variables[j]
                can_layer_size[i][layers['OrderOfCanalizingVariables'][count]] = layers['LayerStructure'][j]
                count+=1

    functions_str = list(map(lambda BF: ''.join(map(str,BF)),functions))

    df = pd.DataFrame({
        'function' : np.repeat(functions_str, n),
        'variable': np.tile(np.arange(n), n_functions).flatten(),
        'edge effectiveness': np.concatenate(edge_effectiveness_per_var).flatten(),
        'input redundancy': np.concatenate(input_redundancy_per_var).flatten(),
        'canalizing strength': np.concatenate(can_strength_per_var).flatten(),
        'number of essential variables': np.concatenate(essential_vars).flatten(),
        'degenerate': np.concatenate(degenerate_vars).flatten(),
        'canalizing layer position': np.concatenate(can_layer_pos).flatten(),
        'number of  dominant variables': np.concatenate(n_dominant_vars).flatten(),
        'canalizing layer size': np.concatenate(can_layer_size).flatten(),
    })

    if keeping_unique_values:
        df = df.drop_duplicates(subset=['input redundancy', 'canalizing strength', 'canalizing layer position', 'canalizing layer size'])
    
    df.to_csv(filename, index=False)

    return df


def idx_to_bits(i, n):
    """
    Convert an integer index to its binary representation as a list of bits.

    The binary representation is returned with the least significant bit
    first, matching the ordering convention used for Boolean function truth
    tables.

    Parameters
    ----------
    i : int
        Integer index to convert.
    n : int
        Number of bits in the output representation.

    Returns
    -------
    list of int
        List of length ``n`` containing the binary representation of ``i``.
    """
    return [(i >> j) & 1 for j in range(n)]


def bits_to_idx(bits):
    """
    Convert a binary representation as a list of bits to an integer index.

    The input bit ordering assumes the least significant bit is first,
    consistent with the output format of ``idx_to_bits``.

    Parameters
    ----------
    bits : iterable of int
        Binary representation of an integer, where each element is 0 or 1.

    Returns
    -------
    int
        Integer index corresponding to the binary representation.
    """
    return sum(b << j for j, b in enumerate(bits))


def get_equivalent_functions(function):
    """
    Generate all NPN-equivalent Boolean functions of a given Boolean function.

    NPN equivalence is defined by allowing negations (N) and permutations (P) of
    input variables and optional negation of the output (N). The function
    generates all unique truth tables obtainable through these transformations
    and returns them as BooleanFunction objects.

    Parameters
    ----------
    function : bf.BooleanFunction or array-like
        Boolean function whose NPN-equivalent functions are generated.

    Returns
    -------
    list of bf.BooleanFunction
        List of unique Boolean functions equivalent to the input function
        under input permutation, input negation, and output negation.
    """
    f_str = list(map(int,(str(function).strip('[]').split(' '))))
    # Determine number of inputs from truth table length:
    n = int(math.log2(len(f_str)))

    # Validate that the truth table length corresponds to an n-input function:
    if 2**n != len(f_str):
        raise ValueError("Truth table length must be a power of 2")

    equivalent_functions = set()

    # Enumerate all input permutations, input negations, and output negation:
    for perm in permutations(range(n)):
        for neg in product([0, 1], repeat=n):
            for out_flip in [0, 1]:
                new_f_str = []

                # Construct transformed truth table:
                for i in range(2**n):
                    bits = idx_to_bits(i, n)

                    # Permute input variables:
                    bits = [bits[p] for p in perm]

                    # Negate selected input variables:
                    bits = [b ^ neg[j] for j, b in enumerate(bits)]
                    idx = bits_to_idx(bits)
                    val = f_str[idx]

                    # Optionally negate the output:
                    val ^= out_flip
                    new_f_str.append(val)
                
                # Store unique transformed truth tables:
                equivalent_functions.add(tuple(new_f_str))

    return [bf.BooleanFunction(np.array(f_str)) for f_str in equivalent_functions]

