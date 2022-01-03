import numpy as np, pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, issparse, isspmatrix_csc, isspmatrix_csr, vstack as sp_vstack
import warnings
import multiprocessing
import ctypes
import json
import os
from copy import deepcopy
from ._cpp_interface import (
    isoforest_cpp_obj, _sort_csc_indices, _reconstruct_csr_sliced,
    _reconstruct_csr_with_categ, _get_has_openmp
)

__all__ = ["IsolationForest"]

### Helpers
def _get_num_dtype(X_num=None, sample_weights=None, column_weights=None):
    if X_num is not None:
        return np.empty(0, dtype=X_num.dtype)
    elif sample_weights is not None:
        return np.empty(0, dtype=column_weights.dtype)
    elif column_weights is not None:
        return np.empty(0, dtype=sample_weights.dtype)
    else:
        return np.empty(0, dtype=ctypes.c_double)

def _get_int_dtype(X_num):
    if (X_num is not None) and (issparse(X_num)):
        return np.empty(0, dtype=X_num.indices.dtype)
    else:
        return np.empty(0, dtype=ctypes.c_size_t)

def _is_row_major(X_num):
    if (X_num is None) or (issparse(X_num)):
        return False
    else:
        return X_num.strides[1] == X_num.dtype.itemsize

def _is_col_major(X_num):
    if (X_num is None) or (issparse(X_num)):
        return False
    else:
        return X_num.strides[0] == X_num.dtype.itemsize

def _copy_if_subview(X_num, prefer_row_major=False):
    ### TODO: the C++ functions should accept a 'leading dimension'
    ### parameter so as to avoid copying the data here
    if (X_num is not None) and (not issparse(X_num)):
        col_major = _is_col_major(X_num)
        leading_dimension = int(X_num.strides[1 if col_major else 0] / X_num.dtype.itemsize)
        if (
                (leading_dimension != X_num.shape[0 if col_major else 1]) or
                (len(X_num.strides) != 2) or
                (not X_num.flags.aligned) or
                (not _is_row_major(X_num) and not _is_col_major(X_num))
            ):
            X_num = X_num.copy()
        if _is_col_major(X_num) != col_major:
            if prefer_row_major:
                X_num = np.ascontiguousarray(X_num)
            else:
                X_num = np.asfortranarray(X_num)
    return X_num

def _all_equal(x, y):
    if x.shape[0] != y.shape[0]:
        return False
    return np.all(x == y)

def _encode_categorical(cl, categories):
    if (cl.shape[0] >= 100) and (cl.dtype.name == "category"):
        if _all_equal(cl.cat.categories, categories):
            return cl.cat.codes
    return pd.Categorical(cl, categories).codes

def _process_nthreads(nthreads, warn_if_no_omp=False):
    if nthreads is None:
        nthreads = 1
    elif nthreads < 0:
        nthreads = multiprocessing.cpu_count() + 1 + nthreads
        if nthreads < 1:
            raise ValueError("Passed invalid 'nthreads'.")

    if (warn_if_no_omp) and (nthreads > 1) and (not _get_has_openmp()):
        msg_omp  = "Attempting to use more than 1 thread, but "
        msg_omp += "package was built without multi-threading "
        msg_omp += "support - see the project's GitHub page for "
        msg_omp += "more information."
        warnings.warn(msg_omp)

    assert nthreads > 0
    assert isinstance(nthreads, int)
    return nthreads

class IsolationForest:
    """
    Isolation Forest model

    Isolation Forest is an algorithm originally developed for outlier detection that consists in splitting
    sub-samples of the data according to some attribute/feature/column at random. The idea is that, the rarer
    the observation, the more likely it is that a random uniform split on some feature would put outliers alone
    in one branch, and the fewer splits it will take to isolate an outlier observation like this. The concept
    is extended to splitting hyperplanes in the extended model (i.e. splitting by more than one column at a time), and to
    guided (not entirely random) splits in the SCiForest and FCF models that aim at isolating outliers faster and/or
    finding clustered outliers.

    This version adds heuristics to handle missing data and categorical variables. Can be used to aproximate pairwise
    distances by checking the depth after which two observations become separated, and to approximate densities by fitting
    trees beyond balanced-tree limit. Offers options to vary between randomized and deterministic splits too.

    Note
    ----
    The default parameters in this software do not correspond to the suggested parameters in
    any of the references.
    In particular, the following default values are likely to cause huge differences when compared to the
    defaults in other software: ``ndim``, ``sample_size``, ``ntrees``. The defaults here are
    nevertheless more likely to result in better models. In order to mimic scikit-learn for example, one
    would need to pass ``ndim=1``, ``sample_size=256``, ``ntrees=100``, ``missing_action="fail"``, ``nthreads=1``.

    Note
    ----
    Shorthands for parameter combinations that match some of the references:

    'iForest' (reference [1]_):
        ``ndim=1``, ``sample_size=256``, ``max_depth=8``, ``ntrees=100``, ``missing_action="fail"``.

    'EIF' (reference [3]_):
        ``ndim=2``, ``sample_size=256``, ``max_depth=8``, ``ntrees=100``, ``missing_action="fail"``,
        ``coefs="uniform"``, ``standardize_data=False`` (plus standardizing the data **before** passing it).
    
    'SCiForest' (reference [4]_):
        ``ndim=2``, ``sample_size=256``, ``max_depth=8``, ``ntrees=100``, ``missing_action="fail"``,
        ``coefs="normal"``, ``ntry=10``, ``prob_pick_avg_gain=1``, ``penalize_range=True``.
        Might provide much better results with ``max_depth=None`` despite the reference's recommendation.

    'FCF' (reference [11]_):
        ``ndim=2``, ``sample_size=256``, ``max_depth=None``, ``ntrees=200``, ``missing_action="fail"``,
        ``coefs="normal"``, ``ntry=1``, ``prob_pick_pooled_gain=1``.
        Might provide similar or better results with ``ndim=1`` and/or sample size as low as 32.
        For the FCF model aimed at imputing missing values,
        might give better results with ``ntry=10`` or higher and much larger sample sizes.
    'RRCF' (reference [12]_):
        ``ndim=1``, ``prob_pick_col_by_range=1``, ``sample_size=256`` or more, ``max_depth=None``,
        ``ntrees=100`` or more, ``missing_action="fail"``. Note however that reference [12]_ proposed a
        different method for calculation of anomaly scores, while this library uses isolation depth just
        like for 'iForest', so results might differ significantly from those of other libraries.
        Nevertheless, experiments in reference [11]_ suggest that isolation depth might be a better
        scoring metric for this model.

    Note
    ----
    The model offers many tunable parameters (see reference [11]_ for a comparison).
    The most likely candidate to tune is
    ``prob_pick_pooled_gain``, for which higher values tend to
    result in a better ability to flag outliers in multimodal datasets, at the expense of poorer
    generalizability to inputs with values outside the variables' ranges to which the model was fit
    (see plots generated from the examples in GitHub notebook for a better idea of the difference). The next candidate to tune is
    ``sample_size`` - the default is to use all rows, but in some datasets introducing sub-sampling can help,
    especially for the single-variable model. In smaller datasets, one might also want to experiment
    with ``weigh_by_kurtosis`` and perhaps lower ``ndim``. If using ``prob_pick_pooled_gain``, models
    are likely to benefit from deeper trees (controlled by ``max_depth``), but using large samples
    and/or deeper trees can result in significantly slower model fitting and predictions - in such cases,
    using ``min_gain`` (with a value like 0.25) with ``max_depth=None`` can offer a better speed/performance
    trade-off than changing ``max_depth``.
    
    If the data has categorical variables and these are more important important for determining
    outlierness compared to numerical columns, one might want to experiment with ``ndim=1``,
    ``categ_split_type="single_categ"``, and ``scoring_metric="density"``.

    For small datasets, one might also want to experiment with ``ndim=1``, ``scoring_metric="adj_depth"``
    and ``penalize_range=True``.

    Note
    ----
    The default parameters will not scale to large datasets. In particular,
    if the amount of data is large, it's suggested to set a smaller sample size for each tree (parameter ``sample_size``)
    and to fit fewer of them (parameter ``ntrees``).
    As well, the default option for 'missing_action' might slow things down significantly.
    See the documentation of the parameters for more details.
    These defaults can also result in very big model sizes in memory and as serialized
    files (e.g. models that weight over 10GB) when the number of rows in the data is large.
    Using fewer trees, smaller sample sizes, and shallower trees can help to reduce model
    sizes if that becomes a problem.

    Note
    ----
    See the documentation of ``predict`` for some considerations when serving models generated through
    this library.

    Parameters
    ----------
    sample_size : str "auto", int, float(0,1), or None
        Sample size of the data sub-samples with which each binary tree will be built. If passing 'None', each
        tree will be built using the full data. Recommended value in [1]_, [2]_, [3]_ is 256, while
        the default value in the author's code in [5]_ is 'None' here.

        If passing "auto", will use the full number of rows in the data, up to 10,000 (i.e.
        will take 'sample_size=min(nrows(X), 10000)') **when calling fit**, and the full amount
        of rows in the data **when calling the variants** ``fit_predict`` or ``fit_transform``.

        If passing ``None``, will take the full number of rows in the data (no sub-sampling).

        If passing a number between zero and one, will assume it means taking a sample size that represents
        that proportion of the rows in the data.

        Hint: seeing a distribution of scores which is on average too far below 0.5 could mean that the
        model needs more trees and/or bigger samples to reach convergence (unless using non-random
        splits, in which case the distribution is likely to be centered around a much lower number),
        or that the distributions in the data are too skewed for random uniform splits.
    ntrees : int
        Number of binary trees to build for the model. Recommended value in [1]_ is 100, while the default value in the
        author's code in [5]_ is 10. In general, the number of trees required for good results
        is higher when (a) there are many columns, (b) there are categorical variables, (c) categorical variables have many
        categories, (d) `ndim` is high, (e) ``prob_pick_pooled_gain`` is used, (f) ``scoring_metric="density"``
        or ``scoring_metric="boxed_density"`` are used.

        Hint: seeing a distribution of scores which is on average too far below 0.5 could mean that the
        model needs more trees and/or bigger samples to reach convergence (unless using non-random
        splits, in which case the distribution is likely to be centered around a much lower number),
        or that the distributions in the data are too skewed for random uniform splits.
    ndim : int
        Number of columns to combine to produce a split. If passing 1, will produce the single-variable model described
        in [1]_ and [2]_, while if passing values greater than 1, will produce the extended model described in [3]_ and [4]_.
        Recommended value in [4]_ is 2, while [3]_ recommends a low value such as 2 or 3. Models with values higher than 1
        are referred hereafter as the extended model (as in [3]_).

        Note that, when using ``ndim>1`` plus ``standardize_data=True``, the variables are standardized at
        each step as suggested in [4]_, which makes the models slightly different than in [3]_.
    ntry : int
        When using ``prob_pick_pooled_gain`` and/or ``prob_pick_avg_gain``, how many variables (with ``ndim=1``)
        or linear combinations (with ``ndim>1``) to try for determining the best one according to gain.
        
        Recommended value in reference [4]_ is 10 (with ``prob_pick_avg_gain``, for outlier detection), while the
        recommended value in reference [11]_ is 1 (with ``prob_pick_pooled_gain``, for outlier detection), and the
        recommended value in reference [9]_ is 10 to 20 (with ``prob_pick_pooled_gain``, for missing value imputations).
    categ_cols : None or array-like
        Columns that hold categorical features, when the data is passed as an array or matrix.
        Categorical columns should contain only integer values with a continuous numeration starting at zero,
        with negative values and NaN taken as missing,
        and the array or list passed here should correspond to the column numbers, with numeration starting
        at zero. The maximum categorical value should not exceed 'INT_MAX' (typically :math:`2^{31}-1`).
        This might be passed either at construction time or when calling ``fit`` or variations of ``fit``.
        
        This is ignored when the input is passed as a ``DataFrame`` as then it will consider columns as
        categorical depending on their dtype (see the documentation for ``fit`` for details).
    max_depth : int, None, or str "auto"
        Maximum depth of the binary trees to grow. If passing None, will build trees until each observation ends alone
        in a terminal node or until no further split is possible. If using "auto", will limit it to the corresponding
        depth of a balanced binary tree with number of terminal nodes corresponding to the sub-sample size (the reason
        being that, if trying to detect outliers, an outlier will only be so if it turns out to be isolated with shorter average
        depth than usual, which corresponds to a balanced tree depth). When a terminal node has more than 1 observation, the
        remaining isolation depth for them is estimated assuming the data and splits are both uniformly random (separation depth
        follows a similar process with expected value calculated as in [6]_). Default setting for [1]_, [2]_, [3]_, [4]_ is "auto",
        but it's recommended to pass higher values if using the model for purposes other than outlier detection.

        Note that models that use ``prob_pick_pooled_gain`` or ``prob_pick_avg_gain`` are likely to benefit from
        deeper trees (larger ``max_depth``), but deeper trees can result in much slower model fitting and
        predictions.

        If using pooled gain, one might want to substitute ``max_depth`` with ``min_gain``.
    ncols_per_tree : None, int, or float(0,1]
        Number of columns to use (have as potential candidates for splitting at each iteration) in each tree,
        somewhat similar to the 'mtry' parameter of random forests.
        In general, this is only relevant when using non-random splits and/or weighted column choices.

        If passing a number between zero and one, will assume it means taking a sample size that represents
        that proportion of the columns in the data. If passing exactly 1, will assume it means taking
        100% of the columns rather than taking 1 column.

        If passing ``None`` (the default) or zero, will use the full number of available columns.
    prob_pick_pooled_gain : float[0, 1]
        This parameter indicates the probability of choosing the threshold on which to split a variable
        (with ``ndim=1``) or a linear combination of variables (when using ``ndim>1``) as the threshold
        that maximizes a pooled standard deviation gain criterion (see references [9]_ and [11]_) on the
        same variable or linear combination, similarly to regression trees such as CART.

        If using ``ntry>1``, will try several variables or linear combinations thereof and choose the one
        in which the largest standardized gain can be achieved.

        For categorical variables with ``ndim=1``, will use shannon entropy instead (like in [7]_).

        Compared to a simple averaged gain, this tends to result in more evenly-divided splits and more clustered
        groups when they are smaller. Recommended to pass higher values when used for imputation of missing values.
        When used for outlier detection, datasets with multimodal distributions usually see better performance
        under this type of splits.
        
        Note that, since this makes the trees more even and thus it takes more steps to produce isolated nodes,
        the resulting object will be heavier. When splits are not made according to any of ``prob_pick_avg_gain``
        or ``prob_pick_pooled_gain``, both the column and the split point are decided at random. Note that, if
        passing value 1 (100%) with no sub-sampling and using the single-variable model,
        every single tree will have the exact same splits.

        Be aware that ``penalize_range`` can also have a large impact when using ``prob_pick_pooled_gain``.

        Be aware also that, if passing a value of 1 (100%) with no sub-sampling and using the single-variable
        model, every single tree will have the exact same splits.

        Under this option, models are likely to produce better results when increasing ``max_depth``.
        Alternatively, one can also control the depth through ``min_gain`` (for which one might want to
        set ``max_depth=None``).

        Important detail: if using either ``prob_pick_avg_gain`` or ``prob_pick_pooled_gain``, the distribution of
        outlier scores is unlikely to be centered around 0.5.
    prob_pick_avg_gain : float[0, 1]
        This parameter indicates the probability of choosing the threshold on which to split a variable
        (with ``ndim=1``) or a linear combination of variables (when using ``ndim>1``) as the threshold
        that maximizes an averaged standard deviation gain criterion (see references [4]_ and [11]_) on the
        same variable or linear combination.

        If using ``ntry>1``, will try several variables or linear combinations thereof and choose the one
        in which the largest standardized gain can be achieved.

        For categorical variables with ``ndim=1``, will take the expected standard deviation that would be
        gotten if the column were converted to numerical by assigning to each category a random
        number :math:`\\sim \\text{Unif}(0, 1)` and calculate gain with those assumed standard deviations.

        Compared to a pooled gain, this tends to result in more cases in which a single observation or very
        few of them are put into one branch. Typically, datasets with outliers defined by extreme values in
        some column more or less independently of the rest, usually see better performance under this type
        of split. Recommended to use sub-samples (parameter ``sample_size``) when
        passing this parameter. Note that, since this will create isolated nodes faster, the resulting object
        will be lighter (use less memory).
        
        When splits are
        not made according to any of ``prob_pick_avg_gain`` or ``prob_pick_pooled_gain``,
        both the column and the split point are decided at random. Default setting for [1]_, [2]_, [3]_ is
        zero, and default for [4]_ is 1. This is the randomization parameter that can be passed to the author's original code in [5]_,
        but note that the code in [5]_ suffers from a mathematical error in the calculation of running standard deviations,
        so the results from it might not match with this library's.
        
        Be aware that, if passing a value of 1 (100%) with no sub-sampling and using the single-variable model, every single tree will have
        the exact same splits.

        Under this option, models are likely to produce better results when increasing ``max_depth``.

        Important detail: if using either ``prob_pick_avg_gain`` or ``prob_pick_pooled_gain``, the distribution of
        outlier scores is unlikely to be centered around 0.5.
    prob_pick_col_by_range : float[0, 1]
        When using ``ndim=1``, this denotes the probability of choosing the column to split with a probability
        proportional to the range spanned by each column within a node as proposed in reference [12]_.

        When using ``ndim>1``, this denotes the probability of choosing columns to create a hyperplane with a
        probability proportional to the range spanned by each column within a node.

        This option is not compatible with categorical data. If passing column weights, the
        effect will be multiplicative. This option is not compatible with ``weigh_by_kurtosis``.

        Be aware that the data is not standardized in any way for the range calculations, thus the scales
        of features will make a large difference under this option, which might not make it suitable for
        all types of data.

        If there are infinite values, all columns having infinite values will be treated as having the
        same weight, and will be chosen before every other column with non-infinite values.

        Note that the proposed RRCF model from [12]_ uses a different scoring metric for producing anomaly
        scores, while this library uses isolation depth regardless of how columns are chosen, thus results
        are likely to be different from those of other software implementations. Nevertheless, as explored
        in [11]_, isolation depth as a scoring metric typically provides better results than the
        "co-displacement" metric from [12]_ under these split types.
    prob_pick_col_by_var : float[0, 1]
        When using ``ndim=1``, this denotes the probability of choosing the column to split with a probability
        proportional to the variance of each column within a node.

        When using ``ndim>1``, this denotes the probability of choosing columns to create a hyperplane with a
        probability proportional to the variance of each column within a node.

        For categorical data, it will calculate the expected variance if the column were converted to
        numerical by assigning to each category a random number :math:`\\sim \\text{Unif}(0, 1)`, which depending on the number of
        categories and their distribution, produces numbers typically a bit smaller than standardized numerical
        variables.

        Note that when using sparse matrices, the calculation of variance will rely on a procedure that
        uses sums of squares, which has less numerical precision than the
        calculation used for dense inputs, and as such, the results might differ slightly.

        Be aware that this calculated variance is not standardized in any way, so the scales of
        features will make a large difference under this option.

        If passing column weights, the effect will be multiplicative. This option is not compatible
        with ``weigh_by_kurtosis``.

        If passing a ``missing_action`` different than "fail", infinite values will be ignored for the
        variance calculation. Otherwise, all columns with infinite values will have the same probability
        and will be chosen before columns with non-infinite values.
    prob_pick_col_by_kurt : float[0, 1]
        When using ``ndim=1``, this denotes the probability of choosing the column to split with a probability
        proportional to the kurtosis of each column **within a node** (unlike the option ``weigh_by_kurtosis``
        which calculates this metric only at the root).

        When using ``ndim>1``, this denotes the probability of choosing columns to create a hyperplane with a
        probability proportional to the kurtosis of each column within a node.

        For categorical data, it will calculate the expected kurtosis if the column were converted to
        numerical by assigning to each category a random number :math:`\\sim \\text{Unif}(0, 1)`.

        Note that when using sparse matrices, the calculation of kurtosis will rely on a procedure that
        uses sums of squares and higher-power numbers, which has less numerical precision than the
        calculation used for dense inputs, and as such, the results might differ slightly.

        If passing column weights, the effect will be multiplicative. This option is not compatible
        with ``weigh_by_kurtosis``.

        If passing a ``missing_action`` different than "fail", infinite values will be ignored for the
        kurtosis calculation. Otherwise, all columns with infinite values will have the same probability
        and will be chosen before columns with non-infinite values.

        Be aware that kurtosis can be a rather slow metric to calculate.
    min_gain : float > 0
        Minimum gain that a split threshold needs to produce in order to proceed with a split. Only used when the splits
        are decided by a gain criterion (either pooled or averaged). If the highest possible gain in the evaluated
        splits at a node is below this  threshold, that node becomes a terminal node.

        This can be used as a more sophisticated depth control when using pooled gain (note that ``max_depth``
        still applies on top of this heuristic).
    missing_action : str, one of "divide" (single-variable only), "impute", "fail", "auto"
        How to handle missing data at both fitting and prediction time. Options are:
        
        ``"divide"``:
            (For the single-variable model only, recommended) Will follow both branches and combine the result with the
            weight given by the fraction of the data that went to each branch when fitting the model.
        ``"impute"``:
            Will assign observations to the branch with the most observations in the single-variable model, or fill in
            missing values with the median of each column of the sample from which the split was made in the extended
            model (recommended for the extended model).
        ``"fail"``:
            Will assume there are no missing values and will trigger undefined behavior if it encounters any.
        ``"auto"``:
            Will use "divide" for the single-variable model and "impute" for the extended model.
        
        In the extended model, infinite values will be treated as missing.
        Passing "fail" will produce faster fitting and prediction times along with decreased
        model object sizes.
        Models from [1]_, [2]_, [3]_, [4]_ correspond to "fail" here.
    new_categ_action : str, one of "weighted" (single-variable only), "impute" (extended only), "smallest", "random"
        What to do after splitting a categorical feature when new data that reaches that split has categories that
        the sub-sample from which the split was done did not have. Options are:

        ``"weighted"``:
            (For the single-variable model only, recommended) Will follow both branches and combine the result with weight given
            by the fraction of the data that went to each branch when fitting the model.
        ``"impute"``:
            (For the extended model only, recommended) Will assign them the median value for that column that was added to the linear
            combination of features.
        ``"smallest"``:
            In the single-variable case will assign all observations with unseen categories in the split to the branch that had
            fewer observations when fitting the model, and in the extended case will assign them the coefficient of the least
            common category.
        ``"random"``:
            Will assing a branch (coefficient in the extended model) at random for each category beforehand, even if no observations
            had that category when fitting the model. Note that this can produce biased results when deciding
            splits by a gain criterion.

            Important: under this option, if the model is fitted to a ``DataFrame``, when calling ``predict``
            on new data which contains new categories (unseen in the data to which the model was fitted),
            they will be added to the model's state on-the-fly. This means that, if calling ``predict`` on data
            which has new categories, there might be inconsistencies in the results if predictions are done in
            parallel or if passing the same data in batches or with different row orders. It also means that
            the ``predict`` function will not be thread-safe (e.g. cannot be used alongside ``joblib`` with a
            backend that uses shared memory).
        ``"auto"``:
            Will select "weighted" for the single-variable model and "impute" for the extended model.
        Ignored when passing 'categ_split_type' = 'single_categ'.
    categ_split_type : str, one of "auto", subset", or "single_categ"
        Whether to split categorical features by assigning sub-sets of them to each branch, or by assigning
        a single category to a branch and the rest to the other branch. For the extended model, whether to
        give each category a coefficient, or only one while the rest get zero.

        If passing ``"auto"``, will select ``"subset"`` for the extended model and ``"single_categ"`` for
        the single-variable model.
    all_perm : bool
        When doing categorical variable splits by pooled gain with ``ndim=1`` (regular model),
        whether to consider all possible permutations of variables to assign to each branch or not. If ``False``,
        will sort the categories by their frequency and make a grouping in this sorted order. Note that the
        number of combinations evaluated (if ``True``) is the factorial of the number of present categories in
        a given column (minus 2). For averaged gain, the best split is always to put the second most-frequent
        category in a separate branch, so not evaluating all  permutations (passing ``False``) will make it
        possible to select other splits that respect the sorted frequency order.
        Ignored when not using categorical variables or not doing splits by pooled gain or using ``ndim > 1``.
    coef_by_prop : bool
        In the extended model, whether to sort the randomly-generated coefficients for categories
        according to their relative frequency in the tree node. This might provide better results when using
        categorical variables with too many categories, but is not recommended, and not reflective of
        real "categorical-ness". Ignored for the regular model (``ndim=1``) and/or when not using categorical
        variables.
    recode_categ : bool
        Whether to re-encode categorical variables even in case they are already passed
        as ``pd.Categorical``. This is recommended as it will eliminate potentially redundant categorical levels if
        they have no observations, but if the categorical variables are already of type ``pd.Categorical`` with only
        the levels that are present, it can be skipped for slightly faster fitting times. You'll likely
        want to pass ``False`` here if merging several models into one through ``append_trees``.
    weights_as_sample_prob : bool
        If passing sample (row) weights when fitting the model, whether to consider those weights as row
        sampling weights (i.e. the higher the weights, the more likely the observation will end up included
        in each tree sub-sample), or as distribution density weights (i.e. putting a weight of two is the same
        as if the row appeared twice, thus higher weight makes it less of an outlier). Note that sampling weight
        is only used when sub-sampling data for each tree, which is not the default in this implementation.
    sample_with_replacement : bool
        Whether to sample rows with replacement or not (not recommended). Note that distance calculations,
        if desired, don't work well with duplicate rows.
    penalize_range : bool
        Whether to penalize (add -1 to the terminal depth) observations at prediction time that have a value
        of the chosen split variable (linear combination in extended model) that falls outside of a pre-determined
        reasonable range in the data being split (given by 2 * range in data and centered around the split point),
        as proposed in [4]_ and implemented in the authors' original code in [5]_. Not used in single-variable model
        when splitting by categorical variables.

        This option is not supported when using density-based outlier scoring metrics.

        It's recommended to turn this off for faster predictions on sparse CSC matrices.

        Note that this can make a very large difference in the results when using ``prob_pick_pooled_gain``.

        Be aware that this option can make the distribution of outlier scores a bit different
        (i.e. not centered around 0.5).
    scoring_metric : str
        Metric to use for determining outlier scores (see reference [13]_). Options are:

        ``"depth"``
            Will use isolation depth as proposed in reference [1]_. This is typically the safest choice
            and plays well with all model types offered by this library.
        ``"density"``
            Will set scores for each terminal node as the ratio between the fraction of points in the sub-sample
            that end up in that node and the fraction of the volume in the feature space which defines
            the node according to the splits that lead to it.
            If using ``ndim=1``, for categorical variables, this is defined in terms
            of number of categories that go towards each side of the split divided by number of categories
            in the observations that reached that node.

            The standardized outlier score from density for a given observation is calculated as the
            negative of the logarithm of the geometric mean from the per-tree densities, which unlike
            the standardized score produced from depth, is unbounded, but just like the standardized
            score form depth, has a natural threshold for definining outlierness, which in this case
            is zero is instead of 0.5. The non-standardized outlier score is calculated as the
            geometric mean, while the per-tree scores are calculated as the density values.
            
            This might lead to better predictions when using ``ndim=1``, particularly in the presence
            of categorical variables. Note however that using density requires more trees for convergence
            of scores (i.e. good results) compared to isolation-based metrics.

            This option is incompatible with ``penalize_range``.
        ``"adj_depth"``
            Will use an adjusted isolation depth that takes into account the number of points that
            go to each side of a given split vs. the fraction of the range of that feature that each
            side of the split occupies, by a metric as follows:
                :math:`d = \\frac{2}{ 1 + \\frac{1}{2 p} }`
            
            Where :math:`p` is defined as:
                :math:`p = \\frac{n_s}{n_t} / \\frac{r_s}{r_t}`
            
            With :math:`n_t` being the number of points that reach a given node, :math:`n_s` the
            number of points that are sent to a given side of the split/branch at that node,
            :math:`r_t` being the range (maximum minus minimum) of the splitting feature or
            linear combination among the points that reached the node, and :math:`r_s` being the
            range of the same feature or linear combination among the points that are sent to this
            same side of the split/branch. This makes each split add a number between zero and two
            to the isolation depth, with this number's probabilistic distribution being centered
            around 1 and thus the expected isolation depth remaing the same as in the original
            ``"depth"`` metric, but having more variability around the extremes.

            Scores (standardized, non-standardized, per-tree) are aggregated in the same way
            as for ``"depth"``.

            This might lead to better predictions when using ``ndim=1``, particularly in the prescence
            of categorical variables and for smaller datasets, and for smaller datasets, might make
            sense to combine it with ``penalize_range=True``.
        ``"adj_density"``
            Will use the same metric from ``"adj_depth"``, but applied multiplicatively instead
            of additively. The expected value for this adjusted density is not strictly the same
            as for isolation, but using the expected isolation depth as standardizing criterion
            tends to produce similar standardized score distributions (centered around 0.5).

            Scores (standardized, non-standardized, per-tree) are aggregated in the same way
            as for ``"depth"``.

            This option is incompatible with ``penalize_range``.
        ``"boxed_ratio"``
            Will set the scores for each terminal node as the ratio between the volume of the boxed
            feature space for the node as defined by the smallest and largest values from the split
            conditions for each column (bounded by the variable ranges in the sample) and the
            variable ranges in the tree sample.
            If using ``ndim=1``, for categorical variables this is defined in terms of number of
            categories.
            If using ``ndim=>1``, this is defined in terms of the maximum achievable value for the
            splitting linear combination determined from the minimum and maximum values for each
            variable among the points in the sample, and as such, it has a rather different meaning
            compared to the score obtained with ``ndim=1`` - boxed ratio scores with ``ndim>1``
            typically provide very poor quality results and this metric is thus not recommended to
            use in the extended model. With 'ndim>1', it also has a tendency of producing too small
            values which round to zero.

            The standardized outlier score from boxed ratio for a given observation is calculated
            simply as the the average from the per-tree boxed ratios. This metric
            has a lower bound of zero and a theorical upper bound of one, but in practice the scores
            tend to be very small numbers close to zero, and its distribution across
            different datasets is rather unpredictable. In order to keep rankings comparable with
            the rest of the metrics, the non-standardized outlier scores are calculated as the
            negative of the average instead. The per-tree scores are calculated as the ratios.

            For better numerical precision, this metric is implemented in a rather computationally
            inefficient way, and using it might increase fitting times significantly, particularly
            when the number of columns in the data is large.

            This metric might lead to better predictions in datasets with many rows when using ``ndim=1``
            and a relatively small ``sample_size``. Note that more trees are required for convergence
            of scores when using this metric. In some datasets, this metric might result in very bad
            predictions, to the point that taking its inverse produces a much better ranking of outliers.

            This option is incompatible with ``penalize_range``.
        ``"boxed_density2"``
            Will set the score as the ratio between the fraction of points within the sample that
            end up in a given terminal node and the boxed ratio metric.

            Aggregation of scores (standardized, non-standardized, per-tree) is done in the same
            way as for density, and it also has a natural threshold at zero for determining
            outliers and inliers.

            This metric is typically usable with 'ndim>1', but tends to produce much bigger values
            compared to 'ndim=1'.

            Albeit unintuitively, in many datasets, one can usually get better results with metric
            ``"boxed_density"`` instead.

            This option is incompatible with ``penalize_range``.
        ``"boxed_density"``
            Will set the score as the ratio between the fraction of points within the sample that
            end up in a  given terminal node and the ratio between the boxed volume of the feature
            space in the sample and the boxed volume of a node given by the split conditions (inverse
            as in ``"boxed_density2"``). This metric does not have any theoretical or intuitive
            justification behind its existence, and it is perhaps ilogical to use it as a
            scoring metric, but tends to produce good results in some datasets.

            The standardized outlier scores are defined as the negative of the geometric mean
            of this metric, while the non-standardized scores are the geometric mean, and the
            per-tree scores are simply the 'density' values.

            This option is incompatible with ``penalize_range``.
    standardize_data : bool
        Whether to standardize the features at each node before creating alinear combination of them as suggested
        in [4]_. This is ignored when using ``ndim=1``.
    weigh_by_kurtosis : bool
        Whether to weigh each column according to the kurtosis obtained in the sub-sample that is selected
        for each tree as briefly proposed in [1]_. Note that this is only done at the beginning of each tree
        sample. For categorical columns, will calculate expected kurtosis if the column were converted to
        numerical by assigning to each category a random number :math:`\\sim \\text{Unif}(0, 1)`.

        Note that when using sparse matrices, the calculation of kurtosis will rely on a procedure that
        uses sums of squares and higher-power numbers, which has less numerical precision than the
        calculation used for dense inputs, and as such, the results might differ slightly.

        Using this option makes the model more likely to pick the columns that have anomalous values
        when viewed as a 1-d distribution, and can bring a large improvement in some datasets.

        This is intended as a cheap feature selector, while the parameter ``prob_pick_col_by_kurt``
        provides the option to do this at each node in the tree for a different overall type of model.

        If passing column weights, the effect will be multiplicative. This option is not compatible
        with randomized column selection proportional to some other per-node metric.

        If passing ``missing_action="fail"`` and the data has infinite values, columns with rows
        having infinite values will get a weight of zero. If passing a different value for missing
        action, infinite values will be ignored in the kurtosis calculation.
    coefs : str, one of "normal" or "uniform"
        For the extended model, whether to sample random coefficients according to a normal distribution :math:`\\sim \\text{Normal}(0, 1)`
        (as proposed in [4]_) or according to a uniform distribution :math:`\\sim \\text{Unif}(-1, +1)` as proposed in [3]_. Ignored for the
        single-variable model. Note that, for categorical variables, the coefficients will be sampled ~ N (0,1)
        regardless - in order for both types of variables to have transformations in similar ranges (which will tend
        to boost the importance of categorical variables), pass ``"uniform"`` here.
    assume_full_distr : bool
        When calculating pairwise distances (see [8]_), whether to assume that the fitted model represents
        a full population distribution (will use a standardizing criterion assuming infinite sample,
        and the results of the similarity between two points at prediction time will not depend on the
        prescence of any third point that is similar to them, but will differ more compared to the pairwise
        distances between points from which the model was fit). If passing 'False', will calculate pairwise distances
        as if the new observations at prediction time were added to the sample to which each tree was fit, which
        will make the distances between two points potentially vary according to other newly introduced points.
        This will not be assumed when the distances are calculated as the model is being fit (see documentation
        for method 'fit_transform').
    build_imputer : bool
        Whether to construct missing-value imputers so that later this same model could be used to impute
        missing values of new (or the same) observations. Be aware that this will significantly increase the memory
        requirements and serialized object sizes. Note that this is not related to 'missing_action' as missing
        values inside the model are treated differently and follow their own imputation or division strategy.
    min_imp_obs : int
        Minimum number of observations with which an imputation value can be produced. Ignored if passing
        'build_imputer' = 'False'.
    depth_imp : str, one of "higher", "lower", "same"
        How to weight observations according to their depth when used for imputing missing values. Passing
        "higher" will weigh observations higher the further down the tree (away from the root node) the
        terminal node is, while "lower" will do the opposite, and "same" will not modify the weights according
        to node depth in the tree. Implemented for testing purposes and not recommended to change
        from the default. Ignored when passing 'build_imputer' = 'False'.
    weigh_imp_rows : str, one of "inverse", "prop", "flat"
        How to weight node sizes when used for imputing missing values. Passing "inverse" will weigh
        a node inversely proportional to the number of observations that end up there, while "proportional"
        will weight them heavier the more observations there are, and "flat" will weigh all nodes the same
        in this regard regardless of how many observations end up there. Implemented for testing purposes
        and not recommended to change from the default. Ignored when passing 'build_imputer' = 'False'.
    random_seed : int
        Seed that will be used for random number generation.
    nthreads : int
        Number of parallel threads to use. If passing a negative number, will use
        the same formula as joblib does for calculating number of threads (which is
        n_cpus + 1 + n_jobs - i.e. pass -1 to use all available threads). Note that, the more threads,
        the more memory will be allocated, even if the thread does not end up being used.
        Be aware that most of the operations are bound by memory bandwidth, which means that
        adding more threads will not result in a linear speed-up. For some types of data
        (e.g. large sparse matrices with small sample sizes), adding more threads might result
        in only a very modest speed up (e.g. 1.5x faster with 4x more threads),
        even if all threads look fully utilized.
    n_estimators : None or int
        Synonym for ``ntrees``, kept for better compatibility with scikit-learn.
    max_samples : None or int
        Synonym for ``sample_size``, kept for better compatibility with scikit-learn.
    n_jobs : None or int
        Synonym for ``nthreads``, kept for better compatibility with scikit-learn.
    random_state : None, int, or RandomState
        Synonym for ``random_seed``, kept for better compatibility with scikit-learn.
    bootstrap : None or bool
        Synonym for ``sample_with_replacement``, kept for better compatibility with scikit-learn.

    Attributes
    ----------
    cols_numeric_ : array(n_num_features,)
        Array with the names of the columns that were taken as numerical
        (Only when fitting the model to a DataFrame object).
    cols_categ_ : array(n_categ_features,)
        Array with the names of the columns that were taken as categorical
        (Only when fitting the model to a DataFrame object).
    is_fitted_ : bool
        Indicator telling whether the model has been fit to data or not.

    References
    ----------
    .. [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest."
           2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
    .. [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation-based anomaly detection."
           ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
    .. [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner. "Extended Isolation Forest."
           arXiv preprint arXiv:1811.02141 (2018).
    .. [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "On detecting clustered anomalies using SCiForest."
           Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
    .. [5] https://sourceforge.net/projects/iforest/
    .. [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
    .. [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
    .. [8] Cortes, David. "Distance approximation using Isolation Forests."
           arXiv preprint arXiv:1910.12362 (2019).
    .. [9] Cortes, David. "Imputing missing values with unsupervised random trees."
           arXiv preprint arXiv:1911.06646 (2019).
    .. [10] https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottom
    .. [11] Cortes, David. "Revisiting randomized choices in isolation forests."
            arXiv preprint arXiv:2110.13402 (2021).
    .. [12] Guha, Sudipto, et al. "Robust random cut forest based anomaly detection on streams."
            International conference on machine learning. PMLR, 2016.
    .. [13] Cortes, David. "Isolation forests: looking beyond tree depth."
            arXiv preprint arXiv:2111.11639 (2021).
    """
    def __init__(self, sample_size = "auto", ntrees = 500, ndim = 3, ntry = 1,
                 categ_cols = None, max_depth = "auto", ncols_per_tree = None,
                 prob_pick_pooled_gain = 0.0, prob_pick_avg_gain = 0.0,
                 prob_pick_col_by_range = 0.0, prob_pick_col_by_var = 0.0,
                 prob_pick_col_by_kurt = 0.0,
                 min_gain = 0., missing_action = "auto", new_categ_action = "auto",
                 categ_split_type = "auto", all_perm = False,
                 coef_by_prop = False, recode_categ = False,
                 weights_as_sample_prob = True, sample_with_replacement = False,
                 penalize_range = False, standardize_data = True,
                 scoring_metric = "depth", weigh_by_kurtosis = False,
                 coefs = "normal", assume_full_distr = True,
                 build_imputer = False, min_imp_obs = 3,
                 depth_imp = "higher", weigh_imp_rows = "inverse",
                 random_seed = 1, nthreads = -1,
                 n_estimators = None, max_samples = None,
                 n_jobs = None, random_state = None, bootstrap = None):
        self.sample_size = sample_size
        self.ntrees = ntrees
        self.ndim = ndim
        self.ntry = ntry
        self.categ_cols = categ_cols
        self.max_depth = max_depth
        self.ncols_per_tree = ncols_per_tree
        self.prob_pick_avg_gain = prob_pick_avg_gain
        self.prob_pick_pooled_gain = prob_pick_pooled_gain
        self.prob_pick_col_by_range = prob_pick_col_by_range
        self.prob_pick_col_by_var = prob_pick_col_by_var
        self.prob_pick_col_by_kurt = prob_pick_col_by_kurt
        self.min_gain = min_gain
        self.missing_action = missing_action
        self.new_categ_action = new_categ_action
        self.categ_split_type = categ_split_type
        self.all_perm = all_perm
        self.coef_by_prop = coef_by_prop
        self.recode_categ = recode_categ
        self.weights_as_sample_prob = weights_as_sample_prob
        self.sample_with_replacement = sample_with_replacement
        self.penalize_range = penalize_range
        self.standardize_data = standardize_data
        self.scoring_metric = scoring_metric
        self.weigh_by_kurtosis = weigh_by_kurtosis
        self.coefs = coefs
        self.assume_full_distr = assume_full_distr
        self.build_imputer = build_imputer
        self.min_imp_obs = min_imp_obs
        self.depth_imp = depth_imp
        self.weigh_imp_rows = weigh_imp_rows
        self.random_seed = random_seed
        self.nthreads = nthreads
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.bootstrap = bootstrap

        self._reset_obj()

    def _init(self, categ_cols = None):
        if categ_cols is not None:
            if self.categ_cols is not None:
                warnings.warn("Passed 'categ_cols' in constructor and fit method. Will take the latter.")
            self.categ_cols = categ_cols
        self._initialize_full(
                 sample_size = self.sample_size if (self.max_samples is None) else self.max_samples,
                 ntrees = self.ntrees if (self.n_estimators is None) else self.n_estimators,
                 ndim = self.ndim, ntry = self.ntry,
                 categ_cols = self.categ_cols,
                 max_depth = self.max_depth, ncols_per_tree = self.ncols_per_tree,
                 prob_pick_avg_gain = self.prob_pick_avg_gain, prob_pick_pooled_gain = self.prob_pick_pooled_gain,
                 prob_pick_col_by_range = self.prob_pick_col_by_range,
                 prob_pick_col_by_var = self.prob_pick_col_by_var,
                 prob_pick_col_by_kurt = self.prob_pick_col_by_kurt,
                 min_gain = self.min_gain, missing_action = self.missing_action, new_categ_action = self.new_categ_action,
                 categ_split_type = self.categ_split_type, all_perm = self.all_perm,
                 coef_by_prop = self.coef_by_prop, recode_categ = self.recode_categ,
                 weights_as_sample_prob = self.weights_as_sample_prob,
                 sample_with_replacement = self.sample_with_replacement if (self.bootstrap is None) else self.bootstrap,
                 penalize_range = self.penalize_range, standardize_data = self.standardize_data,
                 scoring_metric = self.scoring_metric, weigh_by_kurtosis = self.weigh_by_kurtosis,
                 coefs = self.coefs, assume_full_distr = self.assume_full_distr,
                 build_imputer = self.build_imputer, min_imp_obs = self.min_imp_obs,
                 depth_imp = self.depth_imp, weigh_imp_rows = self.weigh_imp_rows,
                 random_seed = self.random_seed if (self.random_state is None) else self.random_state,
                 nthreads = self.nthreads if (self.n_jobs is None) else self.n_jobs)

    def _initialize_full(self, sample_size = None, ntrees = 500, ndim = 3, ntry = 1,
                 categ_cols = None, max_depth = "auto", ncols_per_tree = None,
                 prob_pick_avg_gain = 0.0, prob_pick_pooled_gain = 0.0,
                 prob_pick_col_by_range = 0.0, prob_pick_col_by_var = 0.0,
                 prob_pick_col_by_kurt = 0.0,
                 min_gain = 0., missing_action = "auto", new_categ_action = "auto",
                 categ_split_type = "auto", all_perm = False,
                 coef_by_prop = False, recode_categ = True,
                 weights_as_sample_prob = True, sample_with_replacement = False,
                 penalize_range = True, standardize_data = True,
                 scoring_metric = "depth", weigh_by_kurtosis = False,
                 coefs = "normal", assume_full_distr = True,
                 build_imputer = False, min_imp_obs = 3,
                 depth_imp = "higher", weigh_imp_rows = "inverse",
                 random_seed = 1, nthreads = -1):
        if (sample_size is not None) and (sample_size != "auto"):
            assert sample_size > 0
            if sample_size > 1:
                assert isinstance(sample_size, int)
        if ncols_per_tree is not None:
            assert ncols_per_tree > 0
            if ncols_per_tree > 1:
                assert isinstance(ncols_per_tree, int)
            elif ncols_per_tree == 1:
                ncols_per_tree = None
        assert ntrees > 0
        assert isinstance(ntrees, int)
        if (max_depth != "auto") and (max_depth is not None):
            assert max_depth > 0
            assert isinstance(max_depth, int)
            if (sample_size is not None) and (sample_size != "auto"):
                if not (max_depth < sample_size):
                    warnings.warn("Passed 'max_depth' greater than 'sample_size'. Will be ignored.")
        assert ndim >= 1
        assert isinstance(ndim, int)
        assert ntry >= 1
        assert isinstance(ntry, int)
        if isinstance(random_seed, np.random.RandomState):
            random_seed = random_seed.randint(np.iinfo(np.int32).max)
        if isinstance(random_seed, np.random.Generator):
            random_seed = random_seed.integers(np.iinfo(np.int32).max)
        random_seed = int(random_seed)
        assert random_seed >= 0
        assert isinstance(min_imp_obs, int)
        assert min_imp_obs >= 1

        assert missing_action    in ["divide",        "impute",    "fail",   "auto"]
        assert new_categ_action  in ["weighted",      "smallest",  "random", "impute", "auto"]
        assert categ_split_type  in ["single_categ",  "subset",    "auto"]
        assert coefs             in ["normal",        "uniform"]
        assert depth_imp         in ["lower",         "higher",    "same"]
        assert weigh_imp_rows    in ["inverse",       "prop",      "flat"]
        assert scoring_metric    in ["depth",         "adj_depth", "density", "adj_density",
                                     "boxed_density", "boxed_density2",       "boxed_ratio"]

        assert prob_pick_avg_gain     >= 0
        assert prob_pick_pooled_gain  >= 0
        assert prob_pick_col_by_range >= 0
        assert prob_pick_col_by_var   >= 0
        assert prob_pick_col_by_kurt  >= 0
        assert min_gain               >= 0
        s = prob_pick_avg_gain + prob_pick_pooled_gain
        if s > 1:
            warnings.warn("Split type probabilities sum to more than 1, will standardize them")
            prob_pick_avg_gain     /= s
            prob_pick_pooled_gain  /= s

        s = prob_pick_col_by_range + prob_pick_col_by_var + prob_pick_col_by_kurt
        if s > 1:
            warnings.warn("Column choice probabilities sum to more than 1, will standardize them")
            prob_pick_col_by_range  /= s
            prob_pick_col_by_var    /= s
            prob_pick_col_by_kurt   /= s

        if weigh_by_kurtosis and (prob_pick_col_by_range or prob_pick_col_by_var or prob_pick_col_by_kurt):
            raise ValueError("'weigh_by_kurtosis' is incompatible with by-node column weight criteria.")

        if (ndim == 1) and ((sample_size is None) or (sample_size == "auto")) and ((prob_pick_avg_gain >= 1) or (prob_pick_pooled_gain >= 1)) and (not sample_with_replacement):
            msg  = "Passed parameters for deterministic single-variable splits"
            msg += " with no sub-sampling. "
            msg += "Every tree fitted will end up doing exactly the same splits. "
            msg += "It's recommended to set 'prob_pick_avg_gain' < 1, 'prob_pick_pooled_gain' < 1, "
            msg += "or to use the extended model (ndim > 1)."
            warnings.warn(msg)

        if missing_action == "auto":
            if ndim == 1:
                missing_action = "divide"
            else:
                missing_action = "impute"

        if new_categ_action == "auto":
            if ndim == 1:
                new_categ_action = "weighted"
            else:
                new_categ_action = "impute"

        if (build_imputer) and (missing_action == "fail"):
            raise ValueError("Cannot impute missing values when passing 'missing_action' = 'fail'.")

        if categ_split_type == "auto":
            if ndim == 1:
                categ_split_type = "single_categ"
            else:
                categ_split_type = "subset"
        if ndim == 1:
            if (categ_split_type != "single_categ") and (new_categ_action == "impute"):
                raise ValueError("'new_categ_action' = 'impute' not supported in single-variable model.")
        else:
            if missing_action == "divide":
                raise ValueError("'missing_action' = 'divide' not supported in extended model.")
            if (categ_split_type != "single_categ") and (new_categ_action == "weighted"):
                raise ValueError("'new_categ_action' = 'weighted' not supported in extended model.")

        if penalize_range and scoring_metric in ["density", "adj_density", "boxed_density", "boxed_density2", "boxed_ratio"]:
            raise ValueError("'penalize_range' is incompatible with density scoring.")

        if categ_cols is not None:
            categ_cols = np.array(categ_cols).reshape(-1).astype(int)
            categ_cols.sort()

        ## TODO: for better sklearn compatibility, should have versions of
        ## these with underscores at the end
        self.sample_size             =  sample_size
        self.ntrees                  =  ntrees
        self.ndim                    =  ndim
        self.ntry                    =  ntry
        self.categ_cols              =  categ_cols
        self.max_depth               =  max_depth
        self.ncols_per_tree          =  ncols_per_tree
        self.prob_pick_avg_gain      =  float(prob_pick_avg_gain)
        self.prob_pick_pooled_gain   =  float(prob_pick_pooled_gain)
        self.prob_pick_col_by_range  =  float(prob_pick_col_by_range)
        self.prob_pick_col_by_var    =  float(prob_pick_col_by_var)
        self.prob_pick_col_by_kurt   =  float(prob_pick_col_by_kurt)
        self.min_gain                =  min_gain
        self.missing_action          =  missing_action
        self.new_categ_action        =  new_categ_action
        self.categ_split_type_       =  categ_split_type
        self.coefs                   =  coefs
        self.depth_imp               =  depth_imp
        self.weigh_imp_rows          =  weigh_imp_rows
        self.scoring_metric          =  scoring_metric
        self.min_imp_obs             =  min_imp_obs
        self.random_seed             =  random_seed
        self.nthreads                =  nthreads

        self.all_perm                =  bool(all_perm)
        self.recode_categ            =  bool(recode_categ)
        self.coef_by_prop            =  bool(coef_by_prop)
        self.weights_as_sample_prob  =  bool(weights_as_sample_prob)
        self.sample_with_replacement =  bool(sample_with_replacement)
        self.penalize_range          =  bool(penalize_range)
        self.standardize_data        =  bool(standardize_data)
        self.weigh_by_kurtosis       =  bool(weigh_by_kurtosis)
        self.assume_full_distr       =  bool(assume_full_distr)
        self.build_imputer           =  bool(build_imputer)

        self._reset_obj()

    def _reset_obj(self):
        self.cols_numeric_  =  np.array([])
        self.cols_categ_    =  np.array([])
        self._cat_mapping   =  list()
        self._cat_max_lev   =  np.array([])
        self._ncols_numeric =  0
        self._ncols_categ   =  0
        self.is_fitted_     =  False
        self._ntrees        =  0
        self._cpp_obj       =  isoforest_cpp_obj()
        self._is_extended_  =  self.ndim > 1

    def copy(self):
        """
        Get a deep copy of this object

        Returns
        -------
        copied : obj
            A deep copy of this object
        """
        if not self.is_fitted_:
            self._cpp_obj = isoforest_cpp_obj()
            return deepcopy(self)
        else:
            obj_restore = self._cpp_obj
            obj_new = self._cpp_obj.deepcopy()
            try:
                self._cpp_obj = None
                out = deepcopy(self)
            finally:
                self._cpp_obj = obj_restore
            out._cpp_obj = obj_new
            return out

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Kept for compatibility with scikit-learn.

        Parameters
        ----------
        deep : bool
            Ignored.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        import inspect
        return {param.name:getattr(self, param.name) for param in inspect.signature(self.__init__).parameters.values()}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Kept for compatibility with scikit-learn.

        Note
        ----
        Setting any parameter other than the number of threads will reset the model
        - that is, if it was fitted to some data, the fitted model will be lost,
        and it will need to be refitted before being able to make predictions.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not (len(params) == 1 and ("nthreads" in params or "n_jobs" in params)):
            self.is_fitted_ = False
        valid_params = self.get_params(deep=False)
        for k,v in params.items():
            if k not in valid_params:
                raise ValueError("Invalid parameter: ", k)
            setattr(self, k, v)
        return self

    def __str__(self):
        msg = ""
        if self._is_extended_:
            msg += "Extended "
        msg += "Isolation Forest model"
        if (self.prob_pick_avg_gain + self.prob_pick_pooled_gain) > 0:
            msg += " (using guided splits)"
        msg += "\n"
        if self.ndim > 1:
            msg += "Splitting by %d variables at a time\n" % self.ndim
        if self.is_fitted_:
            msg += "Consisting of %d trees\n" % self._ntrees
            if self._ncols_numeric > 0:
                msg += "Numeric columns: %d\n" % self._ncols_numeric
            if self._ncols_categ:
                msg += "Categorical columns: %d\n" % self._ncols_categ
        return msg

    def __repr__(self):
        return self.__str__()

    def _get_model_obj(self):
        return self._cpp_obj.get_cpp_obj(self._is_extended_)

    def _get_imputer_obj(self):
        return self._cpp_obj.get_imputer()

    def _check_can_use_imputer(self, X_cat):
        categ_split_type = self.categ_split_type
        if categ_split_type == "auto":
            if self.ndim == 1:
                categ_split_type = "single_categ"
            else:
                categ_split_type = "subset"
        if (self.build_imputer) and (self.ndim == 1) and (X_cat is not None) and (X_cat.shape[1]):
            if (categ_split_type != "single_categ") and (self.new_categ_action == "weighted"):
                raise ValueError("Cannot build imputer with 'ndim=1' + 'new_categ_action=weighted'.")
            if self.missing_action == "divide":
                raise ValueError("Cannot build imputer with 'ndim=1' + 'missing_action=divide'.")

    def fit(self, X, y = None, sample_weights = None, column_weights = None, categ_cols = None):
        """
        Fit isolation forest model to data

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data to which to fit the model. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
            If passing a DataFrame, will assume that columns are:
            
             - Numeric, if their dtype is a subtype of NumPy's 'number' or 'datetime64'.
            
             - Categorical, if their dtype is 'object', 'Categorical', or 'bool'. Note that,
               if `Categorical` dtypes are ordered, the order will be ignored here.
            
            Other dtypes are not supported.

            Note that, if passing NumPy arrays, they are used in column-major order (a.k.a. "Fortran arrays"),
            and if they are not already in column-major format, will need to create a copy of the data.
        y : None
            Not used. Kept as argument for compatibility with Scikit-Learn pipelining.
        sample_weights : None or array(n_samples,)
            Sample observation weights for each row of 'X', with higher weights indicating either higher sampling
            probability (i.e. the observation has a larger effect on the fitted model, if using sub-samples), or
            distribution density (i.e. if the weight is two, it has the same effect of including the same data
            point twice), according to parameter 'weights_as_sample_prob' in the model constructor method.
        column_weights : None or array(n_features,)
            Sampling weights for each column in 'X'. Ignored when picking columns by deterministic criterion.
            If passing None, each column will have a uniform weight. If used along with kurtosis weights, the
            effect is multiplicative.
        categ_cols : None or array-like
            Columns that hold categorical features, when the data is passed as an array or matrix.
            Categorical columns should contain only integer values with a continuous numeration starting at zero,
            with negative values and NaN taken as missing,
            and the array or list passed here should correspond to the column numbers, with numeration starting
            at zero. The maximum categorical value should not exceed 'INT_MAX' (typically :math:`2^{31}-1`).
            This might be passed either at construction time or when calling ``fit`` or variations of ``fit``.
            
            This is ignored when the input is passed as a ``DataFrame`` as then it will consider columns as
            categorical depending on their dtype.

        Returns
        -------
        self : obj
            This object.
        """
        self._init(categ_cols)
        nthreads_use = _process_nthreads(self.nthreads, True)
        if (
                self.sample_size is None
                and (sample_weights is not None)
                and (self.weights_as_sample_prob)
        ):
            raise ValueError("Sampling weights are only supported when using sub-samples for each tree.")
        self._reset_obj()
        X_num, X_cat, ncat, sample_weights, column_weights, nrows = self._process_data(X, sample_weights, column_weights)

        self._check_can_use_imputer(X_cat)

        if self.sample_size is None:
            sample_size = nrows
        elif self.sample_size == "auto":
            sample_size = min(nrows, 10000)
            if (sample_weights is not None) and (self.weights_as_sample_prob):
                raise ValueError("Sampling weights are only supported when using sub-samples for each tree.")
        elif self.sample_size <= 1:
            sample_size = int(np.ceil(self.sample_size * nrows))
            if sample_size < 2:
                raise ValueError("Sampling proportion amounts to a single row or less.")
        else:
            sample_size = self.sample_size
        if sample_size > nrows:
            sample_size = nrows
        if self.max_depth == "auto":
            max_depth = 0
            limit_depth = True
        elif self.max_depth is None:
            max_depth = nrows - 1
            limit_depth = False
        else:
            max_depth = self.max_depth
            limit_depth = False
        if max_depth >= sample_size:
            max_depth = 0
            limit_depth = False

        if self.ncols_per_tree is None:
            ncols_per_tree = 0
        elif self.ncols_per_tree <= 1:
            ncols_tot = 0
            if X_num is not None:
                ncols_tot += X_num.shape[1]
            if X_cat is not None:
                ncols_tot += X_cat.shape[1]
            ncols_per_tree = int(np.ceil(self.ncols_per_tree * ncols_tot))
        else:
            ncols_per_tree = self.ncols_per_tree

        if (self.prob_pick_pooled_gain or self.prob_pick_avg_gain) and self.ndim == 1:
            ncols_tot = (X_num.shape[1] if X_num is not None else 0) + (X_cat.shape[1] if X_cat is not None else 0)
            if self.ntry > ncols_tot:
                warnings.warn("Passed 'ntry' larger than number of columns, will decrease it.")

        if isinstance(self.random_state, np.random.RandomState):
            seed = self.random_state.randint(np.iinfo(np.int32).max)
        else:
            seed = self.random_seed

        self._cpp_obj.fit_model(_get_num_dtype(X_num, sample_weights, column_weights),
                                _get_int_dtype(X_num),
                                X_num, X_cat, ncat, sample_weights, column_weights,
                                ctypes.c_size_t(nrows).value,
                                ctypes.c_size_t(self._ncols_numeric).value,
                                ctypes.c_size_t(self._ncols_categ).value,
                                ctypes.c_size_t(self.ndim).value,
                                ctypes.c_size_t(self.ntry).value,
                                self.coefs,
                                ctypes.c_bool(self.coef_by_prop).value,
                                ctypes.c_bool(self.sample_with_replacement).value,
                                ctypes.c_bool(self.weights_as_sample_prob).value,
                                ctypes.c_size_t(sample_size).value,
                                ctypes.c_size_t(self.ntrees).value,
                                ctypes.c_size_t(max_depth).value,
                                ctypes.c_size_t(ncols_per_tree).value,
                                ctypes.c_bool(limit_depth).value,
                                ctypes.c_bool(self.penalize_range).value,
                                ctypes.c_bool(self.standardize_data).value,
                                self.scoring_metric,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(self.weigh_by_kurtosis).value,
                                ctypes.c_double(self.prob_pick_pooled_gain).value,
                                ctypes.c_double(self.prob_pick_avg_gain).value,
                                ctypes.c_double(self.prob_pick_col_by_range).value,
                                ctypes.c_double(self.prob_pick_col_by_var).value,
                                ctypes.c_double(self.prob_pick_col_by_kurt).value,
                                ctypes.c_double(self.min_gain).value,
                                self.missing_action,
                                self.categ_split_type_,
                                self.new_categ_action,
                                ctypes.c_bool(self.build_imputer).value,
                                ctypes.c_size_t(self.min_imp_obs).value,
                                self.depth_imp,
                                self.weigh_imp_rows,
                                ctypes.c_bool(self.build_imputer).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_uint64(seed).value,
                                ctypes.c_int(nthreads_use).value)
        self.is_fitted_ = True
        self._ntrees = self.ntrees
        return self

    def fit_predict(self, X, column_weights = None, output_outlierness = "score",
                    output_distance = None, square_mat = False, output_imputed = False,
                    categ_cols = None):
        """
        Fit the model in-place and produce isolation or separation depths along the way
        
        See the documentation of other methods ('init', 'fit', 'predict', 'predict_distance')
        for details.

        Note
        ----
        The data must NOT contain any duplicate rows.

        Note
        ----
        This function will be faster at predicting average depths than calling 'fit' + 'predict'
        separately when using full row samples.

        Note
        ----
        If using 'penalize_range' = 'True', the resulting scores/depths from this function might differ a bit
        from those of 'fit' + 'predict' ran separately.

        Note
        ----
        Sample weights are not supported for this method.

        Note
        ----
        When using multiple threads, there can be small differences in the predicted scores or
        average depth or separation/distance between runs due to roundoff error.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data to which to fit the model. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
            If passing a DataFrame, will assume that columns are:
            
             - Numeric, if their dtype is a subtype of NumPy's 'number' or 'datetime64'.
            
             - Categorical, if their dtype is 'object', 'Categorical', or 'bool'. Note that,
               if `Categorical` dtypes are ordered, the order will be ignored here.
            
            Other dtypes are not supported.
        column_weights : None or array(n_features,)
            Sampling weights for each column in 'X'. Ignored when picking columns by deterministic criterion.
            If passing None, each column will have a uniform weight. If used along with kurtosis weights, the
            effect is multiplicative.
            Note that, if passing a DataFrame with both numeric and categorical columns, the column names must
            not be repeated, otherwise the column weights passed here will not end up matching.
        output_outlierness : None or str in ["score", "avg_depth"]
            Desired type of outlierness output. If passing "score", will output standardized outlier score.
            If passing "avg_depth" will output average isolation depth without standardizing.
            If passing 'None', will skip outlierness calculations.
        output_distance : None or str in ["dist", "avg_sep"]
            Type of distance output to produce. If passing "dist", will standardize the average separation
            depths. If passing "avg_sep", will output the average separation depth without standardizing it
            (note that lower separation depth means furthest distance). If passing 'None', will skip distance calculations.
        square_mat : bool
            Whether to produce a full square matrix with the distances. If passing 'False', will output
            only the upper triangular part as a 1-d array in which entry (i,j) with 0 <= i < j < n is located at
            position p(i,j) = (i * (n - (i+1)/2) + j - i - 1).
            Ignored when passing 'output_distance' = 'None'.
        output_imputed : bool
            Whether to output the data with imputed missing values. Model object must have been initialized
            with 'build_imputer' = 'True'.
        categ_cols : None or array-like
            Columns that hold categorical features, when the data is passed as an array or matrix.
            Categorical columns should contain only integer values with a continuous numeration starting at zero,
            with negative values and NaN taken as missing,
            and the array or list passed here should correspond to the column numbers, with numeration starting
            at zero. The maximum categorical value should not exceed 'INT_MAX' (typically :math:`2^{31}-1`).
            This might be passed either at construction time or when calling ``fit`` or variations of ``fit``.
            
            This is ignored when the input is passed as a ``DataFrame`` as then it will consider columns as
            categorical depending on their dtype.

        Returns
        -------
        output : array(n_samples,), or dict
            Requested outputs about isolation depth (outlierness), pairwise separation depth (distance), and/or
            imputed missing values. If passing either 'output_distance' or 'output_imputed', will return a dictionary
            with keys "pred" (array(n_samples,)), "dist" (array(n_samples * (n_samples - 1) / 2,) or array(n_samples, n_samples)),
            "imputed" (array-like(n_samples, n_columns)), according to whether each output type is present.
        """
        self._init(categ_cols)
        nthreads_use = _process_nthreads(self.nthreads, True)
        if (
            (self.sample_size is not None) and
            (self.sample_size != "auto") and
            (self.sample_size != 1) and
            (self.sample_size != nrows)
        ):
            raise ValueError("Cannot use 'fit_predict' when the sample size is limited.")
        if self.sample_with_replacement:
            raise ValueError("Cannot use 'fit_predict' or 'fit_transform' when sampling with replacement.")

        if (output_outlierness is None) and (output_distance is None):
            raise ValueError("Must pass at least one of 'output_outlierness' or 'output_distance'.")

        if output_outlierness is not None:
            assert output_outlierness in ["score", "avg_depth"]

        if output_distance is not None:
            assert output_distance in ["dist", "avg_sep"]

        if output_imputed:
            if self.missing_action == "fail":
                raise ValueError("Cannot impute missing values when using 'missing_action' = 'fail'.")
            if not self.build_imputer:
                msg  = "Trying to impute missing values from object "
                msg += "that was initialized with 'build_imputer' = 'False' "
                msg += "- will force 'build_imputer' to 'True'."
                warnings.warn(msg)
                self.build_imputer = True

        self._reset_obj()
        X_num, X_cat, ncat, sample_weights, column_weights, nrows = self._process_data(X, None, column_weights)

        self._check_can_use_imputer(X_cat)

        if (output_imputed) and (issparse(X_num)):
            msg  = "Imputing missing values from CSC matrix on-the-fly can be very slow, "
            msg += "it's recommended if possible to fit the model first and then pass the "
            msg += "same matrix as CSR to 'transform'."
            warnings.warn(msg)

        if self.max_depth == "auto":
            max_depth = 0
            limit_depth = True
        elif self.max_depth is None:
            max_depth = nrows - 1
        else:
            max_depth = self.max_depth
            limit_depth = False
        if max_depth >= nrows:
            max_depth = 0
            limit_depth = False

        if self.ncols_per_tree is None:
            ncols_per_tree = 0
        elif self.ncols_per_tree <= 1:
            ncols_tot = 0
            if X_num is not None:
                ncols_tot += X_num.shape[1]
            if X_cat is not None:
                ncols_tot += X_cat.shape[1]
            ncols_per_tree = int(np.ceil(self.ncols_per_tree * ncols_tot))
        else:
            ncols_per_tree = self.ncols_per_tree

        if (self.prob_pick_pooled_gain or self.prob_pick_avg_gain) and self.ndim == 1:
            ncols_tot = (X_num.shape[1] if X_num is not None else 0) + (X_cat.shape[1] if X_cat is not None else 0)
            if self.ntry > ncols_tot:
                warnings.warn("Passed 'ntry' larger than number of columns, will decrease it.")

        if isinstance(self.random_state, np.random.RandomState):
            seed = self.random_state.randint(np.iinfo(np.int32).max)
        else:
            seed = self.random_seed

        depths, tmat, dmat, X_num, X_cat = self._cpp_obj.fit_model(_get_num_dtype(X_num, None, column_weights),
                                                                   _get_int_dtype(X_num),
                                                                   X_num, X_cat, ncat, None, column_weights,
                                                                   ctypes.c_size_t(nrows).value,
                                                                   ctypes.c_size_t(self._ncols_numeric).value,
                                                                   ctypes.c_size_t(self._ncols_categ).value,
                                                                   ctypes.c_size_t(self.ndim).value,
                                                                   ctypes.c_size_t(self.ntry).value,
                                                                   self.coefs,
                                                                   ctypes.c_bool(self.coef_by_prop).value,
                                                                   ctypes.c_bool(self.sample_with_replacement).value,
                                                                   ctypes.c_bool(self.weights_as_sample_prob).value,
                                                                   ctypes.c_size_t(nrows).value,
                                                                   ctypes.c_size_t(self.ntrees).value,
                                                                   ctypes.c_size_t(max_depth).value,
                                                                   ctypes.c_size_t(ncols_per_tree).value,
                                                                   ctypes.c_bool(limit_depth).value,
                                                                   ctypes.c_bool(self.penalize_range).value,
                                                                   ctypes.c_bool(self.standardize_data).value,
                                                                   self.scoring_metric,
                                                                   ctypes.c_bool(output_distance is not None).value,
                                                                   ctypes.c_bool(output_distance == "dist").value,
                                                                   ctypes.c_bool(square_mat).value,
                                                                   ctypes.c_bool(output_outlierness is not None).value,
                                                                   ctypes.c_bool(output_outlierness == "score").value,
                                                                   ctypes.c_bool(self.weigh_by_kurtosis).value,
                                                                   ctypes.c_double(self.prob_pick_pooled_gain).value,
                                                                   ctypes.c_double(self.prob_pick_avg_gain).value,
                                                                   ctypes.c_double(self.prob_pick_col_by_range).value,
                                                                   ctypes.c_double(self.prob_pick_col_by_var).value,
                                                                   ctypes.c_double(self.prob_pick_col_by_kurt).value,
                                                                   ctypes.c_double(self.min_gain).value,
                                                                   self.missing_action,
                                                                   self.categ_split_type_,
                                                                   self.new_categ_action,
                                                                   ctypes.c_bool(self.build_imputer).value,
                                                                   ctypes.c_size_t(self.min_imp_obs).value,
                                                                   self.depth_imp,
                                                                   self.weigh_imp_rows,
                                                                   ctypes.c_bool(output_imputed).value,
                                                                   ctypes.c_bool(self.all_perm).value,
                                                                   ctypes.c_uint64(seed).value,
                                                                   ctypes.c_int(nthreads_use).value)
        self.is_fitted_ = True
        self._ntrees = self.ntrees

        if (not output_distance) and (not output_imputed):
            return depths
        else:
            outp = {"pred" : depths}
            if output_distance:
                if square_mat:
                    outp["dist"] = dmat
                else:
                    outp["dist"] = tmat
            if output_imputed:
                outp["imputed"] = self._rearrange_imputed(X, X_num, X_cat)
            return outp

    def _process_data(self, X, sample_weights, column_weights):
        ### TODO: this needs a refactoring after introducing 'categ_cols'

        if X.__class__.__name__ == "DataFrame":

            if self.categ_cols is not None:
                warnings.warn("'categ_cols' is ignored when passing a DataFrame as input.")
                self.categ_cols = None

            ### https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas
            X_num = X.select_dtypes(include = [np.number, np.datetime64]).to_numpy()
            if X_num.dtype not in [ctypes.c_double, ctypes.c_float]:
                X_num = X_num.astype(ctypes.c_double)
            if not _is_col_major(X_num):
                X_num = np.asfortranarray(X_num)
            X_cat = X.select_dtypes(include = [pd.CategoricalDtype, "object", "bool"])
            if (X_num.shape[1] + X_cat.shape[1]) == 0:
                raise ValueError("Input data has no columns of numeric or categorical type.")
            elif (X_num.shape[1] + X_cat.shape[1]) < X.shape[1]:
                cols_num = np.array(X.select_dtypes(include = [np.number, np.datetime64]).columns.values)
                cols_cat = np.array(X_cat.columns.values)
                msg  = "Only numeric and categorical columns are supported."
                msg += " Got passed the following types: ["
                msg += ", ".join([str(X[cl].dtype) for cl in X.columns.values if cl not in cols_num and cl not in cols_cat][:3])
                msg += "]\n(Sample problem columns: ["
                msg += ", ".join([str(cl) for cl in X.columns.values if cl not in cols_num and cl not in cols_cat][:3])
                msg += "])"
                raise ValueError(msg)

            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.array(X.columns.values)

            self._ncols_numeric = X_num.shape[1]
            self._ncols_categ   = X_cat.shape[1]
            self.cols_numeric_  = np.array(X.select_dtypes(include = [np.number, np.datetime64]).columns.values)
            self.cols_categ_    = np.array(X.select_dtypes(include = [pd.CategoricalDtype, "object", "bool"]).columns.values)
            if not self._ncols_numeric:
                X_num = None
            else:
                nrows = X_num.shape[0]

            if not self._ncols_categ:
                X_cat = None
            else:
                nrows = X_cat.shape[0]

            has_ordered = False
            if X_cat is not None:
                self._cat_mapping = [None for cl in range(X_cat.shape[1])]
                for cl in range(X_cat.shape[1]):
                    if (X_cat[X_cat.columns[cl]].dtype.name == "category") and (X_cat[X_cat.columns[cl]].dtype.ordered):
                        has_ordered = True
                    if (not self.recode_categ) and (X_cat[X_cat.columns[cl]].dtype.name == "category"):
                        self._cat_mapping[cl] = np.array(X_cat[X_cat.columns[cl]].cat.categories)
                        X_cat = X_cat.assign(**{X_cat.columns[cl] : X_cat[X_cat.columns[cl]].cat.codes})
                    else:
                        cl, self._cat_mapping[cl] = pd.factorize(X_cat[X_cat.columns[cl]])
                        X_cat = X_cat.assign(**{X_cat.columns[cl] : cl})
                    if (self.all_perm
                        and (self.ndim == 1)
                        and (self.prob_pick_pooled_gain)
                    ):
                        if np.math.factorial(self._cat_mapping[cl].shape[0]) > np.iinfo(ctypes.c_size_t).max:
                            msg  = "Number of permutations for categorical variables is larger than "
                            msg += "maximum representable integer. Try using 'all_perm=False'."
                            raise ValueError(msg)
                    # https://github.com/pandas-dev/pandas/issues/30618
                    if self._cat_mapping[cl].__class__.__name__ == "CategoricalIndex":
                        self._cat_mapping[cl] = self._cat_mapping[cl].to_numpy()
                X_cat = X_cat.to_numpy()
                if X_cat.dtype != ctypes.c_int:
                    X_cat = X_cat.astype(ctypes.c_int)
                if not _is_col_major(X_cat):
                    X_cat = np.asfortranarray(X_cat)
                if has_ordered:
                    warnings.warn("Data contains ordered categoricals. These are treated as unordered.")

        else:
            if len(X.shape) != 2:
                raise ValueError("Input data must be two-dimensional.")

            self.n_features_in_ = X.shape[1]

            X_cat = None
            if self.categ_cols is not None:
                if np.max(self.categ_cols) >= X.shape[1]:
                    raise ValueError("'categ_cols' contains indices higher than the number of columns in 'X'.")
                self.cols_numeric_ = np.setdiff1d(np.arange(X.shape[1]), self.categ_cols)
                if issparse(X) and not isspmatrix_csc(X):
                    X = csc_matrix(X)
                X_cat = X[:, self.categ_cols]
                X = X[:, self.cols_numeric_]

            if X.shape[1]:
                if issparse(X):
                    avoid_sort = False
                    if not isspmatrix_csc(X):
                        warnings.warn("Sparse matrices are only supported in CSC format, will be converted.")
                        X = csc_matrix(X)
                        avoid_sort = True
                    if X.nnz == 0:
                        raise ValueError("'X' has no non-zero entries")

                    if ((X.indptr.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]) or
                        (X.indices.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]) or
                        (X.indptr.dtype != X.indices.dtype) or
                        (X.data.dtype not in [ctypes.c_double, ctypes.c_float])
                    ):
                        X = X.copy()
                    if X.data.dtype not in [ctypes.c_double, ctypes.c_float]:
                        X.data    = X.data.astype(ctypes.c_double)
                    if (X.indptr.dtype != X.indices.dtype) or (X.indices.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]):
                        X.indices = X.indices.astype(ctypes.c_size_t)
                    if (X.indptr.dtype != X.indices.dtype) or (X.indptr.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]):
                        X.indptr  = X.indptr.astype(ctypes.c_size_t)
                    if not avoid_sort:
                        _sort_csc_indices(X)
                
                else:
                    if (X.__class__.__name__ == "ndarray") and (X.dtype not in [ctypes.c_double, ctypes.c_float]):
                        X = X.astype(ctypes.c_double)
                    if (X.__class__.__name__ != "ndarray") or (not _is_col_major(X)):
                        X = np.asfortranarray(X)
                    if X.dtype not in [ctypes.c_double, ctypes.c_float]:
                        X = X.astype(ctypes.c_double)

            self._ncols_numeric = X.shape[1]
            self._ncols_categ   = 0 if (X_cat is None) else X_cat.shape[1]
            if self.categ_cols is None:
                self.cols_numeric_  = np.array([])
            self.cols_categ_    = np.array([])
            self._cat_mapping   = list()

            if (self._ncols_numeric + self._ncols_categ) == 0:
                raise ValueError("'X' has zero columns.")

            if X.shape[1]:
                X_num = X
                nrows = X_num.shape[0]
            else:
                X_num = None
            
            if X_cat is not None:
                if issparse(X_cat):
                    X_cat = X_cat.toarray()
                if np.any(np.isnan(X_cat)):
                    X_cat = X_cat.copy()
                    X_cat[np.isnan(X_cat)] = -1
                if X_cat.dtype != ctypes.c_int:
                    X_cat = X_cat.astype(ctypes.c_int)
                if not _is_col_major(X_cat):
                    X_cat = np.asfortranarray(X_cat)
                self._cat_max_lev = np.max(X_cat, axis=0)
                if np.any(self._cat_max_lev < 0):
                    warnings.warn("Some categorical columns contain only missing values.")
                nrows = X_cat.shape[0]

        if nrows == 0:
            raise ValueError("Input data has zero rows.")
        elif nrows < 3:
            raise ValueError("Input data must have at least 3 rows.")
        elif (self.sample_size is not None) and (self.sample_size != "auto"):
            if self.sample_size > nrows:
                warnings.warn("Input data has fewer rows than sample_size, will forego sub-sampling.")

        if X_cat is not None:
            if self.categ_cols is None:
                ncat = np.array([self._cat_mapping[cl].shape[0] for cl in range(X_cat.shape[1])], dtype = ctypes.c_int)
            else:
                if self._cat_max_lev is None:
                    self._cat_max_lev = []
                if not isinstance(self._cat_max_lev, np.ndarray):
                    self._cat_max_lev = np.array(self._cat_max_lev)
                ncat = (self._cat_max_lev + 1).clip(0)
                if ncat.dtype != ctypes.c_int:
                    ncat = ncat.astype(ctypes.c_int)
        else:
            ncat = None

        if sample_weights is not None:
            sample_weights = np.array(sample_weights).reshape(-1)
            if (X_num is not None) and (X_num.dtype != sample_weights.dtype):
                sample_weights = sample_weights.astype(X_num.dtype)
            if sample_weights.dtype not in [ctypes.c_double, ctypes.c_float]:
                sample_weights = sample_weights.astype(ctypes.c_double)
            if sample_weights.shape[0] != nrows:
                raise ValueError("'sample_weights' has different number of rows than 'X'.")

        ncols = 0
        if X_num is not None:
            ncols += X_num.shape[1]
        if X_cat is not None:
            ncols += X_cat.shape[1]

        if column_weights is not None:
            column_weights = np.array(column_weights).reshape(-1)
            if (X_num is not None) and (X_num.dtype != column_weights.dtype):
                column_weights = column_weights.astype(X_num.dtype)
            if column_weights.dtype not in [ctypes.c_double, ctypes.c_float]:
                column_weights = column_weights.astype(ctypes.c_double)
            if ncols != column_weights.shape[0]:
                raise ValueError("'column_weights' has %d entries, but data has %d columns." % (column_weights.shape[0], ncols))
            if (X_num is not None) and (X_cat is not None):
                column_weights = np.r_[column_weights[X.columns.values == self.cols_numeric_],
                                       column_weights[X.columns.values == self.cols_categ_]]

        if (sample_weights is not None) and (column_weights is not None) and (sample_weights.dtype != column_weights.dtype):
            sample_weights = sample_weights.astype(ctypes.c_double)
            column_weights = column_weights.astype(ctypes.c_double)

        if self.ndim > 1:
            if self.ndim > ncols:
                msg  = "Model was meant to take %d variables for each split, but data has %d columns."
                msg += " Will decrease number of splitting variables to match number of columns."
                msg = msg % (self.ndim, ncols)
                warnings.warn(msg)
                self.ndim = ncols
                if self.ndim < 2:
                    self._is_extended_ = False

        X_num = _copy_if_subview(X_num, False)
        X_cat = _copy_if_subview(X_cat, False)

        return X_num, X_cat, ncat, sample_weights, column_weights, nrows

    def _process_data_new(self, X, allow_csr = True, allow_csc = True, prefer_row_major = False,
                          keep_new_cat_levels = False):
        if X.__class__.__name__ == "DataFrame":
            if ((self.cols_numeric_.shape[0] + self.cols_categ_.shape[0]) > 0) and (self.categ_cols is None):
                if self.categ_cols is None:
                    missing_cols = np.setdiff1d(np.r_[self.cols_numeric_, self.cols_categ_], np.array(X.columns.values))
                    if missing_cols.shape[0] > 0:
                        raise ValueError("Input data is missing %d columns - example: [%s]" % (missing_cols.shape[0], ", ".join(missing_cols[:3])))
                else:
                    if X.shape[1] < (self.cols_numeric_.shape[0] + self.cols_categ_.shape[0]):
                        raise ValueError("Error: expected input with %d columns - got: %d." %
                                         ((self.cols_numeric_.shape[0] + self.cols_categ_.shape[0]), X.shape[1]))

                if self._ncols_numeric > 0:
                    if self.categ_cols is None:
                        X_num = X[self.cols_numeric_].to_numpy()
                    else:
                        X_num = X.iloc[:, self.cols_numeric_].to_numpy()
                    
                    if X_num.dtype not in [ctypes.c_double, ctypes.c_float]:
                        X_num = X_num.astype(ctypes.c_double)
                    if (not prefer_row_major) and (not _is_col_major(X_num)):
                        X_num = np.asfortranarray(X_num)
                    nrows = X_num.shape[0]
                else:
                    X_num = None

                if self._ncols_categ > 0:
                    if self.categ_cols is None:
                        X_cat = X[self.cols_categ_]

                        if (not keep_new_cat_levels) and \
                        (
                            (self.new_categ_action == "impute" and self.missing_action == "impute")
                                or
                            (self.new_categ_action == "weighted" and
                             self.categ_split_type_ != "single_categ"
                             and self.missing_action == "divide")
                        ):
                            for cl in range(self._ncols_categ):
                                X_cat = X_cat.assign(**{
                                    self.cols_categ_[cl] : _encode_categorical(X_cat[self.cols_categ_[cl]],
                                                                               self._cat_mapping[cl])
                                })
                        else:
                            for cl in range(self._ncols_categ):
                                X_cat = X_cat.assign(**{
                                    self.cols_categ_[cl] : pd.Categorical(X_cat[self.cols_categ_[cl]])
                                })
                                new_levs = np.setdiff1d(X_cat[self.cols_categ_[cl]].cat.categories, self._cat_mapping[cl])
                                if new_levs.shape[0]:
                                    self._cat_mapping[cl] = np.r_[self._cat_mapping[cl], new_levs]
                                X_cat = X_cat.assign(**{
                                    self.cols_categ_[cl] : _encode_categorical(X_cat[self.cols_categ_[cl]],
                                                                               self._cat_mapping[cl])
                                })

                    else:
                        X_cat = X.iloc[:, self.categ_cols]
                    
                    X_cat = X_cat.to_numpy()
                    if X_cat.dtype != ctypes.c_int:
                        X_cat = X_cat.astype(ctypes.c_int)
                    if (not prefer_row_major) and (not _is_col_major(X_cat)):
                        X_cat = np.asfortranarray(X_cat)
                    nrows = X_cat.shape[0]
                else:
                    X_cat = None

            elif self._ncols_categ == 0:
                if X.shape[1] < self._ncols_numeric:
                    raise ValueError("Input has different number of columns than data to which model was fit.")
                X_num = X.to_numpy()
                if X_num.dtype not in [ctypes.c_double, ctypes.c_float]:
                    X_num = X_num.astype(ctypes.c_double)
                if (not prefer_row_major) and (not _is_col_major(X_num)):
                    X_num = np.asfortranarray(X_num)
                X_cat = None
                nrows = X_num.shape[0]
            elif self._ncols_numeric == 0:
                if X.shape[1] < self._ncols_categ:
                    raise ValueError("Input has different number of columns than data to which model was fit.")
                X_cat = X.to_numpy()[:, :self._ncols_categ]
                if X_cat.dtype  != ctypes.c_int:
                    X_cat = X_cat.astype(ctypes.c_int)
                if (not prefer_row_major) and (not _is_col_major(X_cat)):
                    X_cat = np.asfortranarray(X_cat)
                X_num = None
                nrows = X_cat.shape[0]
            else:
                nrows = X.shape[0]
                X_num = X.iloc[:, self.cols_numeric_].to_numpy()
                X_cat = X.iloc[:, self.categ_cols].to_numpy()
                if X_num.dtype not in [ctypes.c_double, ctypes.c_float]:
                    X_num = X_num.astype(ctypes.c_double)
                if (not prefer_row_major) and (not _is_col_major(X_num)):
                    X_num = np.asfortranarray(X_num)
                if X_cat.dtype  != ctypes.c_int:
                    X_cat = X_cat.astype(ctypes.c_int)
                if (not prefer_row_major) and (not _is_col_major(X_cat)):
                    X_cat = np.asfortranarray(X_cat)

            if (X_num is not None) and (X_cat is not None) and (_is_col_major(X_num) != _is_col_major(X_cat)):
                if prefer_row_major:
                    X_num = np.ascontiguousarray(X_num)
                    X_cat = np.ascontiguousarray(X_cat)
                else:
                    X_num = np.asfortranarray(X_num)
                    X_cat = np.asfortranarray(X_cat)

        else:
            if (self._ncols_categ > 0) and (self.categ_cols is None):
                raise ValueError("Model was fit to DataFrame with categorical columns, but new input is a numeric array/matrix.")
            if len(X.shape) != 2:
                raise ValueError("Input data must be two-dimensional.")
            if (self.categ_cols is None) and (X.shape[1] < self._ncols_numeric):
                raise ValueError("Input has different number of columns than data to which model was fit.")
            
            if self.categ_cols is None:
                X_cat = None
            else:
                if issparse(X) and (not isspmatrix_csc(X)) and (not isspmatrix_csr(X)):
                    X = csc_matrix(X)
                X_cat = X[:, self.categ_cols]
                if issparse(X_cat):
                    X_cat = X_cat.toarray()
                X = X[:, self.cols_numeric_]

            X_num = None
            if X.shape[1]:
                if issparse(X):
                    avoid_sort = False
                    if isspmatrix_csr(X) and not allow_csr:
                        warnings.warn("Cannot predict from CSR sparse matrix, will convert to CSC.")
                        X = csc_matrix(X)
                        avoid_sort = True
                    elif isspmatrix_csc(X) and not allow_csc:
                        warnings.warn("Method supports sparse matrices only in CSR format, will convert sparse format.")
                        X = csr_matrix(X)
                        avoid_sort = True
                    elif (not isspmatrix_csc(X)) and (not isspmatrix_csr(X)):
                        msg  = "Sparse matrix inputs only supported as "
                        if allow_csc:
                            msg += "CSC"
                            if allow_csr:
                                msg += " or CSR"
                        else:
                            msg += "CSR"
                        msg += " format, will convert to "
                        if allow_csc:
                            msg += "CSC."
                            warnings.warn(msg)
                            X = csc_matrix(X)
                        else:
                            msg += "CSR."
                            warnings.warn(msg)
                            X = csr_matrix(X)
                        avoid_sort = True

                    if ((X.indptr.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]) or
                        (X.indices.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]) or
                        (X.indptr.dtype != X.indices.dtype) or
                        (X.data.dtype not in [ctypes.c_double, ctypes.c_float])
                    ):
                        X = X.copy()
                    if X.data.dtype not in [ctypes.c_double, ctypes.c_float]:
                        X.data    = X.data.astype(ctypes.c_double)
                    if (X.indptr.dtype != X.indices.dtype) or (X.indices.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]):
                        X.indices = X.indices.astype(ctypes.c_size_t)
                    if (X.indptr.dtype != X.indices.dtype) or (X.indptr.dtype not in [ctypes.c_int, np.int64, ctypes.c_size_t]):
                        X.indptr  = X.indptr.astype(ctypes.c_size_t)
                    if not avoid_sort:
                        _sort_csc_indices(X)
                    X_num = X
                
                else:
                    if not isinstance(X, np.ndarray):
                        if prefer_row_major:
                            X = np.array(X)
                        else:
                            X = np.asfortranarray(X)
                    if X.dtype not in [ctypes.c_double, ctypes.c_float]:
                        X = X.astype(ctypes.c_double)
                    if (not prefer_row_major) and (not _is_col_major(X)):
                        X = np.asfortranarray(X)
                    X_num = X
                nrows = X_num.shape[0]

        if X_cat is not None:
            nrows = X_cat.shape[0]
            if np.any(np.isnan(X_cat)):
                X_cat = X_cat.copy()
                X_cat[np.isnan(X_cat)] = -1

            if (X_num is not None) and (isspmatrix_csc(X_num)):
                prefer_row_major = False


            if (self.categ_cols is not None) and np.any(X_cat > self._cat_max_lev.reshape((1,-1))):
                X_cat[X_cat > self._cat_max_lev] = -1
            if X_cat.dtype != ctypes.c_int:
                X_cat = X_cat.astype(ctypes.c_int)
            if (not prefer_row_major) and (not _is_col_major(X_cat)):
                X_cat = np.asfortranarray(X_cat)

        X_num = _copy_if_subview(X_num, prefer_row_major)
        X_cat = _copy_if_subview(X_cat, prefer_row_major)

        if (X_num is not None) and (isspmatrix_csc(X_num)) and (X_cat is not None) and (not _is_col_major(X_cat)):
            X_cat = np.asfortranarray(X_cat)
        if (nrows > 1) and (X_cat is not None) and (X_num is not None) and (not isspmatrix_csc(X_num)):
            if prefer_row_major:
                if _is_row_major(X_num) != _is_row_major(X_cat):
                    X_num = np.ascontiguousarray(X_num)
                    X_cat = np.ascontiguousarray(X_cat)
            else:
                if _is_col_major(X_num) != _is_col_major(X_cat):
                    X_num = np.asfortranarray(X_num)
                    X_cat = np.asfortranarray(X_cat)

        return X_num, X_cat, nrows

    def _rearrange_imputed(self, orig, X_num, X_cat):
        if orig.__class__.__name__ == "DataFrame":
            ncols_imputed = 0
            if X_num is not None:
                if (self.cols_numeric_ is not None) and (self.cols_numeric_.shape[0]):
                    df_num = pd.DataFrame(X_num, columns = self.cols_numeric_ if (self.categ_cols is None) else orig.columns.values[self.cols_numeric_])
                else:
                    df_num = pd.DataFrame(X_num)
                ncols_imputed += df_num.shape[1]
            if X_cat is not None:
                if self.categ_cols is None:
                    df_cat = pd.DataFrame(X_cat, columns = self.cols_categ_)
                    for cl in range(self.cols_categ_.shape[0]):
                        df_cat[self.cols_categ_[cl]] = pd.Categorical.from_codes(df_cat[self.cols_categ_[cl]], self._cat_mapping[cl])
                else:
                    df_cat = pd.DataFrame(X_cat, columns = orig.columns.values[self.categ_cols])
                ncols_imputed += df_cat.shape[1]
            
            if orig.columns.values.shape[0] != ncols_imputed:
                if self.categ_cols is None:
                    cols_new = np.setdiff1d(orig.columns.values, np.r_[self.cols_numeric_, self.cols_categ_])
                else:
                    cols_new = orig.columns[(self._ncols_numeric + self._ncols_categ):]
                if (X_num is not None) and (X_cat is None):
                    out = pd.concat([df_num, orig[cols_new]], axis = 1)
                elif (X_num is None) and (X_cat is not None):
                    out = pd.concat([df_cat, orig[cols_new]], axis = 1)
                else:
                    out = pd.concat([df_num, df_cat, orig[cols_new]], axis = 1)
                out = out[orig.columns.values]
                return out

            if (X_num is not None) and (X_cat is None):
                return df_num[orig.columns.values]
            elif (X_num is None) and (X_cat is not None):
                return df_cat[orig.columns.values]
            else:
                df = pd.concat([df_num, df_cat], axis = 1)
                df = df[orig.columns.values]
                return df

        else: ### not DataFrame

            if issparse(orig):
                outp = orig.copy()
                if (self.categ_cols is None) and (orig.shape[1] == self._ncols_numeric):
                    outp.data[:] = X_num.data
                elif self.categ_cols is None:
                    if isspmatrix_csr(orig):
                        _reconstruct_csr_sliced(
                            outp.data,
                            outp.indptr,
                            X_num.data if (X_num is not None) else np.empty(0, dtype=outp.data.dtype),
                            X_num.indptr if (X_num is not None) else np.zeros(1, dtype=outp.indptr.dtype),
                            outp.shape[0]
                        )
                    else:
                        outp[:, :self._ncols_numeric] = X_num
                else:
                    if isspmatrix_csr(orig):
                        _reconstruct_csr_with_categ(
                            outp.data,
                            outp.indices,
                            outp.indptr,
                            X_num.data if (X_num is not None) else np.empty(0, dtype=outp.data.dtype),
                            X_num.indices if (X_num is not None) else np.empty(0, dtype=outp.indices.dtype),
                            X_num.indptr if (X_num is not None) else np.zeros(1, dtype=outp.indptr.dtype),
                            X_cat,
                            self.cols_numeric_.astype(ctypes.c_size_t) if (self.cols_numeric_ is not None) else np.empty(0, dtype=ctypes.c_size_t),
                            self.categ_cols.astype(ctypes.c_size_t),
                            outp.shape[0], outp.shape[1],
                             _is_col_major(X_cat),
                        )
                    else:
                        if np.any(X_cat < 0):
                            X_cat = X_cat.astype("float")
                            X_cat[X_cat < 0] = np.nan
                        outp[:, self.categ_cols] = X_cat
                        if X_num is not None:
                            outp[:, self.cols_numeric_] = X_num
                return outp
            
            else:
                if (self.categ_cols is None) and (orig.shape[1] == self._ncols_numeric):
                    return X_num
                elif self.categ_cols is None:
                    outp = orig.copy()
                    outp[:, :self._ncols_numeric] = X_num[:, :self._ncols_numeric]
                else:
                    outp = orig.copy()
                    if np.any(X_cat < 0):
                        X_cat = X_cat.astype("float")
                        X_cat[X_cat < 0] = np.nan
                    outp[:, self.categ_cols] = X_cat
                    if X_num is not None:
                        outp[:, self.cols_numeric_] = X_num[:, :self._ncols_numeric]
                return outp


    def predict(self, X, output = "score"):
        """
        Predict outlierness based on average isolation depth or density

        Calculates the approximate depth that it takes to isolate an observation according to the
        fitted model splits, or the average density of the branches in which observations fall.
        Can output either the average depth/density, or a standardized outlier score
        based on whether it takes more or fewer splits than average to isolate observations. In the
        standardized outlier score for density-based metrics, values closer to 1 indicate more outlierness,
        while values closer to 0.5 indicate average outlierness, and close to 0 more averageness
        (harder to isolate).
        When using ``scoring_metric="density"``, the standardized outlier scores are instead unbounded,
        with larger values indicating more outlierness and a natural threshold of zero for determining
        inliers and outliers.

        Note
        ----
        Depending on the model parameters, it might be possible to convert the models to 'treelite' format
        for faster predictions or for easier model serving. See method ``to_treelite`` for details.

        Note
        ----
        If the model was built with 'nthreads>1', this prediction function will
        use OpenMP for parallelization. In a linux setup, one usually has GNU's "gomp" as OpenMP as backend, which
        will hang when used in a forked process - for example, if one tries to call this prediction function from
        'flask'+'gunicorn', which uses process forking for parallelization, it will cause the whole application to freeze;
        and if using kubernetes on top of a different backend such as 'falcon', might cause it to run slower than
        needed or to hang too. A potential fix in these cases is to set the number of threads to 1 in the object
        (e.g. 'model.nthreads = 1'), or to use a different version of this library compiled without OpenMP
        (requires manually altering the 'setup.py' file), or to use a non-GNU OpenMP backend. This should not
        be an issue when using this library normally in e.g. a jupyter notebook.
        
        Note
        ----
        The more threads that are set for the model, the higher the memory requirements will be as each
        thread will allocate an array with one entry per row.
        
        Note
        ----
        In order to save memory when fitting and serializing models, the functionality for outputting
        terminal node number will generate index mappings on the fly for all tree nodes, even if passing only
        1 row, so it's only recommended for batch predictions.

        Note
        ----
        The outlier scores/depth predict functionality is optimized for making predictions on one or a
        few rows at a time - for making large batches of predictions, it might be faster to use the
        'fit_predict' functionality.

        Note
        ----
        If using non-random splits (parameters ``prob_pick_avg_gain``, ``prob_pick_pooled_gain``)
        and/or range penalizations (which are off by default), the distribution of scores might
        not be centered around 0.5.

        Note
        ----
        When making predictions on CSC matrices with many rows using multiple threads, there
        can be small differences between runs due to roundoff error.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Observations for which to predict outlierness or average isolation depth. Can pass
            a NumPy array, Pandas DataFrame, or SciPy sparse CSC or CSR matrix.

            If 'X' is sparse and one wants to obtain the outlier score or average depth or tree
            numbers, it's highly recommended to pass it in CSC format as it will be much faster
            when the number of trees or rows is large.

            While the 'X' used by ``fit`` always needs to be in column-major order, predictions
            can be done on data that is in either row-major or column-major orders, with row-major
            being faster for dense data.
        output : str, one of "score", "avg_depth", "tree_num", "tree_depths"
            Desired type of output. Options are:

            ``"score"``:
                Will output standardized outlier scores. For all scoring metrics, higher values
                indicate more outlierness.

            ``"avg_depth"``:
                Will output unstandardized average isolation depths. For ``scoring_metric="density"``,
                will output the geometric mean instead. See the documentation for ``scoring_metric``,
                for more details about the calculation for other metrics.
                For all scoring metrics, higher values indicate less outlierness.

            ``"tree_num"``:
                Will output the index of the terminal node under each tree in the model.

            ``"tree_depths"``:
                Will output non-standardized per-tree isolation depths or densities or log-densities
                (note that they will not include range penalties from ``penalize_range=True``).
                See the documentation for ``scoring_metric`` for details about the calculation
                for each metrics.

        Returns
        -------
        score : array(n_samples,) or array(n_samples, n_trees)
            Requested output type for each row accoring to parameter 'output' (outlier scores,
            average isolation depth, terminal node indices, or per-tree isolation depths).
        """
        assert self.is_fitted_
        assert output in ["score", "avg_depth", "tree_num", "tree_depths"]
        nthreads_use = _process_nthreads(self.nthreads)
        X_num, X_cat, nrows = self._process_data_new(X, prefer_row_major = True, keep_new_cat_levels = False)
        if (output in ["tree_num", "tree_depths"]) and (self.ndim == 1):
            if self.missing_action == "divide":
                raise ValueError("Cannot output tree numbers/depths when using 'missing_action' = 'divide'.")
            if (self._ncols_categ > 0) and (self.new_categ_action == "weighted") and (self.categ_split_type_ != "single_categ"):
                raise ValueError("Cannot output tree numbers/depths when using 'new_categ_action' = 'weighted'.")
            if (nrows == 1) and (output == "tree_num"):
                warnings.warn("Predicting tree number is slow, not recommended to do for 1 row at a time.")

        depths, tree_num, tree_depths = self._cpp_obj.predict(
                                            _get_num_dtype(X_num, None, None), _get_int_dtype(X_num),
                                            X_num, X_cat, self._is_extended_,
                                            ctypes.c_size_t(nrows).value,
                                            ctypes.c_int(nthreads_use).value,
                                            ctypes.c_bool(output == "score").value,
                                            ctypes.c_bool(output == "tree_num").value,
                                            ctypes.c_bool(output == "tree_depths").value
                                        )

        if output in ["score", "avg_depth"]:
            return depths
        elif output == "tree_depths":
            return tree_depths
        else:
            return tree_num

    def decision_function(self, X):
        """
        Wrapper for 'predict' with 'output=score'

        This function is kept for compatibility with SciKit-Learn.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Observations for which to predict outlierness or average isolation depth. Can pass
            a NumPy array, Pandas DataFrame, or SciPy sparse CSC or CSR matrix.

        Returns
        -------
        score : array(n_samples,)
            Outlier scores for the rows in 'X' (the higher, the most anomalous).
        """
        return self.predict(X, output="score")

    def predict_distance(self, X, output = "dist", square_mat = False, X_ref = None):
        """
        Predict approximate distances between points

        Predict approximate pairwise distances between points or individual distances between
        two sets of points based on how many
        splits it takes to separate them. Can output either the average number of paths,
        or a standardized metric (in the same way as the outlier score) in which values closer
        to zero indicate nearer points, closer to one further away points, and closer to 0.5
        average distance.

        Note
        ----
        The more threads that are set for the model, the higher the memory requirement will be as each
        thread will allocate an array with one entry per combination.

        Note
        ----
        While in theory it should be possible to make this computation relatively fast by precomputing
        results for each pair of terminal nodes in a given tree, the procedure here is based on
        calculating this metric on-the-fly as each pair of observations is passed down a tree, which
        makes it relatively slow, and thus not recommended for real-time usage.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Observations for which to calculate approximate pairwise distances,
            or first group for distances between sets of points. Can pass
            a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
        output : str, one of "dist", "avg_sep"
            Type of output to produce. If passing "dist", will standardize the average separation
            depths. If passing "avg_sep", will output the average separation depth without standardizing it
            (note that lower separation depth means furthest distance).
        square_mat : bool
            Whether to produce a full square matrix with the pairwise distances. If passing 'False', will output
            only the upper triangular part as a 1-d array in which entry (i,j) with 0 <= i < j < n is located at
            position p(i,j) = (i * (n - (i+1)/2) + j - i - 1).
            Ignored when passing ``X_ref``.
        X_ref : array or array-like (n_ref, n_features)
            Second group of observations. If passing it, will calculate distances between each point in
            ``X`` and each point in ``X_ref``. If passing ``None`` (the default), will calculate
            pairwise distances between the points in ``X``.
            Must be of the same type as ``X`` (e.g. array, DataFrame, CSC).

        Returns
        -------
        dist : array(n_samples * (n_samples - 1) / 2,) or array(n_samples, n_samples) or array(n_samples, n_ref)
            Approximate distances or average separation depth between points, according to
            parameter 'output'. Shape and size depends on parameter ``square_mat``, or ``X_ref`` if passed.
        """
        assert self.is_fitted_
        assert output in ["dist", "avg_sep"]
        nthreads_use = _process_nthreads(self.nthreads)

        if X_ref is None:
            nobs_group1 = 0
        else:
            if X.__class__ != X_ref.__class__:
                raise ValueError("'X' and 'X_ref' must be of the same class.")
            nobs_group1 = X.shape[0]
            if X.__class__.__name__ == "DataFrame":
                X = X.append(X_ref, ignore_index = True)
            elif issparse(X):
                X = sp_vstack([X, X_ref])
            else:
                X = np.vstack([X, X_ref])

        X_num, X_cat, nrows = self._process_data_new(X, allow_csr = False, prefer_row_major = False, keep_new_cat_levels = False)
        if nrows == 1:
            raise ValueError("Cannot calculate pairwise distances for only 1 row.")

        tmat, dmat, rmat = self._cpp_obj.dist(_get_num_dtype(X_num, None, None), _get_int_dtype(X_num),
                                              X_num, X_cat, self._is_extended_,
                                              ctypes.c_size_t(nrows).value,
                                              ctypes.c_int(nthreads_use).value,
                                              ctypes.c_bool(self.assume_full_distr).value,
                                              ctypes.c_bool(output == "dist").value,
                                              ctypes.c_bool(square_mat).value,
                                              ctypes.c_size_t(nobs_group1).value)
        
        if X_ref is not None:
            return rmat
        elif square_mat:
            return dmat
        else:
            return tmat

    def transform(self, X):
        """
        Impute missing values in the data using isolation forest model

        Note
        ----
        In order to use this functionality, the model must have been built with imputation capabilities ('build_imputer' = 'True').

        Note
        ----
        Categorical columns, if imputed with a model fit to a DataFrame, will always come out
        with pandas categorical dtype.

        Note
        ----
        The input may contain new columns (i.e. not present when the model was fitted),
        which will be output as-is.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data for which missing values should be imputed. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSR matrix.

            If the model was fit to a DataFrame with categorical columns, must also be a DataFrame.

        Returns
        -------
        X_imputed : array or array-like (n_samples, n_features)
            Object of the same type and dimensions as 'X', but with missing values already imputed. Categorical
            columns will be output as pandas's 'Categorical' regardless of their dtype in 'X'.
        """
        assert self.is_fitted_
        if not self.build_imputer:
            raise ValueError("Cannot impute missing values with model that was built with 'build_imputer' =  'False'.")
        if self.missing_action == "fail":
            raise ValueError("Cannot impute missing values when using 'missing_action' = 'fail'.")
        nthreads_use = _process_nthreads(self.nthreads)

        X_num, X_cat, nrows = self._process_data_new(X, allow_csr = True, allow_csc = False, prefer_row_major = True, keep_new_cat_levels = False)
        if X.__class__.__name__ != "DataFrame":
            if X_num is not None:
                if X_num.shape[1] == self._ncols_numeric:
                    X_num = X_num.copy()
                else:
                    X_num = X_num[:, :self._ncols_numeric].copy()
            if X_cat is not None:
                X_cat = X_cat.copy()
        X_num, X_cat = self._cpp_obj.impute(_get_num_dtype(X_num, None, None), _get_int_dtype(X_num),
                                            X_num, X_cat,
                                            ctypes.c_bool(self._is_extended_).value,
                                            ctypes.c_size_t(nrows).value,
                                            ctypes.c_int(nthreads_use).value)
        return self._rearrange_imputed(X, X_num, X_cat)

    def fit_transform(self, X, y = None, column_weights = None, categ_cols = None):
        """
        SciKit-Learn pipeline-compatible version of 'fit_predict'

        Will fit the model and output imputed missing values. Intended to be used as part of SciKit-learn
        pipelining. Note that this is just a wrapper over 'fit_predict' with parameter 'output_imputed' = 'True'.
        See the documentation of 'fit_predict' for details.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data to which to fit the model and whose missing values need to be imputed. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix (see the documentation of ``fit`` for more details).

            If the model was fit to a DataFrame with categorical columns, must also be a DataFrame.
        y : None
            Not used. Kept for compatibility with SciKit-Learn.
        column_weights : None or array(n_features,)
            Sampling weights for each column in 'X'. Ignored when picking columns by deterministic criterion.
            If passing None, each column will have a uniform weight. If used along with kurtosis weights, the
            effect is multiplicative.
            Note that, if passing a DataFrame with both numeric and categorical columns, the column names must
            not be repeated, otherwise the column weights passed here will not end up matching.
        categ_cols : None or array-like
            Columns that hold categorical features, when the data is passed as an array or matrix.
            Categorical columns should contain only integer values with a continuous numeration starting at zero,
            with negative values and NaN taken as missing,
            and the array or list passed here should correspond to the column numbers, with numeration starting
            at zero. The maximum categorical value should not exceed 'INT_MAX' (typically :math:`2^{31}-1`).
            This might be passed either at construction time or when calling ``fit`` or variations of ``fit``.
            
            This is ignored when the input is passed as a ``DataFrame`` as then it will consider columns as
            categorical depending on their dtype.

        Returns
        -------
        imputed : array-like(n_samples, n_columns)
            Input data 'X' with missing values imputed according to the model.
        """
        if (self.sample_size is None) or (self.sample_size == "auto"):
            outp = self.fit_predict(X = X, column_weights = column_weights, output_imputed = True)
            return outp["imputed"]
        else:
            self.fit(X = X, column_weights = column_weights)
            return self.transform(X)

    def partial_fit(self, X, sample_weights = None, column_weights = None):
        """
        Add additional (single) tree to isolation forest model

        Adds a single tree fit to the full (non-subsampled) data passed here. Must
        have the same columns as previously-fitted data.

        Note
        ----
        If constructing trees with different sample sizes, the outlier scores will not be centered around
        0.5 and might have a very skewed distribution. The standardizing constant for the scores will be
        taken according to the sample size passed in the construction argument (if that is ``None`` or
        ``"auto"``, will then set it as the sample size of the first tree).

        Note
        ----
        This function is not thread-safe - that is, it will produce problems if one tries to call
        this function on the same model object in parallel through e.g. ``joblib`` with a shared-memory
        backend (which is not the default for joblib).

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data to which to fit the new tree. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
            If passing a DataFrame, will assume that columns are:
            
             - Numeric, if their dtype is a subtype of NumPy's 'number' or 'datetime64'.
            
             - Categorical, if their dtype is 'object', 'Categorical', or 'bool'. Note that,
               if `Categorical` dtypes are ordered, the order will be ignored here.
               Categorical columns, if any, may have new categories.
            
            Other dtypes are not supported.
        sample_weights : None or array(n_samples,)
            Sample observation weights for each row of 'X', with higher weights indicating
            distribution density (i.e. if the weight is two, it has the same effect of including the same data
            point twice). If not 'None', model must have been built with 'weights_as_sample_prob' = 'False'.
        column_weights : None or array(n_features,)
            Sampling weights for each column in 'X'. Ignored when picking columns by deterministic criterion.
            If passing None, each column will have a uniform weight. If used along with kurtosis weights, the
            effect is multiplicative.

        Returns
        -------
        self : obj
            This object.
        """
        if not self.is_fitted_:
            self._init()
        if (sample_weights is not None) and (self.weights_as_sample_prob):
            raise ValueError("Cannot use sampling weights with 'partial_fit'.")

        if not self.is_fitted_:
            trees_restore = self.ntrees
            try:
                self.ntrees = 1
                self.fit(X = X, sample_weights = sample_weights, column_weights = column_weights)
            finally:
                self.ntrees = trees_restore
            return self
        
        X_num, X_cat, nrows = self._process_data_new(X, allow_csr = False, prefer_row_major = False, keep_new_cat_levels = True)
        if sample_weights is not None:
            sample_weights = np.array(sample_weights).reshape(-1)
            if (X_num is not None) and (X_num.dtype != sample_weights.dtype):
                sample_weights = sample_weights.astype(X_num.dtype)
            if sample_weights.dtype not in [ctypes.c_double, ctypes.c_float]:
                sample_weights = sample_weights.astype(ctypes.c_double)
            assert sample_weights.shape[0] == X.shape[0]
        if column_weights is not None:
            column_weights = np.array(column_weights).reshape(-1)
            if (X_num is not None) and (X_num.dtype != column_weights.dtype):
                column_weights = column_weights.astype(X_num.dtype)
            if column_weights.dtype not in [ctypes.c_double, ctypes.c_float]:
                column_weights = column_weights.astype(ctypes.c_double)
            assert column_weights.shape[0] == X.shape[1]
        if (sample_weights is not None) and (column_weights is not None) and (sample_weights.dtype != column_weights.dtype):
            sample_weights = sample_weights.astype(ctypes.c_double)
            column_weights = column_weights.astype(ctypes.c_double)
        ncat = None
        if self._ncols_categ > 0:
            ncat = np.array([arr.shape[0] for arr in self._cat_mapping]).astype(ctypes.c_int)
        if (ncat is None) and (X_cat is not None) and (X_cat.shape[1]):
            ncat = X_cat.max(axis=0).clip(0)
        if self.max_depth == "auto":
            max_depth = 0
            limit_depth = True
        elif self.max_depth is None:
            max_depth = nrows - 1
        else:
            max_depth = self.max_depth
            limit_depth = False

        if self.ncols_per_tree is None:
            ncols_per_tree = 0
        elif self.ncols_per_tree <= 1:
            ncols_tot = 0
            if X_num is not None:
                ncols_tot += X_num.shape[1]
            if X_cat is not None:
                ncols_tot += X_cat.shape[1]
            ncols_per_tree = int(np.ceil(self.ncols_per_tree * ncols_tot))
        else:
            ncols_per_tree = self.ncols_per_tree

        if (self.prob_pick_pooled_gain or self.prob_pick_avg_gain) and self.ndim == 1:
            ncols_tot = (X_num.shape[1] if X_num is not None else 0) + (X_cat.shape[1] if X_cat is not None else 0)
            if self.ntry > ncols_tot:
                warnings.warn("Passed 'ntry' larger than number of columns, will decrease it.")

        if isinstance(self.random_state, np.random.RandomState):
            seed = self.random_state.randint(np.iinfo(np.int32).max)
        else:
            seed = self.random_seed
        seed += self._ntrees

        self._cpp_obj.fit_tree(_get_num_dtype(X_num, sample_weights, column_weights),
                               _get_int_dtype(X_num),
                               X_num, X_cat, ncat, sample_weights, column_weights,
                               ctypes.c_size_t(nrows).value,
                               ctypes.c_size_t(self._ncols_numeric).value,
                               ctypes.c_size_t(self._ncols_categ).value,
                               ctypes.c_size_t(self.ndim).value,
                               ctypes.c_size_t(self.ntry).value,
                               self.coefs,
                               ctypes.c_bool(self.coef_by_prop).value,
                               ctypes.c_size_t(max_depth).value,
                               ctypes.c_size_t(ncols_per_tree).value,
                               ctypes.c_bool(limit_depth).value,
                               ctypes.c_bool(self.penalize_range).value,
                               ctypes.c_bool(self.standardize_data),
                               ctypes.c_bool(self.weigh_by_kurtosis).value,
                               ctypes.c_double(self.prob_pick_pooled_gain).value,
                               ctypes.c_double(self.prob_pick_avg_gain).value,
                               ctypes.c_double(getattr(self, "prob_pick_col_by_range", 0.)).value,
                               ctypes.c_double(getattr(self, "prob_pick_col_by_var", 0.)).value,
                               ctypes.c_double(getattr(self, "prob_pick_col_by_kurt", 0.)).value,
                               ctypes.c_double(self.min_gain).value,
                               self.missing_action,
                               self.categ_split_type_,
                               self.new_categ_action,
                               ctypes.c_bool(self.build_imputer).value,
                               ctypes.c_size_t(self.min_imp_obs).value,
                               self.depth_imp,
                               self.weigh_imp_rows,
                               ctypes.c_bool(self.all_perm).value,
                               ctypes.c_int(seed).value)
        self._ntrees += 1
        return self

    def get_num_nodes(self):
        """
        Get number of nodes per tree

        Gets the number of nodes per tree, along with the number of terminal nodes.

        Returns
        -------
        nodes : tuple(array(n_trees,), array(n_trees,))
            A tuple in which the first element denotes the total number of nodes
            in each tree, and the second element denotes the number of terminal
            nodes. Both are returned as arrays having one entry per tree.
        """
        assert self.is_fitted_
        n_nodes, n_terminal = self._cpp_obj.get_n_nodes(ctypes.c_bool(self._is_extended_).value,
                                                        ctypes.c_int(self.nthreads).value)
        return n_nodes, n_terminal

    def append_trees(self, other):
        """
        Appends isolation trees from another Isolation Forest model into this one

        This function is intended for merging models **that use the same hyperparameters** but
        were fitted to different subsets of data.

        In order for this to work, both models must have been fit to data in the same format - 
        that is, same number of columns, same order of the columns, and same column types, although
        not necessarily same object classes (e.g. can mix ``np.array`` and ``scipy.sparse.csc_matrix``).

        If the data has categorical variables, the models should have been built with parameter
        ``recode_categ=False`` in the class constructor,
        and the categorical columns passed as type ``pd.Categorical`` with the same encoding -
        otherwise different models might be using different encodings for each categorical column,
        which will not be preserved as only the trees will be appended without any associated metadata.

        Note
        ----
        This function will not perform any checks on the inputs, and passing two incompatible
        models (e.g. fit to different numbers of columns) will result in wrong results and
        potentially crashing the Python process when using it.

        Note
        ----
        This function is not thread-safe - that is, it will produce problems if one tries to call
        this function on the same model object in parallel through e.g. ``joblib`` with a shared-memory
        backend (which is not the default for joblib).

        Parameters
        ----------
        other : IsolationForest
            Another Isolation Forest model from which trees will be appended to this model.
            It will not be modified during the call to this function.

        Returns
        -------
        self : obj
            This object.
        """
        assert self.is_fitted_
        assert other.is_fitted_
        assert isinstance(other, IsolationForest)

        if (self._is_extended_) != (other._is_extended_):
            raise ValueError("Cannot mix extended and regular isolation forest models (ndim=1).")

        if self.cols_categ_.shape[0]:
            warnings.warn("Merging models with categorical features might give wrong results.")

        self._cpp_obj.append_trees_from_other(other._cpp_obj, self._is_extended_)
        self._ntrees += other._ntrees

        return self

    def export_model(self, file, add_metada_file = False):
        """
        Export Isolation Forest model

        Save Isolation Forest model to a serialized file along with its
        metadata, in order to be re-used in Python or in the R or the C++ versions of this package.
        
        This function is not suggested to be used for passing models to and from Python -
        in such case, one can use ``pickle`` instead, although the function
        still works correctly for serializing objects between Python processes.
        
        Note that, if the model was fitted to a ``DataFrame``, the column names must be
        something exportable as JSON, and must be something that R could
        use as column names (for example, using integers as column names is valid in pandas
        but not in R).
        
        Can optionally generate a JSON file with metadata such as the column names and the
        levels of categorical variables, which can be inspected visually in order to detect
        potential issues (e.g. character encoding) or to make sure that the columns are of
        the right types.

        The metadata file, if produced, will contain, among other things, the encoding that was used for
        categorical columns - this is under ``data_info.cat_levels``, as an array of arrays by column,
        with the first entry for each column corresponding to category 0, second to category 1,
        and so on (the C++ version takes them as integers). When passing ``categ_cols``, there
        will be no encoding but it will save the maximum category integer and the column
        numbers instead of names.
        
        The serialized file can be used in the C++ version by reading it as a binary file
        and de-serializing its contents using the C++ function 'deserialize_combined'
        (recommended to use 'inspect_serialized_object' beforehand).

        Be aware that this function will write raw bytes from memory as-is without compression,
        so the file sizes can end up being much larger than when using ``pickle``.
        
        The metadata is not used in the C++ version, but is necessary for the R and Python versions.

        Note
        ----
        While in earlier versions of this library this functionality used to be faster than
        ``pickle``, starting with version 0.3.0, this function and ``pickle`` should have
        similar timings and it's recommended to use ``pickle`` for serializing objects
        across Python processes.

        Note
        ----
        **Important:** The model treats boolean variables as categorical. Thus, if the model was fit
        to a ``DataFrame`` with boolean columns, when importing this model into C++, they need to be
        encoded in the same order - e.g. the model might encode ``True`` as zero and ``False``
        as one - you need to look at the metadata for this. Also, if using some of Pandas' own
        Boolean types, these might end up as non-boolean categorical, and if importing the model into R,
        you might need to pass values as e.g. ``"True"`` instead of ``TRUE`` (look at the ``.metadata``
        file to determine this).

        Note
        ----
        The files produced by this function will be compatible between:
        
        * Different operating systems.

        * Different compilers.

        * Different Python/R versions.

        * Systems with different 'size_t' width (e.g. 32-bit and 64-bit),
          as long as the file was produced on a system that was either 32-bit or 64-bit,
          and as long as each saved value fits within the range of the machine's 'size_t' type.

        * Systems with different 'int' width,
          as long as the file was produced on a system that was 16-bit, 32-bit, or 64-bit,
          and as long as each saved value fits within the range of the machine's int type.

        * Systems with different bit endianness (e.g. x86 and PPC64 in non-le mode).

        * Versions of this package from 0.3.0 onwards, **but only forwards compatible**
          (e.g. a model saved with versions 0.3.0 to 0.3.5 can be loaded under version
          0.3.6, but not the other way around, and attempting to do so will cause crashes
          and memory curruptions without an informative error message). **This last point applies
          also to models saved through pickle**. Note that loading a
          model produced by an earlier version of the library might be slightly slower.

        But will not be compatible between:

        * Systems with different floating point numeric representations
          (e.g. standard IEEE754 vs. a base-10 system).

        * Versions of this package earlier than 0.3.0.

        This pretty much guarantees that a given file can be serialized and de-serialized
        in the same machine in which it was built, regardless of how the library was compiled.

        Reading a serialized model that was produced in a platform with different
        characteristics (e.g. 32-bit vs. 64-bit) will be much slower.

        Note
        ----
        On Windows, if compiling this library with a compiler other than MSVC or MINGW,
        there might be issues exporting models larger than 2GB.

        Parameters
        ----------
        file : str
            The output file path into which to export the model. Must be a file name, not a
            file handle.
        add_metada_file : bool
            Whether to generate a JSON file with metadata, which will have
            the same name as the model but will end in '.metadata'. This file is not used by the
            de-serialization function, it's only meant to be inspected manually, since such contents
            will already be written in the produced model file.

        Returns
        -------
        self : obj
            This object.
        """
        assert self.is_fitted_
        file = os.path.expanduser(file)
        metadata = self._export_metadata()
        if add_metada_file:
            with open(file + ".metadata", "w") as of:
                json.dump(metadata, of, indent=4)
        metadata = json.dumps(metadata)
        metadata = metadata.encode('utf-8')
        self._cpp_obj.serialize_obj(file, metadata, self.ndim > 1, has_imputer=self.build_imputer)
        return self

    @staticmethod
    def import_model(file):
        """
        Load an Isolation Forest model exported from R or Python

        Loads a serialized Isolation Forest model as produced and exported
        by the function ``export_model`` or by the R version of this package.
        Note that the metadata must be something
        importable in Python - e.g. column names must be valid for Pandas.
        
        It's recommended to generate a '.metadata' file (passing ``add_metada_file=True``) and
        to visually inspect said file in any case.

        See the documentation for ``export_model`` for details about compatibility
        of the generated files across different machines and versions.

        Note
        ----
        This is a static class method - that is, it should be called like this:
            ``iso = IsolationForest.import_model(...)``
        (i.e. no parentheses after `IsolationForest`)

        Note
        ----
        While in earlier versions of this library this functionality used to be faster than
        ``pickle``, starting with version 0.3.0, this function and ``pickle`` should have
        similar timings and it's recommended to use ``pickle`` for serializing objects
        across Python processes.
        
        Parameters
        ----------
        file : str
            The input file path containing an exported model along with its metadata file.
            Must be a file name, not a file handle.

        Returns
        -------
        iso : IsolationForest
            An Isolation Forest model object reconstructed from the serialized file
            and ready to use.
        """
        file = os.path.expanduser(file)
        obj = IsolationForest()
        metadata = obj._cpp_obj.deserialize_obj(file)
        metadata = json.loads(metadata)
        obj._take_metadata(metadata)
        return obj

    def generate_sql(self, enclose="doublequotes", output_tree_num = False, tree = None,
                     table_from = None, select_as = "outlier_score",
                     column_names = None, column_names_categ = None):
        """
        Generate SQL statements representing the model prediction function

        Generate SQL statements - either separately per tree (the default),
        for a single tree if needed (if passing ``tree``), or for all trees
        concatenated together (if passing ``table_from``). Can also be made
        to output terminal node numbers (numeration starting at zero).

        Note
        ----
        Making predictions through SQL is much less efficient than from the model
        itself, as each terminal node will have to check all of the conditions
        that lead to it instead of passing observations down a tree.

        Note
        ----
        If constructed with the default arguments, the model will not perform any
        sub-sampling, which can lead to very big trees. If it was fit to a large
        dataset, the generated SQL might consist of gigabytes of text, and might
        lay well beyond the character limit of commands accepted by SQL vendors.

        Note
        ----
        The generated SQL statements will not include range penalizations, thus
        predictions might differ from calls to ``predict`` when using
        ``penalize_range=True``.

        Note
        ----
        The generated SQL statements will only include handling of missing values
        when using ``missing_action="impute"``. When using the single-variable
        model with categorical variables + subset splits, the rule buckets might be
        incomplete due to not including categories that were not present in a given
        node - this last point can be avoided by using ``new_categ_action="smallest"``,
        ``new_categ_action="random"``, or ``missing_action="impute"`` (in the latter
        case will treat them as missing, but the ``predict`` function might treat
        them differently).

        Note
        ----
        The resulting statements will include all the tree conditions as-is,
        with no simplification. Thus, there might be lots of redundant conditions
        in a given terminal node (e.g. "X > 2" and "X > 1", the second of which is
        redundant).

        Note
        ----
        If using ``scoring_metric="density"`` or ``scoring_metric="boxed_ratio"`` plus
        ``output_tree_num=False``, the outputs will correspond to the logarithm of the
        density rather than the density.

        Parameters
        ----------
        enclose : str
            With which symbols to enclose the column names in the select statement
            so as to make them SQL compatible in case they include characters like dots.
            Options are:

            ``"doublequotes"``:
                Will enclose them as ``"column_name"`` - this will work for e.g. PostgreSQL.

            ``"squarebraces"``:
                Will enclose them as ``[column_name]`` - this will work for e.g. SQL Server.

            ``"none"``:
                Will output the column names as-is (e.g. ``column_name``)
        output_tree_num : bool
            Whether to make the statements return the terminal node number
            instead of the isolation depth. The numeration will start at zero.
        tree : int or None
            Tree for which to generate SQL statements. If passed, will generate
            the statements only for that single tree. If passing 'None', will
            generate statements for all trees in the model.
        table_from : str or None
            If passing this, will generate a single select statement for the
            outlier score from all trees, selecting the data from the table
            name passed here. In this case, will always output the outlier
            score, regardless of what is passed under ``output_tree_num``.
        select_as : str
            Alias to give to the generated outlier score in the select statement.
            Ignored when not passing ``table_from``.
        column_names : None or list[str]
            Column names to use for the **numeric** columns.
            If not passed and the model was fit to a ``DataFrame``, will use the column
            names from that ``DataFrame``, which can be found under ``self.cols_numeric_``.
            If not passing it and the model was fit to data in a format other than
            ``DataFrame``, the columns will be named "column_N" in the resulting
            SQL statement. Note that the names will be taken verbatim - this function will
            not do any checks for whether they constitute valid SQL or not, and will not
            escape characters such as double quotation marks.
        column_names_categ : None or list[str]
            Column names to use for the **categorical** columns.
            If not passed, will use the column names from the ``DataFrame`` to which the
            model was fit. These can be found under ``self.cols_categ_``.

        Returns
        -------
        sql : list[str] or str
            A list of SQL statements for each tree as strings, or the SQL statement
            for a single tree if passing 'tree', or a single select-from SQL statement
            with all the trees concatenated if passing ``table_from``.
        """
        assert self.is_fitted_
        
        single_tree = False
        if tree is not None:
            if isinstance(tree, float):
                tree = int(tree)
            assert isinstance(tree, int)
            assert tree >= 0
            assert tree < self._ntrees
            single_tree = True
        else:
            tree = 0
        output_tree_num = bool(output_tree_num)

        if self._ncols_numeric:
            if column_names is not None:
                if len(column_names) != self._ncols_numeric:
                    raise ValueError("'column_names' must have %d entries." % self._ncols_numeric)
            else:
                if self.cols_numeric_.shape[0]:
                    column_names = self.cols_numeric_
                else:
                    column_names = ["column_" + str(cl) for cl in range(self._ncols_numeric)]
        else:
            column_names = []

        if self.cols_categ_.shape[0]:
            if column_names_categ is not None:
                if len(column_names_categ) != self.cols_categ_.shape[0]:
                    raise ValueError("'column_names_categ' must have %d entries." % self.cols_categ_.shape[0])
            else:
                column_names_categ = self.cols_categ_
            categ_levels = [[str(lev) for lev in mp] for mp in self._cat_mapping]
        else:
            column_names_categ = []
            categ_levels = []

        assert enclose in ["doublequotes", "squarebraces", "none"]
        if enclose != "none":
            enclose_left  = '"' if (enclose == "doublequotes") else '['
            enclose_right = '"' if (enclose == "doublequotes") else ']'
            column_names = [enclose_left + cl + enclose_right for cl in column_names]
            column_names_categ = [enclose_left + cl + enclose_right for cl in column_names_categ]

        nthreads_use = _process_nthreads(self.nthreads)

        out = [s for s in self._cpp_obj.generate_sql(self.ndim > 1,
                                                     column_names, column_names_categ, categ_levels,
                                                     output_tree_num, single_tree, tree, nthreads_use)]
        if single_tree:
            return out[0]
        return out

    def to_treelite(self, use_float32 = False):
        """
        Convert model to 'treelite' format

        Converts an IsolationForest model to a 'treelite' object, which can be compiled into a small
        standalone runtime library for smaller models and usually faster predictions:

            https://treelite.readthedocs.io/en/latest/index.html


        A couple notes about this conversion:

            - It is only possible to convert to 'treelite' when using ``ndim=1`` (which is not the default).
            - The 'treelite' and 'treelite_runtime' libraries must be installed for this to work.
            - The options for handling missing values in 'treelite' are more limited.
              This function will always produce models that force ``missing_action="impute"``, regardless
              of how the IsolationForest model itself handles them.
            - The options for handling unseen categories in categorical variables are also more
              limited in 'treelite'. It's not possible to convert models that use ``new_categ_action="weighted"``,
              and categories that were not present within the training data (which are not meant to be passed to
              'treelite') will always be sent to the right side of the split, which might produce different
              results from ``predict``.
            - Some features such as range penalizations will not be kept in the 'treelite' model.
            - While this library always uses C 'double' precision (typically 'float64') for model objects and
              prediction outputs, 'treelite' (a) can use 'float32' precision, (b) converts floating point numbers
              to a decimal representation and back to floating point; which combined can result in some precision
              loss which leads to producing slightly different predictions from the ``predict`` function in this
              package.
            - The output returned from a compiled 'treelite' model when calling ``predict`` will be the
              average isolation depth, as it does not (yet?) support the standardized outlier score from
              isolation forests. If using ``scoring_metric="density"``, the output should match with the
              standardized outlier score however.
            - If the model was fit to a DataFrame having a mixture of numerical and categorical columns, the
              resulting 'treelite' object will be built assuming all the numerical columns come before the
              categorical columns, regardless of which order they originally had in the data that was passed to
              'fit'. In such cases, it is possible to check the order of the columns under attributes
              ``self.cols_numeric_`` and ``self.cols_categ_``.
            - Categorical columns in 'treelite' are passed as integer values. if the model was fit to a DataFrame
              with categorical columns, the encoding that is used can be found under ``self._cat_mapping``.
            - The 'treelite' object returned by this function will not yet have been compiled. It's necessary to
              call ``compile`` and ``export_lib`` afterwards in order to be able to use it.

        Parameters
        ----------
        use_float32 : bool
            Whether to use 'float32' type for the model. This is typically faster but has less precision
            than the typical 'float64' (outside of this conversion, models from this library always use
            'float64').

        Returns
        -------
        model : obj
            A 'treelite' model object.
        """
        assert self.ndim == 1
        assert self.is_fitted_

        if (self._ncols_categ and
            self.categ_split_type_ != "single_categ" and
            self.new_categ_action not in ["smallest", "random"]
        ):
            raise ValueError("Cannot convert to 'treelite' with the current parameters for categorical columns.")

        if self.missing_action != "impute":
            warnings.warn("'treelite' conversion will switch 'missing_action' to 'impute'.")
        if self.penalize_range:
            warnings.warn("'penalize_range' is ignored (assumed 'False') for 'treelite' conversion.")

        import treelite

        float_dtype = 'float32' if bool(use_float32) else 'float64'

        num_node_info = np.empty(6, dtype=ctypes.c_double)
        n_nodes = self.get_num_nodes()[0]

        if self.categ_cols is None:
            mapping_num_cols = np.arange(self._ncols_numeric)
            mapping_cat_cols = np.arange(self._ncols_numeric, self._ncols_numeric + self._ncols_categ)
        else:
            mapping_num_cols = np.setdiff1d(np.arange(self._ncols_numeric + self._ncols_categ),
                                            self.categ_cols, assume_unique=True)
            mapping_cat_cols = np.array(self.categ_cols).reshape(-1).astype(int)

        builder = treelite.ModelBuilder(
            num_feature = self._ncols_numeric + self._ncols_categ,
            average_tree_output = True,
            threshold_type = float_dtype,
            leaf_output_type = float_dtype
        )
        for tree_ix in range(self._ntrees):
            tree = treelite.ModelBuilder.Tree(threshold_type = float_dtype, leaf_output_type = float_dtype)
            for node_ix in range(n_nodes[tree_ix]):
                cat_left = self._cpp_obj.get_node(tree_ix, node_ix, num_node_info)
                
                if num_node_info[0] == 1:
                    tree[node_ix].set_leaf_node(num_node_info[1], leaf_value_type = float_dtype)
                
                elif num_node_info[0] == 0:
                    tree[node_ix].set_numerical_test_node(
                        feature_id = mapping_num_cols[int(num_node_info[1])],
                        opname = "<=",
                        threshold = num_node_info[2],
                        threshold_type = float_dtype,
                        default_left = bool(num_node_info[3]),
                        left_child_key = int(num_node_info[4]),
                        right_child_key = int(num_node_info[5])
                    )

                else:
                    tree[node_ix].set_categorical_test_node(
                        feature_id = mapping_cat_cols[int(num_node_info[1])],
                        left_categories = cat_left,
                        default_left = bool(num_node_info[3]),
                        left_child_key = int(num_node_info[4]),
                        right_child_key = int(num_node_info[5])
                    )

            tree[0].set_root()
            builder.append(tree)
        model = builder.commit()
        return model

    def drop_imputer(self):
        """
        Drops the imputer sub-object from this model object

        Drops the imputer sub-object from this model object, if it was fitted with data imputation
        capabilities. The imputer, if constructed, is likely to be a very heavy object which might
        not be needed for all purposes.

        Returns
        -------
        self : obj
            This object
        """
        self._cpp_obj.drop_imputer()
        return self

    def subset_trees(self, trees_take):
        """
        Subset trees of a given model

        Creates a new model containing only selected trees of this
        model object.

        Parameters
        ----------
        trees_take : array_like(n,)
            Indices of the trees of this model to copy over to the new model.
            Must be integers with numeration starting at zero.

        Returns
        -------
        new_model : obj
            A new IsolationForest model object, containing only the subset of trees
            from this object that was specified under 'trees_take'.
        """
        assert self.is_fitted_
        trees_take = np.array(trees_take).reshape(-1).astype(ctypes.c_size_t)
        if not trees_take.shape[0]:
            raise ValueError("'trees_take' is empty.")
        if trees_take.max() >= self.ntrees:
            raise ValueError("Attempting to take tree indices that the model does not have.")
        new_cpp_obj = self._cpp_obj.subset_model(trees_take, self.ndim>1, self.build_imputer)
        old_cpp_obj = self._cpp_obj
        try:
            self._cpp_obj = None
            new_obj = deepcopy(self)
            new_obj._cpp_obj = new_cpp_obj
        finally:
            self._cpp_obj = old_cpp_obj
        return new_obj

    ### https://github.com/numpy/numpy/issues/19069
    def _is_np_int(self, el):
        return (
            np.issubdtype(el.__class__, int) or
            np.issubdtype(el.__class__, np.integer) or
            np.issubdtype(el.__class__, np.int8) or
            np.issubdtype(el.__class__, np.int16) or
            np.issubdtype(el.__class__, np.int16) or
            np.issubdtype(el.__class__, np.int32) or
            np.issubdtype(el.__class__, np.int64) or
            np.issubdtype(el.__class__, np.uint8) or
            np.issubdtype(el.__class__, np.uint16) or
            np.issubdtype(el.__class__, np.uint16) or
            np.issubdtype(el.__class__, np.uint32) or
            np.issubdtype(el.__class__, np.uint64)
        )

    def _denumpify_list(self, lst):
        return [int(el) if self._is_np_int(el) else el for el in lst]

    def _export_metadata(self):
        if (self.max_depth is not None) and (self.max_depth != "auto"):
            self.max_depth = int(self.max_depth)

        data_info = {
            "ncols_numeric" : int(self._ncols_numeric), ## is in c++
            "ncols_categ" : int(self._ncols_categ),  ## is in c++
            "cols_numeric" : list(self.cols_numeric_),
            "cols_categ" : list(self.cols_categ_),
            "cat_levels" : [list(m) for m in self._cat_mapping],
            "categ_cols" : [] if self.categ_cols is None else list(self.categ_cols),
            "categ_max" : [] if self._cat_max_lev is None else list(self._cat_max_lev)
        }

        ### Beaware of np.int64, which looks like a Python integer but is not accepted by json
        data_info["cols_numeric"] = self._denumpify_list(data_info["cols_numeric"])
        data_info["cols_categ"] = self._denumpify_list(data_info["cols_categ"])
        data_info["categ_cols"] = self._denumpify_list(data_info["categ_cols"])
        data_info["categ_max"] = self._denumpify_list(data_info["categ_max"])
        if len(data_info["cat_levels"]):
            data_info["cat_levels"] = [self._denumpify_list(lst) for lst in data_info["cat_levels"]]
        if len(data_info["categ_cols"]):
            data_info["categ_cols"] = self._denumpify_list(data_info["categ_cols"])

        model_info = {
            "ndim" : int(self.ndim),
            "nthreads" : _process_nthreads(self.nthreads),
            "build_imputer" : bool(self.build_imputer)
        }

        params = {
            "sample_size" : self.sample_size,
            "ntrees" : int(self._ntrees),  ## is in c++
            "ntry" : int(self.ntry),
            "max_depth" : self.max_depth,
            "ncols_per_tree" : self.ncols_per_tree,
            "prob_pick_avg_gain" : float(self.prob_pick_avg_gain),
            "prob_pick_pooled_gain" : float(self.prob_pick_pooled_gain),
            "prob_pick_col_by_range" : float(self.prob_pick_col_by_range),
            "prob_pick_col_by_var" : float(self.prob_pick_col_by_var),
            "prob_pick_col_by_kurt" : float(self.prob_pick_col_by_kurt),
            "min_gain" : float(self.min_gain),
            "missing_action" : self.missing_action,  ## is in c++
            "new_categ_action" : self.new_categ_action,  ## is in c++
            "categ_split_type" : self.categ_split_type_,  ## is in c++
            "coefs" : self.coefs,
            "depth_imp" : self.depth_imp,
            "weigh_imp_rows" : self.weigh_imp_rows,
            "min_imp_obs" : int(self.min_imp_obs),
            "random_seed" : self.random_seed,
            "all_perm" : self.all_perm,
            "coef_by_prop" : self.coef_by_prop,
            "weights_as_sample_prob" : self.weights_as_sample_prob,
            "sample_with_replacement" : self.sample_with_replacement,
            "penalize_range" : self.penalize_range,
            "standardize_data" : self.standardize_data,
            "scoring_metric" : self.scoring_metric,
            "weigh_by_kurtosis" : self.weigh_by_kurtosis,
            "assume_full_distr" : self.assume_full_distr,
        }

        if params["max_depth"] == "auto":
            params["max_depth"] = 0

        return {"data_info" : data_info, "model_info" : model_info, "params" : params}

    def _take_metadata(self, metadata):
        self._ncols_numeric = metadata["data_info"]["ncols_numeric"]
        self._ncols_categ = metadata["data_info"]["ncols_categ"]
        self.cols_numeric_ = np.array(metadata["data_info"]["cols_numeric"])
        self.cols_categ_ = np.array(metadata["data_info"]["cols_categ"])
        self._cat_mapping = [np.array(lst) for lst in metadata["data_info"]["cat_levels"]]
        self.categ_cols = np.array(metadata["data_info"]["categ_cols"]).reshape(-1).astype(int) if len(metadata["data_info"]["categ_cols"]) else None
        self._cat_max_lev = np.array(metadata["data_info"]["categ_max"]).reshape(-1).astype(int) if (self.categ_cols is not None) else []

        self.ndim = metadata["model_info"]["ndim"]
        self.nthreads = _process_nthreads(metadata["model_info"]["nthreads"])
        self.build_imputer = metadata["model_info"]["build_imputer"]

        self.sample_size = metadata["params"]["sample_size"]
        self.ntrees = metadata["params"]["ntrees"]
        self._ntrees = self.ntrees
        self.ntry = metadata["params"]["ntry"]
        self.max_depth = metadata["params"]["max_depth"]
        self.ncols_per_tree = metadata["params"]["ncols_per_tree"]
        self.prob_pick_avg_gain = metadata["params"]["prob_pick_avg_gain"]
        self.prob_pick_pooled_gain = metadata["params"]["prob_pick_pooled_gain"]
        try:
            self.prob_pick_col_by_range = metadata["params"]["prob_pick_col_by_range"]
        except:
            self.prob_pick_col_by_range = 0.0
        try:
            self.prob_pick_col_by_var = metadata["params"]["prob_pick_col_by_var"]
        except:
            self.prob_pick_col_by_var = 0.0
        try:
            self.prob_pick_col_by_kurt = metadata["params"]["prob_pick_col_by_kurt"]
        except:
            self.prob_pick_col_by_kurt = 0.0
        self.min_gain = metadata["params"]["min_gain"]
        self.missing_action = metadata["params"]["missing_action"]
        self.new_categ_action = metadata["params"]["new_categ_action"]
        self.categ_split_type = metadata["params"]["categ_split_type"]
        self.categ_split_type_ = self.categ_split_type
        self.coefs = metadata["params"]["coefs"]
        self.depth_imp = metadata["params"]["depth_imp"]
        self.weigh_imp_rows = metadata["params"]["weigh_imp_rows"]
        self.min_imp_obs = metadata["params"]["min_imp_obs"]
        self.random_seed = metadata["params"]["random_seed"]
        self.all_perm = metadata["params"]["all_perm"]
        self.coef_by_prop = metadata["params"]["coef_by_prop"]
        self.weights_as_sample_prob = metadata["params"]["weights_as_sample_prob"]
        self.sample_with_replacement = metadata["params"]["sample_with_replacement"]
        self.penalize_range = metadata["params"]["penalize_range"]
        try:
            self.standardize_data = metadata["params"]["standardize_data"]
        except:
            self.standardize_data = True
        try:
            self.scoring_metric = metadata["params"]["scoring_metric"]
        except:
            self.scoring_metric = "depth"
        self.weigh_by_kurtosis = metadata["params"]["weigh_by_kurtosis"]
        self.assume_full_distr = metadata["params"]["assume_full_distr"]

        if "prob_split_avg_gain" in metadata["params"].keys():
            if metadata["params"]["prob_split_avg_gain"] > 0:
                msg = "'prob_split_avg_gain' has been deprecated in favor of 'prob_pick_avg_gain' + 'ntry'."
                if self.ndim > 1:
                    msg += " Be sure to change these parameters if refitting this model or adding trees."
                warnings.warn(msg)
        if "prob_split_pooled_gain" in metadata["params"].keys():
            if metadata["params"]["prob_split_pooled_gain"] > 0:
                msg = "'prob_split_pooled_gain' has been deprecated in favor of 'prob_pick_pooled_gain' + 'ntry'."
                if self.ndim > 1:
                    msg += " Be sure to change these parameters if refitting this model or adding trees."
                warnings.warn(msg)

        self.is_fitted_ = True
        self._is_extended_ = self.ndim > 1
        return self

    def __is_fitted__(self):
        return self.is_fitted_
