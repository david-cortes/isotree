import numpy as np, pandas as pd
from scipy.sparse import csc_matrix, issparse, isspmatrix_csc, isspmatrix_csr
import warnings
import multiprocessing
import ctypes
from ._cpp_interface import isoforest_cpp_obj

class IsolationForest:
    """
    Isolation Forest model

    Isolation Forest is an algorithm originally developed for outlier detection that consists in splitting
    sub-samples of the data according to some attribute/feature/column at random. The idea is that, the rarer
    the observation, the more likely it is that a random uniform split on some feature would put outliers alone
    in one branch, and the fewer splits it will take to isolate an outlier observation like this. The concept
    is extended to splitting hyperplanes in the extended model (i.e. splitting by more than one column at a time), and to
    guided (not entirely random) splits in the SCiForest model that aim at isolating outliers faster and
    finding clustered outliers.

    This version adds heuristics to handle missing data and categorical variables. Can be used to aproximate pairwise
    distances by checking the depth after which two observations become separated, and to approximate densities by fitting
    trees beyond balanced-tree limit. Offers options to vary between randomized and deterministic splits too.

    Note
    ----
    The default parameters set up for this implementation will not scale to large datasets. In particular,
    if the amount of data is large, you might want to set a smaller sample size for each tree, and fit fewer of them.
    If using the single-variable model, you might also want to set 'prob_pick_pooled_gain' = 0, or perhaps replace it
    with 'prob_split_pooled_gain'. See the documentation of the parameters for more details.

    Parameters
    ----------
    sample_size : int or None
        Sample size of the data sub-samples with which each binary tree will be built. If passing 'None', each
        tree will be built using the full data. Recommended value in [1], [2], [3] is 256, while
        the default value in the author's code in [5] is 'None' here.
    ntrees : int
        Number of binary trees to build for the model. Recommended value in [1] is 100, while the default value in the
        author's code in [5] is 10.
    ndim : int
        Number of columns to combine to produce a split. If passing 1, will produce the single-variable model described
        in [1] and [2], while if passing values greater than 1, will produce the extended model described in [3] and [4].
        Recommended value in [4] is 2, while [3] recommends a low value such as 2 or 3. Models with values higher than 1
        are referred hereafter as the extended model (as in [3]).
    ntry : int
        In the extended model with non-random splits, how many random combinations to try for determining the best gain.
        Only used when deciding splits by gain (see documentation for parameters 'prob_pick_avg_gain' and 'prob_pick_pooled_gain').
        Recommended value in [4] is 10. Ignored for single-variable model.
    max_depth : int, None, or str "auto"
        Maximum depth of the binary trees to grow. If passing None, will build trees until each observation ends alone
        in a terminal node or until no further split is possible. If using "auto", will limit it to the corresponding
        depth of a balanced binary tree with number of terminal nodes corresponding to the sub-sample size (the reason
        being that, if trying to detect outliers, an outlier will only be so if it turns out to be isolated with shorter average
        depth than usual, which corresponds to a balanced tree depth) When a terminal node has more than 1 observation, the
        remaining isolation depth for them is estimated assuming the data and splits are both uniformly random (separation depth
        follows a similar process with expected value calculated as in [6]). Default setting for [1], [2], [3], [4] is "auto",
        but it's recommended to pass higher values if using the model for purposes other than outlier detection.
    prob_pick_avg_gain : float(0, 1)
        Probability of making each split in the single-variable model by choosing a column and split point in that
        same column as both the column and split point that gives the largest averaged gain (as proposed in [4]) across
        all available columns and possible splits in each column. Note that this implies evaluating every single column
        in the sample data when this type of split happens, which will potentially make the model fitting much slower,
        but has no impact on prediction time. For categorical variables, will take the expected standard deviation that
        would be gotten if the column were converted to numerical by assigning to each category a random number ~ Unif(0, 1)
        and calculate gain with those assumed standard deviations. For the extended model, this parameter indicates the probability that the
        split point in the chosen linear combination of variables will be decided by this averaged gain criterion. Compared to
        a pooled average, this tends to result in more cases in which a single observation or very few of them are put into
        one branch. When splits are not made according to any of 'prob_pick_avg_gain', 'prob_pick_pooled_gain', 'prob_split_avg_gain',
        'prob_split_pooled_gain', both the column and the split point are decided at random. Default setting for [1], [2], [3] is
        zero, and default for [4] is 1. This is the randomization parameter that can be passed to the author's original code in [5].
        Note that, if passing value 1 (100%) and using the single-variable model, every single tree will have the exact same splits.
    prob_pick_pooled_gain : float(0, 1)
        Probability of making each split in the single-variable model by choosing a column and split point in that
        same column as both the column and split point that gives the largest pooled gain (as used in decision tree
        classifiers such as C4.5 in [7]) across all available columns and possible splits in each column. Note
        that this implies evaluating every single column in the sample data when this type of split happens, which
        will potentially make the model fitting much slower, but has no impact on prediction time. For categorical
        variables, will use shannon entropy instead (like in [7]). For the extended model, this parameter indicates the probability
        that the split point in the chosen linear combination of variables will be decided by this pooled gain
        criterion. Compared to a simple average, this tends to result in more evenly-divided splits and more clustered
        groups when they are smaller. When splits are not made according to any of 'prob_pick_avg_gain',
        'prob_pick_pooled_gain', 'prob_split_avg_gain', 'prob_split_pooled_gain', both the column and the split point
        are decided at random. Note that, if passing value 1 (100%) and using the single-variable model, every single tree will have
        the exact same splits.
    prob_split_avg_gain : float(0, 1)
        Probability of making each split by selecting a column at random and determining the split point as
        that which gives the highest averaged gain. Not supported for the extended model as the splits are on
        linear combinations of variables. See the documentation for parameter 'prob_pick_avg_gain' for more details.
    prob_split_pooled_gain : float(0, 1)
        Probability of making each split by selecting a column at random and determining the split point as
        that which gives the highest pooled gain. Not supported for the extended model as the splits are on
        linear combinations of variables. See the documentation for parameter 'prob_pick_pooled_gain' for more details.
    missing_action : str, one of "divide" (single-variable only), "impute", "fail", "auto"
        How to handle missing data at both fitting and prediction time. Options are a) "divide" (for the single-variable
        model only, recommended), which will follow both branches and combine the result with the weight given by the fraction of
        the data that went to each branch when fitting the model, b) "impute", which will assign observations to the
        branch with the most observations in the single-variable model, or fill in missing values with the median
        of each column of the sample from which the split was made in the extended model (recommended), c) "fail" which will assume
        there are no missing values and will trigger undefined behavior if it encounters any, d) "auto", which will use "divide" for
        the single-variable model and "impute" for the extended model. In the extended model, infinite alues will be treated as
        missing. Note that passing "fail" might crash the Python process if there turn out to be
        missing values, but will otherwise produce faster fitting and prediction times along with decreased model object sizes.
        Models from [1], [2], [3], [4] correspond to "fail" here.
    new_categ_action : str, one of "weighted" (single-variable only), "impute" (extended only), "smallest", "random"
        What to do after splitting a categorical feature when new data that reaches that split has categories that
        the sub-sample from which the split was done did not have. Options are a) "weighted" (for the single-variable
        model only, recommended), which will follow both branches and combine the result with weight given by the fraction of the
        data that went to each branch when fitting the model, b) "impute" (for the extended model only, recommended) which will assign
        them the median value for that column that was added to the linear combination of features, c) "smallest", which
        in the single-variable case will assign all observations with unseen categories in the split to the branch that had
        fewer observations when fitting the model, and in the extended case will assign them the coefficient of the least common
        category, d) "random", which will assing a branch (coefficient in the extended model) at random for
        each category beforehand, even if no observations had that category when fitting the model, e) "auto" which will select
        "weighted" for the single-variable model and "impute" for the extended model. Ignored when
        passing 'categ_split_type' = 'single_categ'.
    categ_split_type : str, one of "subset" or "single_categ"
        Whether to split categorical features by assigning sub-sets of them to each branch, or by assigning
        a single category to a branch and the rest to the other branch. For the extended model, whether to
        give each category a coefficient, or only one while the rest get zero.
    all_perm : bool
        When doing categorical variable splits by pooled gain, whether to consider all possible permutations
        of variables to assign to each branch or not. If 'False', will sort the categories by their frequency
        and make a grouping in this sorted order. Note that the number of combinations evaluated (if 'True')
        is the factorial of the number of present categories in a given column (minus 2). For averaged gain, the
        best split is always to put the second most-frequent category in a separate branch, so not evaluating all
        permutations (passing 'False') will make it possible to select other splits that respect the sorted frequency order.
        Ignored when not using categorical variables or not doing splits by pooled gain.
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
        Whether to penalize (add +1 to the terminal depth) observations at prediction time that have a value
        of the chosen split variable (linear combination in extended model) that falls outside of a pre-determined
        reasonable range in the data being split (given by 2 * range in data and centered around the split point),
        as proposed in [4] and implemented in the authors' original code in [5]. Not used in single-variable model
        when splitting by categorical variables.
    weigh_by_kurtosis : bool
        Whether to weigh each column according to the kurtosis obtained in the sub-sample that is selected
        for each tree as briefly proposed in [1]. Note that this is only done at the beginning of each tree
        sample, so if not using sub-samples, it's better to pass column weights calculated externally. For
        categorical columns, will calculate expected kurtosis if the column was converted to numerical by
        assigning to each category a random number ~ Unif(0, 1).
    coefs : str, one of "normal" or "uniform"
        For the extended model, whether to sample random coefficients according to a normal distribution ~ N(0, 1)
        (as proposed in [3]) or according to a uniform distribution ~ Unif(-1, +1) as proposed in [4]. Ignored for the
        single-variable model.
    assume_full_distr : bool
        When calculating pairwise distances, whether to assume that the fitted model represents
        a full population distribution (will use a standardizing criterion assuming infinite sample,
        and the results of the similarity between two points at prediction time will not depend on the
        prescence of any third point that is similar to them, but will differ more compared to the pairwise
        distances between points from which the model was fit). If passing 'False', will calculate pairwise distances
        as if the new observations at prediction time were added to the sample to which each tree was fit, which
        will make the distances between two points potentially vary according to other newly introduced points.
        This will not be assumed when the distances are calculated as the model is being fit (see documentation
        for method 'fit_transform').
    random_seed : int
        Seed that will be used to generate random numbers used by the model.
    nthreads : int
        Number of parallel threads to use. If passing a negative number, will use
        the maximum number of available threads in the system. Note that, the more threads,
        the more memory will be allocated, even if the thread does not end up being used.

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
    [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
    [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation-based anomaly detection." ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
    [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner. "Extended Isolation Forest." arXiv preprint arXiv:1811.02141 (2018).
    [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "On detecting clustered anomalies using SCiForest." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
    [5] https://sourceforge.net/projects/iforest/
    [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
    [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
    """
    def __init__(self, sample_size = None, ntrees = 500, ndim = 3, ntry = 3, max_depth = "auto",
                 prob_pick_avg_gain = 0.0, prob_pick_pooled_gain = 0.25,
                 prob_split_avg_gain = 0.0, prob_split_pooled_gain = 0.0,
                 missing_action = "auto", new_categ_action = "auto",
                 categ_split_type = "subset", all_perm = False,
                 weights_as_sample_prob = True, sample_with_replacement = False,
                 penalize_range = True, weigh_by_kurtosis = False,
                 coefs = "normal", assume_full_distr = True,
                 random_seed = 1, nthreads = -1):
        if sample_size is not None:
            assert sample_size > 0
            assert isinstance(sample_size, int)
        assert ntrees > 0
        assert isinstance(ntrees, int)
        if (max_depth != "auto") and (max_depth is not None):
            assert max_depth > 0
            assert isinstance(max_depth, int)
            if sample_size is not None:
                assert max_depth < sample_size
        assert ndim >= 1
        assert isinstance(ndim, int)
        assert ntry >= 1
        assert isinstance(ntry, int)
        assert random_seed >= 1

        assert missing_action    in ["divide",        "impute",   "fail",   "auto"]
        assert new_categ_action  in ["weighted",      "smallest", "random", "impute", "auto"]
        assert categ_split_type  in ["single_categ",  "subset"]
        assert coefs             in ["normal",        "uniform"]

        assert prob_pick_avg_gain     >= 0
        assert prob_pick_pooled_gain  >= 0
        assert prob_split_avg_gain    >= 0
        assert prob_split_pooled_gain >= 0
        s = prob_pick_avg_gain + prob_pick_pooled_gain + prob_split_avg_gain + prob_split_pooled_gain
        if s > 1:
            warnings.warn("Split type probabilities sum to more than 1, will standardize them")
            prob_pick_avg_gain     /= s
            prob_pick_pooled_gain  /= s
            prob_split_avg_gain    /= s
            prob_split_pooled_gain /= s

        if (ndim == 1) and ((prob_pick_avg_gain >= 1) or (prob_pick_pooled_gain >= 1)):
            msg  = "Passed parameters for deterministic single-variable splits. "
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

        if ndim == 1:
            if new_categ_action == "impute":
                raise ValueError("'new_categ_action' = 'impute' not supported in single-variable model.")
        else:
            if (prob_split_avg_gain > 0) or (prob_split_pooled_gain > 0):
                msg  = "Non-zero values for 'prob_split_avg_gain' "
                msg += "and 'prob_split_pooled_gain' not meaningful in "
                msg += "extended model."
                raise ValueError(msg)
            if missing_action == "divide":
                raise ValueError("'missing_action' = 'divide' not supported in extended model.")
            if new_categ_action == "weighted":
                raise ValueError("'new_categ_action' = 'weighted' not supported in extended model.")

        if nthreads is None:
            nthreads = 1
        elif nthreads < 1:
            nthreads = multiprocessing.cpu_count()

        assert nthreads > 0
        assert isinstance(nthreads, int)

        self.sample_size             =  sample_size
        self.ntrees                  =  ntrees
        self.ndim                    =  ndim
        self.ntry                    =  ntry
        self.max_depth               =  max_depth
        self.prob_pick_avg_gain      =  prob_pick_avg_gain
        self.prob_pick_pooled_gain   =  prob_pick_pooled_gain
        self.prob_split_avg_gain     =  prob_split_avg_gain
        self.prob_split_pooled_gain  =  prob_split_pooled_gain
        self.missing_action          =  missing_action
        self.new_categ_action        =  new_categ_action
        self.categ_split_type        =  categ_split_type
        self.coefs                   =  coefs
        self.random_seed             =  random_seed
        self.nthreads                =  nthreads

        self.all_perm                =  bool(all_perm)
        self.weights_as_sample_prob  =  bool(weights_as_sample_prob)
        self.sample_with_replacement =  bool(sample_with_replacement)
        self.penalize_range          =  bool(penalize_range)
        self.weigh_by_kurtosis       =  bool(weigh_by_kurtosis)
        self.assume_full_distr       =  bool(assume_full_distr)

        self.cols_numeric_  =  np.array([])
        self.cols_categ_    =  np.array([])
        self._cat_mapping   =  list()
        self._ncols_numeric =  0
        self._ncols_categ   =  0
        self.is_fitted_     =  False
        self._cpp_obj       =  isoforest_cpp_obj()
        self._is_extended_  =  ndim > 1

    def __str__(self):
        msg = ""
        if self._is_extended_:
            msg += "Extended "
        msg += "Isolation Forest model"
        if (self.prob_pick_avg_gain + self.prob_pick_pooled_gain) > 0 or \
            (self.ndim == 1 and (self.prob_split_avg_gain + self.prob_split_pooled_gain) > 0):
            msg += " (using guided splits)"
        msg += "\n"
        if self.ndim > 1:
            msg += "Splitting by %d variables at a time\n" % self.ndim
        if self.is_fitted_:
            msg += "Consisting of %d trees\n" % self.ntrees
            if self._ncols_numeric > 0:
                msg += "Numeric columns: %d\n" % self._ncols_numeric
            if self._ncols_categ:
                msg += "Categorical columns: %d\n" % self._ncols_categ
        return msg

    def __repr__(self):
        return self.__str__()

    def _get_model_obj(self):
        return self._cpp_obj.get_cpp_obj(self._is_extended_)

    def fit(self, X, sample_weights = None, column_weights = None):
        """
        Fit isolation forest model to data

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data to which to fit the model. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
            If passing a DataFrame, will assume that columns are categorical if their dtype is 'object', 'Categorical', or 'bool',
            and will assume they are numerical if their dtype is a subtype of NumPy's 'number' or 'datetime64'.
            Other dtypes are not supported.
        sample_weights : None or array(n_samples,)
            Sample observation weights for each row of 'X', with higher weights indicating either higher sampling
            probability (i.e. the observation has a larger effect on the fitted model, if using sub-samples), or
            distribution density (i.e. if the weight is two, it has the same effect of including the same data
            point twice), according to parameter 'weights_as_sample_prob' in the model constructor method.
        column_weights : None or array(n_features,)
            Sampling weights for each column in 'X'. Ignored when picking columns by deterministic criterion.
            If passing None, each column will have a uniform weight. Cannot be used when weighting by kurtosis.

        Returns
        -------
        self : obj
            This object.
        """
        if self.sample_size is None and sample_weights is not None and self.weights_as_sample_prob:
            raise ValueError("Sampling weights are only supported when using sub-samples for each tree.")
        if column_weights is not None and self.weigh_by_kurtosis:
            raise ValueError("Cannot pass column weights when weighting columns by kurtosis.")
        X_num, X_cat, ncat, sample_weights, column_weights, nrows = self._process_data(X, sample_weights, column_weights)

        if self.sample_size is None:
            sample_size = nrows
        else:
            sample_size = self.sample_size
        if self.max_depth == "auto":
            max_depth = 0
            limit_depth = True
        elif self.max_depth is None:
            max_depth = nrows - 1
        else:
            max_depth = self.max_depth
            limit_depth = False

        self._cpp_obj.fit_model(X_num, X_cat, ncat, sample_weights, column_weights,
                                ctypes.c_size_t(nrows).value,
                                ctypes.c_size_t(self._ncols_numeric).value,
                                ctypes.c_size_t(self._ncols_categ).value,
                                ctypes.c_size_t(self.ndim).value,
                                ctypes.c_size_t(self.ntry).value,
                                self.coefs,
                                ctypes.c_bool(self.sample_with_replacement).value,
                                ctypes.c_bool(self.weights_as_sample_prob).value,
                                ctypes.c_size_t(sample_size).value,
                                ctypes.c_size_t(self.ntrees).value,
                                ctypes.c_size_t(max_depth).value,
                                ctypes.c_bool(limit_depth).value,
                                ctypes.c_bool(self.penalize_range).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(False).value,
                                ctypes.c_bool(self.weigh_by_kurtosis).value,
                                ctypes.c_double(self.prob_pick_avg_gain).value,
                                ctypes.c_double(self.prob_split_avg_gain).value,
                                ctypes.c_double(self.prob_pick_pooled_gain).value,
                                ctypes.c_double(self.prob_split_pooled_gain).value,
                                self.categ_split_type,
                                self.new_categ_action,
                                self.missing_action,
                                ctypes.c_bool(self.all_perm).value,
                                ctypes.c_uint64(self.random_seed).value,
                                ctypes.c_int(self.nthreads).value)
        self.is_fitted_ = True
        return self

    def fit_transform(self, X, column_weights = None, output_outlierness = "score", output_distance = None, square_mat = False):
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
        Sample weights are not supported for this method.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data to which to fit the model. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
            If passing a DataFrame, will assume that columns are categorical if their dtype is 'object', 'Categorical', or 'bool',
            and will assume they are numerical if their dtype is a subtype of NumPy's 'number' or 'datetime64'.
            Other dtypes are not supported.
        column_weights : None or array(n_features,)
            Sampling weights for each column in 'X'. Ignored when picking columns by deterministic criterion.
            If passing None, each column will have a uniform weight. Cannot be used when weighting by kurtosis.
            Note that, if passing a DataFrame with both numeric and categorical columns, the column names must
            not be repeated, otherwise the column weights passed here will not end up matching.
        output_outlierness : None or str in ["score", "avg_path"]
            Desired type of outlierness output. If passing "score", will output standardized outlier score.
            If passing "avg_path" will output average isolation path without standardizing.
            If passing 'None', will skip outlierness calculations.
        output_distance : None or str in ["dist", "avg_sep"]
            Type of distance output to produce. If passing "dist", will standardize the average separation
            depths. If passing "avg_sep", will output the average separation depth without standardizing it
            (note that lower separation depth means furthest distance). If passing 'None', will skip distance calculations.
        square_mat : bool
            Whether to produce a full square matrix with the distances. If passing 'False', will output
            only the upper triangular part as a 1-d array in which entry 0 <= i < j < n is located at
            position p(i,j) = (i * (n - (i+1)/2) + j - i - 1). Ignored when passing 'output_distance' = 'None'.

        Returns
        -------
        output : array(n_samples,), and/or array(n_samples * (n_samples - 1) / 2,) or array(n_samples, n_samples)
            Requested outputs about isolation depth (outlierness) and/or pairwise separation depth (distance).
            If passing only one of 'output_outlierness' or 'output_distance', will output 1 array according to
            the parameters, whereas if passing both, will output a tuple with the first element being the result
            from 'output_outlierness', and the second element being the result of 'output_distance'.
        """
        if self.sample_size is not None:
            raise ValueError("Cannot use 'fit_transform' when the sample size is limited.")
        if self.sample_with_replacement:
            raise ValueError("Cannot use 'fit_transform' when sampling with replacement.")
        if column_weights is not None and self.weigh_by_kurtosis:
            raise ValueError("Cannot pass column weights when weighting columns by kurtosis.")

        if (output_outlierness is None) and (output_distance is None):
            raise ValueError("Must pass at least one of 'output_outlierness' or 'output_distance'.")

        if output_outlierness is not None:
            assert output_outlierness in ["score", "avg_path"]

        if output_distance is not None:
            assert output_distance in ["dist", "avg_sep"]

        X_num, X_cat, ncat, sample_weights, column_weights, nrows = self._process_data(X, None, column_weights)

        if self.max_depth == "auto":
            max_depth = 0
            limit_depth = True
        elif self.max_depth is None:
            max_depth = nrows - 1
        else:
            max_depth = self.max_depth
            limit_depth = False

        depths, tmat, dmat = self._cpp_obj.fit_model(X_num, X_cat, ncat, None, column_weights,
                                                     ctypes.c_size_t(nrows).value,
                                                     ctypes.c_size_t(self._ncols_numeric).value,
                                                     ctypes.c_size_t(self._ncols_categ).value,
                                                     ctypes.c_size_t(self.ndim).value,
                                                     ctypes.c_size_t(self.ntry).value,
                                                     self.coefs,
                                                     ctypes.c_bool(self.sample_with_replacement).value,
                                                     ctypes.c_bool(self.weights_as_sample_prob).value,
                                                     ctypes.c_size_t(nrows).value,
                                                     ctypes.c_size_t(self.ntrees).value,
                                                     ctypes.c_size_t(max_depth).value,
                                                     ctypes.c_bool(limit_depth).value,
                                                     ctypes.c_bool(self.penalize_range).value,
                                                     ctypes.c_bool(output_distance is not None).value,
                                                     ctypes.c_bool(output_distance == "dist").value,
                                                     ctypes.c_bool(square_mat).value,
                                                     ctypes.c_bool(output_outlierness is not None).value,
                                                     ctypes.c_bool(output_outlierness == "score").value,
                                                     ctypes.c_bool(self.weigh_by_kurtosis).value,
                                                     ctypes.c_double(self.prob_pick_avg_gain).value,
                                                     ctypes.c_double(self.prob_split_avg_gain).value,
                                                     ctypes.c_double(self.prob_pick_pooled_gain).value,
                                                     ctypes.c_double(self.prob_split_pooled_gain).value,
                                                     self.categ_split_type,
                                                     self.new_categ_action,
                                                     self.missing_action,
                                                     ctypes.c_bool(self.all_perm).value,
                                                     ctypes.c_uint64(self.random_seed).value,
                                                     ctypes.c_int(self.nthreads).value)
        self.is_fitted_ = True

        if (output_outlierness is not None) and (output_distance is not None):
            if square_mat:
                return depths, dmat
            else:
                return depths, tmat
        elif (output_distance is not None):
            if square_mat:
                return dmat
            else:
                return tmat
        else:
            return depths

    def _process_data(self, X, sample_weights, column_weights):
        ## TODO: must reorder column weights if the columns are split
        if X.__class__.__name__ == "DataFrame":
            ### https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas
            X_num = X.select_dtypes(include = [np.number, np.datetime64]).to_numpy()
            X_num = np.asfortranarray(X_num).astype(ctypes.c_double)
            X_cat = X.select_dtypes(include = [pd.CategoricalDtype, "object", "bool"]).copy()
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

            if X_cat is not None:
                self._cat_mapping = [None for cl in range(X_cat.shape[1])]
                for cl in range(X_cat.shape[1]):
                    X_cat[X_cat.columns[cl]], self._cat_mapping[cl] = pd.factorize(X_cat[X_cat.columns[cl]])
                X_cat = np.asfortranarray(X_cat).astype(ctypes.c_int)

        else:
            if len(X.shape) != 2:
                raise ValueError("Input data must be two-dimensional.")

            if X.__class__.__name__ == "ndarray":
                X = np.asfortranarray(X).astype(ctypes.c_double)
            elif issparse(X):
                if isspmatrix_csc(X):
                    X.sort_indices()
                else:
                    warnings.warn("Sparse matrices are only supported in CSC format, will be converted.")
                    X = csc_matrix(X)
                X.data    = X.data.astype(ctypes.c_double)
                X.indices = X.indices.astype(ctypes.c_size_t)
                X.indptr  = X.indptr.astype(ctypes.c_size_t)
            else:
                X = np.asfortranarray(np.array(X)).astype(ctypes.c_double)

            self._ncols_numeric = X.shape[1]
            self._ncols_categ   = 0
            self.cols_numeric_  = np.array([])
            self.cols_categ_    = np.array([])
            self._cat_mapping   = list()

            X_num = X
            X_cat = None
            nrows = X_num.shape[0]

        if nrows == 0:
            raise ValueError("Input data has zero rows.")
        elif nrows < 5:
            raise ValueError("Input data has too few rows.")
        elif self.sample_size is not None:
            if self.sample_size > nrows:
                warnings.warn("Input data has fewer rows than sample_size, will increase sample_size.")
                self.sample_size = None

        if X_cat is not None:
            ncat = np.array([self._cat_mapping[cl].shape[0] for cl in range(X_cat.shape[1])], dtype = ctypes.c_int)
        else:
            ncat = None

        if sample_weights is not None:
            sample_weights = np.array(sample_weights).reshape(-1).astype(ctypes.c_double)
            if sample_weights.shape[0] != nrows:
                raise ValueError("'sample_weights' has different number of rows than 'X'.")

        ncols = 0
        if X_num is not None:
            ncols += X_num.shape[1]
        if X_cat is not None:
            ncols += X_cat.shape[1]

        if column_weights is not None:
            column_weights = np.array(column_weights).reshape(-1).astype(ctypes.c_double)
            if ncols != column_weights.shape[0]:
                raise ValueError("'column_weights' has %d entries, but data has %d columns." % (column_weights.shape[0], ncols))
            if (X_num is not None) and (X_cat is not None):
                column_weights = np.r_[column_weights[X.columns.values == self.cols_numeric_],
                                       column_weights[X.columns.values == self.cols_categ_]]

        if self.ndim > 1:
            if self.ndim > ncols:
                raise ValueError("Model was meant to take %d variables for each split, but data has %d columns." % (self.ndim, ncols))

        return X_num, X_cat, ncat, sample_weights, column_weights, nrows

    def _process_data_new(self, X, allow_csr = True):
        if X.__class__.__name__ == "DataFrame":
            if (self.cols_numeric_.shape[0] + self.cols_categ_.shape[0]) > 0:
                missing_cols = np.setdiff1d(np.array(X.columns.values), np.r_[self.cols_numeric_, self.cols_categ_])
                if missing_cols.shape[0] > 0:
                    raise ValueError("Input data is missing %d columns - example: [%s]" % (missing_cols.shape[0], ", ".join(missing_cols[:3])))

                if self._ncols_numeric > 0:
                    X_num = X[self.cols_numeric_].to_numpy()
                    X_num = np.asfortranarray(X_num).astype(ctypes.c_double)
                    nrows = X_num.shape[0]
                else:
                    X_num = None

                if self._ncols_categ > 0:
                    X_cat = X[self.cols_categ_].copy()
                    for cl in range(self._ncols_categ):
                        X_cat[self.cols_categ_[cl]] = pd.Categorical(X_cat[self.cols_categ_[cl]], self._cat_mapping[cl]).codes
                    X_cat = X_cat.to_numpy()
                    X_cat = np.asfortranarray(X_cat).astype(ctypes.c_int)
                    nrows = X_cat.shape[0]
                else:
                    X_cat = None

            else:
                if X.shape[1] != self._ncols_numeric:
                    raise ValueError("Input has different number of columns than data to which model was fit.")
                X_num = X.to_numpy()
                X_num = np.asfortranarray(X_num).astype(ctypes.c_double)
                X_cat = None
                nrows = X_num.shape[0]


        else:
            if self._ncols_categ > 0:
                raise ValueError("Model was fit to DataFrame with categorical columns, but new input is a numeric array/matrix.")
            if len(X.shape) != 2:
                raise ValueError("Input data must be two-dimensional.")
            if X.shape[1] != self._ncols_numeric:
                raise ValueError("Input has different number of columns than data to which model was fit.")
            
            X_cat = None
            if issparse(X):
                if isspmatrix_csr(X) and not allow_csr:
                    warnings.warn("Cannot predict from CSR sparse matrix, will convert to CSC.")
                    X = csc_matrix(X)
                elif (not isspmatrix_csc(X)) and (not isspmatrix_csr(X)):
                    msg  = "Sparse matrix inputs only supported as CSC"
                    if allow_csr:
                        msg += " or CSR"
                    msg += " format, will convert to CSC."
                    warnings.warn(msg)
                    X = csc_matrix(X)
                else:
                    X.sort_indices()
                X.data    = X.data.astype(ctypes.c_double)
                X.indices = X.indices.astype(ctypes.c_size_t)
                X.indptr  = X.indptr.astype(ctypes.c_size_t)
                X_num     = X
            else:
                if X.__class__.__name__ != "ndarray":
                    X = np.array(X)
                X_num = np.asfortranarray(X).astype(ctypes.c_double)
            nrows = X_num.shape[0]

        return X_num, X_cat, nrows

    def predict(self, X, output = "score"):
        """
        Predict outlierness based on average isolation depth

        Calculates the approximate depth that it takes to isolate an observation according to the
        fitted model splits. Can output either the average depth, or a standardized outlier score
        based on whether it takes more or fewer splits than average to isolate observations. In the
        standardized outlier score metric, values closer to 1 indicate more outlierness, while values
        closer to 0.5 indicate average outlierness, and close to 0 more averageness (harder to isolate).
        
        Note
        ----
        The more threads that are set for the model, the higher the memory requirements will be as each
        thread will allocate an array with one entry per row.
        
        Note
        ----
        Predictions for sparse data will be much slower than for dense data. Not recommended to pass
        sparse matrices unless they are too big to fit in memory.
        
        Note
        ----
        In order to save memory when fitting and serializing models, the functionality for outputting
        terminal node number will generate index mappings on the fly for all tree nodes, even if passing only
        1 row, so it's only recommended for batch predictions.
        
        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Observations for which to predict outlierness or average isolation path. Can pass
            a NumPy array, Pandas DataFrame, or SciPy sparse CSC or CSR matrix.
        output : str, one of "score", "avg_path", "tree_num"
            Desired type of output. If passing "score", will output standardized outlier score.
            If passing "avg_path" will output average isolation path without standardizing. If
            passing "tree_num", will output the index of the terminal node under each tree in
            the model.

        Returns
        -------
        score : array(n_samples,) or array(n_samples, n_trees)
            Requested output type for each row accoring to parameter 'output' (outlier scores,
            average isolation depth, or terminal node indices).
        """
        assert self.is_fitted_
        assert output in ["score", "avg_path", "tree_num"]
        X_num, X_cat, nrows = self._process_data_new(X)
        if output == "tree_num":
            if self.missing_action == "divide":
                raise ValueError("Cannot output tree number when using 'missing_action' = 'divide'.")
            if self.new_categ_action == "weighted":
                raise ValueError("Cannot output tree number when using 'new_categ_action' = 'weighted'.")
            if nrows == 1:
                warnings.warn("Predicting tree number is slow, not recommended to do for 1 row at a time.")

        depths, tree_num = self._cpp_obj.predict(X_num, X_cat, self._is_extended_,
                                                 ctypes.c_size_t(nrows).value,
                                                 ctypes.c_int(self.nthreads).value,
                                                 ctypes.c_bool(output == "score").value,
                                                 ctypes.c_bool(output == "tree_num").value)

        if output in ["score", "avg_path"]:
            return depths
        else:
            return tree_num

    def predict_distance(self, X, output = "dist", square_mat = False):
        """
        Predict approximate pairwise distances

        Predict approximate pairwise distances between points based on how many
        splits it takes to separate them. Can output either the average number of paths,
        or a standardized metric (in the same way as the outlier score) in which values closer
        to zero indicate nearer points, closer to one further away points, and closer to 0.5
        average distance.

        Note
        ----
        The more threads that are set for the model, the higher the memory requirement will be as each
        thread will allocate an array with one entry per combination.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Observations for which to calculate approximate pairwise distances. Can pass
            a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
        output : str, one of "dist", "avg_sep"
            Type of output to produce. If passing "dist", will standardize the average separation
            depths. If passing "avg_sep", will output the average separation depth without standardizing it
            (note that lower separation depth means furthest distance).
        square_mat : bool
            Whether to produce a full square matrix with the distances. If passing 'False', will output
            only the upper triangular part as a 1-d array in which entry 0 <= i < j < n is located at
            position p(i,j) = (i * (n - (i+1)/2) + j - i - 1).

        Returns
        -------
        dist : array(n_samples * (n_samples - 1) / 2,) or array(n_samples, n_samples)
            Approximate pairwise distances or average separation depth, according to
            parameter 'output'. Shape and size depends on paramnter 'square_mat'.
        """
        assert self.is_fitted_
        if not self._is_extended_:
            if self.new_categ_action == "weighted" and self.missing_action != "divide":
                msg  = "Cannot predict distances when using "
                msg += "'new_categ_action' = 'weighted' "
                msg += "if 'missing_action' != 'divide'."
                raise ValueError(msg)
        assert output in ["dist", "avg_sep"]
        X_num, X_cat, nrows = self._process_data_new(X, allow_csr = False)

        if nrows == 1:
            raise ValueError("Cannot calculate pairwise distances for only 1 row.")

        tmat, dmat = self._cpp_obj.dist(X_num, X_cat, self._is_extended_,
                                        ctypes.c_size_t(nrows).value,
                                        ctypes.c_int(self.nthreads).value,
                                        ctypes.c_bool(self.assume_full_distr).value,
                                        ctypes.c_bool(output == "dist").value,
                                        ctypes.c_bool(square_mat).value)
        
        if square_mat:
            return dmat
        else:
            return tmat

    def partial_fit(self, X, sample_weights = None, column_weights = None):
        """
        Add additional (single) tree to isolation forest model

        Adds a single tree fit to the full (non-subsampled) data passed here. Must
        have the same columns as previously-fitted data.

        Parameters
        ----------
        X : array or array-like (n_samples, n_features)
            Data to which to fit the new tree. Can pass a NumPy array, Pandas DataFrame, or SciPy sparse CSC matrix.
            If passing a DataFrame, will assume that columns are categorical if their dtype is 'object', 'Categorical', or 'bool',
            and will assume they are numerical if their dtype is a subtype of NumPy's 'number' or 'datetime64'.
            Other dtypes are not supported.
        sample_weights : None or array(n_samples,)
            Sample observation weights for each row of 'X', with higher weights indicating
            distribution density (i.e. if the weight is two, it has the same effect of including the same data
            point twice). If not 'None', model must have been built with 'weights_as_sample_prob' = 'False'.
        column_weights : None or array(n_features,)
            Sampling weights for each column in 'X'. Ignored when picking columns by deterministic criterion.
            If passing None, each column will have a uniform weight. Cannot be used when weighting by kurtosis.

        Returns
        -------
        self : obj
            This object.
        """
        if (sample_weights is not None) and (self.weights_as_sample_prob):
            raise ValueError("Cannot use sampling weights with 'partial_fit'.")
        if (column_weights is not None) and (self.weigh_by_kurtosis):
            raise ValueError("Cannot pass column weights when weighting columns by kurtosis.")

        if not self.is_fitted_:
            return self.fit(X = X, sample_weights = sample_weights, column_weights = column_weights)
        
        X_num, X_cat, nrows = self._process_data_new(X, allow_csr = False)
        if sample_weights is not None:
            sample_weights = sample_weights.reshape(-1).astype(ctypes.c_double)
            assert sample_weights.shape[0] == X.shape[0]
        if column_weights is not None:
            column_weights = column_weights.reshape(-1).astype(ctypes.c_double)
            assert column_weights.shape[0] == X.shape[1]
        ncat = None
        if self._ncols_categ > 0:
            ncat = np.array([arr.shape[0] for arr in self._cat_mapping]).astype(ctypes.c_int)
        if self.max_depth == "auto":
            max_depth = 0
            limit_depth = True
        elif self.max_depth is None:
            max_depth = nrows - 1
        else:
            max_depth = self.max_depth
            limit_depth = False

        self._cpp_obj.fit_tree(X_num, X_cat, ncat, sample_weights, column_weights,
                               ctypes.c_size_t(nrows).value,
                               ctypes.c_size_t(self._ncols_numeric).value,
                               ctypes.c_size_t(self._ncols_categ).value,
                               ctypes.c_size_t(self.ndim).value,
                               ctypes.c_size_t(self.ntry).value,
                               self.coefs,
                               ctypes.c_size_t(max_depth).value,
                               ctypes.c_bool(limit_depth).value,
                               ctypes.c_bool(self.penalize_range).value,
                               ctypes.c_bool(self.weigh_by_kurtosis).value,
                               ctypes.c_double(self.prob_pick_avg_gain).value,
                               ctypes.c_double(self.prob_split_avg_gain).value,
                               ctypes.c_double(self.prob_pick_pooled_gain).value,
                               ctypes.c_double(self.prob_split_pooled_gain).value,
                               self.categ_split_type,
                               self.new_categ_action,
                               self.missing_action,
                               ctypes.c_bool(self.all_perm).value,
                               ctypes.c_int(self.nthreads).value)
        self.ntrees += 1
        return self
