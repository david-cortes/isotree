/*    Isolation forests and variations thereof, with adjustments for incorporation
*     of categorical variables and missing values.
*     Writen for C++11 standard and aimed at being used in R and Python.
*     
*     This library is based on the following works:
*     [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation forest."
*         2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
*     [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation-based anomaly detection."
*         ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
*     [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner.
*         "Extended Isolation Forest."
*         arXiv preprint arXiv:1811.02141 (2018).
*     [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "On detecting clustered anomalies using SCiForest."
*         Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
*     [5] https://sourceforge.net/projects/iforest/
*     [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
*     [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
*     [8] Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
*     [9] Cortes, David. "Imputing missing values with unsupervised random trees." arXiv preprint arXiv:1911.06646 (2019).
* 
*     BSD 2-Clause License
*     Copyright (c) 2019-2021, David Cortes
*     All rights reserved.
*     Redistribution and use in source and binary forms, with or without
*     modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright notice, this
*       list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright notice,
*       this list of conditions and the following disclaimer in the documentation
*       and/or other materials provided with the distribution.
*     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
*     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
*     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "isotree.hpp"

/*  Fit Isolation Forest model, or variant of it such as SCiForest
* 
* Parameters:
* ===========
* - model_outputs (out)
*       Pointer to already allocated isolation forest model object for single-variable splits.
*       If fitting the extended model, pass NULL (must pass 'model_outputs_ext'). Can later add
*       additional trees through function 'add_tree'.
* - model_outputs_ext (out)
*       Pointer to already allocated extended isolation forest model object (for multiple-variable splits).
*       Note that if 'ndim' = 1, must use instead the single-variable model object.
*       If fitting the single-variable model, pass NULL (must pass 'model_outputs'). Can later add
*       additional trees through function 'add_tree'.
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data to which to fit the model. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.).
*       Pass NULL if there are no dense numeric columns (must also pass 'ncols_numeric' = 0 if there's
*       no sparse numeric data either).
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - ncols_numeric
*       Number of numeric columns in the data (whether they come in a sparse matrix or dense array).
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data to which to fit the model. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.).
*       Pass NULL if there are no categorical columns (must also pass 'ncols_categ' = 0).
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must also be present
*       (note that they are not treated as being ordinal, this is just an encoding). Missing values
*       should be encoded as negative numbers such as (-1).
* - ncols_categ
*       Number of categorical columns in the data.
* - ncat[ncols_categ]
*       Number of categories in each categorical column. E.g. if the highest code for a column is '4',
*       the number of categories for that column is '5' (zero is one category).
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns.
* - Xc_indptr[ncols_numeric + 1]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end.
*       Pass NULL if there are no sparse numeric columns.
* - ndim
*       How many dimensions (columns) to use for making a split. Must pass 'ndim' = 1 for
*       the single-variable model. Note that the model object pointer passed must also
*       agree with the value passed to 'ndim'.
* - ntry
*       In the split-criterion extended model, how many random hyperplanes to evaluate in
*       order to decide which one is best to take. Ignored for the single-variable case
*       and for random splits.
* - coef_type
*       For the extended model, whether to sample random coefficients according to a normal distribution ~ N(0, 1)
*       (as proposed in [3]) or according to a uniform distribution ~ Unif(-1, +1) as proposed in [4]. Ignored for the
*       single-variable model.
* - sample_weights[nrows]
*       Weights for the rows when building a tree, either as sampling importances when using
*       sub-samples for each tree (i.e. passing weight '2' makes a row twice as likely to be included
*       in a random sub-sample), or as density measurement (i.e. passing weight '2' is the same as if
*       the row appeared twice, thus it's less of an outlier) - how this is taken is determined
*       through parameter 'weight_as_sample'.
*       Pass NULL if the rows all have uniform weights.
* - with_replacement
*       Whether to produce sub-samples with replacement or not.
* - weight_as_sample
*       If passing 'sample_weights', whether to consider those weights as row sampling weights (i.e. the higher
*       the weights, the more likely the observation will end up included in each tree sub-sample), or as distribution
*       density weights (i.e. putting a weight of two is the same as if the row appeared twice, thus higher weight makes it
*       less of an outlier). Note that sampling weight is only used when sub-sampling data for each tree.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'categ_data'.
* - sample_size
*       Sample size of the data sub-samples with which each binary tree will be built. When a terminal node has more than
*       1 observation, the remaining isolation depth for them is estimated assuming the data and splits are both uniformly
*       random (separation depth follows a similar process with expected value calculated as in [6]). If passing zero,
*       will set it to 'nrows'. Recommended value in [1], [2], [3] is 256, while the default value in the author's code
*       in [5] is 'nrows' here.
* - ntrees
*       Number of binary trees to build for the model. Recommended value in [1] is 100, while the default value in the
*       author's code in [5] is 10.
* - max_depth
*       Maximum depth of the binary trees to grow. Will get overwritten if passing 'limit_depth' = 'true'.
* - ncols_per_tree
*       Number of columns to use (have as potential candidates for splitting at each iteration) in each tree,
*       similar to the 'mtry' parameter of random forests.
*       In general, this is only relevant when using non-random splits and/or weighting by kurtosis.
*       If passing zero, will use the full number of available columns.
*       Recommended value: 0.
* - limit_depth
*       Whether to automatically set the maximum depth to the corresponding depth of a balanced binary tree with number of
*       terminal nodes corresponding to the sub-sample size (the reason being that, if trying to detect outliers, an outlier
*       will only be so if it turns out to be isolated with shorter average depth than usual, which corresponds to a balanced
*       tree depth). Default setting for [1], [2], [3], [4] is 'true', but it's recommended to pass higher values if
*       using the model for purposes other than outlier detection.
* - penalize_range
*       Whether to penalize (add -1 to the terminal depth) observations at prediction time that have a value
*       of the chosen split variable (linear combination in extended model) that falls outside of a pre-determined
*       reasonable range in the data being split (given by 2 * range in data and centered around the split point),
*       as proposed in [4] and implemented in the authors' original code in [5]. Not used in single-variable model
*       when splitting by categorical variables. Note that this can make a very large difference in the results
*       when using `prob_pick_pooled_gain`.
* - standardize_dist
*       If passing 'tmat' (see documentation for it), whether to standardize the resulting average separation
*       depths in order to produce a distance metric or not, in the same way this is done for the outlier score.
* - tmat[nrows * (nrows - 1) / 2]
*       Array in which to calculate average separation depths or standardized distance metric (see documentation
*       for 'standardize_dist') as the model is being fit. Pass NULL to avoid doing these calculations alongside
*       the regular model process. If passing this output argument, the sample size must be the same as the number
*       of rows, and there cannot be sample weights. If not NULL, must already be initialized to zeros. As the
*       output is a symmetric matrix, this function will only fill in the upper-triangular part, in which
*       entry 0 <= i < j < n will be located at position
*           p(i,j) = (i * (n - (i+1)/2) + j - i - 1).
*       Can be converted to a dense square matrix through function 'tmat_to_dense'.
* - output_depths[nrows]
*       Array in which to calculate average path depths or standardized outlierness metric (see documentation
*       for 'standardize_depth') as the model is being fit. Pass NULL to avoid doing these calculations alongside
*       the regular model process. If passing this output argument, the sample size must be the same as the number
*       of rows. If not NULL, must already be initialized to zeros.
* - standardize_depth
*       If passing 'output_depths', whether to standardize the results as proposed in [1], in order to obtain
*       a metric in which the more outlier is an observation, the closer this standardized metric will be to 1,
*       with average observations obtaining 0.5. If passing 'false' here, the numbers in 'output_depths' will be
*       the average depth of each row across all trees.
* - col_weights[ncols_numeric + ncols_categ]
*       Sampling weights for each column, assuming all the numeric columns come before the categorical columns.
*       Ignored when picking columns by deterministic criterion.
*       If passing NULL, each column will have a uniform weight. Cannot be used when weighting by kurtosis.
* - weigh_by_kurt
*       Whether to weigh each column according to the kurtosis obtained in the sub-sample that is selected
*       for each tree as briefly proposed in [1]. Note that this is only done at the beginning of each tree
*       sample, so if not using sub-samples, it's better to pass column weights calculated externally. For
*       categorical columns, will calculate expected kurtosis if the column was converted to numerical by
*       assigning to each category a random number ~ Unif(0, 1).
* - prob_pick_by_gain_avg
*       Probability of making each split in the single-variable model by choosing a column and split point in that
*       same column as both the column and split point that gives the largest averaged gain (as proposed in [4]) across
*       all available columns and possible splits in each column. Note that this implies evaluating every single column
*       in the sample data when this type of split happens, which will potentially make the model fitting much slower,
*       but has no impact on prediction time. For categorical variables, will take the expected standard deviation that
*       would be gotten if the column were converted to numerical by assigning to each category a random number ~ Unif(0, 1)
*       and calculate gain with those assumed standard deviations. For the extended model, this parameter indicates the probability that the
*       split point in the chosen linear combination of variables will be decided by this averaged gain criterion. Compared to
*       a pooled average, this tends to result in more cases in which a single observation or very few of them are put into
*       one branch.  Recommended to use sub-samples (parameter `sample_size`) when passing this parameter. When splits are
*       not made according to any of 'prob_pick_by_gain_avg', 'prob_pick_by_gain_pl', 'prob_split_by_gain_avg', 'prob_split_by_gain_pl',
*       both the column and the split point are decided at random.
*       Default setting for [1], [2], [3] is zero, and default for [4] is 1. This is the randomization parameter that can
*       be passed to the author's original code in [5]. Note that, if passing value 1 (100%) with no sub-sampling and using the
*       single-variable model, every single tree will have the exact same splits.
*       Important detail: if using either 'prob_pick_avg_gain' or 'prob_pick_pooled_gain', the distribution of
*       outlier scores is unlikely to be centered around 0.5.
* - prob_split_by_gain_avg
*       Probability of making each split by selecting a column at random and determining the split point as
*       that which gives the highest averaged gain. Not supported for the extended model as the splits are on
*       linear combinations of variables. See the documentation for parameter 'prob_pick_by_gain_avg' for more details.
* - prob_pick_by_gain_pl
*       Probability of making each split in the single-variable model by choosing a column and split point in that
*       same column as both the column and split point that gives the largest pooled gain (as used in decision tree
*       classifiers such as C4.5 in [7]) across all available columns and possible splits in each column. Note
*       that this implies evaluating every single column in the sample data when this type of split happens, which
*       will potentially make the model fitting much slower, but has no impact on prediction time. For categorical
*       variables, will use shannon entropy instead (like in [7]). For the extended model, this parameter indicates the probability
*       that the split point in the chosen linear combination of variables will be decided by this pooled gain
*       criterion. Compared to a simple average, this tends to result in more evenly-divided splits and more clustered
*       groups when they are smaller. Recommended to pass higher values when used for imputation of missing values.
*       When used for outlier detection, higher values of this parameter result in models that are able to better flag
*       outliers in the training data, but generalize poorly to outliers in new data and to values of variables
*       outside of the ranges from the training data. Passing small 'sample_size' and high values of this parameter will
*       tend to flag too many outliers. When splits are not made according to any of 'prob_pick_by_gain_avg',
*       'prob_pick_by_gain_pl', 'prob_split_by_gain_avg', 'prob_split_by_gain_pl', both the column and the split point
*       are decided at random. Note that, if passing value 1 (100%) with no sub-sampling and using the single-variable model,
*       every single tree will have the exact same splits.
*       Be aware that 'penalize_range' can also have a large impact when using 'prob_pick_pooled_gain'.
*       Important detail: if using either 'prob_pick_avg_gain' or 'prob_pick_pooled_gain', the distribution of
*       outlier scores is unlikely to be centered around 0.5.
* - prob_split_by_gain_pl
*       Probability of making each split by selecting a column at random and determining the split point as
*       that which gives the highest pooled gain. Not supported for the extended model as the splits are on
*       linear combinations of variables. See the documentation for parameter 'prob_pick_by_gain_pl' for more details.
* - min_gain
*       Minimum gain that a split threshold needs to produce in order to proceed with a split. Only used when the splits
*       are decided by a gain criterion (either pooled or averaged). If the highest possible gain in the evaluated
*       splits at a node is below this  threshold, that node becomes a terminal node.
* - missing_action
*       How to handle missing data at both fitting and prediction time. Options are a) "Divide" (for the single-variable
*       model only, recommended), which will follow both branches and combine the result with the weight given by the fraction of
*       the data that went to each branch when fitting the model, b) "Impute", which will assign observations to the
*       branch with the most observations in the single-variable model, or fill in missing values with the median
*       of each column of the sample from which the split was made in the extended model (recommended), c) "Fail" which will assume
*       there are no missing values and will trigger undefined behavior if it encounters any. In the extended model, infinite
*       values will be treated as missing. Note that passing "fail" might crash the process if there turn out to be
*       missing values, but will otherwise produce faster fitting and prediction times along with decreased model object sizes.
*       Models from [1], [2], [3], [4] correspond to "Fail" here.
* - cat_split_type
*       Whether to split categorical features by assigning sub-sets of them to each branch, or by assigning
*       a single category to a branch and the rest to the other branch. For the extended model, whether to
*       give each category a coefficient, or only one while the rest get zero.
* - new_cat_action
*       What to do after splitting a categorical feature when new data that reaches that split has categories that
*       the sub-sample from which the split was done did not have. Options are a) "Weighted" (recommended), which
*       in the single-variable model will follow both branches and combine the result with weight given by the fraction of the
*       data that went to each branch when fitting the model, and in the extended model will assign
*       them the median value for that column that was added to the linear combination of features, b) "Smallest", which will
*       assign all observations with unseen categories in the split to the branch that had fewer observations when
*       fitting the model, c) "Random", which will assing a branch (coefficient in the extended model) at random for
*       each category beforehand, even if no observations had that category when fitting the model. Ignored when
*       passing 'cat_split_type' = 'SingleCateg'.
* - all_perm
*       When doing categorical variable splits by pooled gain with 'ndim=1' (regular model),
*       whether to consider all possible permutations of variables to assign to each branch or not. If 'false',
*       will sort the categories by their frequency and make a grouping in this sorted order. Note that the
*       number of combinations evaluated (if 'true') is the factorial of the number of present categories in
*       a given column (minus 2). For averaged gain, the best split is always to put the second most-frequent
*       category in a separate branch, so not evaluating all  permutations (passing 'false') will make it
*       possible to select other splits that respect the sorted frequency order.
*       The total number of combinations must be a number that can fit into a 'size_t' variable - for x64-64
*       systems, this means no column can have more than 20 different categories if using 'all_perm=true',
*       but note that this is not checked within the function.
*       Ignored when not using categorical variables or not doing splits by pooled gain or using 'ndim>1'.
* - coef_by_prop
*       In the extended model, whether to sort the randomly-generated coefficients for categories
*       according to their relative frequency in the tree node. This might provide better results when using
*       categorical variables with too many categories, but is not recommended, and not reflective of
*       real "categorical-ness". Ignored for the regular model ('ndim=1') and/or when not using categorical
*       variables.
* - imputer (out)
*       Pointer to already-allocated imputer object, which can be used to produce missing value imputations
*       in new data. Pass NULL if no missing value imputations are required. Note that this is not related to
*       'missing_action' as missing values inside the model are treated differently and follow their own imputation
*       or division strategy.
* - min_imp_obs
*       Minimum number of observations with which an imputation value can be produced. Ignored if passing
*       'build_imputer' = 'false'.
* - depth_imp
*       How to weight observations according to their depth when used for imputing missing values. Passing
*       "Higher" will weigh observations higher the further down the tree (away from the root node) the
*       terminal node is, while "lower" will do the opposite, and "Sane" will not modify the weights according
*       to node depth in the tree. Implemented for testing purposes and not recommended to change
*       from the default. Ignored when not passing 'impute_nodes'.
* - weigh_imp_rows
*       How to weight node sizes when used for imputing missing values. Passing "Inverse" will weigh
*       a node inversely proportional to the number of observations that end up there, while "Proportional"
*       will weight them heavier the more observations there are, and "Flat" will weigh all nodes the same
*       in this regard regardless of how many observations end up there. Implemented for testing purposes
*       and not recommended to change from the default. Ignored when not passing 'impute_nodes'.
* - impute_at_fit
*       Whether to impute missing values in the input data as the model is being built. If passing 'true',
*       then 'sample_size' must be equal to 'nrows'. Values in the arrays passed to 'numeric_data',
*       'categ_data', and 'Xc', will get overwritten with the imputations produced.
* - random_seed
*       Seed that will be used to generate random numbers used by the model.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used.
*       Be aware that most of the operations are bound by memory bandwidth, which means that
*       adding more threads will not result in a linear speed-up. For some types of data
*       (e.g. large sparse matrices with small sample sizes), adding more threads might result
*       in only a very modest speed up (e.g. 1.5x faster with 4x more threads),
*       even if all threads look fully utilized.
*       Ignored when not building with OpenMP support.
* 
* Returns
* =======
* Will return macro 'EXIT_SUCCESS' (typically =0) upon completion.
* If the process receives an interrupt signal, will return instead
* 'EXIT_FAILURE' (typically =1). If you do not have any way of determining
* what these values correspond to, you can use the functions
* 'return_EXIT_SUCESS' and 'return_EXIT_FAILURE', which will return them
* as integers.
* 
* References
* ==========
* [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*     "Isolation forest."
*     2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
* [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*     "Isolation-based anomaly detection."
*     ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
* [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner.
*     "Extended Isolation Forest."
*     arXiv preprint arXiv:1811.02141 (2018).
* [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*     "On detecting clustered anomalies using SCiForest."
*     Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
* [5] https://sourceforge.net/projects/iforest/
* [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
* [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
* [8] Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
*/
template <class real_t, class sparse_ix>
int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees,
                size_t max_depth, size_t ncols_per_tree,
                bool   limit_depth, bool penalize_range,
                bool   standardize_dist, double tmat[],
                double output_depths[], bool standardize_depth,
                real_t col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, int nthreads)
{
    if (prob_pick_by_gain_avg < 0 || prob_split_by_gain_avg < 0 ||
        prob_pick_by_gain_pl < 0  || prob_split_by_gain_pl < 0)
        throw std::runtime_error("Cannot pass negative probabilities.\n");
    if (ndim == 0 && model_outputs == NULL)
        throw std::runtime_error("Must pass 'ndim>0' in the extended model.\n");


    /* TODO: this function should also accept the array as a memoryview with a
       leading dimension that might not correspond to the number of columns,
       so as to avoid having to make deep copies of memoryviews in python. */

    /* calculate maximum number of categories to use later */
    int max_categ = 0;
    for (size_t col = 0; col < ncols_categ; col++)
        max_categ = (ncat[col] > max_categ)? ncat[col] : max_categ;

    bool calc_dist = tmat != NULL;

    if (calc_dist || sample_size == 0)
        sample_size = nrows;

    /* put data in structs to shorten function calls */
    InputData<real_t, sparse_ix>
              input_data     = {numeric_data, ncols_numeric, categ_data, ncat, max_categ, ncols_categ,
                                nrows, ncols_numeric + ncols_categ, sample_weights,
                                weight_as_sample, col_weights,
                                Xc, Xc_ind, Xc_indptr,
                                0, 0, std::vector<double>(),
                                std::vector<char>(), 0};
    ModelParams model_params = {with_replacement, sample_size, ntrees, ncols_per_tree,
                                limit_depth? log2ceil(sample_size) : max_depth? max_depth : (sample_size - 1),
                                penalize_range, random_seed, weigh_by_kurt,
                                prob_pick_by_gain_avg, (model_outputs == NULL)? 0 : prob_split_by_gain_avg,
                                prob_pick_by_gain_pl,  (model_outputs == NULL)? 0 : prob_split_by_gain_pl,
                                min_gain, cat_split_type, new_cat_action, missing_action, all_perm,
                                (model_outputs != NULL)? 0 : ndim, (model_outputs != NULL)? 0 : ntry,
                                coef_type, coef_by_prop, calc_dist, (bool)(output_depths != NULL), impute_at_fit,
                                depth_imp, weigh_imp_rows, min_imp_obs};

    /* if using weights as sampling probability, build a binary tree for faster sampling */
    if (input_data.weight_as_sample && input_data.sample_weights != NULL)
    {
        build_btree_sampler(input_data.btree_weights_init, input_data.sample_weights,
                            input_data.nrows, input_data.log2_n, input_data.btree_offset);
    }

    /* if imputing missing values on-the-fly, need to determine which are missing */
    std::vector<ImputedData<sparse_ix>> impute_vec;
    std::unordered_map<size_t, ImputedData<sparse_ix>> impute_map;
    if (model_params.impute_at_fit)
        check_for_missing(input_data, impute_vec, impute_map, nthreads);

    /* store model data */
    if (model_outputs != NULL)
    {
        model_outputs->trees.resize(ntrees);
        model_outputs->trees.shrink_to_fit();
        model_outputs->new_cat_action = new_cat_action;
        model_outputs->cat_split_type = cat_split_type;
        model_outputs->missing_action = missing_action;
        model_outputs->exp_avg_depth  = expected_avg_depth(sample_size);
        model_outputs->exp_avg_sep = expected_separation_depth(model_params.sample_size);
        model_outputs->orig_sample_size = input_data.nrows;
    }

    else
    {
        model_outputs_ext->hplanes.resize(ntrees);
        model_outputs_ext->hplanes.shrink_to_fit();
        model_outputs_ext->new_cat_action = new_cat_action;
        model_outputs_ext->cat_split_type = cat_split_type;
        model_outputs_ext->missing_action = missing_action;
        model_outputs_ext->exp_avg_depth  = expected_avg_depth(sample_size);
        model_outputs_ext->exp_avg_sep = expected_separation_depth(model_params.sample_size);
        model_outputs_ext->orig_sample_size = input_data.nrows;
    }

    if (imputer != NULL)
        initialize_imputer(*imputer, input_data, ntrees, nthreads);

    /* initialize thread-private memory */
    if ((size_t)nthreads > ntrees)
        nthreads = (int)ntrees;
    #ifdef _OPENMP
        std::vector<WorkerMemory<ImputedData<sparse_ix>>> worker_memory(nthreads);
    #else
        std::vector<WorkerMemory<ImputedData<sparse_ix>>> worker_memory(1);
    #endif

    /* Global variable that determines if the procedure receives a stop signal */
    SignalSwitcher ss = SignalSwitcher();

    /* grow trees */
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic) shared(model_outputs, model_outputs_ext, worker_memory, input_data, model_params)
    for (size_t_for tree = 0; tree < ntrees; tree++)
    {
        if (interrupt_switch)
            continue; /* Cannot break with OpenMP==2.0 (MSVC) */

        if (
            model_params.impute_at_fit &&
            input_data.n_missing &&
            !worker_memory[omp_get_thread_num()].impute_vec.size() &&
            !worker_memory[omp_get_thread_num()].impute_map.size()
            )
        {
            #ifdef _OPENMP
            if (nthreads > 1)
            {
                worker_memory[omp_get_thread_num()].impute_vec = impute_vec;
                worker_memory[omp_get_thread_num()].impute_map = impute_map;
            }

            else
            #endif
            {
                worker_memory[0].impute_vec = std::move(impute_vec);
                worker_memory[0].impute_map = std::move(impute_map);
            }
        }

        fit_itree((model_outputs != NULL)? &model_outputs->trees[tree] : NULL,
                  (model_outputs_ext != NULL)? &model_outputs_ext->hplanes[tree] : NULL,
                  worker_memory[omp_get_thread_num()],
                  input_data,
                  model_params,
                  (imputer != NULL)? &(imputer->imputer_tree[tree]) : NULL,
                  tree);

        if ((model_outputs != NULL))
            model_outputs->trees[tree].shrink_to_fit();
        else
            model_outputs_ext->hplanes[tree].shrink_to_fit();
    }

    /* check if the procedure got interrupted */
    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    if ((model_outputs != NULL))
        model_outputs->trees.shrink_to_fit();
    else
        model_outputs_ext->hplanes.shrink_to_fit();

    /* if calculating similarity/distance, now need to reduce and average */
    if (calc_dist)
        gather_sim_result< PredictionData<real_t, sparse_ix>, InputData<real_t, sparse_ix> >
                         (NULL, &worker_memory,
                          NULL, &input_data,
                          model_outputs, model_outputs_ext,
                          tmat, NULL, 0,
                          model_params.ntrees, false,
                          standardize_dist, nthreads);

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* same for depths */
    if (output_depths != NULL)
    {
        #ifdef _OPENMP
        if (nthreads > 1)
        {
            for (WorkerMemory<ImputedData<sparse_ix>> &w : worker_memory)
            {
                if (w.row_depths.size())
                {
                    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(input_data, output_depths, w, worker_memory)
                    for (size_t_for row = 0; row < input_data.nrows; row++)
                        output_depths[row] += w.row_depths[row];
                }
            }
        }
        else
        #endif
        {
            std::copy(worker_memory[0].row_depths.begin(), worker_memory[0].row_depths.end(), output_depths);
        }

        if (standardize_depth)
        {
            double depth_divisor = (double)ntrees * ((model_outputs != NULL)?
                                                     model_outputs->exp_avg_depth : model_outputs_ext->exp_avg_depth);
            for (size_t_for row = 0; row < nrows; row++)
                output_depths[row] = std::exp2( - output_depths[row] / depth_divisor );
        }

        else
        {
            double ntrees_dbl = (double) ntrees;
            for (size_t_for row = 0; row < nrows; row++)
                output_depths[row] /= ntrees_dbl;
        }
    }

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* if imputing missing values, now need to reduce and write final values */
    if (model_params.impute_at_fit)
    {
        #ifdef _OPENMP
        if (nthreads > 1)
        {
            for (WorkerMemory<ImputedData<sparse_ix>> &w : worker_memory)
                combine_tree_imputations(w, impute_vec, impute_map, input_data.has_missing, nthreads);
        }

        else
        #endif
        {
            impute_vec = std::move(worker_memory[0].impute_vec);
            impute_map = std::move(worker_memory[0].impute_map);
        }

        apply_imputation_results(impute_vec, impute_map, *imputer, input_data, nthreads);
    }

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    return EXIT_SUCCESS;
}


/* Add additional trees to already-fitted isolation forest model
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the trees are are to be added to an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'. Note that this function is not thread-safe,
*       so it cannot be run in parallel for the same model object.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the trees are are to be added to an single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'. Note that this function is not thread-safe,
*       so it cannot be run in parallel for the same model object.
* - numeric_data
*       Pointer to numeric data to which to fit this additional tree. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.).
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
*       If the model from 'fit_iforest' was fit to numeric data, must pass numeric data with the same number
*       of columns, either as dense or as sparse arrays.
* - ncols_numeric
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - categ_data
*       Pointer to categorical data to which to fit this additional tree. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.).
*       Pass NULL if there are no categorical columns. The encoding must be the same as was used
*       in the data to which the model was fit.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
*       If the model from 'fit_iforest' was fit to categorical data, must pass categorical data with the same number
*       of columns and the same category encoding.
* - ncols_categ
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - ncat
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns.
* - Xc_indptr[ncols_numeric + 1]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end.
*       Pass NULL if there are no sparse numeric columns.
* - ndim
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - ntry
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - coef_type
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - sample_weights
*       Weights for the rows when adding this tree, either as sampling importances when using
*       sub-samples for each tree (i.e. passing weight '2' makes a row twice as likely to be included
*       in a random sub-sample), or as density measurement (i.e. passing weight '2' is the same as if
*       the row appeared twice, thus it's less of an outlier) - how this is taken is determined
*       through parameter 'weight_as_sample' that was passed to 'fit_iforest.
*       Pass NULL if the rows all have uniform weights.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'categ_data'.
* - max_depth
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - ncols_per_tree
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - limit_depth
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - penalize_range
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - col_weights
*       Sampling weights for each column, assuming all the numeric columns come before the categorical columns.
*       Ignored when picking columns by deterministic criterion.
*       If passing NULL, each column will have a uniform weight. Cannot be used when weighting by kurtosis.
* - weigh_by_kurt
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_by_gain_avg
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_split_by_gain_avg
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_by_gain_pl
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_split_by_gain_pl
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - min_gain
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - missing_action
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - cat_split_type
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - new_cat_action
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - depth_imp
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - weigh_imp_rows
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - all_perm
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - coef_by_prop
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - impute_nodes
*       Pointer to already-allocated imputation nodes for the tree that will be built. Note that the number of
*       entries in the imputation object must match the number of fitted trees when it is used.  Pass
*       NULL if no imputation node is required.
* - min_imp_obs
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - random_seed
*       Seed that will be used to generate random numbers used by the model.
*/
template <class real_t, class sparse_ix>
int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
             real_t numeric_data[],  size_t ncols_numeric,
             int    categ_data[],    size_t ncols_categ,    int ncat[],
             real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
             real_t sample_weights[], size_t nrows,
             size_t max_depth,     size_t ncols_per_tree,
             bool   limit_depth,   bool penalize_range,
             real_t col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
             double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, std::vector<ImputeNode> *impute_nodes, size_t min_imp_obs,
             uint64_t random_seed)
{
    if (prob_pick_by_gain_avg < 0 || prob_split_by_gain_avg < 0 ||
        prob_pick_by_gain_pl < 0  || prob_split_by_gain_pl < 0)
        throw std::runtime_error("Cannot pass negative probabilities.\n");
    if (ndim == 0 && model_outputs == NULL)
        throw std::runtime_error("Must pass 'ndim>0' in the extended model.\n");

    int max_categ = 0;
    for (size_t col = 0; col < ncols_categ; col++)
        max_categ = (ncat[col] > max_categ)? ncat[col] : max_categ;

    InputData<real_t, sparse_ix>
              input_data     = {numeric_data, ncols_numeric, categ_data, ncat, max_categ, ncols_categ,
                                nrows, ncols_numeric + ncols_categ, sample_weights,
                                false, col_weights,
                                Xc, Xc_ind, Xc_indptr,
                                0, 0, std::vector<double>(),
                                std::vector<char>(), 0};
    ModelParams model_params = {false, nrows, (size_t)1, ncols_per_tree,
                                max_depth? max_depth : (nrows - 1),
                                penalize_range, random_seed, weigh_by_kurt,
                                prob_pick_by_gain_avg, (model_outputs == NULL)? 0 : prob_split_by_gain_avg,
                                prob_pick_by_gain_pl,  (model_outputs == NULL)? 0 : prob_split_by_gain_pl,
                                min_gain, cat_split_type, new_cat_action, missing_action, all_perm,
                                (model_outputs != NULL)? 0 : ndim, (model_outputs != NULL)? 0 : ntry,
                                coef_type, coef_by_prop, false, false, false, depth_imp, weigh_imp_rows, min_imp_obs};

    std::unique_ptr<WorkerMemory<ImputedData<sparse_ix>>> workspace = std::unique_ptr<WorkerMemory<ImputedData<sparse_ix>>>(new WorkerMemory<ImputedData<sparse_ix>>);

    size_t last_tree;
    if (model_outputs != NULL)
    {
        last_tree = model_outputs->trees.size();
        model_outputs->trees.emplace_back();
    }

    else
    {
        last_tree = model_outputs_ext->hplanes.size();
        model_outputs_ext->hplanes.emplace_back();
    }

    fit_itree((model_outputs != NULL)? &model_outputs->trees.back() : NULL,
              (model_outputs_ext != NULL)? &model_outputs_ext->hplanes.back() : NULL,
              *workspace,
              input_data,
              model_params,
              impute_nodes,
              last_tree);

    if ((model_outputs != NULL))
        model_outputs->trees.back().shrink_to_fit();
    else
        model_outputs_ext->hplanes.back().shrink_to_fit();

    return EXIT_SUCCESS;
}

template <class InputData, class WorkerMemory>
void fit_itree(std::vector<IsoTree>    *tree_root,
               std::vector<IsoHPlane>  *hplane_root,
               WorkerMemory             &workspace,
               InputData                &input_data,
               ModelParams              &model_params,
               std::vector<ImputeNode> *impute_nodes,
               size_t                   tree_num)
{
    /* initialize array for depths if called for */
    if (!workspace.ix_arr.size() && model_params.calc_depth)
        workspace.row_depths.resize(input_data.nrows, 0);

    /* choose random sample of rows */
    if (!workspace.ix_arr.size()) workspace.ix_arr.resize(model_params.sample_size);
    if (input_data.log2_n > 0)
        workspace.btree_weights.assign(input_data.btree_weights_init.begin(),
                                       input_data.btree_weights_init.end());
    workspace.rnd_generator.seed(model_params.random_seed + tree_num);
    workspace.rbin  = std::uniform_real_distribution<double>(0, 1);
    sample_random_rows(workspace.ix_arr, input_data.nrows, model_params.with_replacement,
                       workspace.rnd_generator, workspace.ix_all,
                       (input_data.weight_as_sample)? input_data.sample_weights : NULL,
                       workspace.btree_weights, input_data.log2_n, input_data.btree_offset,
                       workspace.is_repeated);
    workspace.st  = 0;
    workspace.end = model_params.sample_size - 1;

    /* in some cases, it's not possible to use column weights even if they are given */
    bool avoid_col_weights = (tree_root != NULL && model_params.ndim < 2 &&
                              (model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl) >= 1)
                                ||
                             (hplane_root != NULL && model_params.ndim >= input_data.ncols_tot);
    if (input_data.col_weights != NULL && !avoid_col_weights)
        workspace.col_sampler.initialize(input_data.col_weights, input_data.ncols_tot);


    /* set expected tree size and add root node */
    {
        size_t exp_nodes = 2 * model_params.sample_size;
        if (model_params.sample_size >= (SIZE_MAX / (size_t)2))
            exp_nodes = SIZE_MAX;
        else if (model_params.max_depth <= (size_t)30)
            exp_nodes = std::min(exp_nodes, pow2(model_params.max_depth));
        if (tree_root != NULL)
        {
            tree_root->reserve(exp_nodes);
            tree_root->emplace_back();
        }
        else
        {
            hplane_root->reserve(exp_nodes);
            hplane_root->emplace_back();
        }
        if (impute_nodes != NULL)
        {
            impute_nodes->reserve(exp_nodes);
            impute_nodes->emplace_back((size_t) 0);
        }
    }

    /* initialize array with candidate categories if not already done */
    if (!workspace.categs.size())
        workspace.categs.resize(input_data.max_categ);

    /* IMPORTANT!!!!!
       The standard library implementation is likely going to use the ziggurat method
       for normal sampling, which has some state memory in the **distribution object itself**
       in addition to the state memory from the RNG engine. DO NOT avoid re-generating this
       object on each tree, despite being inefficient, because then it can cause seed
       irreproducibility when the number of splitting dimensions is odd and the number
       of threads is more than 1. This is a very hard issue to debug since everything
       works fine depending on the order in which trees are assigned to threads.
       DO NOT PUT THESE LINES BELOW THE NEXT IF. */
    if (hplane_root != NULL)
    {
        if (input_data.ncols_categ || model_params.coef_type == Normal)
            workspace.coef_norm = std::normal_distribution<double>(0, 1);
        if (model_params.coef_type == Uniform)
            workspace.coef_unif = std::uniform_real_distribution<double>(-1, 1);
    }

    /* for the extended model, initialize extra vectors and objects */
    if (hplane_root != NULL && !workspace.comb_val.size())
    {
        workspace.comb_val.resize(model_params.sample_size);
        workspace.col_take.resize(model_params.ndim);
        workspace.col_take_type.resize(model_params.ndim);

        if (input_data.ncols_numeric)
        {
            workspace.ext_offset.resize(input_data.ncols_tot);
            workspace.ext_coef.resize(input_data.ncols_tot);
            workspace.ext_mean.resize(input_data.ncols_tot);
        }

        if (input_data.ncols_categ)
        {
            workspace.ext_fill_new.resize(input_data.max_categ);
            switch(model_params.cat_split_type)
            {
                case SingleCateg:
                {
                    workspace.chosen_cat.resize(input_data.max_categ);
                    break;
                }

                case SubSet:
                {
                    workspace.ext_cat_coef.resize(input_data.ncols_tot);
                    for (std::vector<double> &v : workspace.ext_cat_coef)
                        v.resize(input_data.max_categ);
                    break;
                }
            }
        }

        workspace.ext_fill_val.resize(input_data.ncols_tot);

    }

    /* if it contains missing values, also have to set an array of weights,
       which will be modified during iterations when there are NAs.
       If there are already density weights, need to standardize them to sum up to
       the sample size here */
    long double weight_scaling = 0;
    if (model_params.missing_action == Divide || (input_data.sample_weights != NULL && !input_data.weight_as_sample))
    {
        workspace.weights_map.clear();

        /* if the sub-sample size is small relative to the full sample size, use a mapping */
        if (model_params.sample_size < input_data.nrows / 4)
        {
            if (input_data.sample_weights != NULL && !input_data.weight_as_sample)
            {
                for (const size_t ix : workspace.ix_arr)
                {
                    weight_scaling += input_data.sample_weights[ix];
                    workspace.weights_map[ix] = input_data.sample_weights[ix];
                }
                weight_scaling = (long double)model_params.sample_size / weight_scaling;
                for (auto &w : workspace.weights_map)
                    w.second *= weight_scaling;

            }

            else
            {
                for (const size_t ix : workspace.ix_arr)
                    workspace.weights_map[ix] = 1;
            }

        }

        /* if the sub-sample size is large, fill a full array matching to the sample size */
        else
        {
            if (!workspace.weights_arr.size())
            {
                if (input_data.sample_weights != NULL && !input_data.weight_as_sample)
                {
                    workspace.weights_arr.assign(input_data.sample_weights, input_data.sample_weights + input_data.nrows);
                    weight_scaling = std::accumulate(workspace.ix_arr.begin(),
                                                     workspace.ix_arr.end(),
                                                     (long double)0,
                                                     [&input_data](const long double a, const size_t b){return a + (long double)input_data.sample_weights[b];}
                                                     );
                    weight_scaling = (long double)model_params.sample_size / weight_scaling;
                    for (double &w : workspace.weights_arr)
                        w *= weight_scaling;
                }

                else
                {
                    workspace.weights_arr.resize(input_data.nrows, (double)1);
                }

            }

            else
            {
                if (input_data.sample_weights != NULL && !input_data.weight_as_sample)
                {
                    for (const size_t ix : workspace.ix_arr)
                    {
                        weight_scaling += input_data.sample_weights[ix];
                        workspace.weights_arr[ix] = input_data.sample_weights[ix];
                    }
                    weight_scaling = (long double)model_params.sample_size / weight_scaling;
                    for (double &w : workspace.weights_arr)
                        w *= weight_scaling;

                }

                else
                {
                    /* Note: while not all of them need to be overwritten, this is faster
                       (sub-sample size was already determined to be at least 1/4 of the sample size) */
                    std::fill(workspace.weights_arr.begin(), workspace.weights_arr.end(), (double)1);
                }
            }
        }
    }

    /* if producing distance/similarity, also need to initialize the triangular matrix */
    if (model_params.calc_dist && !workspace.tmat_sep.size())
        workspace.tmat_sep.resize((input_data.nrows * (input_data.nrows - 1)) / 2, 0);

    /* make space for buffers if not already allocated */
    if (
            (model_params.prob_split_by_gain_avg > 0 || model_params.prob_pick_by_gain_avg > 0 ||
             model_params.prob_split_by_gain_pl > 0  || model_params.prob_pick_by_gain_pl > 0  ||
             model_params.weigh_by_kurt || hplane_root != NULL)
                &&
            (!workspace.buffer_dbl.size() && !workspace.buffer_szt.size() && !workspace.buffer_chr.size())
        )
    {
        size_t min_size_dbl = 0;
        size_t min_size_szt = 0;
        size_t min_size_chr = 0;

        bool gain = model_params.prob_split_by_gain_avg > 0 || model_params.prob_pick_by_gain_avg > 0 ||
                    model_params.prob_split_by_gain_pl > 0  || model_params.prob_pick_by_gain_pl > 0;

        if (input_data.ncols_categ)
        {
            min_size_szt = 2 * input_data.max_categ;
            min_size_dbl = input_data.max_categ + 1;
            if (gain && model_params.cat_split_type == SubSet)
                min_size_chr = input_data.max_categ;
        }

        if (input_data.Xc_indptr != NULL && gain)
        {
            min_size_szt = std::max(min_size_szt, model_params.sample_size);
            min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
        }

        if (gain && (model_params.ntry > 1 ||
                     model_params.prob_pick_by_gain_avg > 0 ||
                     model_params.prob_split_by_gain_avg > 0 ||
                     (model_params.ndim < 2 && model_params.prob_pick_by_gain_pl > 0) ||
                     model_params.min_gain > 0)
        )
        {
            min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
            if (model_params.ndim < 2 && input_data.Xc_indptr != NULL)
                min_size_dbl = std::max(min_size_dbl, (size_t)2*model_params.sample_size);
        }

        /* for the extended model */
        if (hplane_root != NULL)
        {
            min_size_dbl = std::max(min_size_dbl, pow2(log2ceil(input_data.ncols_tot) + 1));
            if (model_params.missing_action != Fail)
            {
                min_size_szt = std::max(min_size_szt, model_params.sample_size);
                min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
            }

            if (input_data.ncols_categ && model_params.cat_split_type == SubSet)
            {
                min_size_szt = std::max(min_size_szt, 2 * (size_t)input_data.max_categ + 1);
                min_size_dbl = std::max(min_size_dbl, (size_t)input_data.max_categ);
            }

            if (model_params.weigh_by_kurt)
                min_size_szt = std::max(min_size_szt, input_data.ncols_tot);

            if (gain && (workspace.weights_arr.size() || workspace.weights_map.size()))
            {
                workspace.sample_weights.resize(model_params.sample_size);
                min_size_szt = std::max(min_size_szt, model_params.sample_size);
            }
        }

        /* now resize */
        if (workspace.buffer_dbl.size() < min_size_dbl)
            workspace.buffer_dbl.resize(min_size_dbl);

        if (workspace.buffer_szt.size() < min_size_szt)
            workspace.buffer_szt.resize(min_size_szt);

        if (workspace.buffer_chr.size() < min_size_chr)
            workspace.buffer_chr.resize(min_size_chr);

        /* for guided column choice, need to also remember the best split so far */
        if (
            model_params.cat_split_type == SubSet &&
            (
                model_params.prob_pick_by_gain_avg  || 
                model_params.prob_pick_by_gain_pl
            )
           )
        {
            workspace.this_split_categ.resize(input_data.max_categ);
        }

    }

    /* weigh columns by kurtosis in the sample if required */
    if (model_params.weigh_by_kurt && !avoid_col_weights)
    {
        std::vector<double> kurt_weights(input_data.ncols_numeric + input_data.ncols_categ, 0.);

        if (model_params.ncols_per_tree == 0 || model_params.ncols_per_tree >= input_data.ncols_tot)
        {

            if (input_data.Xc_indptr == NULL)
            {

                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                        kurt_weights[col] = calc_kurtosis(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                          input_data.numeric_data + col * input_data.nrows,
                                                          model_params.missing_action);
                    else if (workspace.weights_arr.size())
                        kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   input_data.numeric_data + col * input_data.nrows,
                                                                   model_params.missing_action, workspace.weights_arr);
                    else
                        kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   input_data.numeric_data + col * input_data.nrows,
                                                                   model_params.missing_action, workspace.weights_map);
                }
            }

            else
            {
                std::sort(workspace.ix_arr.begin(), workspace.ix_arr.end());
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                        kurt_weights[col] = calc_kurtosis(workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                          input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                          model_params.missing_action);
                    else if (workspace.weights_arr.size())
                        kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                   input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                   model_params.missing_action, workspace.weights_arr);
                    else
                        kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                   input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                   model_params.missing_action, workspace.weights_map);
                }
            }

            for (size_t col = 0; col < input_data.ncols_categ; col++)
            {
                if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis(workspace.ix_arr.data(), workspace.st, workspace.end,
                                      input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                      workspace.buffer_szt.data(), workspace.buffer_dbl.data(),
                                      model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator);
                else if (workspace.weights_arr.size())
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                               workspace.buffer_dbl.data(),
                                               model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                               workspace.weights_arr);
                else
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                               workspace.buffer_dbl.data(),
                                               model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                               workspace.weights_map);
            }

            for (auto &w : kurt_weights) w = std::fmax(1e-8, -1. + w);
            workspace.col_sampler.initialize(kurt_weights.data(), kurt_weights.size());
        }

        

        else
        {
            std::vector<size_t> cols_take(model_params.ncols_per_tree);
            std::vector<size_t> buffer1;
            std::vector<bool> buffer2;
            sample_random_rows(cols_take, input_data.ncols_tot, false,
                               workspace.rnd_generator, buffer1,
                               (double*)NULL, kurt_weights, /* <- will not get used */
                               (size_t)0, (size_t)0, buffer2);

            if (input_data.Xc_indptr != NULL)
                std::sort(workspace.ix_arr.begin(), workspace.ix_arr.end());

            for (size_t col : cols_take)
            {
                if (col < input_data.ncols_numeric)
                {
                    if (input_data.Xc_indptr == NULL)
                    {
                        if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                            kurt_weights[col] = calc_kurtosis(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                              input_data.numeric_data + col * input_data.nrows,
                                                              model_params.missing_action);
                        else if (workspace.weights_arr.size())
                            kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                       input_data.numeric_data + col * input_data.nrows,
                                                                       model_params.missing_action, workspace.weights_arr);
                        else
                            kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                       input_data.numeric_data + col * input_data.nrows,
                                                                       model_params.missing_action, workspace.weights_map);
                        kurt_weights[col] = std::fmax(1e-8, -1. + kurt_weights[col]);
                    }

                    else
                    {
                        if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                            kurt_weights[col] = calc_kurtosis(workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                              model_params.missing_action);
                        else if (workspace.weights_arr.size())
                            kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                       input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                       model_params.missing_action, workspace.weights_arr);
                        else
                            kurt_weights[col] = calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                       input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                       model_params.missing_action, workspace.weights_map);
                        kurt_weights[col] = std::fmax(1e-8, -1. + kurt_weights[col]);
                    }
                }

                else
                {
                    if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                        kurt_weights[col] =
                            calc_kurtosis(workspace.ix_arr.data(), workspace.st, workspace.end,
                                          input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                          input_data.ncat[col - input_data.ncols_numeric],
                                          workspace.buffer_szt.data(), workspace.buffer_dbl.data(),
                                          model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator);
                    else if (workspace.weights_arr.size())
                        kurt_weights[col] =
                            calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                  input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                                  input_data.ncat[col - input_data.ncols_numeric],
                                                  workspace.buffer_dbl.data(),
                                                  model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                                  workspace.weights_arr);
                    else
                        kurt_weights[col] =
                            calc_kurtosis_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                  input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                                  input_data.ncat[col - input_data.ncols_numeric],
                                                  workspace.buffer_dbl.data(),
                                                  model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                                  workspace.weights_map);
                    kurt_weights[col] = std::fmax(1e-8, -1. + kurt_weights[col]);
                }
            }

            workspace.col_sampler.initialize(kurt_weights.data(), kurt_weights.size());
        }
    }

    workspace.col_sampler.initialize(input_data.ncols_tot);
    /* TODO: this can be done more efficiently when sub-sampling columns */
    if (! (model_params.weigh_by_kurt && !avoid_col_weights))
        workspace.col_sampler.leave_m_cols(model_params.ncols_per_tree, workspace.rnd_generator);
    workspace.try_all = false;
    if (hplane_root != NULL && model_params.ndim >= input_data.ncols_tot)
        workspace.try_all = true;


    if (tree_root != NULL)
        split_itree_recursive(*tree_root,
                              workspace,
                              input_data,
                              model_params,
                              impute_nodes,
                              0);
    else
        split_hplane_recursive(*hplane_root,
                               workspace,
                               input_data,
                               model_params,
                               impute_nodes,
                               0);

    /* if producing imputation structs, only need to keep the ones for terminal nodes */
    if (impute_nodes != NULL)
        drop_nonterminal_imp_node(*impute_nodes, tree_root, hplane_root);
}
