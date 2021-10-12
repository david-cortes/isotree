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

/* Standard headers */
#include <cstddef>
#include <cstdint>
#include <vector>
using std::size_t;

/*  The library has overloaded functions supporting different input types.
    Note that, while 'float' type is supported, it will
    be slower to fit models to them as the models internally use
    'double' and 'long double', and it's not recommended to use.

    In order to use the library with different types than the ones
    suggested here, add something like this before including the
    library header:
      #define real_t float
      #define sparse_ix int
      #include "isotree.hpp"
    The header may be included multiple times if required. */
#ifndef real_t
    #define real_t double     /* supported: float, double */
#endif
#ifndef sparse_ix
    #define sparse_ix size_t  /* supported: int, int64_t, size_t */
#endif

#ifndef ISOTREE_H
#define ISOTREE_H

#ifdef _WIN32
    #define ISOTREE_EXPORTED __declspec(dllimport)
#else
    #define ISOTREE_EXPORTED 
#endif


/* Types used through the package - zero is the suggested value (when appropriate) */
typedef enum  NewCategAction {Weighted=0,  Smallest=11,    Random=12}  NewCategAction; /* Weighted means Impute in the extended model */
typedef enum  MissingAction  {Divide=21,   Impute=22,      Fail=0}     MissingAction;  /* Divide is only for non-extended model */
typedef enum  ColType        {Numeric=31,  Categorical=32, NotUsed=0}  ColType;
typedef enum  CategSplit     {SubSet=0,    SingleCateg=41}             CategSplit;
typedef enum  GainCriterion  {Averaged=51, Pooled=52,      NoCrit=0}   Criterion;      /* For guided splits */
typedef enum  CoefType       {Uniform=61,  Normal=0}                   CoefType;       /* For extended model */
typedef enum  UseDepthImp    {Lower=71,    Higher=0,       Same=72}    UseDepthImp;    /* For NA imputation */
typedef enum  WeighImpRows   {Inverse=0,   Prop=81,        Flat=82}    WeighImpRows;   /* For NA imputation */

/* Notes about new categorical action:
*  - For single-variable case, if using 'Smallest', can then pass data at prediction time
*    having categories that were never in the training data (as an integer higher than 'ncat'
*    for that column), but if using 'Random' or 'Weighted', these must be passed as NA (int < 0)
*  - For extended case, 'Weighted' becomes a weighted imputation instead, and if using either
*    'Weighted' or 'Smallest', can pass newer, unseen categories at prediction time too.
*  - If using 'Random', cannot pass new categories at prediction time.
*  - If using 'Weighted' for single-variable case, cannot predict similarity with a value
*    for MissingAction other than 'Divide'. */



/* Structs that are output (modified) from the main function */
typedef struct IsoTree {
    ColType  col_type = NotUsed;
    size_t   col_num;
    double   num_split;
    std::vector<char> cat_split;
    int      chosen_cat;
    size_t   tree_left;
    size_t   tree_right;
    double   pct_tree_left;
    double   score;        /* will not be integer when there are weights or early stop */
    double   range_low  = -HUGE_VAL;
    double   range_high =  HUGE_VAL;
    double   remainder; /* only used for distance/similarity */

    IsoTree() = default;

} IsoTree;

typedef struct IsoHPlane {
    std::vector<size_t>   col_num;
    std::vector<ColType>  col_type;
    std::vector<double>   coef;
    std::vector<double>   mean;
    std::vector<std::vector<double>> cat_coef;
    std::vector<int>      chosen_cat;
    std::vector<double>   fill_val;
    std::vector<double>   fill_new;

    double   split_point;
    size_t   hplane_left;
    size_t   hplane_right;
    double   score;        /* will not be integer when there are weights or early stop */
    double   range_low  = -HUGE_VAL;
    double   range_high =  HUGE_VAL;
    double   remainder; /* only used for distance/similarity */

    IsoHPlane() = default;
} IsoHPlane;

typedef struct IsoForest {
    std::vector< std::vector<IsoTree> > trees;
    NewCategAction    new_cat_action;
    CategSplit        cat_split_type;
    MissingAction     missing_action;
    double            exp_avg_depth;
    double            exp_avg_sep;
    size_t            orig_sample_size;
    bool              has_range_penalty;
    IsoForest() = default;
} IsoForest;

typedef struct ExtIsoForest {
    std::vector< std::vector<IsoHPlane> > hplanes;
    NewCategAction    new_cat_action;
    CategSplit        cat_split_type;
    MissingAction     missing_action;
    double            exp_avg_depth;
    double            exp_avg_sep;
    size_t            orig_sample_size;
    bool              has_range_penalty;
    ExtIsoForest() = default;
} ExtIsoForest;

typedef struct ImputeNode {
    std::vector<double>  num_sum;
    std::vector<double>  num_weight;
    std::vector<std::vector<double>>  cat_sum;
    std::vector<double>  cat_weight;
    size_t               parent;
    ImputeNode() = default;
} ImputeNode; /* this is for each tree node */

typedef struct Imputer {
    size_t               ncols_numeric;
    size_t               ncols_categ;
    std::vector<int>     ncat;
    std::vector<std::vector<ImputeNode>> imputer_tree;
    std::vector<double>  col_means;
    std::vector<int>     col_modes;
    Imputer() = default;
} Imputer;

#endif /* ISOTREE_H */

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
*       The largest value here should be smaller than the largest possible value of 'size_t'.
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
*       (as proposed in [4]) or according to a uniform distribution ~ Unif(-1, +1) as proposed in [3]. Ignored for the
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
*       Models that use 'prob_pick_by_gain_pl' or 'prob_pick_by_gain_avg' are likely to benefit from
*       deeper trees (larger 'max_depth'), but deeper trees can result in much slower model fitting and
*       predictions.
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
*       tree depth). Default setting for [1], [2], [3], [4] is 'true', but it's recommended to pass 'false' here
*       and higher values for 'max_depth' if using the model for purposes other than outlier detection.
*       Note that, if passing 'limit_depth=true', then 'max_depth' is ignored.
* - penalize_range
*       Whether to penalize (add -1 to the terminal depth) observations at prediction time that have a value
*       of the chosen split variable (linear combination in extended model) that falls outside of a pre-determined
*       reasonable range in the data being split (given by 2 * range in data and centered around the split point),
*       as proposed in [4] and implemented in the authors' original code in [5]. Not used in single-variable model
*       when splitting by categorical variables. Note that this can make a very large difference in the results
*       when using `prob_pick_pooled_gain`.
* - standardize_data
*       Whether to standardize the features at each node before creating a linear combination of them as suggested
*       in [4]. This is ignored when using 'ndim=1'.
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
*       outliers in the training data, but generalize poorly to outliers in new data (including out-of-bag samples
*       for each tree) and to values of variables outside of the ranges from the training data. Passing small 
*       'sample_size' and high values of this parameter will tend to flag too many outliers. When splits are not
*       made according to any of 'prob_pick_by_gain_avg', 'prob_pick_by_gain_pl', 'prob_split_by_gain_avg', 'prob_split_by_gain_pl',
*       both the column and the split point are decided at random. Note that, if passing value 1 (100%) with no 
*       sub-sampling and using the single-variable model, every single tree will have the exact same splits.
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
ISOTREE_EXPORTED
int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees,
                size_t max_depth,   size_t ncols_per_tree,
                bool   limit_depth, bool penalize_range, bool standardize_data,
                bool   standardize_dist, double tmat[],
                real_t output_depths[], bool standardize_depth,
                double col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, int nthreads);



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
* - imputer
*       Pointer to already-allocated imputer object, as it was output from function 'fit_model' while
*       producing either 'model_outputs' or 'model_outputs_ext'.
*       Pass NULL if the model was built without imputer.
* - min_imp_obs
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - random_seed
*       Seed that will be used to generate random numbers used by the model.
*/
ISOTREE_EXPORTED
int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
             real_t numeric_data[],  size_t ncols_numeric,
             int    categ_data[],    size_t ncols_categ,    int ncat[],
             real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
             real_t sample_weights[], size_t nrows,
             size_t max_depth,     size_t ncols_per_tree,
             bool   limit_depth,   bool penalize_range, bool standardize_data,
             real_t col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
             double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, Imputer *imputer, size_t min_imp_obs,
             uint64_t random_seed);


/* Predict outlier score, average depth, or terminal node numbers
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data for which to make predictions. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data for which to make predictions. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no categorical columns.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
* - is_col_major
*       Whether 'numeric_data' and 'categ_data' come in column-major order, like the data to which the
*       model was fit. If passing 'false', will assume they are in row-major order. Note that most of
*       the functions in this library work only with column-major order, but here both are suitable
*       and row-major is preferred. Both arrays must have the same orientation (row/column major).
*       If there is numeric sparse data in combination with categorical dense data and there are many
*       rows, it is recommended to pass the categorical data in column major order, as it will take
*       a faster route.
*       If passing 'is_col_major=true', must also provide 'ld_numeric' and/or 'ld_categ'.
* - ld_numeric
*       Leading dimension of the array 'numeric_data', if it is passed in row-major format.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'numeric_data + n*ld_numeric'). If passing
*       'numeric_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows. This is ignored when passing numeric
*       data in sparse format.
* - ld_categ
*       Leading dimension of the array 'categ_data', if it is passed in row-major format.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'categ_data + n*ld_categ'). If passing
*       'categ_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows.
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSC format.
* - Xc_indptr[ncols_categ + 1]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end.
*       Pass NULL if there are no sparse numeric columns in CSC format.
* - Xr[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSR format (row-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'. 
* - Xr_ind[nnz]
*       Pointer to column indices to which each non-zero entry in 'Xr' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSR format.
* - Xr_indptr[nrows + 1]
*       Pointer to row index pointers that tell at entry [row] where does row 'row'
*       start and at entry [row + 1] where does row 'row' end.
*       Pass NULL if there are no sparse numeric columns in CSR format.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'Xr, 'categ_data'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* - standardize
*       Whether to standardize the average depths for each row according to their relative magnitude
*       compared to the expected average, in order to obtain an outlier score. If passing 'false',
*       will output the average depth instead.
*       Ignored when not passing 'output_depths'.
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - output_depths[nrows] (out)
*       Pointer to array where the output average depths or outlier scores will be written into
*       (the return type is controlled according to parameter 'standardize').
*       Must already be initialized to zeros, and should always be passed when calling
*       this function (it is not optional).
* - tree_num[nrows * ntrees] (out)
*       Pointer to array where the output terminal node numbers will be written into.
*       Note that the mapping between tree node and terminal tree node is not stored in
*       the model object for efficiency reasons, so this mapping will be determined on-the-fly
*       when passing this parameter, and as such, there will be some overhead regardless of
*       the actual number of rows. Output will be in column-major order ([nrows, ntrees]).
*       This will not be calculable when using 'ndim==1' alongside with either
*       'missing_action==Divide' or 'new_categ_action=Weighted'.
*       Pass NULL if this type of output is not needed.
* - per_tree_depths[nrows * ntrees] (out)
*       Pointer to array where to output per-tree depths or expected depths for each row.
*       Note that these will not include range penalities ('penalize_range=true').
*       Output will be in row-major order ([nrows, ntrees]).
*       This will not be calculable when using 'ndim==1' alongside with either
*       'missing_action==Divide' or 'new_categ_action=Weighted'.
*       Pass NULL if this type of output is not needed.
*/
ISOTREE_EXPORTED
void predict_iforest(real_t numeric_data[], int categ_data[],
                     bool is_col_major, size_t ld_numeric, size_t ld_categ,
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[],
                     double per_tree_depths[]);



/* Get the number of nodes present in a given model, per tree
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'.
* - n_nodes[ntrees] (out)
*       Number of nodes in tree of the model, including non-terminal nodes.
* - n_terminal[ntrees] (out)
*       Number of terminal nodes in each tree of the model.
* - nthreads
*       Number of parallel threads to use.
*/
ISOTREE_EXPORTED void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads);
ISOTREE_EXPORTED void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads);



/* Calculate distance or similarity between data points
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data for which to make calculations. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.),
*       and the column order must be the same as in the data that was used to fit the model.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data for which to make calculations. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.),
*       and the column order must be the same as in the data that was used to fit the model.
*       Pass NULL if there are no categorical columns.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSC format.
* - Xc_indptr[ncols_categ + 1]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end.
*       Pass NULL if there are no sparse numeric columns in CSC format.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'Xr, 'categ_data'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* - assume_full_distr
*       Whether to assume that the fitted model represents a full population distribution (will use a
*       standardizing criterion assuming infinite sample, and the results of the similarity between two points
*       at prediction time will not depend on the prescence of any third point that is similar to them, but will
*       differ more compared to the pairwise distances between points from which the model was fit). If passing
*       'false', will calculate pairwise distances as if the new observations at prediction time were added to
*       the sample to which each tree was fit, which will make the distances between two points potentially vary
*       according to other newly introduced points.
* - standardize_dist
*       Whether to standardize the resulting average separation depths between rows according
*       to the expected average separation depth in a similar way as when predicting outlierness,
*       in order to obtain a standardized distance. If passing 'false', will output the average
*       separation depth instead.
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the calculations are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the calculations are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - tmat[nrows * (nrows - 1) / 2] (out)
*       Pointer to array where the resulting pairwise distances or average separation depths will
*       be written into. As the output is a symmetric matrix, this function will only fill in the
*       upper-triangular part, in which entry 0 <= i < j < n will be located at position
*           p(i,j) = (i * (n - (i+1)/2) + j - i - 1).
*       Can be converted to a dense square matrix through function 'tmat_to_dense'.
*       The array must already be initialized to zeros.
*       If calculating distance/separation from a group of points to another group of points,
*       pass NULL here and use 'rmat' instead.
* - rmat[nrows1 * nrows2] (out)
*       Pointer to array where to write the distances or separation depths between each row in
*       one set of observations and each row in a different set of observations. If doing these
*       calculations for all pairs of observations/rows, pass 'rmat' instead.
*       Will take the first group of observations as the rows in this matrix, and the second
*       group as the columns. The groups are assumed to be in the same data arrays, with the
*       first group corresponding to the earlier rows there.
*       This matrix will be used in row-major order (i.e. entries 1..n_from contain the first row).
*       Must be already initialized to zeros.
*       Ignored when 'tmat' is passed.
* - n_from
*       When calculating distances between two groups of points, this indicates the number of
*       observations/rows belonging to the first group (the rows in 'rmat'), which will be
*       assumed to be the first 'n_from' rows.
*       Ignored when 'tmat' is passed.
*/
ISOTREE_EXPORTED
void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, int nthreads, bool assume_full_distr, bool standardize_dist,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from);


/* Impute missing values in new data
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric] (in, out)
*       Pointer to numeric data in which missing values will be imputed. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
*       Imputations will overwrite values in this same array.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data in which missing values will be imputed. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no categorical columns.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
*       Imputations will overwrite values in this same array.
* - is_col_major
*       Whether 'numeric_data' and 'categ_data' come in column-major order, like the data to which the
*       model was fit. If passing 'false', will assume they are in row-major order. Note that most of
*       the functions in this library work only with column-major order, but here both are suitable
*       and row-major is preferred. Both arrays must have the same orientation (row/column major).
* - ncols_categ
*       Number of categorical columns in the data.
* - ncat[ncols_categ]
*       Number of categories in each categorical column. E.g. if the highest code for a column is '4',
*       the number of categories for that column is '5' (zero is one category).
*       Must be the same as was passed to 'fit_iforest'.
* - Xr[nnz] (in, out)
*       Pointer to numeric data in sparse numeric matrix in CSR format (row-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
*       Imputations will overwrite values in this same array.
* - Xr_ind[nnz]
*       Pointer to column indices to which each non-zero entry in 'Xr' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSR format.
* - Xr_indptr[nrows + 1]
*       Pointer to row index pointers that tell at entry [row] where does row 'row'
*       start and at entry [row + 1] where does row 'row' end.
*       Pass NULL if there are no sparse numeric columns in CSR format.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'Xr, 'categ_data'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - impute_nodes
*       Pointer to fitted imputation node obects for the same trees as in 'model_outputs' or 'model_outputs_ext',
*       as produced from function 'fit_iforest',
*/
ISOTREE_EXPORTED
void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer);


/* Append trees from one model into another
* 
* Parameters
* ==========
* - model (in, out)
*       Pointer to isolation forest model wich has already been fit through 'fit_iforest'.
*       The trees from 'other' will be merged into this (will be at the end of vector member 'trees').
*       Both 'model' and 'other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - other
*       Pointer to isolation forest model which has already been fit through 'fit_iforest'.
*       The trees from this object will be added into 'model' (this object will not be modified).
*       Both 'model' and 'other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - ext_model (in, out)
*       Pointer to extended isolation forest model which has already been fit through 'fit_iforest'.
*       The trees/hyperplanes from 'ext_other' will be merged into this (will be at the end of vector member 'hplanes').
*       Both 'ext_model' and 'ext_other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - ext_other
*       Pointer to extended isolation forest model which has already been fit through 'fit_iforest'.
*       The trees/hyperplanes from this object will be added into 'ext_model' (this object will not be modified).
*       Both 'ext_model' and 'ext_other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - imputer (in, out)
*       Pointer to imputation object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest'.
*       The imputation nodes from 'iother' will be merged into this (will be at the end of vector member 'imputer_tree').
*       Hyperparameters related to imputation might differ between 'imputer' and 'iother' ('imputer' will preserve its
*       hyperparameters after the merge).
*       Pass NULL if this is not to be used.
* - iother
*       Pointer to imputation object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest'.
*       The imputation nodes from this object will be added into 'imputer' (this object will not be modified).
*       Hyperparameters related to imputation might differ between 'imputer' and 'iother' ('imputer' will preserve its
*       hyperparameters after the merge).
*       Pass NULL if this is not to be used.
*/
ISOTREE_EXPORTED
void merge_models(IsoForest*     model,      IsoForest*     other,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                  Imputer*       imputer,    Imputer*       iother);

/* Create a model containing a sub-set of the trees from another model
* 
* Parameters
* ==========
* - model (in)
*       Pointer to isolation forest model wich has already been fit through 'fit_iforest',
*       from which the desired trees will be copied into a new model object.
*       Pass NULL if using the extended model.
* - ext_model (in)
*       Pointer to extended isolation forest model which has already been fit through 'fit_iforest',
*       from which the desired trees will be copied into a new model object.
*       Pass NULL if using the single-variable model.
* - imputer (in)
*       Pointer to imputation object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest'.
*       Pass NULL if the model was built without an imputer.
* - model_new (out)
*       Pointer to already-allocated isolation forest model, which will be reset and to
*       which the selected trees from 'model' will be copied.
*       Pass NULL if using the extended model.
* - ext_model_new (out)
*       Pointer to already-allocated extended isolation forest model, which will be reset and to
*       which the selected hyperplanes from 'ext_model' will be copied.
*       Pass NULL if using the single-variable model.
* - imputer_new (out)
*       Pointer to already-allocated imputation object, which will be reset and to
*       which the selected nodes from 'imputer' (matching to those of either 'model'
*       or 'ext_model') will be copied.
*       Pass NULL if the model was built without an imputer.
*/
ISOTREE_EXPORTED
void subset_model(IsoForest*     model,      IsoForest*     model_new,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                  Imputer*       imputer,    Imputer*       imputer_new,
                  size_t *trees_take, size_t ntrees_take);


/* Functions to inspect serialized objects
* 
* Parameters
* ==========
* - serialized_bytes (in)
*       A model from this library, serialized through the functions available since
*       version 0.3.0, in any of the varieties offered by the library (as separate
*       objects or as combined objects with metadata).
* - is_isotree_model (out)
*       Whether the input 'serialized_bytes' is a serialized model from this library.
* - is_compatible (out)
*       Whether the serialized model is compatible (i.e. can be de-serialized) with the
*       current setup.
*       Serialized models are compatible between:
*        - Different operating systems.
*        - Different compilers.
*        - Systems with different 'size_t' width (e.g. 32-bit and 64-bit),
*          as long as the file was produced on a system that was either 32-bit or 64-bit,
*          and as long as each saved value fits within the range of the machine's 'size_t' type.
*        - Systems with different 'int' width,
*          as long as the file was produced on a system that was 16-bit, 32-bit, or 64-bit,
*          and as long as each saved value fits within the range of the machine's int type.
*        - Systems with different bit endianness (e.g. x86 and PPC64 in non-le mode).
*        - Versions of this package from 0.3.0 onwards.
*       But are not compatible between:
*        - Systems with different floating point numeric representations
*          (e.g. standard IEEE754 vs. a base-10 system).
*        - Versions of this package earlier than 0.3.0.
*       This pretty much guarantees that a given file can be serialized and de-serialized
*       in the same machine in which it was built, regardless of how the library was compiled.
*       Reading a serialized model that was produced in a platform with different
*       characteristics (e.g. 32-bit vs. 64-bit) will be much slower however.
* - has_combined_objects (out)
*       Whether the serialized model is in the format of combined objects (as produced by the
*       functions named 'serialized_combined') or in the format of separate objects (as produced
*       by the functions named 'serialized_<model>').
*       If if is in the format of combined objects, must be de-serialized through the functions
*       named 'deserialize_combined'; ohterwise, must be de-serialized through the functions
*       named 'deserialize_<model>'.
*       Note that the Python and R interfaces of this library use the combined objects format
*       when serializing to files.
* - has_IsoForest (out)
*       Whether the serialized bytes include an 'IsoForest' object. If it has 'has_combined_objects=true',
*       might include additional objects.
* - has_ExtIsoForest (out)
*       Whether the serialized bytes include an 'ExtIsoForest' object. If it has 'has_combined_objects=true',
*       might include additional objects.
* - has_Imputer (out)
*       Whether the serialized bytes include an 'Imputer' object. If it has 'has_combined_objects=true',
*       might include additional objects.
* - has_metadata (out)
*       Whether the serialized bytes include additional metadata in the form of a 'char' array.
*       This can only be present when having 'has_combined_objects=true'.
* - size_metadata (out)
*       When the serialized bytes contain metadata, this denotes the size of the metadata (number
*       of bytes that it contains).
*/
ISOTREE_EXPORTED
void inspect_serialized_object
(
    const char *serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
);
ISOTREE_EXPORTED
void inspect_serialized_object
(
    FILE *serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
);
ISOTREE_EXPORTED
void inspect_serialized_object
(
    std::istream &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
);
ISOTREE_EXPORTED
void inspect_serialized_object
(
    const std::string &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
);

/* Serialization and de-serialization functions (individual objects)
*
* Parameters
* ==========
* - model (in or out depending on function)
*       A model object to serialize (when it has 'const' qualifier), after being fitted through
*       function 'fit_iforest'; or an already-allocated object (should be initialized through
*       the default constructor) into which a serialized object of the same class will be
*       de-serialized. In the latter case, the contents of this object will be overwritten.
*       Note that this will only be able to load models generated with isotree version 0.3.0
*       and later, and that these serialized models are forwards compatible but not backwards
*       compatible (that is, a model saved with 0.3.0 can be loaded with 0.3.6, but not the other
*       way around).
* - output (out)
*       A writable object or stream in which to save/persist/serialize the
*       model or imputer object. In the functions that do not take this as a parameter,
*       it will be returned as a string containing the raw bytes.
*       Should be opened in binary mode.
*       Note: on Windows, if compiling this library with a compiler other than MSVC or MINGW,
*       there might be issues writing models to FILE pointers if the models are larger than 2GB.
* - in (in)
*       An readable object or stream which contains the serialized/persisted model or
*       imputer object which will be de-serialized. Should be opened in binary mode.
* 
* Returns
* =======
* (Only for functions 'determine_serialized_size')
* Size that the model or imputer object will use when serialized, intended to be
* used for allocating arrays beforehand when serializing to 'char'.
*/
ISOTREE_EXPORTED
size_t determine_serialized_size(const IsoForest &model);
ISOTREE_EXPORTED
size_t determine_serialized_size(const ExtIsoForest &model);
ISOTREE_EXPORTED
size_t determine_serialized_size(const Imputer &model);
ISOTREE_EXPORTED
void serialize_IsoForest(const IsoForest &model, char *out);
ISOTREE_EXPORTED
void serialize_IsoForest(const IsoForest &model, FILE *out);
ISOTREE_EXPORTED
void serialize_IsoForest(const IsoForest &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_IsoForest(const IsoForest &model);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, const char *in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, const std::string &in);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, char *out);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, FILE *out);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_ExtIsoForest(const ExtIsoForest &model);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const char *in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const std::string &in);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, char *out);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, FILE *out);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_Imputer(const Imputer &model);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, const char *in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, const std::string &in);


/* Serialization and de-serialization functions (combined objects)
*
* Parameters
* ==========
* - model (in or out depending on function)
*       A single-variable model object to serialize or de-serialize.
*       If the serialized object contains this type of object, it must be
*       passed, as an already-allocated object (initialized through the default
*       constructor function).
*       When de-serializing, can check if it needs to be passed through function
*       'inspect_serialized_object'.
*       If using the extended model, should pass NULL.
*       Must pass one of 'model' or 'model_ext'.
* - model_ext (in or out depending on function)
*       An extended model object to serialize or de-serialize.
*       If using the single-variable model, should pass NULL.
*       Must pass one of 'model' or 'model_ext'.
* - imputer (in or out depending on function)
*       An imputer object to serialize or de-serialize.
*       Like 'model' and 'model_ext', must also be passed when de-serializing
*       if the serialized bytes contain such object.
* - optional_metadata (in or out depending on function)
*       Optional metadata to write at the end of the file, which will be written
*       unformatted (it is assumed files are in binary mode).
*       Pass NULL if there is no metadata.
* - size_optional_metadata (in or out depending on function)
*       Size of the optional metadata, if passed. Pass zero if there is no metadata.
* - serialized_model (in)
*       A single-variable model which was serialized to raw bytes in the separate-objects
*       format, using function 'serialize_IsoForest'.
*       Pass NULL if using the extended model.
*       Must pass one of 'serialized_model' or 'serialized_model_ext'.
*       Note that if it was produced on a platform with different characteristics than
*       the one in which this function is being called (e.g. different 'size_t' width or
*       different endianness), it will be re-serialized during the function call, which
*       can be slow and use a lot of memory.
* - serialized_model_ext (in)
*       An extended model which was serialized to raw bytes in the separate-objects
*       format, using function 'serialize_ExtIsoForest'.
*       Pass NULL if using the single-variable model.
*       Must pass one of 'serialized_model' or 'serialized_model_ext'.
* - serialized_imputer (in)
*       An imputer object which was serialized to raw bytes in the separate-objects
*       format, using function 'serialize_Imputer'.
* - output (out)
*       A writable object or stream in which to save/persist/serialize the
*       model objects. In the functions that do not take this as a parameter,
*       it will be returned as a string containing the raw bytes.
*       Should be opened in binary mode.
* - in (in)
*       An readable object or stream which contains the serialized/persisted model
*       objects which will be de-serialized. Should be opened in binary mode.
* 
* Returns
* =======
* (Only for functions 'determine_serialized_size')
* Size that the objects will use when serialized, intended to be
* used for allocating arrays beforehand when serializing to 'char'.
*/
ISOTREE_EXPORTED
size_t determine_serialized_size_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
size_t determine_serialized_size_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    char *out
);
ISOTREE_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
);
ISOTREE_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
);
ISOTREE_EXPORTED
std::string serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
);
ISOTREE_EXPORTED
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
);
ISOTREE_EXPORTED
std::string serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    const char* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    FILE* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    std::istream &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    const std::string &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);


/* Serialize additional trees into previous serialized bytes
*
* Parameters
* ==========
* - model (in)
*       A model object to re-serialize, which had already been serialized into
*       'serialized_bytes' with fewer trees than it currently has, and then
*       additional trees added through functions such as 'add_tree' or 'merge_models'.
* - serialized_bytes (in) / old_bytes (out)
*       Serialized version of 'model', which had previously been produced with
*       fewer trees than it currently has and then additional trees added through
*       functions such as 'add_tree' or 'merge_models'.
*       Must have been produced in a setup with the same characteristics (e.g. width
*       of 'int' and 'size_t', endianness, etc.).
* - old_ntrees
*       Number of trees which were serialized from 'model' into 'serialized_bytes'
*       before. Trees that come after this index are assumed to be the additional
*       trees to serialize.
* 
* Returns
* =======
* - For functions 'check_can_undergo_incremental_serialization', whether the serialized
*   object can be incrementally serialized.
* - For functions 'determine_serialized_size_additional_trees', additional size (in addition
*   to current size) that the new serialized objects will have if they undergo incremental
*   serialization.
*/
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const IsoForest &model, const char *serialized_bytes);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const ExtIsoForest &model, const char *serialized_bytes);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const Imputer &model, const char *serialized_bytes);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const IsoForest &model, size_t old_ntrees);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const ExtIsoForest &model, size_t old_ntrees);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const Imputer &model, size_t old_ntrees);
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, std::string &old_bytes);


/* Translate isolation forest model into a single SQL select statement
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - table_from
*       Table name from where the columns used in the model will be selected.
* - select_as
*       Alias to give to the outlier score in the select statement.
* - numeric_colnames
*       Names to use for the numerical columns.
* - categ_colnames
*       Names to use for the categorical columns.
* - categ_levels
*       Names to use for the levels/categories of each categorical column. These will be enclosed
*       in single quotes.
* - index1
*       Whether to make the node numbers start their numeration at 1 instead of 0 in the
*       resulting statement. If passing 'output_tree_num=false', this will only affect the
*       commented lines which act as delimiters. If passing 'output_tree_num=true', will also
*       affect the results (which will also start at 1).
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* 
* Returns
* =======
* A string with the corresponding SQL statement that will calculate the outlier score
* from the model.
*/
ISOTREE_EXPORTED
std::string generate_sql_with_select_from(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                          std::string &table_from, std::string &select_as,
                                          std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                          std::vector<std::vector<std::string>> &categ_levels,
                                          bool index1, int nthreads);


/* Translate model trees into SQL select statements
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - numeric_colnames
*       Names to use for the numerical columns.
* - categ_colnames
*       Names to use for the categorical columns.
* - categ_levels
*       Names to use for the levels/categories of each categorical column. These will be enclosed
*       in single quotes.
* - output_tree_num
*       Whether to output the terminal node number instead of the separation depth at each node.
* - index1
*       Whether to make the node numbers start their numeration at 1 instead of 0 in the
*       resulting statement. If passing 'output_tree_num=false', this will only affect the
*       commented lines which act as delimiters. If passing 'output_tree_num=true', will also
*       affect the results (which will also start at 1).
* - single_tree
*       Whether to generate the select statement for a single tree of the model instead of for
*       all. The tree number to generate is to be passed under 'tree_num'.
* - tree_num
*       Tree number for which to generate an SQL select statement, if passing 'single_tree=true'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* 
* Returns
* =======
* A vector containing at each element the SQL statement for the corresponding tree in the model.
* If passing 'single_tree=true', will contain only one element, corresponding to the tree given
* in 'tree_num'. The statements will be node-by-node, with commented-out separators using '---'
* as delimiters and including the node number as part of the comment.
*/
ISOTREE_EXPORTED
std::vector<std::string> generate_sql(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                      std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                      std::vector<std::vector<std::string>> &categ_levels,
                                      bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                      int nthreads);
