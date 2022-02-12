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
*     [8] Cortes, David.
*         "Distance approximation using Isolation Forests."
*         arXiv preprint arXiv:1910.12362 (2019).
*     [9] Cortes, David.
*         "Imputing missing values with unsupervised random trees."
*         arXiv preprint arXiv:1911.06646 (2019).
*     [10] https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottom
*     [11] Cortes, David.
*          "Revisiting randomized choices in isolation forests."
*          arXiv preprint arXiv:2110.13402 (2021).
*     [12] Guha, Sudipto, et al.
*          "Robust random cut forest based anomaly detection on streams."
*          International conference on machine learning. PMLR, 2016.
*     [13] Cortes, David.
*          "Isolation forests: looking beyond tree depth."
*          arXiv preprint arXiv:2111.11639 (2021).
*     [14] Ting, Kai Ming, Yue Zhu, and Zhi-Hua Zhou.
*          "Isolation kernel and its effect on SVM"
*          Proceedings of the 24th ACM SIGKDD
*          International Conference on Knowledge Discovery & Data Mining. 2018.
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
    #define sparse_ix int  /* supported: int, int64_t, size_t */
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
typedef enum  CoefType       {Uniform=61,  Normal=0}                   CoefType;       /* For extended model */
typedef enum  UseDepthImp    {Lower=71,    Higher=0,       Same=72}    UseDepthImp;    /* For NA imputation */
typedef enum  WeighImpRows   {Inverse=0,   Prop=81,        Flat=82}    WeighImpRows;   /* For NA imputation */
typedef enum  ScoringMetric  {Depth=0,     Density=92,     BoxedDensity=94, BoxedDensity2=96, BoxedRatio=95,
                              AdjDepth=91, AdjDensity=93}              ScoringMetric;

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

typedef struct SingleTreeIndex {
    std::vector<size_t> terminal_node_mappings;
    std::vector<double> node_distances;
    std::vector<double> node_depths;
    std::vector<size_t> reference_points;
    std::vector<size_t> reference_indptr;
    std::vector<size_t> reference_mapping;
    size_t n_terminal;
} TreeNodeIndex;

typedef struct TreesIndexer {
    std::vector<SingleTreeIndex> indices;
    TreesIndexer() = default;
} TreesIndexer;

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
*       When using any of 'prob_pick_by_gain_pl', 'prob_pick_by_gain_avg', 'prob_pick_by_full_gain', 'prob_pick_by_dens', how many variables (with 'ndim=1')
*       or linear combinations (with 'ndim>1') to try for determining the best one according to gain.
*       Recommended value in reference [4] is 10 (with 'prob_pick_by_gain_avg', for outlier detection), while the
*       recommended value in reference [11] is 1 (with 'prob_pick_by_gain_pl', for outlier detection), and the
*       recommended value in reference [9] is 10 to 20 (with 'prob_pick_by_gain_pl', for missing value imputations).
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
*       Whether to sample rows with replacement or not (not recommended). Note that distance calculations,
*       if desired, don't work well with duplicate rows.
* - weight_as_sample
*       If passing sample (row) weights when fitting the model, whether to consider those weights as row
*       sampling weights (i.e. the higher the weights, the more likely the observation will end up included
*       in each tree sub-sample), or as distribution density weights (i.e. putting a weight of two is the same
*       as if the row appeared twice, thus higher weight makes it less of an outlier, but does not give it a
*       higher chance of being sampled if the data uses sub-sampling).
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
*       Note that models that use 'prob_pick_by_gain_pl' or 'prob_pick_by_gain_avg' are likely to benefit from
*       deeper trees (larger 'max_depth'), but deeper trees can result in much slower model fitting and
*       predictions.
*       If using pooled gain, one might want to substitute 'max_depth' with 'min_gain'.
* - ncols_per_tree
*       Number of columns to use (have as potential candidates for splitting at each iteration) in each tree,
*       similar to the 'mtry' parameter of random forests.
*       In general, this is only relevant when using non-random splits and/or weighted column choices.
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
*       when using 'prob_pick_by_gain_pl'.
*       This option is not supported when using density-based outlier scoring metrics.
* - standardize_data
*       Whether to standardize the features at each node before creating a linear combination of them as suggested
*       in [4]. This is ignored when using 'ndim=1'.
* - scoring_metric
*       Metric to use for determining outlier scores (see reference [13]).
*       If passing 'Depth', will use isolation depth as proposed in reference [1]. This is typically the safest choice
*       and plays well with all model types offered by this library.
*       If passing 'Density', will set scores for each terminal node as the ratio between the fraction of points in the sub-sample
*       that end up in that node and the fraction of the volume in the feature space which defines
*       the node according to the splits that lead to it.
*       If using 'ndim=1', for categorical variables, 'Density' is defined in terms
*       of number of categories that go towards each side of the split divided by number of categories
*       in the observations that reached that node.
*       The standardized outlier score from 'Density' for a given observation is calculated as the
*       negative of the logarithm of the geometric mean from the per-tree densities, which unlike
*       the standardized score produced from 'Depth', is unbounded, but just like the standardized
*       score form 'Depth', has a natural threshold for definining outlierness, which in this case
*       is zero is instead of 0.5. The non-standardized outlier score for 'Density' is calculated as the
*       geometric mean, while the per-tree scores are calculated as the density values.
*       'Density' might lead to better predictions when using 'ndim=1', particularly in the presence
*       of categorical variables. Note however that using 'Density' requires more trees for convergence
*       of scores (i.e. good results) compared to isolation-based metrics.
*       'Density' is incompatible with 'penalize_range=true'.
*       If passing 'AdjDepth', will use an adjusted isolation depth that takes into account the number of points that
*       go to each side of a given split vs. the fraction of the range of that feature that each
*       side of the split occupies, by a metric as follows: 'd = 2/ (1 + 1/(2*p))'
*       where 'p' is defined as 'p = (n_s / n_t) / (r_s / r_t)
*       with 'n_t' being the number of points that reach a given node, 'n_s' the
*       number of points that are sent to a given side of the split/branch at that node,
*       'r_t' being the range (maximum minus minimum) of the splitting feature or
*       linear combination among the points that reached the node, and 'r_s' being the
*       range of the same feature or linear combination among the points that are sent to this
*       same side of the split/branch. This makes each split add a number between zero and two
*       to the isolation depth, with this number's probabilistic distribution being centered
*       around 1 and thus the expected isolation depth remaing the same as in the original
*       'Depth' metric, but having more variability around the extremes.
*       Scores (standardized, non-standardized, per-tree) for 'AdjDepth' are aggregated in the same way
*       as for 'Depth'.
*       'AdjDepth' might lead to better predictions when using 'ndim=1', particularly in the prescence
*       of categorical variables and for smaller datasets, and for smaller datasets, might make
*       sense to combine it with 'penalize_range=true'.
*       If passing 'AdjDensity', will use the same metric from 'AdjDepth', but applied multiplicatively instead
*       of additively. The expected value for 'AdjDepth' is not strictly the same
*       as for isolation, but using the expected isolation depth as standardizing criterion
*       tends to produce similar standardized score distributions (centered around 0.5).
*       Scores (standardized, non-standardized, per-tree) from 'AdjDensity' are aggregated in the same way
*       as for 'Depth'.
*       'AdjDepth' is incompatible with 'penalize_range=true'.
*       If passing 'BoxedRatio', will set the scores for each terminal node as the ratio between the volume of the boxed
*       feature space for the node as defined by the smallest and largest values from the split
*       conditions for each column (bounded by the variable ranges in the sample) and the
*       variable ranges in the tree sample.
*       If using 'ndim=1', for categorical variables 'BoxedRatio' is defined in terms of number of categories.
*       If using 'ndim=>1', 'BoxedRatio' is defined in terms of the maximum achievable value for the
*       splitting linear combination determined from the minimum and maximum values for each
*       variable among the points in the sample, and as such, it has a rather different meaning
*       compared to the score obtained with 'ndim=1' - 'BoxedRatio' scores with 'ndim>1'
*       typically provide very poor quality results and this metric is thus not recommended to
*       use in the extended model. With 'ndim>1', 'BoxedRatio' also has a tendency of producing too small
*       values which round to zero.
*       The standardized outlier score from 'BoxedRatio' for a given observation is calculated
*       simply as the the average from the per-tree boxed ratios. 'BoxedRatio' metric
*       has a lower bound of zero and a theorical upper bound of one, but in practice the scores
*       tend to be very small numbers close to zero, and its distribution across
*       different datasets is rather unpredictable. In order to keep rankings comparable with
*       the rest of the metrics, the non-standardized outlier scores for 'BoxedRatio' are calculated as the
*       negative of the average instead. The per-tree 'BoxedRatio' scores are calculated as the ratios.
*       'BoxedRatio' can be calculated in a fast-but-not-so-precise way, and in a low-but-precise
*       way, which is controlled by parameter 'fast_bratio'. Usually, both should give the
*       same results, but in some fatasets, the fast way can lead to numerical inaccuracies
*       due to roundoffs very close to zero.
*       'BoxedRatio' might lead to better predictions in datasets with many rows when using 'ndim=1'
*       and a relatively small 'sample_size'. Note that more trees are required for convergence
*       of scores when using 'BoxedRatio'. In some datasets, 'BoxedRatio' metric might result in very bad
*       predictions, to the point that taking its inverse produces a much better ranking of outliers.
*       'BoxedRatio' option is incompatible with 'penalize_range'.
*       If passing 'BoxedDensity2', will set the score as the ratio between the fraction of points within the sample that
*       end up in a given terminal node and the 'BoxedRatio' metric.
*       Aggregation of scores (standardized, non-standardized, per-tree) for 'BoxedDensity2' is done in the same
*       way as for 'Density', and it also has a natural threshold at zero for determining
*       outliers and inliers.
*       'BoxedDensity2' is typically usable with 'ndim>1', but tends to produce much bigger values
*       compared to 'ndim=1'.
*       Albeit unintuitively, in many datasets, one can usually get better results with metric
*       'BoxedDensity' instead.
*       The calculation of 'BoxedDensity2' is also controlled by 'fast_bratio'.
*       'BoxedDensity2' incompatible with 'penalize_range'.
*       If passing 'BoxedDensity', will set the score as the ratio between the fraction of points within the sample that
*       end up in a  given terminal node and the ratio between the boxed volume of the feature
*       space in the sample and the boxed volume of a node given by the split conditions (inverse
*       as in 'BoxedDensity2'). This metric does not have any theoretical or intuitive
*       justification behind its existence, and it is perhaps ilogical to use it as a
*       scoring metric, but tends to produce good results in some datasets.
*       The standardized outlier scores for 'BoxedDensity' are defined as the negative of the geometric mean,
*       while the non-standardized scores are the geometric mean, and the per-tree scores are simply the 'density' values.
*       The calculation of 'BoxedDensity' is also controlled by 'fast_bratio'.
*       'BoxedDensity' option is incompatible with 'penalize_range'.
* - fast_bratio
*       When using "boxed" metrics for scoring, whether to calculate them in a fast way through
*       cumulative sum of logarithms of ratios after each split, or in a slower way as sum of
*       logarithms of a single ratio per column for each terminal node.
*       Usually, both methods should give the same results, but in some datasets, particularly
*       when variables have too small or too large ranges, the first method can be prone to
*       numerical inaccuracies due to roundoff close to zero.
*       Note that this does not affect calculations for models with 'ndim>1', since given the
*       split types, the calculation for them is different.
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
*       If passing NULL, each column will have a uniform weight. If used along with kurtosis weights, the
*       effect is multiplicative.
* - weigh_by_kurt
*       Whether to weigh each column according to the kurtosis obtained in the sub-sample that is selected
*       for each tree as briefly proposed in [1]. Note that this is only done at the beginning of each tree
*       sample. For categorical columns, will calculate expected kurtosis if the column were converted to
*       numerical by assigning to each category a random number ~ Unif(0, 1).
*       This is intended as a cheap feature selector, while the parameter 'prob_pick_col_by_kurt'
*       provides the option to do this at each node in the tree for a different overall type of model.
*       If passing column weights or weighted column choices ('prob_pick_col_by_range', 'prob_pick_col_by_var'),
*       the effect will be multiplicative. This option is not compatible with 'prob_pick_col_by_kurt'.
*       If passing 'missing_action=fail' and the data has infinite values, columns with rows
*       having infinite values will get a weight of zero. If passing a different value for missing
*       action, infinite values will be ignored in the kurtosis calculation.
*       If using 'missing_action=Impute', the calculation of kurtosis will not use imputed values
*       in order not to favor columns with missing values (which would increase kurtosis by all having
*       the same central value).
* - prob_pick_by_gain_pl
*       This parameter indicates the probability of choosing the threshold on which to split a variable
*       (with 'ndim=1') or a linear combination of variables (when using 'ndim>1') as the threshold
*       that maximizes a pooled standard deviation gain criterion (see references [9] and [11]) on the
*       same variable or linear combination, similarly to regression trees such as CART.
*       If using 'ntry>1', will try several variables or linear combinations thereof and choose the one
*       in which the largest standardized gain can be achieved.
*       For categorical variables with 'ndim=1', will use shannon entropy instead (like in [7]).
*       Compared to a simple averaged gain, this tends to result in more evenly-divided splits and more clustered
*       groups when they are smaller. Recommended to pass higher values when used for imputation of missing values.
*       When used for outlier detection, datasets with multimodal distributions usually see better performance
*       under this type of splits.
*       Note that, since this makes the trees more even and thus it takes more steps to produce isolated nodes,
*       the resulting object will be heavier. When splits are not made according to any of 'prob_pick_by_gain_avg',
*       'prob_pick_by_gain_pl', 'prob_pick_by_full_gain', 'prob_pick_by_dens', both the column and the split point are decided at random.
*       Note that, if passing value 1 (100%) with no sub-sampling and using the single-variable model,
*       every single tree will have the exact same splits.
*       Be aware that 'penalize_range' can also have a large impact when using 'prob_pick_by_gain_pl'.
*       Be aware also that, if passing a value of 1 (100%) with no sub-sampling and using the single-variable
*       model, every single tree will have the exact same splits.
*       Under this option, models are likely to produce better results when increasing 'max_depth'.
*       Alternatively, one can also control the depth through 'min_gain' (for which one might want to
*       set 'max_depth=0').
*       Important detail: if using any of 'prob_pick_by_gain_avg', 'prob_pick_by_gain_pl', 'prob_pick_by_full_gain',
*       'prob_pick_by_dens', the distribution of outlier scores is unlikely to be centered around 0.5.
* - prob_pick_by_gain_avg
*       This parameter indicates the probability of choosing the threshold on which to split a variable
*       (with 'ndim=1') or a linear combination of variables (when using 'ndim>1') as the threshold
*       that maximizes an averaged standard deviation gain criterion (see references [4] and [11]) on the
*       same variable or linear combination.
*       If using 'ntry>1', will try several variables or linear combinations thereof and choose the one
*       in which the largest standardized gain can be achieved.
*       For categorical variables with 'ndim=1', will take the expected standard deviation that would be
*       gotten if the column were converted to numerical by assigning to each category a random
*       number ~ Unif(0, 1) and calculate gain with those assumed standard deviations.
*       Compared to a pooled gain, this tends to result in more cases in which a single observation or very
*       few of them are put into one branch. Typically, datasets with outliers defined by extreme values in
*       some column more or less independently of the rest, usually see better performance under this type
*       of split. Recommended to use sub-samples (parameter 'sample_size') when
*       passing this parameter. Note that, since this will create isolated nodes faster, the resulting object
*       will be lighter (use less memory).
*       When splits are not made according to any of 'prob_pick_by_gain_avg', 'prob_pick_by_gain_pl',
*       'prob_pick_by_full_gain', 'prob_pick_by_dens', both the column and the split point are decided at random.
*       Default setting for [1], [2], [3] is zero, and default for [4] is 1.
*       This is the randomization parameter that can be passed to the author's original code in [5],
*       but note that the code in [5] suffers from a mathematical error in the calculation of running standard deviations,
*       so the results from it might not match with this library's.
*       Be aware that, if passing a value of 1 (100%) with no sub-sampling and using the single-variable model,
*       every single tree will have the exact same splits.
*       Under this option, models are likely to produce better results when increasing 'max_depth'.
*       Important detail: if using any of 'prob_pick_by_gain_avg', 'prob_pick_by_gain_pl',
*       'prob_pick_by_full_gain', 'prob_pick_by_dens', the distribution of outlier scores is unlikely to be centered around 0.5.
* - prob_pick_by_full_gain
*       This parameter indicates the probability of choosing the threshold on which to split a variable
*       (with 'ndim=1') or a linear combination of variables (when using 'ndim>1') as the threshold
*       that minimizes the pooled sums of variances of all columns (or a subset of them if using
*       'ncols_per_tree').
*       In general, 'prob_pick_by_full_gain' is much slower to evaluate than the other gain types, and does not tend to
*       lead to better results. When using 'prob_pick_by_full_gain', one might want to use a different scoring
*       metric (particulatly 'Density', 'BoxedDensity2' or 'BoxedRatio'). Note that
*       the variance calculations are all done through the (exact) sorted-indices approach, while is much
*       slower than the (approximate) histogram approach used by other decision tree software.
*       Be aware that the data is not standardized in any way for the range calculations, thus the scales
*       of features will make a large difference under 'prob_pick_by_full_gain', which might not make it suitable for
*       all types of data.
*       'prob_pick_by_full_gain' is not compatible with categorical data, and 'min_gain' does not apply to it.
*       When splits are not made according to any of 'prob_pick_by_gain_avg', 'prob_pick_by_gain_pl',
*       'prob_pick_by_full_gain', 'prob_pick_by_dens', both the column and the split point are decided at random.
*       Default setting for [1], [2], [3], [4] is zero.
* - prob_pick_dens
*       This parameter indicates the probability of choosing the threshold on which to split a variable
*       (with 'ndim=1') or a linear combination of variables (when using 'ndim>1') as the threshold
*       that maximizes the pooled densities of the branch distributions.
*       The 'min_gain' option does not apply to this type of splits.
*       When splits are not made according to any of 'prob_pick_by_gain_avg', 'prob_pick_by_gain_pl',
*       'prob_pick_by_full_gain', 'prob_pick_by_dens', both the column and the split point are decided at random.
*       Default setting for [1], [2], [3], [4] is zero.
* - prob_pick_col_by_range
*       When using 'ndim=1', this denotes the probability of choosing the column to split with a probability
*       proportional to the range spanned by each column within a node as proposed in reference [12].
*       When using 'ndim>1', this denotes the probability of choosing columns to create a hyperplane with a
*       probability proportional to the range spanned by each column within a node.
*       This option is not compatible with categorical data. If passing column weights, the
*       effect will be multiplicative.
*       Be aware that the data is not standardized in any way for the range calculations, thus the scales
*       of features will make a large difference under this option, which might not make it suitable for
*       all types of data.
*       Note that the proposed RRCF model from [12] uses a different scoring metric for producing anomaly
*       scores, while this library uses isolation depth regardless of how columns are chosen, thus results
*       are likely to be different from those of other software implementations. Nevertheless, as explored
*       in [11], isolation depth as a scoring metric typically provides better results than the
*       "co-displacement" metric from [12] under these split types.
* - prob_pick_col_by_var
*       When using 'ndim=1', this denotes the probability of choosing the column to split with a probability
*       proportional to the variance of each column within a node.
*       When using 'ndim>1', this denotes the probability of choosing columns to create a hyperplane with a
*       probability proportional to the variance of each column within a node.
*       For categorical data, it will calculate the expected variance if the column were converted to
*       numerical by assigning to each category a random number ~ Unif(0, 1), which depending on the number of
*       categories and their distribution, produces numbers typically a bit smaller than standardized numerical
*       variables.
*       Note that when using sparse matrices, the calculation of variance will rely on a procedure that
*       uses sums of squares, which has less numerical precision than the
*       calculation used for dense inputs, and as such, the results might differ slightly.
*       Be aware that this calculated variance is not standardized in any way, so the scales of
*       features will make a large difference under this option.
*       If there are infinite values, all columns having infinite values will be treated as having the
*       same weight, and will be chosen before every other column with non-infinite values.
*       If passing column weights , the effect will be multiplicative.
*       If passing a 'missing_action' different than 'fail', infinite values will be ignored for the
*       variance calculation. Otherwise, all columns with infinite values will have the same probability
*       and will be chosen before columns with non-infinite values.
* - prob_pick_col_by_kurt
*       When using 'ndim=1', this denotes the probability of choosing the column to split with a probability
*       proportional to the kurtosis of each column **within a node** (unlike the option 'weigh_by_kurtosis'
*       which calculates this metric only at the root).
*       When using 'ndim>1', this denotes the probability of choosing columns to create a hyperplane with a
*       probability proportional to the kurtosis of each column within a node.
*       For categorical data, it will calculate the expected kurtosis if the column were converted to
*       numerical by assigning to each category a random number ~ Unif(0, 1).
*       Note that when using sparse matrices, the calculation of kurtosis will rely on a procedure that
*       uses sums of squares and higher-power numbers, which has less numerical precision than the
*       calculation used for dense inputs, and as such, the results might differ slightly.
*       If passing column weights, the effect will be multiplicative. This option is not compatible
*       with 'weigh_by_kurtosis'.
*       If passing a 'missing_action' different than 'fail', infinite values will be ignored for the
*       variance calculation. Otherwise, all columns with infinite values will have the same probability
*       and will be chosen before columns with non-infinite values.
*       If using 'missing_action=Impute', the calculation of kurtosis will not use imputed values
*       in order not to favor columns with missing values (which would increase kurtosis by all having
*       the same central value).
*       Be aware that kurtosis can be a rather slow metric to calculate.
* - min_gain
*       Minimum gain that a split threshold needs to produce in order to proceed with a split.
*       Only used when the splits are decided by a variance gain criterion ('prob_pick_by_gain_pl' or
*       'prob_pick_by_gain_avg', but not 'prob_pick_by_full_gain' nor 'prob_pick_by_dens').
*       If the highest possible gain in the evaluated splits at a node is below this  threshold,
*       that node becomes a terminal node.
*       This can be used as a more sophisticated depth control when using pooled gain (note that 'max_depth'
*       still applies on top of this heuristic).
* - missing_action
*       How to handle missing data at both fitting and prediction time. Options are a) 'Divide' (for the single-variable
*       model only, recommended), which will follow both branches and combine the result with the weight given by the fraction of
*       the data that went to each branch when fitting the model, b) 'Impute', which will assign observations to the
*       branch with the most observations in the single-variable model (but imputed values will also be used for
*       gain calculations), or fill in missing values with the median of each column of the sample from which the
*       split was made in the extended model (recommended) (but note that the calculation of medians does not take 
*       into account sample weights when using 'weights_as_sample_prob=false', and note that when using a gain
*       criterion for splits with 'ndim=1', it will use the imputed values in the calculation), c) 'Fail' which will
*       assume that there are no missing values and will trigger undefined behavior if it encounters any.
*       In the extended model, infinite values will be treated as missing.
*       Note that passing 'Fail' might crash the process if there turn out to be missing values, but will otherwise
*       produce faster fitting and prediction times along with decreased model object sizes.
*       Models from [1], [2], [3], [4] correspond to 'Fail' here.
* - cat_split_type
*       Whether to split categorical features by assigning sub-sets of them to each branch, or by assigning
*       a single category to a branch and the rest to the other branch. For the extended model, whether to
*       give each category a coefficient, or only one while the rest get zero.
* - new_cat_action
*       What to do after splitting a categorical feature when new data that reaches that split has categories that
*       the sub-sample from which the split was done did not have. Options are a) "Weighted" (recommended), which
*       in the single-variable model will follow both branches and combine the result with weight given by the fraction of the
*       data that went to each branch when fitting the model, and in the extended model will assign
*       them the median value for that column that was added to the linear combination of features (but note that
*       this median calculation does not use sample weights when using 'weights_as_sample_prob=false'),
*       b) "Smallest", which will assign all observations with unseen categories in the split to the branch that
*       had fewer observations when fitting the model, c) "Random", which will assing a branch (coefficient in the
*       extended model) at random for each category beforehand, even if no observations had that category when
*       fitting the model. Ignored when passing 'cat_split_type' = 'SingleCateg'.
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
* - use_long_double
*       Whether to use 'long double' (extended precision) type for more precise calculations about
*       standard deviations, means, ratios, weights, gain, and other potential aggregates. This makes
*       such calculations accurate to a larger number of decimals (provided that the compiler used has
*       wider long doubles than doubles) and it is highly recommended to use when the input data has
*       a number of rows or columns exceeding 2^53 (an unlikely scenario), and also highly recommended
*       to use when the input data has problematic scales (e.g. numbers that differ from each other by
*       something like 10^-100 or columns that include values like 10^100 and 10^-100 and still need to
*       be sensitive to a difference of 10^-100), but will make the calculations slower, the more so in
*       platforms in which 'long double' is a software-emulated type (e.g. Power8 platforms).
*       Note that some platforms (most notably windows with the msvc compiler) do not make any difference
*       between 'double' and 'long double'.
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
                ScoringMetric scoring_metric, bool fast_bratio,
                bool   standardize_dist, double tmat[],
                double output_depths[], bool standardize_depth,
                real_t col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                double prob_pick_by_full_gain, double prob_pick_by_dens,
                double prob_pick_col_by_range, double prob_pick_col_by_var,
                double prob_pick_col_by_kurt,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, bool use_long_double, int nthreads);



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
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data to which to fit this additional tree. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.).
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
*       If the model from 'fit_iforest' was fit to numeric data, must pass numeric data with the same number
*       of columns, either as dense or as sparse arrays.
* - ncols_numeric
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Cannot be changed from
*       what was originally passed to 'fit_iforest'.
* - categ_data[nrows * ncols_categ]
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
* - ncat[ncols_categ]
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). May contain new categories,
*       but should keep the same encodings that were used for previous categories.
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
* - standardize_data
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - fast_bratio
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - col_weights
*       Sampling weights for each column, assuming all the numeric columns come before the categorical columns.
*       Ignored when picking columns by deterministic criterion.
*       If passing NULL, each column will have a uniform weight. If used along with kurtosis weights, the
*       effect is multiplicative.
* - weigh_by_kurt
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_by_gain_pl
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_by_gain_avg
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_by_full_gain
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_by_dens
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_col_by_range
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_col_by_var
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
* - prob_pick_col_by_kurt
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
* - indexer
*       Indexer object associated to the model object ('model_outputs' or 'model_outputs_ext'), which will
*       be updated with the new tree to add.
*       If 'indexer' has reference points, these must be passed again here in order to index them.
*       Pass NULL if the model has no associated indexer.
* - ref_numeric_data[nref * ncols_numeric]
*       Pointer to numeric data for reference points. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'ref_is_col_major').
*       Pass NULL if there are no dense numeric columns or no reference points.
*       Can only pass one of 'ref_numeric_data' or 'ref_Xc' + 'ref_Xc_ind' + 'ref_Xc_indptr'.
*       If 'indexer' is passed, it has reference points, and the data to which the model was fit had
*       numeric columns, then numeric data for reference points must be passed (in either dense or sparse format).
* - ref_categ_data[nref * ncols_categ]
*       Pointer to categorical data for reference points. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'ref_is_col_major').
*       Pass NULL if there are no categorical columns or no reference points.
*       If 'indexer' is passed, it has reference points, and the data to which the model was fit had
*       categorical columns, then 'ref_categ_data' must be passed.
* - ref_is_col_major
*       Whether 'ref_numeric_data' and/or 'ref_categ_data' are in column-major order. If numeric data is
*       passed in sparse format, categorical data must be passed in column-major format. If passing dense
*       data, row-major format is preferred as it will be faster. If the data is passed in row-major format,
*       must also pass 'ref_ld_numeric' and/or 'ref_ld_categ'.
*       If both 'ref_numeric_data' and 'ref_categ_data' are passed, they must have the same orientation
*       (row-major or column-major).
* - ref_ld_numeric
*       Leading dimension of the array 'ref_numeric_data', if it is passed in row-major format.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'ref_numeric_data + n*ref_ld_numeric'). If passing
*       'ref_numeric_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows. This is ignored when passing numeric
*       data in sparse format.
* - ref_ld_categ
*       Leading dimension of the array 'ref_categ_data', if it is passed in row-major format.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'ref_categ_data + n*ref_ld_categ'). If passing
*       'ref_categ_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows.
* - ref_Xc[ref_nnz]
*       Pointer to numeric data for reference points in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns for reference points or no reference points.
*       Can only pass one of 'ref_numeric_data' or 'ref_Xc' + 'ref_Xc_ind' + 'ref_Xc_indptr'.
* - ref_Xc_ind[ref_nnz]
*       Pointer to row indices to which each non-zero entry in 'ref_Xc' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSC format for reference points or no reference points.
* - ref_Xc_indptr[ref_nnz]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end.
*       Pass NULL if there are no sparse numeric columns in CSC format for reference points or no reference points.
* - random_seed
*       Seed that will be used to generate random numbers used by the model.
* - use_long_double
*       Same parameter as for 'fit_iforest' (see the documentation in there for details). Can be changed from
*       what was originally passed to 'fit_iforest'.
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
             bool   fast_bratio,
             real_t col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
             double prob_pick_by_full_gain, double prob_pick_by_dens,
             double prob_pick_col_by_range, double prob_pick_col_by_var,
             double prob_pick_col_by_kurt,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, Imputer *imputer, size_t min_imp_obs,
             TreesIndexer *indexer,
             real_t ref_numeric_data[], int ref_categ_data[],
             bool ref_is_col_major, size_t ref_ld_numeric, size_t ref_ld_categ,
             real_t ref_Xc[], sparse_ix ref_Xc_ind[], sparse_ix ref_Xc_indptr[],
             uint64_t random_seed, bool use_long_double);


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
*       Should always be passed when calling this function (it is not optional).
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
* - indexer
*       Pointer to associated tree indexer for the model being used, if it was constructed,
*       which can be used to speed up tree numbers/indices predictions.
*       This is ignored when not passing 'tree_num'.
*       Pass NULL if the indexer has not been constructed.
*/
ISOTREE_EXPORTED
void predict_iforest(real_t numeric_data[], int categ_data[],
                     bool is_col_major, size_t ld_numeric, size_t ld_categ,
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[],
                     double per_tree_depths[],
                     TreesIndexer *indexer);



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
ISOTREE_EXPORTED void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept;
ISOTREE_EXPORTED void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept;



/* Calculate distance or similarity or kernel/proximity between data points
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data for which to make calculations. If not using 'indexer', must be
*       ordered by columns like Fortran, not ordered by rows like C (i.e. entries 1..n contain
*       column 0, n+1..2n column 1, etc.), while if using 'indexer', may be passed in either
*       row-major or column-major format (with row-major being faster).
*       If categorical data is passed, must be in the same storage order (row-major / column-major)
*       as numerical data (whether dense or sparse).
*       The column order must be the same as in the data that was used to fit the model.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data for which to make calculations. If not using 'indexer', must be
*       ordered by columns like Fortran, not ordered by rows like C (i.e. entries 1..n contain
*       column 0, n+1..2n column 1, etc.), while if using 'indexer', may be passed in either
*       row-major or column-major format (with row-major being faster).
*       If numerical data is passed, must be in the same storage order (row-major / column-major)
*       as categorical data (whether the numerical data is dense or sparse).
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
*       Pass NULL if there are no categorical columns.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed),
*       or optionally in CSR format (row-compressed) if using 'indexer' and passing 'is_col_major=false'
*       (not recommended as the calculations will be slower if sparse data is passed as CSR).
*       If categorical data is passed, must be in the same storage order (row-major or CSR / column-major or CSC)
*       as numerical data (whether dense or sparse).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds
*       (column indices if 'Xc' is in CSR format).
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSC or CSR format.
* - Xc_indptr[ncols_categ + 1]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end
*       (row index pointers if 'Xc' is passed in CSR format).
*       Pass NULL if there are no sparse numeric columns in CSC or CSR format.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'categ_data'.
* - use_long_double
*       Whether to use 'long double' (extended precision) type for the calculations. This makes them
*       more accurate (provided that the compiler used has wider long doubles than doubles), but
*       slower - especially in platforms in which 'long double' is a software-emulated type (e.g.
*       Power8 platforms).
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used (with one exception being kernel calculations
*       with respect to reference points in an idexer). Ignored when not building with OpenMP support.
* - assume_full_distr
*       Whether to assume that the fitted model represents a full population distribution (will use a
*       standardizing criterion assuming infinite sample, and the results of the similarity between two points
*       at prediction time will not depend on the prescence of any third point that is similar to them, but will
*       differ more compared to the pairwise distances between points from which the model was fit). If passing
*       'false', will calculate pairwise distances as if the new observations at prediction time were added to
*       the sample to which each tree was fit, which will make the distances between two points potentially vary
*       according to other newly introduced points.
*       This was added for experimentation purposes only and it's not recommended to pass 'false'.
*       Note that when calculating distances using 'indexer', there
*       might be slight discrepancies between the numbers produced with or without the indexer due to what
*       are considered "additional" observations in this calculation.
*       This is ignored when passing 'as_kernel=true'.
* - standardize_dist
*       Whether to standardize the resulting average separation depths between rows according
*       to the expected average separation depth in a similar way as when predicting outlierness,
*       in order to obtain a standardized distance. If passing 'false', will output the average
*       separation depth instead.
*       If passing 'as_kernel=true', this indicates whether to output a fraction (if 'true') or
*       the raw number of matching trees (if 'false').
* - as_kernel
*       Whether to calculate the "similarities" as isolation kernel or proximity matrix, which counts
*       the proportion of trees in which two observations end up in the same terminal node. This is
*       typically much faster than separation-based distance, but is typically not as good quality.
*       Note that, for kernel calculations, the indexer is only used if it has reference points stored on it.
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the calculations are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the calculations are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - tmat[nrows * (nrows - 1) / 2] (out)
*       Pointer to array where the resulting pairwise distances or average separation depths or kernels will
*       be written into. As the output is a symmetric matrix, this function will only fill in the
*       upper-triangular part, in which entry 0 <= i < j < n will be located at position
*           p(i,j) = (i * (n - (i+1)/2) + j - i - 1).
*       Can be converted to a dense square matrix through function 'tmat_to_dense'.
*       The array must already be initialized to zeros.
*       If calculating distance/separation from a group of points to another group of points,
*       pass NULL here and use 'rmat' instead.
* - rmat[nrows1 * nrows2] (out)
*       Pointer to array where to write the distances or separation depths or kernels between each row in
*       one set of observations and each row in a different set of observations. If doing these
*       calculations for all pairs of observations/rows, pass 'tmat' instead.
*       Will take the first group of observations as the rows in this matrix, and the second
*       group as the columns. The groups are assumed to be in the same data arrays, with the
*       first group corresponding to the earlier rows there.
*       This matrix will be used in row-major order (i.e. entries 1..nrows2 contain the first row from nrows1).
*       Must be already initialized to zeros.
*       If passing 'use_indexed_references=true' plus an indexer object with reference points, this
*       array should have dimension [nrows, n_references].
*       Ignored when 'tmat' is passed.
* - n_from
*       When calculating distances between two groups of points, this indicates the number of
*       observations/rows belonging to the first group (the rows in 'rmat'), which will be
*       assumed to be the first 'n_from' rows.
*       Ignored when 'tmat' is passed or when 'use_indexed_references=true' plus an indexer with
*       references are passed.
* - use_indexed_references
*       Whether to calculate distances with respect to reference points stored in the indexer
*       object, if it has any. This is only supported with 'assume_full_distr=true' or with 'as_kernel=true'.
*       If passing 'use_indexed_references=true', then 'tmat' must be NULL, and 'rmat' must
*       be of dimension [nrows, n_references].
* - indexer
*       Pointer to associated tree indexer for the model being used, if it was constructed,
*       which can be used to speed up distance calculations, assuming that it was built with
*       option 'with_distances=true'. If it does not contain node distances, it will not be used.
*       Pass NULL if the indexer has not been constructed or was constructed with 'with_distances=false'.
*       If it contains reference points and passing 'use_indexed_references=true', distances will be
*       calculated between between the input data passed here and the reference points stored in this object.
*       If passing 'as_kernel=true', the indexer can only be used for calculating kernels with respect to
*       reference points in the indexer, otherwise it will not be used (which also means that the data must be
*       passed in column-major order for all kernel calculations that are not with respect to reference points
*       from an indexer).
* - is_col_major
*       Whether the data comes in column-major order. If using 'indexer', predictions are also possible
*       (and are even faster for the case of dense-only data) if passing the data in row-major format.
*       Without 'indexer' (and with 'as_kernel=true' but without reference points in the idnexer), data
*       may only be passed in column-major format.
*       If there is sparse numeric data, it is highly suggested to pass it in CSC/column-major format.
* - ld_numeric
*       If passing 'is_col_major=false', this indicates the leading dimension of the array 'numeric_data'.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'numeric_data + n*ld_numeric'). If passing
*       'numeric_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows. This is ignored when passing numeric
*       data in sparse format.
*       Note that data in row-major order is only accepted when using 'indexer'.
* - ld_categ
*       If passing 'is_col_major=false', this indicates the leading dimension of the array 'categ_data'.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'categ_data + n*ld_categ'). If passing
*       'categ_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows.
*       Note that data in row-major order is only accepted when using 'indexer'.
*/
ISOTREE_EXPORTED
void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, bool use_long_double, int nthreads,
                     bool assume_full_distr, bool standardize_dist, bool as_kernel,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from, bool use_indexed_references,
                     TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ);

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
* - use_long_double
*       Whether to use 'long double' (extended precision) type for the calculations. This makes them
*       more accurate (provided that the compiler used has wider long doubles than doubles), but
*       slower - especially in platforms in which 'long double' is a software-emulated type (e.g.
*       Power8 platforms).
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
                           size_t nrows, bool use_long_double, int nthreads,
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
* - indexer (in, out)
*       Pointer to indexer object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest' or through another specialized function.
*       The imputation nodes from 'ind_other' will be merged into this (will be at the end of vector member 'indices').
*       Reference points should not differ between 'indexer' and 'ind_other'.
*       Pass NULL if this is not to be used.
* - ind_other
*       Pointer to indexer object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest' or through another specialized function.
*       The imputation nodes from this object will be added into 'imputer' (this object will not be modified).
*       Reference points should not differ between 'indexer' and 'ind_other'.
*       Pass NULL if this is not to be used.
*/
ISOTREE_EXPORTED
void merge_models(IsoForest*     model,      IsoForest*     other,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                  Imputer*       imputer,    Imputer*       iother,
                  TreesIndexer*  indexer,    TreesIndexer*  ind_other);

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
* - indexer (in)
*       Pointer to indexer object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest' or through another specialized funcction.
*       Pass NULL if the model was built without an indexer.
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
* - indexer_new (out)
*       Pointer to already-allocated indexer object, which will be reset and to
*       which the selected nodes from 'indexer' (matching to those of either 'model'
*       or 'ext_model') will be copied.
*       Pass NULL if the model was built without an indexer.
*/
ISOTREE_EXPORTED
void subset_model(IsoForest*     model,      IsoForest*     model_new,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                  Imputer*       imputer,    Imputer*       imputer_new,
                  TreesIndexer*  indexer,    TreesIndexer*  indexer_new,
                  size_t *trees_take, size_t ntrees_take);

/* Build indexer for faster terminal node predictions and/or distance calculations
* 
* Parameters
* ==========
* - indexer
*       Pointer or reference to an indexer object which will be associated to a fitted model and in
*       which indices for terminal nodes and potentially node distances will be stored.
* - model / model_outputs / model_outputs_ext
*       Pointer or reference to a fitted model object for which an indexer will be built.
* - nthreads
*       Number of parallel threads to use. This operation will only be multi-threaded when passing
*       'with_distances=true'.
* - with_distances
*       Whether to also pre-calculate node distances in order to speed up 'calc_similarity' (distances).
*       Note that this will consume a lot more memory and make the resulting object significantly
*       heavier.
*/
ISOTREE_EXPORTED
void build_tree_indices(TreesIndexer &indexer, const IsoForest &model, int nthreads, const bool with_distances);
ISOTREE_EXPORTED
void build_tree_indices(TreesIndexer &indexer, const ExtIsoForest &model, int nthreads, const bool with_distances);
ISOTREE_EXPORTED
void build_tree_indices
(
    TreesIndexer *indexer,
    const IsoForest *model_outputs,
    const ExtIsoForest *model_outputs_ext,
    int nthreads,
    const bool with_distances
);
/* Gets the number of reference points stored in an indexer object */
ISOTREE_EXPORTED
size_t get_number_of_reference_points(const TreesIndexer &indexer) noexcept;


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
    bool &has_Indexer,
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
    bool &has_Indexer,
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
    bool &has_Indexer,
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
    bool &has_Indexer,
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
size_t determine_serialized_size(const IsoForest &model) noexcept;
ISOTREE_EXPORTED
size_t determine_serialized_size(const ExtIsoForest &model) noexcept;
ISOTREE_EXPORTED
size_t determine_serialized_size(const Imputer &model) noexcept;
ISOTREE_EXPORTED
size_t determine_serialized_size(const TreesIndexer &model) noexcept;
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
ISOTREE_EXPORTED
void serialize_Indexer(const TreesIndexer &model, char *out);
ISOTREE_EXPORTED
void serialize_Indexer(const TreesIndexer &model, FILE *out);
ISOTREE_EXPORTED
void serialize_Indexer(const TreesIndexer &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_Indexer(const TreesIndexer &model);
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, const char *in);
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, const std::string &in);


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
    const TreesIndexer *indexer,
    const size_t size_optional_metadata
) noexcept;
ISOTREE_EXPORTED
size_t determine_serialized_size_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *serialized_indexer,
    const size_t size_optional_metadata
) noexcept;
ISOTREE_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const TreesIndexer *indexer,
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
    const TreesIndexer *indexer,
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
    const TreesIndexer *indexer,
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
    const TreesIndexer *indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *serialized_indexer,
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
    const char *serialized_indexer,
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
    const char *serialized_indexer,
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
    TreesIndexer *indexer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    FILE* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    TreesIndexer *indexer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    std::istream &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    TreesIndexer *indexer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    const std::string &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    TreesIndexer *indexer,
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
size_t determine_serialized_size_additional_trees(const IsoForest &model, size_t old_ntrees);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const ExtIsoForest &model, size_t old_ntrees);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const Imputer &model, size_t old_ntrees);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const TreesIndexer &model, size_t old_ntrees);
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_Indexer(const TreesIndexer &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_Indexer(const TreesIndexer &model, std::string &old_bytes);


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


ISOTREE_EXPORTED
void set_reference_points(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
                          const bool with_distances,
                          real_t *numeric_data, int *categ_data,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          real_t *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
                          real_t *Xr, sparse_ix *Xr_ind, sparse_ix *Xr_indptr,
                          size_t nrows, int nthreads);
