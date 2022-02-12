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
*     Copyright (c) 2019-2022, David Cortes
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
template <class real_t, class sparse_ix>
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
                uint64_t random_seed, bool use_long_double, int nthreads)
{
    if (use_long_double && !has_long_double()) {
        use_long_double = false;
        fprintf(stderr, "Passed 'use_long_double=true', but library was compiled without long double support.\n");
    }
    #ifndef NO_LONG_DOUBLE
    if (likely(!use_long_double))
    #endif
        return fit_iforest_internal<real_t, sparse_ix, double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, with_replacement, weight_as_sample,
            nrows, sample_size, ntrees,
            max_depth, ncols_per_tree,
            limit_depth, penalize_range, standardize_data,
            scoring_metric, fast_bratio,
            standardize_dist, tmat,
            output_depths, standardize_depth,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            all_perm, imputer, min_imp_obs,
            depth_imp, weigh_imp_rows, impute_at_fit,
            random_seed, nthreads
        );
    #ifndef NO_LONG_DOUBLE
    else
        return fit_iforest_internal<real_t, sparse_ix, long double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, with_replacement, weight_as_sample,
            nrows, sample_size, ntrees,
            max_depth, ncols_per_tree,
            limit_depth, penalize_range, standardize_data,
            scoring_metric, fast_bratio,
            standardize_dist, tmat,
            output_depths, standardize_depth,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            all_perm, imputer, min_imp_obs,
            depth_imp, weigh_imp_rows, impute_at_fit,
            random_seed, nthreads
        );
    #endif
}

template <class real_t, class sparse_ix, class ldouble_safe>
int fit_iforest_internal(
                IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees,
                size_t max_depth, size_t ncols_per_tree,
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
                uint64_t random_seed, int nthreads)
{
    if (
        prob_pick_by_gain_avg  < 0 || prob_pick_by_gain_pl  < 0 ||
        prob_pick_by_full_gain < 0 || prob_pick_by_dens     < 0 ||
        prob_pick_col_by_range < 0 ||
        prob_pick_col_by_var   < 0 || prob_pick_col_by_kurt < 0
    ) {
        throw std::runtime_error("Cannot pass negative probabilities.\n");
    }
    if (prob_pick_col_by_range && ncols_categ)
        throw std::runtime_error("'prob_pick_col_by_range' is not compatible with categorical data.\n");
    if (prob_pick_by_full_gain && ncols_categ)
        throw std::runtime_error("'prob_pick_by_full_gain' is not compatible with categorical data.\n");
    if (prob_pick_col_by_kurt && weigh_by_kurt)
        throw std::runtime_error("'weigh_by_kurt' and 'prob_pick_col_by_kurt' cannot be used together.\n");
    if (ndim == 0 && model_outputs == NULL)
        throw std::runtime_error("Must pass 'ndim>0' in the extended model.\n");
    if (penalize_range &&
        (scoring_metric == Density ||
         scoring_metric == AdjDensity ||
         is_boxed_metric(scoring_metric))
    )
        throw std::runtime_error("'penalize_range' is incompatible with density scoring.\n");
    if (with_replacement) {
        if (tmat != NULL)
            throw std::runtime_error("Cannot calculate distance while sampling with replacement.\n");
        if (output_depths != NULL)
            throw std::runtime_error("Cannot make predictions at fit time when sampling with replacement.\n");
        if (impute_at_fit)
            throw std::runtime_error("Cannot impute at fit time when sampling with replacement.\n");
    }
    if (sample_size != 0 && sample_size < nrows) {
        if (output_depths != NULL)
            throw std::runtime_error("Cannot produce outlier scores at fit time when using sub-sampling.\n");
        if (tmat != NULL)
            throw std::runtime_error("Cannot calculate distances at fit time when using sub-sampling.\n");
        if (impute_at_fit)
            throw std::runtime_error("Cannot produce missing data imputations at fit time when using sub-sampling.\n");
    }


    /* TODO: this function should also accept the array as a memoryview with a
       leading dimension that might not correspond to the number of columns,
       so as to avoid having to make deep copies of memoryviews in python and to
       allow using pointers to columns of dataframes in R and Python. */

    /* calculate maximum number of categories to use later */
    int max_categ = 0;
    for (size_t col = 0; col < ncols_categ; col++)
        max_categ = (ncat[col] > max_categ)? ncat[col] : max_categ;

    bool calc_dist = tmat != NULL;

    if (sample_size == 0)
        sample_size = nrows;

    if (model_outputs != NULL)
        ntry = std::min(ntry, ncols_numeric + ncols_categ);

    if (ncols_per_tree == 0)
        ncols_per_tree = ncols_numeric + ncols_categ;

    /* put data in structs to shorten function calls */
    InputData<real_t, sparse_ix>
              input_data     = {numeric_data, ncols_numeric, categ_data, ncat, max_categ, ncols_categ,
                                nrows, ncols_numeric + ncols_categ, sample_weights,
                                weight_as_sample, col_weights,
                                Xc, Xc_ind, Xc_indptr,
                                0, 0, std::vector<double>(),
                                std::vector<char>(), 0, NULL,
                                (double*)NULL, (double*)NULL, (int*)NULL, std::vector<double>(),
                                std::vector<double>(), std::vector<double>(),
                                std::vector<size_t>(), std::vector<size_t>()};
    ModelParams model_params = {with_replacement, sample_size, ntrees, ncols_per_tree,
                                limit_depth? log2ceil(sample_size) : max_depth? max_depth : (sample_size - 1),
                                penalize_range, standardize_data, random_seed, weigh_by_kurt,
                                prob_pick_by_gain_avg, prob_pick_by_gain_pl,
                                prob_pick_by_full_gain, prob_pick_by_dens,
                                prob_pick_col_by_range, prob_pick_col_by_var,
                                prob_pick_col_by_kurt,
                                min_gain, cat_split_type, new_cat_action, missing_action,
                                scoring_metric, fast_bratio, all_perm,
                                (model_outputs != NULL)? 0 : ndim, ntry,
                                coef_type, coef_by_prop, calc_dist, (bool)(output_depths != NULL), impute_at_fit,
                                depth_imp, weigh_imp_rows, min_imp_obs};

    /* if calculating full gain, need to produce copies of the data in row-major order */
    if (prob_pick_by_full_gain)
    {
        if (input_data.Xc_indptr == NULL)
            colmajor_to_rowmajor(input_data.numeric_data, input_data.nrows, input_data.ncols_numeric, input_data.X_row_major);
        else
            colmajor_to_rowmajor(input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                 input_data.nrows, input_data.ncols_numeric,
                                 input_data.Xr, input_data.Xr_ind, input_data.Xr_indptr);
    }

    /* if using weights as sampling probability, build a binary tree for faster sampling */
    if (input_data.weight_as_sample && input_data.sample_weights != NULL)
    {
        build_btree_sampler(input_data.btree_weights_init, input_data.sample_weights,
                            input_data.nrows, input_data.log2_n, input_data.btree_offset);
    }

    /* same for column weights */
    /* TODO: this should also save the kurtoses when using 'prob_pick_col_by_kurt' */
    ColumnSampler<ldouble_safe> base_col_sampler;
    if (
        col_weights != NULL ||
        (model_params.weigh_by_kurt && model_params.sample_size == input_data.nrows && !model_params.with_replacement &&
         (model_params.ncols_per_tree >= input_data.ncols_tot / (model_params.ntrees * 2)))
    )
    {
        bool avoid_col_weights = (model_outputs != NULL && model_params.ntry >= model_params.ncols_per_tree &&
                                  model_params.prob_pick_by_gain_avg  + model_params.prob_pick_by_gain_pl +
                                  model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens >= 1)
                                    ||
                                 (model_outputs == NULL && model_params.ndim >= model_params.ncols_per_tree)
                                    ||
                                 (model_params.ncols_per_tree == 1);
        if (!avoid_col_weights)
        {
            if (model_params.weigh_by_kurt && model_params.sample_size == input_data.nrows && !model_params.with_replacement)
            {
                RNG_engine rnd_generator(random_seed);
                std::vector<double> kurt_weights = calc_kurtosis_all_data<InputData<real_t, sparse_ix>, ldouble_safe>(input_data, model_params, rnd_generator);
                if (col_weights != NULL)
                {
                    for (size_t col = 0; col < input_data.ncols_tot; col++)
                    {
                        if (kurt_weights[col] <= 0) continue;
                        kurt_weights[col] *= col_weights[col];
                        kurt_weights[col]  = std::fmax(kurt_weights[col], 1e-100);
                    }
                }
                base_col_sampler.initialize(kurt_weights.data(), input_data.ncols_tot);

                if (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var)
                {
                    input_data.all_kurtoses = std::move(kurt_weights);
                }
            }

            else
            {
                base_col_sampler.initialize(input_data.col_weights, input_data.ncols_tot);
            }

            input_data.preinitialized_col_sampler = &base_col_sampler;
        }
    }

    /* in some cases, all trees will need to calculate variable ranges for all columns */
    /* TODO: the model might use 'leave_m_cols', or have 'prob_pick_col_by_range<1', in which
       case it might not be beneficial to do this beforehand. Find out when the expected gain
       from doing this here is not beneficial. */
    /* TODO: move this to a different file, it doesn't belong here */
    std::vector<double> variable_ranges_low;
    std::vector<double> variable_ranges_high;
    std::vector<int> variable_ncats;
    if (
        model_params.sample_size == input_data.nrows && !model_params.with_replacement &&
        (model_params.ncols_per_tree >= input_data.ncols_numeric) &&
        ((model_params.prob_pick_col_by_range && input_data.ncols_numeric)
            ||
         is_boxed_metric(model_params.scoring_metric))
    )
    {
        variable_ranges_low.resize(input_data.ncols_numeric);
        variable_ranges_high.resize(input_data.ncols_numeric);

        std::unique_ptr<unsigned char[]> buffer_cats;
        size_t adj_col;
        if (is_boxed_metric(model_params.scoring_metric))
        {
            variable_ncats.resize(input_data.ncols_categ);
            buffer_cats = std::unique_ptr<unsigned char[]>(new unsigned char[input_data.max_categ]);
        }

        if (base_col_sampler.col_indices.empty())
            base_col_sampler.initialize(input_data.ncols_tot);

        bool unsplittable;
        size_t n_tried_numeric = 0;
        size_t col;
        base_col_sampler.prepare_full_pass();
        while (base_col_sampler.sample_col(col))
        {
            if (col < input_data.ncols_numeric)
            {
                if (input_data.Xc_indptr == NULL)
                {
                    get_range(input_data.numeric_data + nrows*col,
                              input_data.nrows,
                              model_params.missing_action,
                              variable_ranges_low[col],
                              variable_ranges_high[col],
                              unsplittable);
                }

                else
                {
                    get_range(col, input_data.nrows,
                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                              model_params.missing_action,
                              variable_ranges_low[col],
                              variable_ranges_high[col],
                              unsplittable);
                }

                n_tried_numeric++;

                if (unsplittable)
                {
                    variable_ranges_low[col] = 0;
                    variable_ranges_high[col] = 0;
                    base_col_sampler.drop_col(col);
                }
            }

            else
            {
                if (!is_boxed_metric(model_params.scoring_metric))
                {
                    if (n_tried_numeric >= input_data.ncols_numeric)
                        break;
                    else
                        continue;
                }
                adj_col = col - input_data.ncols_numeric;


                variable_ncats[adj_col] = count_ncateg_in_col(input_data.categ_data + input_data.nrows*adj_col,
                                                              input_data.nrows, input_data.ncat[adj_col],
                                                              buffer_cats.get());
                if (variable_ncats[adj_col] <= 1)
                    base_col_sampler.drop_col(col);
            }
        }

        input_data.preinitialized_col_sampler = &base_col_sampler;
        if (input_data.ncols_numeric) {
            input_data.range_low = variable_ranges_low.data();
            input_data.range_high = variable_ranges_high.data();
        }
        if (input_data.ncols_categ) {
            input_data.ncat_ = variable_ncats.data();
        }
    }

    /* if imputing missing values on-the-fly, need to determine which are missing */
    std::vector<ImputedData<sparse_ix, ldouble_safe>> impute_vec;
    hashed_map<size_t, ImputedData<sparse_ix, ldouble_safe>> impute_map;
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
        model_outputs->scoring_metric = scoring_metric;
        if (
            model_outputs->scoring_metric != Density &&
            model_outputs->scoring_metric != BoxedDensity &&
            model_outputs->scoring_metric != BoxedDensity2 &&
            model_outputs->scoring_metric != BoxedRatio
        )
            model_outputs->exp_avg_depth  = expected_avg_depth<ldouble_safe>(sample_size);
        else
            model_outputs->exp_avg_depth  = 1;
        model_outputs->exp_avg_sep = expected_separation_depth<ldouble_safe>(model_params.sample_size);
        model_outputs->orig_sample_size = input_data.nrows;
        model_outputs->has_range_penalty = penalize_range;
    }

    else
    {
        model_outputs_ext->hplanes.resize(ntrees);
        model_outputs_ext->hplanes.shrink_to_fit();
        model_outputs_ext->new_cat_action = new_cat_action;
        model_outputs_ext->cat_split_type = cat_split_type;
        model_outputs_ext->missing_action = missing_action;
        model_outputs_ext->scoring_metric = scoring_metric;
        if (
            model_outputs_ext->scoring_metric != Density &&
            model_outputs_ext->scoring_metric != BoxedDensity &&
            model_outputs_ext->scoring_metric != BoxedDensity2 &&
            model_outputs_ext->scoring_metric != BoxedRatio
        )
            model_outputs_ext->exp_avg_depth  = expected_avg_depth<ldouble_safe>(sample_size);
        else
            model_outputs_ext->exp_avg_depth  = 1;
        model_outputs_ext->exp_avg_sep = expected_separation_depth<ldouble_safe>(model_params.sample_size);
        model_outputs_ext->orig_sample_size = input_data.nrows;
        model_outputs_ext->has_range_penalty = penalize_range;
    }

    if (imputer != NULL)
        initialize_imputer<decltype(input_data), ldouble_safe>(
            *imputer, input_data, ntrees, nthreads
        );

    /* initialize thread-private memory */
    if ((size_t)nthreads > ntrees)
        nthreads = (int)ntrees;
    #ifdef _OPENMP
        std::vector<WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>> worker_memory(nthreads);
    #else
        std::vector<WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>> worker_memory(1);
    #endif

    /* Global variable that determines if the procedure receives a stop signal */
    SignalSwitcher ss = SignalSwitcher();

    /* For exception handling */
    bool threw_exception = false;
    std::exception_ptr ex = NULL;

    /* grow trees */
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic) shared(model_outputs, model_outputs_ext, worker_memory, input_data, model_params, threw_exception, ex)
    for (size_t_for tree = 0; tree < (decltype(tree))ntrees; tree++)
    {
        if (interrupt_switch || threw_exception)
            continue; /* Cannot break with OpenMP==2.0 (MSVC) */

        try
        {
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

            fit_itree<decltype(input_data), typename std::remove_pointer<decltype(worker_memory.data())>::type, ldouble_safe>(
                      (model_outputs != NULL)? &model_outputs->trees[tree] : NULL,
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

        catch (...)
        {
            #pragma omp critical
            {
                if (!threw_exception)
                {
                    threw_exception = true;
                    ex = std::current_exception();
                }
            }
        }
    }

    /* check if the procedure got interrupted */
    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* check if some exception was thrown */
    if (threw_exception)
        std::rethrow_exception(ex);

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
                          standardize_dist, false, nthreads);

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
            for (auto &w : worker_memory)
            {
                if (w.row_depths.size())
                {
                    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(input_data, output_depths, w, worker_memory)
                    for (size_t_for row = 0; row < (decltype(row))input_data.nrows; row++)
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
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] = std::exp2( - output_depths[row] / depth_divisor );
        }

        else
        {
            double ntrees_dbl = (double) ntrees;
            for (size_t row = 0; row < nrows; row++)
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
            for (auto &w : worker_memory)
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
template <class real_t, class sparse_ix>
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
             uint64_t random_seed, bool use_long_double)
{
    if (use_long_double && !has_long_double()) {
        use_long_double = false;
        fprintf(stderr, "Passed 'use_long_double=true', but library was compiled without long double support.\n");
    }
    #ifndef NO_LONG_DOUBLE
    if (likely(!use_long_double))
    #endif
        return add_tree_internal<real_t, sparse_ix, double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, nrows,
            max_depth,     ncols_per_tree,
            limit_depth,   penalize_range, standardize_data,
            fast_bratio,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            depth_imp, weigh_imp_rows,
            all_perm, imputer, min_imp_obs,
            indexer,
            ref_numeric_data, ref_categ_data,
            ref_is_col_major, ref_ld_numeric, ref_ld_categ,
            ref_Xc, ref_Xc_ind, ref_Xc_indptr,
            random_seed
        );
    #ifndef NO_LONG_DOUBLE
    else
        return add_tree_internal<real_t, sparse_ix, long double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, nrows,
            max_depth,     ncols_per_tree,
            limit_depth,   penalize_range, standardize_data,
            fast_bratio,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            depth_imp, weigh_imp_rows,
            all_perm, imputer, min_imp_obs,
            indexer,
            ref_numeric_data, ref_categ_data,
            ref_is_col_major, ref_ld_numeric, ref_ld_categ,
            ref_Xc, ref_Xc_ind, ref_Xc_indptr,
            random_seed
        );
    #endif
}

template <class real_t, class sparse_ix, class ldouble_safe>
int add_tree_internal(
             IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
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
             uint64_t random_seed)
{
    if (
        prob_pick_by_gain_avg  < 0  || prob_pick_by_gain_pl  < 0 ||
        prob_pick_by_full_gain < 0  || prob_pick_by_dens     < 0 ||
        prob_pick_col_by_range < 0  ||
        prob_pick_col_by_var   < 0  || prob_pick_col_by_kurt < 0
    ) {
        throw std::runtime_error("Cannot pass negative probabilities.\n");
    }
    if (prob_pick_col_by_range && ncols_categ)
        throw std::runtime_error("'prob_pick_col_by_range' is not compatible with categorical data.\n");
    if (prob_pick_by_full_gain && ncols_categ)
        throw std::runtime_error("'prob_pick_by_full_gain' is not compatible with categorical data.\n");
    if (prob_pick_col_by_kurt && weigh_by_kurt)
        throw std::runtime_error("'weigh_by_kurt' and 'prob_pick_col_by_kurt' cannot be used together.\n");
    if (ndim == 0 && model_outputs == NULL)
        throw std::runtime_error("Must pass 'ndim>0' in the extended model.\n");
    if (indexer != NULL && !indexer->indices.empty() && !indexer->indices.front().reference_points.empty()) {
        if (ref_numeric_data == NULL && ref_categ_data == NULL && ref_Xc_indptr == NULL)
            throw std::runtime_error("'indexer' has reference points. Those points must be passed to index them in the new tree to add.\n");
    }

    std::vector<ImputeNode> *impute_nodes = NULL;

    int max_categ = 0;
    for (size_t col = 0; col < ncols_categ; col++)
        max_categ = (ncat[col] > max_categ)? ncat[col] : max_categ;

    if (model_outputs != NULL)
        ntry = std::min(ntry, ncols_numeric + ncols_categ);

    if (ncols_per_tree == 0)
        ncols_per_tree = ncols_numeric + ncols_categ;

    if (indexer != NULL && indexer->indices.empty())
        indexer = NULL;

    InputData<real_t, sparse_ix>
              input_data     = {numeric_data, ncols_numeric, categ_data, ncat, max_categ, ncols_categ,
                                nrows, ncols_numeric + ncols_categ, sample_weights,
                                false, col_weights,
                                Xc, Xc_ind, Xc_indptr,
                                0, 0, std::vector<double>(),
                                std::vector<char>(), 0, NULL,
                                (double*)NULL, (double*)NULL, (int*)NULL, std::vector<double>(),
                                std::vector<double>(), std::vector<double>(),
                                std::vector<size_t>(), std::vector<size_t>()};
    ModelParams model_params = {false, nrows, (size_t)1, ncols_per_tree,
                                max_depth? max_depth : (nrows - 1),
                                penalize_range, standardize_data, random_seed, weigh_by_kurt,
                                prob_pick_by_gain_avg, prob_pick_by_gain_pl,
                                prob_pick_by_full_gain, prob_pick_by_dens,
                                prob_pick_col_by_range, prob_pick_col_by_var,
                                prob_pick_col_by_kurt,
                                min_gain, cat_split_type, new_cat_action, missing_action,
                                (model_outputs != NULL)? model_outputs->scoring_metric : model_outputs_ext->scoring_metric,
                                fast_bratio, all_perm,
                                (model_outputs != NULL)? 0 : ndim, ntry,
                                coef_type, coef_by_prop, false, false, false, depth_imp, weigh_imp_rows, min_imp_obs};

    if (prob_pick_by_full_gain)
    {
        if (input_data.Xc_indptr == NULL)
            colmajor_to_rowmajor(input_data.numeric_data, input_data.nrows, input_data.ncols_numeric, input_data.X_row_major);
        else
            colmajor_to_rowmajor(input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                 input_data.nrows, input_data.ncols_numeric,
                                 input_data.Xr, input_data.Xr_ind, input_data.Xr_indptr);
    }

    std::unique_ptr<WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>> workspace(
        new WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>()
    );

    size_t last_tree;
    bool added_tree = false;
    try
    {
        if (model_outputs != NULL)
        {
            last_tree = model_outputs->trees.size();
            model_outputs->trees.emplace_back();
            added_tree = true;
        }

        else
        {
            last_tree = model_outputs_ext->hplanes.size();
            model_outputs_ext->hplanes.emplace_back();
            added_tree = true;
        }

        if (imputer != NULL)
        {
            imputer->imputer_tree.emplace_back();
            impute_nodes = &(imputer->imputer_tree.back());
        }

        if (indexer != NULL)
        {
            indexer->indices.emplace_back();
        }

        SignalSwitcher ss = SignalSwitcher();
        check_interrupt_switch(ss);

        fit_itree<decltype(input_data), typename std::remove_pointer<decltype(workspace.get())>::type, ldouble_safe>(
                  (model_outputs != NULL)? &model_outputs->trees.back() : NULL,
                  (model_outputs_ext != NULL)? &model_outputs_ext->hplanes.back() : NULL,
                  *workspace,
                  input_data,
                  model_params,
                  impute_nodes,
                  last_tree);

        check_interrupt_switch(ss);

        if (model_outputs != NULL) {
            model_outputs->trees.back().shrink_to_fit();
            model_outputs->has_range_penalty = model_outputs->has_range_penalty || penalize_range;
        }
        else {
            model_outputs_ext->hplanes.back().shrink_to_fit();
            model_outputs_ext->has_range_penalty = model_outputs_ext->has_range_penalty || penalize_range;
        }

        if (imputer != NULL)
            imputer->imputer_tree.back().shrink_to_fit();

        if (indexer != NULL)
        {
            if (model_outputs != NULL)
                build_terminal_node_mappings_single_tree(indexer->indices.back().terminal_node_mappings,
                                                         indexer->indices.back().n_terminal,
                                                         model_outputs->trees.back());
            else
                build_terminal_node_mappings_single_tree(indexer->indices.back().terminal_node_mappings,
                                                         indexer->indices.back().n_terminal,
                                                         model_outputs_ext->hplanes.back());

            check_interrupt_switch(ss);


            if (!indexer->indices.front().node_distances.empty())
            {
                std::vector<size_t> temp;
                temp.reserve(indexer->indices.back().n_terminal);
                if (model_outputs != NULL) {
                    build_dindex(
                        temp,
                        indexer->indices.back().terminal_node_mappings,
                        indexer->indices.back().node_distances,
                        indexer->indices.back().node_depths,
                        indexer->indices.back().n_terminal,
                        model_outputs->trees.back()
                    );
                }
                else {
                    build_dindex(
                        temp,
                        indexer->indices.back().terminal_node_mappings,
                        indexer->indices.back().node_distances,
                        indexer->indices.back().node_depths,
                        indexer->indices.back().n_terminal,
                        model_outputs_ext->hplanes.back()
                    );
                }
            }

            check_interrupt_switch(ss);
            if (!indexer->indices.front().reference_points.empty())
            {
                size_t n_ref = indexer->indices.front().reference_points.size();
                std::vector<sparse_ix> terminal_indices(n_ref);
                std::unique_ptr<double[]> ignored(new double[n_ref]);
                if (model_outputs != NULL)
                {
                    IsoForest single_tree_model;
                    single_tree_model.new_cat_action = model_outputs->new_cat_action;
                    single_tree_model.cat_split_type = model_outputs->cat_split_type;
                    single_tree_model.missing_action = model_outputs->missing_action;
                    single_tree_model.trees.push_back(model_outputs->trees.back());

                    predict_iforest(ref_numeric_data, ref_categ_data,
                                    ref_is_col_major, ref_ld_numeric, ref_ld_categ,
                                    ref_Xc, ref_Xc_ind, ref_Xc_indptr,
                                    (real_t*)NULL, (sparse_ix*)NULL, (sparse_ix*)NULL,
                                    n_ref, 1, false,
                                    &single_tree_model, (ExtIsoForest*)NULL,
                                    ignored.get(), terminal_indices.data(),
                                    (double*)NULL,
                                    indexer);
                }

                else
                {
                    ExtIsoForest single_tree_model;
                    single_tree_model.new_cat_action = model_outputs_ext->new_cat_action;
                    single_tree_model.cat_split_type = model_outputs_ext->cat_split_type;
                    single_tree_model.missing_action = model_outputs_ext->missing_action;
                    single_tree_model.hplanes.push_back(model_outputs_ext->hplanes.back());

                    predict_iforest(ref_numeric_data, ref_categ_data,
                                    ref_is_col_major, ref_ld_numeric, ref_ld_categ,
                                    ref_Xc, ref_Xc_ind, ref_Xc_indptr,
                                    (real_t*)NULL, (sparse_ix*)NULL, (sparse_ix*)NULL,
                                    n_ref, 1, false,
                                    (IsoForest*)NULL, &single_tree_model,
                                    ignored.get(), terminal_indices.data(),
                                    (double*)NULL,
                                    indexer);
                }

                ignored.reset();
                indexer->indices.back().reference_points.assign(terminal_indices.begin(), terminal_indices.end());
                indexer->indices.back().reference_points.shrink_to_fit();
                build_ref_node(indexer->indices.back());
            }

            check_interrupt_switch(ss);
        }
    }

    catch (...)
    {
        if (added_tree)
        {
            if (model_outputs != NULL)
                model_outputs->trees.pop_back();
            else
                model_outputs_ext->hplanes.pop_back();
            if (imputer != NULL) {
                if (model_outputs != NULL)
                    imputer->imputer_tree.resize(model_outputs->trees.size());
                else
                    imputer->imputer_tree.resize(model_outputs_ext->hplanes.size());
            }
            if (indexer != NULL) {
                if (model_outputs != NULL)
                    indexer->indices.resize(model_outputs->trees.size());
                else
                    indexer->indices.resize(model_outputs_ext->hplanes.size());
            }
        }
        throw;
    }

    return EXIT_SUCCESS;
}

template <class InputData, class WorkerMemory, class ldouble_safe>
void fit_itree(std::vector<IsoTree>    *tree_root,
               std::vector<IsoHPlane>  *hplane_root,
               WorkerMemory             &workspace,
               InputData                &input_data,
               ModelParams              &model_params,
               std::vector<ImputeNode> *impute_nodes,
               size_t                   tree_num)
{
    /* initialize array for depths if called for */
    if (workspace.ix_arr.empty() && model_params.calc_depth)
        workspace.row_depths.resize(input_data.nrows, 0);

    /* choose random sample of rows */
    if (workspace.ix_arr.empty()) workspace.ix_arr.resize(model_params.sample_size);
    if (input_data.log2_n > 0)
        workspace.btree_weights.assign(input_data.btree_weights_init.begin(),
                                       input_data.btree_weights_init.end());
    workspace.rnd_generator.seed(model_params.random_seed + tree_num);
    workspace.rbin  = UniformUnitInterval(0, 1);
    sample_random_rows<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                       workspace.ix_arr, input_data.nrows, model_params.with_replacement,
                       workspace.rnd_generator, workspace.ix_all,
                       (input_data.weight_as_sample)? input_data.sample_weights : NULL,
                       workspace.btree_weights, input_data.log2_n, input_data.btree_offset,
                       workspace.is_repeated);
    workspace.st  = 0;
    workspace.end = model_params.sample_size - 1;

    /* in some cases, it's not possible to use column weights even if they are given,
       because every single column will always need to be checked or end up being used. */
    bool avoid_col_weights = (tree_root != NULL && model_params.ntry >= model_params.ncols_per_tree &&
                              model_params.prob_pick_by_gain_avg  + model_params.prob_pick_by_gain_pl +
                              model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens >= 1)
                                ||
                             (tree_root == NULL && model_params.ndim >= model_params.ncols_per_tree)
                                ||
                             (model_params.ncols_per_tree == 1);
    if (input_data.preinitialized_col_sampler == NULL)
    {
        if (input_data.col_weights != NULL && !avoid_col_weights && !model_params.weigh_by_kurt)
            workspace.col_sampler.initialize(input_data.col_weights, input_data.ncols_tot);
    }


    /* set expected tree size and add root node */
    {
        size_t exp_nodes = mult2(model_params.sample_size);
        if (model_params.sample_size >= div2(SIZE_MAX))
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
    if (workspace.categs.empty())
        workspace.categs.resize(input_data.max_categ);

    /* initialize array with per-node column weights if needed */
    if ((model_params.prob_pick_col_by_range ||
         model_params.prob_pick_col_by_var ||
         model_params.prob_pick_col_by_kurt) && workspace.node_col_weights.empty())
    {
        workspace.node_col_weights.resize(input_data.ncols_tot);
        if (tree_root != NULL || model_params.standardize_data || model_params.missing_action != Fail)
        {
            workspace.saved_stat1.resize(input_data.ncols_numeric);
            workspace.saved_stat2.resize(input_data.ncols_numeric);
        }
    }

    /* IMPORTANT!!!!!
       The standard library implementation is likely going to use the Box-Muller method
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
            workspace.coef_norm = StandardNormalDistr(0, 1);
        if (model_params.coef_type == Uniform)
            workspace.coef_unif = UniformMinusOneToOne(-1, 1);
    }

    /* for the extended model, initialize extra vectors and objects */
    if (hplane_root != NULL && workspace.comb_val.empty())
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

    /* If there are density weights, need to standardize them to sum up to
       the sample size here. Note that weights for missing values with 'Divide'
       are only initialized on-demand later on. */
    workspace.changed_weights = false;
    if (hplane_root == NULL) workspace.weights_map.clear();

    ldouble_safe weight_scaling = 0;
    if (input_data.sample_weights != NULL && !input_data.weight_as_sample)
    {
        workspace.changed_weights = true;

        /* For the extended model, if there is no sub-sampling, these weights will remain
           constant throughout and do not need to be re-generated. */
        if (!(  hplane_root != NULL &&
                (!workspace.weights_map.empty() || !workspace.weights_arr.empty()) &&
                model_params.sample_size == input_data.nrows && !model_params.with_replacement
              )
            )
        {
            workspace.weights_map.clear();

            /* if the sub-sample size is small relative to the full sample size, use a mapping */
            if (input_data.Xc_indptr != NULL && model_params.sample_size < input_data.nrows / 50)
            {
                for (const size_t ix : workspace.ix_arr)
                    weight_scaling += input_data.sample_weights[ix];
                weight_scaling = (ldouble_safe)model_params.sample_size / weight_scaling;
                workspace.weights_map.reserve(workspace.ix_arr.size());
                for (const size_t ix : workspace.ix_arr)
                    workspace.weights_map[ix] = input_data.sample_weights[ix] * weight_scaling;
            }

            /* if the sub-sample size is large, fill a full array matching to the sample size */
            else
            {
                if (workspace.weights_arr.empty())
                {
                    workspace.weights_arr.assign(input_data.sample_weights, input_data.sample_weights + input_data.nrows);
                    weight_scaling = std::accumulate(workspace.ix_arr.begin(),
                                                     workspace.ix_arr.end(),
                                                     (ldouble_safe)0,
                                                     [&input_data](const ldouble_safe a, const size_t b){return a + (ldouble_safe)input_data.sample_weights[b];}
                                                     );
                    weight_scaling = (ldouble_safe)model_params.sample_size / weight_scaling;
                    for (double &w : workspace.weights_arr)
                        w *= weight_scaling;
                }

                else
                {
                    for (const size_t ix : workspace.ix_arr)
                    {
                        weight_scaling += input_data.sample_weights[ix];
                        workspace.weights_arr[ix] = input_data.sample_weights[ix];
                    }
                    weight_scaling = (ldouble_safe)model_params.sample_size / weight_scaling;
                    for (double &w : workspace.weights_arr)
                        w *= weight_scaling;
                }
            }
        }
    }

    /* if producing distance/similarity, also need to initialize the triangular matrix */
    if (model_params.calc_dist && workspace.tmat_sep.empty())
        workspace.tmat_sep.resize((input_data.nrows * (input_data.nrows - 1)) / 2, 0);

    /* make space for buffers if not already allocated */
    if (
            (model_params.prob_pick_by_gain_avg    > 0  ||
             model_params.prob_pick_by_gain_pl     > 0  ||
             model_params.prob_pick_by_full_gain   > 0  ||
             model_params.prob_pick_by_dens        > 0  ||
             model_params.prob_pick_col_by_range   > 0  ||
             model_params.prob_pick_col_by_var     > 0  ||
             model_params.prob_pick_col_by_kurt    > 0  ||
             model_params.weigh_by_kurt || hplane_root != NULL)
                &&
            (workspace.buffer_dbl.empty() && workspace.buffer_szt.empty() && workspace.buffer_chr.empty())
        )
    {
        size_t min_size_dbl = 0;
        size_t min_size_szt = 0;
        size_t min_size_chr = 0;

        bool gain = model_params.prob_pick_by_gain_avg  > 0 ||
                    model_params.prob_pick_by_gain_pl   > 0 ||
                    model_params.prob_pick_by_full_gain > 0 ||
                    model_params.prob_pick_by_dens      > 0;

        if (input_data.ncols_categ)
        {
            min_size_szt = (size_t)2 * (size_t)input_data.max_categ;
            min_size_dbl = input_data.max_categ + 1;
            if (gain && model_params.cat_split_type == SubSet)
                min_size_chr = input_data.max_categ;
        }

        if (input_data.Xc_indptr != NULL && gain)
        {
            min_size_szt = std::max(min_size_szt, model_params.sample_size);
            min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
        }

        /* TODO: revisit if this covers all the cases */
        if (model_params.ntry > 1 || gain)
        {
            min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
            if (model_params.ndim < 2 && input_data.Xc_indptr != NULL)
                min_size_dbl = std::max(min_size_dbl, (size_t)2*model_params.sample_size);
        }

        /* for sampled column choices */
        if (model_params.prob_pick_col_by_var)
        {
            if (input_data.ncols_categ) {
                min_size_szt = std::max(min_size_szt, (size_t)input_data.max_categ + 1);
                min_size_dbl = std::max(min_size_dbl, (size_t)input_data.max_categ + 1);
            }
        }

        if (model_params.prob_pick_col_by_kurt)
        {
            if (input_data.ncols_categ) {
                min_size_szt = std::max(min_size_szt, (size_t)input_data.max_categ + 1);
                min_size_dbl = std::max(min_size_dbl, (size_t)input_data.max_categ);
            }

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
                min_size_szt = std::max(min_size_szt, (size_t)2 * (size_t)input_data.max_categ + (size_t)1);
                min_size_dbl = std::max(min_size_dbl, (size_t)input_data.max_categ);
            }

            if (model_params.weigh_by_kurt)
                min_size_szt = std::max(min_size_szt, input_data.ncols_tot);

            if (gain && (!workspace.weights_arr.empty() || !workspace.weights_map.empty()))
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
                model_params.prob_pick_by_gain_pl   ||
                model_params.prob_pick_by_full_gain ||
                model_params.prob_pick_by_dens
            )
           )
        {
            workspace.this_split_categ.resize(input_data.max_categ);
        }

    }

    /* Other potentially necessary buffers */
    if (
        tree_root != NULL && model_params.missing_action == Impute &&
        (model_params.prob_pick_by_gain_avg  || model_params.prob_pick_by_gain_pl ||
         model_params.prob_pick_by_full_gain || model_params.prob_pick_by_dens) &&
        input_data.Xc_indptr == NULL && input_data.ncols_numeric && workspace.imputed_x_buffer.empty()
    )
    {
        workspace.imputed_x_buffer.resize(input_data.nrows);
    }

    if (model_params.prob_pick_by_full_gain && workspace.col_indices.empty())
        workspace.col_indices.resize(model_params.ncols_per_tree);

    if (
        (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var) &&
        model_params.weigh_by_kurt &&
        model_params.sample_size == input_data.nrows && !model_params.with_replacement &&
        (model_params.ncols_per_tree == input_data.ncols_tot) &&
        !input_data.all_kurtoses.empty()
    ) {
        workspace.tree_kurtoses = input_data.all_kurtoses.data();
    }
    else {
        workspace.tree_kurtoses = NULL;
    }

    /* weigh columns by kurtosis in the sample if required */
    /* TODO: this one could probably be refactored to use the function in the helpers */
    std::vector<double> kurt_weights;
    bool avoid_leave_m_cols = false;
    if (
        model_params.weigh_by_kurt &&
        !avoid_col_weights &&
        (input_data.preinitialized_col_sampler == NULL
            ||
         ((model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var) && workspace.tree_kurtoses == NULL))
    )
    {
        kurt_weights.resize(input_data.ncols_numeric + input_data.ncols_categ, 0.);

        if (model_params.ncols_per_tree >= input_data.ncols_tot)
        {

            if (input_data.Xc_indptr == NULL)
            {

                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                        kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                                          input_data.numeric_data + col * input_data.nrows,
                                                          model_params.missing_action);
                    else if (!workspace.weights_arr.empty())
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, decltype(workspace.weights_arr), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   input_data.numeric_data + col * input_data.nrows,
                                                                   model_params.missing_action, workspace.weights_arr);
                    else
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                   decltype(workspace.weights_map), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   input_data.numeric_data + col * input_data.nrows,
                                                                   model_params.missing_action, workspace.weights_map);
                }
            }

            else
            {
                std::sort(workspace.ix_arr.begin(), workspace.ix_arr.end());
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                        kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                          typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                          ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                          input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                          model_params.missing_action);
                    else if (!workspace.weights_arr.empty())
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                   typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                   decltype(workspace.weights_arr), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                   input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                   model_params.missing_action, workspace.weights_arr);
                    else
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                   typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                   decltype(workspace.weights_map), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                   input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                   model_params.missing_action, workspace.weights_map);
                }
            }

            for (size_t col = 0; col < input_data.ncols_categ; col++)
            {
                if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis<ldouble_safe>(
                                      workspace.ix_arr.data(), workspace.st, workspace.end,
                                      input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                      workspace.buffer_szt.data(), workspace.buffer_dbl.data(),
                                      model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator);
                else if (!workspace.weights_arr.empty())
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                               workspace.buffer_dbl.data(),
                                               model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                               workspace.weights_arr);
                else
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                               workspace.buffer_dbl.data(),
                                               model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                               workspace.weights_map);
            }

            for (auto &w : kurt_weights) w = (w == -HUGE_VAL)? 0. : std::fmax(1e-8, -1. + w);
            if (input_data.col_weights != NULL)
            {
                for (size_t col = 0; col < input_data.ncols_tot; col++)
                {
                    if (kurt_weights[col] <= 0) continue;
                    kurt_weights[col] *= input_data.col_weights[col];
                    kurt_weights[col] = std::fmax(kurt_weights[col], 1e-100);
                }
            }
            workspace.col_sampler.initialize(kurt_weights.data(), kurt_weights.size());
        }

        

        else
        {
            std::vector<size_t> cols_take(model_params.ncols_per_tree);
            std::vector<size_t> buffer1;
            std::vector<bool> buffer2;
            sample_random_rows<double, double>(
                               cols_take, input_data.ncols_tot, false,
                               workspace.rnd_generator, buffer1,
                               (double*)NULL, kurt_weights, /* <- will not get used */
                               (size_t)0, (size_t)0, buffer2);

            if (
                model_params.sample_size == input_data.nrows &&
                !model_params.with_replacement &&
                !input_data.all_kurtoses.empty()
            )
            {
                for (size_t col : cols_take)
                    kurt_weights[col] = input_data.all_kurtoses[col];
                goto skip_kurt_calculations;
            }

            if (input_data.Xc_indptr != NULL)
                std::sort(workspace.ix_arr.begin(), workspace.ix_arr.end());

            for (size_t col : cols_take)
            {
                if (col < input_data.ncols_numeric)
                {
                    if (input_data.Xc_indptr == NULL)
                    {
                        if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                            kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                                                              workspace.ix_arr.data(), workspace.st, workspace.end,
                                                              input_data.numeric_data + col * input_data.nrows,
                                                              model_params.missing_action);
                        else if (!workspace.weights_arr.empty())
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                       decltype(workspace.weights_arr), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                       input_data.numeric_data + col * input_data.nrows,
                                                                       model_params.missing_action, workspace.weights_arr);
                        else
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                       decltype(workspace.weights_map), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                       input_data.numeric_data + col * input_data.nrows,
                                                                       model_params.missing_action, workspace.weights_map);
                    }

                    else
                    {
                        if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                            kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                              typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                              ldouble_safe>(
                                                              workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                              model_params.missing_action);
                        else if (!workspace.weights_arr.empty())
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                       typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                       decltype(workspace.weights_arr), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                       input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                       model_params.missing_action, workspace.weights_arr);
                        else
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                       typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                       decltype(workspace.weights_map), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                       input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                       model_params.missing_action, workspace.weights_map);
                    }
                }

                else
                {
                    if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                        kurt_weights[col] =
                            calc_kurtosis<ldouble_safe>(
                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                          input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                          input_data.ncat[col - input_data.ncols_numeric],
                                          workspace.buffer_szt.data(), workspace.buffer_dbl.data(),
                                          model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator);
                    else if (!workspace.weights_arr.empty())
                        kurt_weights[col] =
                            calc_kurtosis_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                   input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                                   input_data.ncat[col - input_data.ncols_numeric],
                                                   workspace.buffer_dbl.data(),
                                                   model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                                   workspace.weights_arr);
                    else
                        kurt_weights[col] =
                            calc_kurtosis_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                   input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                                   input_data.ncat[col - input_data.ncols_numeric],
                                                   workspace.buffer_dbl.data(),
                                                   model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                                   workspace.weights_map);
                }

                /* Note to self: don't move this  to outside of the braces, as it needs to assign a weight
                   of zero to the columns that were not selected, thus it should only do this clipping
                   for columns that are chosen. */
                if (kurt_weights[col] == -HUGE_VAL)
                {
                    kurt_weights[col] = 0;
                }

                else
                {
                    kurt_weights[col] = std::fmax(1e-8, -1. + kurt_weights[col]);
                    if (input_data.col_weights != NULL)
                    {
                        kurt_weights[col] *= input_data.col_weights[col];
                        kurt_weights[col] = std::fmax(kurt_weights[col], 1e-100);
                    }
                }
            }

            skip_kurt_calculations:
            workspace.col_sampler.initialize(kurt_weights.data(), kurt_weights.size());
            avoid_leave_m_cols = true;
        }

        if (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var)
        {
            workspace.tree_kurtoses = kurt_weights.data();
        }
    }

    bool col_sampler_is_fresh = true;
    if (input_data.preinitialized_col_sampler == NULL) {
        workspace.col_sampler.initialize(input_data.ncols_tot);
    }
    else {
        workspace.col_sampler = *((ColumnSampler<ldouble_safe>*)input_data.preinitialized_col_sampler);
        col_sampler_is_fresh = false;
    }
    /* TODO: this can be done more efficiently when sub-sampling columns */
    if (!avoid_leave_m_cols)
        workspace.col_sampler.leave_m_cols(model_params.ncols_per_tree, workspace.rnd_generator);
    if (model_params.ncols_per_tree < input_data.ncols_tot) col_sampler_is_fresh = false;
    workspace.try_all = false;
    if (hplane_root != NULL && model_params.ndim >= input_data.ncols_tot)
        workspace.try_all = true;

    if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
    {
        workspace.density_calculator.initialize(model_params.max_depth,
                                                input_data.ncols_categ? input_data.max_categ : 0,
                                                tree_root != NULL && input_data.ncols_categ,
                                                model_params.scoring_metric);
    }

    else if (is_boxed_metric(model_params.scoring_metric))
    {
        if (tree_root != NULL)
            workspace.density_calculator.initialize_bdens(input_data,
                                                          model_params,
                                                          workspace.ix_arr,
                                                          workspace.col_sampler);
        else
            workspace.density_calculator.initialize_bdens_ext(input_data,
                                                              model_params,
                                                              workspace.ix_arr,
                                                              workspace.col_sampler,
                                                              col_sampler_is_fresh);
    }

    if (tree_root != NULL)
    {
        split_itree_recursive<InputData, WorkerMemory, ldouble_safe>(
                              *tree_root,
                              workspace,
                              input_data,
                              model_params,
                              impute_nodes,
                              0);
    }

    else
    {
        split_hplane_recursive<InputData, WorkerMemory, ldouble_safe>(
                               *hplane_root,
                               workspace,
                               input_data,
                               model_params,
                               impute_nodes,
                               0);
    }

    /* if producing imputation structs, only need to keep the ones for terminal nodes */
    if (impute_nodes != NULL)
        drop_nonterminal_imp_node(*impute_nodes, tree_root, hplane_root);
}
