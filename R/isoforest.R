#' @importFrom parallel detectCores
#' @importFrom stats predict
#' @importFrom utils head
#' @importFrom Rcpp evalCpp
#' @useDynLib isotree, .registration=TRUE

#' @title Create Isolation Forest Model
#' @description Isolation Forest is an algorithm originally developed for outlier detection that consists in splitting
#' sub-samples of the data according to some attribute/feature/column at random. The idea is that, the rarer
#' the observation, the more likely it is that a random uniform split on some feature would put outliers alone
#' in one branch, and the fewer splits it will take to isolate an outlier observation like this. The concept
#' is extended to splitting hyperplanes in the extended model (i.e. splitting by more than one column at a time), and to
#' guided (not entirely random) splits in the SCiForest and FCF models that aim at isolating outliers faster and/or
#' finding clustered outliers.
#' 
#' This version adds heuristics to handle missing data and categorical variables. Can be used to aproximate pairwise
#' distances by checking the depth after which two observations become separated, and to approximate densities by fitting
#' trees beyond balanced-tree limit. Offers options to vary between randomized and deterministic splits too.
#' 
#' \bold{Important:} The default parameters in this software do not correspond to the suggested parameters in
#' any of the references (see section "Matching models from references").
#' In particular, the following default values are likely to cause huge differences when compared to the
#' defaults in other software: `ndim`, `sample_size`, `ntrees`. The defaults here are
#' nevertheless more likely to result in better models. In order to mimic scikit-learn for example, one
#' would need to pass `ndim=1`, `sample_size=256`, `ntrees=100`, `missing_action="fail"`, `nthreads=1`.
#' 
#' Note that the default parameters will not scale to large datasets. In particular,
#' if the amount of data is large, it's suggested to set a smaller sample size for each tree (parameter `sample_size`),
#' and to fit fewer of them (parameter `ntrees`).
#' As well, the default option for `missing_action` might slow things down significantly
#' (see below for details).
#' These defaults can also result in very big model sizes in memory and as serialized
#' files (e.g. models that weight over 10GB) when the number of rows in the data is large.
#' Using fewer trees, smaller sample sizes, and shallower trees can help to reduce model
#' sizes if that becomes a problem.
#' 
#' The model offers many tunable parameters (see reference [11] for a comparison).
#' The most likely candidate to tune is
#' `prob_pick_pooled_gain`, for which higher values tend to
#' result in a better ability to flag outliers in multimodal datasets, at the expense of poorer
#' generalizability to inputs with values outside the variables' ranges to which the model was fit
#' (see plots generated from the examples for a better idea of the difference). The next candidate to tune is
#' `sample_size` - the default is to use all rows, but in some datasets introducing sub-sampling can help,
#' especially for the single-variable model. In smaller datasets, one might also want to experiment
#' with `weigh_by_kurtosis` and perhaps lower `ndim`.If using `prob_pick_pooled_gain`, models
#' are likely to benefit from deeper trees (controlled by `max_depth`), but using large samples
#' and/or deeper trees can result in significantly slower model fitting and predictions - in such cases,
#' using `min_gain` (with a value like 0.25) with `max_depth=NULL` can offer a better speed/performance
#' trade-off than changing `max_depth`.
#' 
#' @section Matching models from references:
#' Shorthands for parameter combinations that match some of the references:\itemize{
#' \item 'iForest' (reference [1]): `ndim=1`, `sample_size=256`, `max_depth=8`, `ntrees=100`, `missing_action="fail"`.
#' \item 'EIF' (reference [3]): `ndim=2`, `sample_size=256`, `max_depth=8`, `ntrees=100`, `missing_action="fail"`,
#' `coefs="uniform"`, `standardize_data=False` (plus standardizing the data \bold{before} passing it).
#' \item 'SCiForest' (reference [4]): `ndim=2`, `sample_size=256`, `max_depth=8`, `ntrees=100`, `missing_action="fail"`,
#' `coefs="normal"`, `ntry=10`, `prob_pick_avg_gain=1`, `penalize_range=True`.
#' Might provide much better results with `max_depth=NULL` despite the reference's recommendation.
#' \item 'FCF' (reference [11]): `ndim=2`, `sample_size=256`, `max_depth=NULL`, `ntrees=200`,
#' `missing_action="fail"`, `coefs="normal"`, `ntry=1`, `prob_pick_pooled_gain=1`.
#' Might provide similar or better results with `ndim=1`.
#' For the FCF model aimed at imputing missing values,
#' might give better results with `ntry=10` or higher and much larger sample sizes.
#' }
#' @section Model serving considerations:
#' If the model is built with `nthreads>1`, the prediction function \link{predict.isolation_forest} will
#' use OpenMP for parallelization. In a linux setup, one usually has GNU's "gomp" as OpenMP as backend, which
#' will hang when used in a forked process - for example, if one tries to call this prediction function from
#' `RestRserve`, which uses process forking for parallelization, it will cause the whole application to freeze;
#' and if using kubernetes on top of a different backend such as plumber, might cause it to run slower than
#' needed or to hang too. A potential fix in these cases is to set the number of threads to 1 in the object
#' (e.g. `model$nthreads <- 1L`), or to use a different version of this library compiled without OpenMP
#' (requires manually altering the `Makevars` file), or to use a non-GNU OpenMP backend. This should not
#' be an issue when using this library normally in e.g. an RStudio session.
#' 
#' In order to make model objects serializable (i.e. usable with `save`, `saveRDS`, and similar), these model
#' objects keep serialized raw bytes from which their underlying heap-allocated C++ object (which does not
#' survive serializations) can be reconstructed. For model serving, one would usually want to drop these
#' serialized bytes after having loaded a model through `readRDS` or similar (note that reconstructing the
#' C++ object will first require calling \link{isotree.restore.handle}, which is done automatically when
#' calling `predict` and similar), as they can increase memory usage by a large amount. These redundant raw bytes
#' can be dropped as follows: `model$cpp_obj$serialized <- NULL` (and an additional
#' `model$cpp_obj$imp_ser <- NULL` when using `build_imputer=TRUE`). After that, one might want to force garbage
#' collection through `gc()`.
#' @details If requesting outlier scores or depths or separation/distance while fitting the
#' model and using multiple threads, there can be small differences in the predicted
#' scores/depth/separation/distance between runs due to roundoff error.
#' 
#' While it's not possible to get a quick overview of the size of the resulting
#' model object, one can get a lower bound of its size in RAM by checking the
#' following: `2*NROW(model$cpp_obj$serialized) / 1024^2` (which should estimate
#' the size in megabytes).
#' @param data Data to which to fit the model. Supported inputs type are:\itemize{
#' \item A `data.frame`, also accepted as `data.table` or `tibble`.
#' \item A `matrix` object from base R.
#' \item A sparse matrix in CSC format, either from package `Matrix` (class `dgCMatrix`) or
#' from package `SparseM` (class `matrix.csc`).
#' }
#' 
#' If passing a `data.frame`, will assume that columns are:
#' \itemize{
#'   \item Numerical, if they are of types `numeric`, `integer`, `Date`, `POSIXct`.
#'   \item Categorical, if they are of type `character`, `factor`, `bool`. Note that,
#'   if factors are ordered, the order will be ignored here.
#' }
#' Other input and column types are not supported.
#' @param sample_size Sample size of the data sub-samples with which each binary tree will be built.
#' Recommended value in references [1], [2], [3], [4] is 256, while the default value in the author's code in reference [5] is
#' `nrow(data)`.
#' 
#' If passing `NULL`, will take the full number of rows in the data (no sub-sampling).
#' 
#' If passing a number between zero and one, will assume it means taking a sample size that represents
#' that proportion of the rows in the data.
#' 
#' Note that sub-sampling is incompatible with `output_score`, `output_dist`, and `output_imputations`,
#' and if any of those options is requested, `sample_size` will be overriden.
#' 
#' Hint: seeing a distribution of scores which is on average too far below 0.5 could mean that the
#' model needs more trees and/or bigger samples to reach convergence (unless using non-random
#' splits, in which case the distribution is likely to be centered around a much lower number),
#' or that the distributions in the data are too skewed for random uniform splits.
#' @param ntrees Number of binary trees to build for the model. Recommended value in reference [1] is 100, while the
#' default value in the author's code in reference [5] is 10. In general, the number of trees required for good results
#' is higher when (a) there are many columns, (b) there are categorical variables, (c) categorical variables have many
#' categories, (d) `ndim` is high, (e) `prob_pick_pooled_gain` is used.
#' 
#' Hint: seeing a distribution of scores which is on average too far below 0.5 could mean that the
#' model needs more trees and/or bigger samples to reach convergence (unless using non-random
#' splits, in which case the distribution is likely to be centered around a much lower number),
#' or that the distributions in the data are too skewed for random uniform splits.
#' @param ndim Number of columns to combine to produce a split. If passing 1, will produce the single-variable model described
#' in references [1] and [2], while if passing values greater than 1, will produce the extended model described in
#' references [3] and [4].
#' Recommended value in reference [4] is 2, while [3] recommends a low value such as 2 or 3. Models with values higher than 1
#' are referred hereafter as the extended model (as in reference [3]).
#' 
#' If passing `NULL`, will assume it means using the full number of columns in the data.
#' 
#' Note that, when using `ndim>1` plus `standardize_data=TRUE`, the variables are standardized at each step
#' as suggested in [4], which makes the models slightly different than in [3].
#' @param ntry When using `prob_pick_pooled_gain` and/or `prob_pick_avg_gain`, how many variables (with `ndim=1`)
#' or linear combinations (with `ndim>1`) to try for determining the best one according to gain.
#' 
#' Recommended value in reference [4] is 10 (with `prob_pick_avg_gain`, for outlier detection), while the
#' recommended value in reference [11] is 1 (with `prob_pick_pooled_gain`, for outlier detection), and the
#' recommended value in reference [9] is 10 to 20 (with `prob_pick_pooled_gain`, for missing value imputations).
#' @param categ_cols Columns that hold categorical features,
#' when the data is passed as a matrix (either dense or sparse).
#' Can be passed as an integer vector (numeration starting at 1)
#' denoting the indices of the columns that are categorical, or as a character vector denoting the
#' names of the columns that are categorical, assuming that `data` has column names.
#' 
#' Categorical columns should contain only integer values with a continuous numeration starting at \bold{zero}
#' (not at one as is typical in R packages), and with negative values and NA/NaN taken as missing.
#' The maximum categorical value should not exceed `.Machine$integer.max` (typically \eqn{2^{31}-1}{2^31-1}).
#' 
#' This is ignored when the input is passed as a `data.frame` as then it will consider columns as
#' categorical depending on their type/class (see the documentation for `data` for details).
#' @param max_depth Maximum depth of the binary trees to grow. By default, will limit it to the corresponding
#' depth of a balanced binary tree with number of terminal nodes corresponding to the sub-sample size (the reason
#' being that, if trying to detect outliers, an outlier will only be so if it turns out to be isolated with shorter average
#' depth than usual, which corresponds to a balanced tree depth).  When a terminal node has more than 1 observation,
#' the remaining isolation depth for them is estimated assuming the data and splits are both uniformly random
#' (separation depth follows a similar process with expected value calculated as in reference [6]). Default setting
#' for references [1], [2], [3], [4] is the same as the default here, but it's recommended to pass higher values if
#' using the model for purposes other than outlier detection.
#' 
#' If passing `NULL` or zero, will not limit the depth of the trees (that is, will grow them until each
#' observation is isolated or until no further split is possible).
#' 
#' Note that models that use `prob_pick_pooled_gain` or `prob_pick_avg_gain` are likely to benefit from
#' deeper trees (larger `max_depth`), but deeper trees can result in much slower model fitting and
#' predictions.
#' 
#' If using pooled gain, one might want to substitute `max_depth` with `min_gain`.
#' @param ncols_per_tree Number of columns to use (have as potential candidates for splitting at each iteration) in each tree,
#' somewhat similar to the 'mtry' parameter of random forests.
#' In general, this is only relevant when using non-random splits and/or weighting by kurtosis.
#' 
#' If passing a number between zero and one, will assume it means taking a sample size that represents
#' that proportion of the columns in the data. Note that, if passing exactly 1, will assume it means taking
#' 100\% of the columns, not taking a single columns.
#' 
#' If passing `NULL`, will use the full number of columns in the data.
#' @param prob_pick_pooled_gain his parameter indicates the probability of choosing the threshold on which to split a variable
#' (with `ndim=1`) or a linear combination of variables (when using `ndim>1`) as the threshold
#' that maximizes a pooled standard deviation gain criterion (see references [9] and [11]) on the
#' same variable or linear combination, similarly to regression trees such as CART.
#' 
#' If using `ntry>1`, will try several variables or linear combinations thereof and choose the one
#' in which the largest standardized gain can be achieved.
#' 
#' For categorical variables with `ndim=1`, will use shannon entropy instead (like in [7]).
#' 
#' Compared to a simple averaged gain, this tends to result in more evenly-divided splits and more clustered
#' groups when they are smaller. Recommended to pass higher values when used for imputation of missing values.
#' When used for outlier detection, datasets with multimodal distributions usually see better performance
#' under this type of splits.
#' 
#' Note that, since this makes the trees more even and thus it takes more steps to produce isolated nodes,
#' the resulting object will be heavier. When splits are not made according to any of `prob_pick_avg_gain`
#' or `prob_pick_pooled_gain`, both the column and the split point are decided at random. Note that, if
#' passing value 1 (100\%) with no sub-sampling and using the single-variable model,
#' every single tree will have the exact same splits.
#' 
#' Be aware that `penalize_range` can also have a large impact when using `prob_pick_pooled_gain`.
#' 
#' Be aware also that, if passing a value of 1 (100%) with no sub-sampling and using the single-variable
#' model, every single tree will have the exact same splits.
#' 
#' Under this option, models are likely to produce better results when increasing `max_depth`.
#' Alternatively, one can also control the depth through `min_gain` (for which one might want to
#' set `max_depth=NULL`).
#' 
#' Important detail: if using either `prob_pick_avg_gain` or `prob_pick_pooled_gain`, the distribution of
#' outlier scores is unlikely to be centered around 0.5.
#' @param prob_pick_avg_gain This parameter indicates the probability of choosing the threshold on which to split a variable
#' (with `ndim=1`) or a linear combination of variables (when using `ndim>1`) as the threshold
#' that maximizes an averaged standard deviation gain criterion (see references [4] and [11]) on the
#' same variable or linear combination.
#' 
#' If using `ntry>1`, will try several variables or linear combinations thereof and choose the one
#' in which the largest standardized gain can be achieved.
#' 
#' For categorical variables with `ndim=1`, will take the expected standard deviation that would be
#' gotten if the column were converted to numerical by assigning to each category a random
#' number `~ Unif(0, 1)` and calculate gain with those assumed standard deviations.
#' 
#' Compared to a pooled gain, this tends to result in more cases in which a single observation or very
#' few of them are put into one branch. Typically, datasets with outliers defined by extreme values in
#' some column more or less independently of the rest, usually see better performance under this type
#' of split. Recommended to use sub-samples (parameter `sample_size`) when
#' passing this parameter. Note that, since this will create isolated nodes faster, the resulting object
#' will be lighter (use less memory).
#' 
#' When splits are
#' not made according to any of `prob_pick_avg_gain` or `prob_pick_pooled_gain`,
#' both the column and the split point are decided at random. Default setting for [1], [2], [3] is
#' zero, and default for [4] is 1. This is the randomization parameter that can be passed to the author's original code in [5],
#' but note that the code in [5] suffers from a mathematical error in the calculation of running standard deviations,
#' so the results from it might not match with this library's.
#' 
#' Be aware that, if passing a value of 1 (100\%) with no sub-sampling and using the single-variable model, every single tree will have
#' the exact same splits.
#' 
#' Under this option, models are likely to produce better results when increasing `max_depth`.
#' 
#' Important detail: if using either `prob_pick_avg_gain` or `prob_pick_pooled_gain`, the distribution of
#' outlier scores is unlikely to be centered around 0.5.
#' @param min_gain Minimum gain that a split threshold needs to produce in order to proceed with a split. Only used when the splits
#' are decided by a gain criterion (either pooled or averaged). If the highest possible gain in the evaluated
#' splits at a node is below this  threshold, that node becomes a terminal node.
#' 
#' This can be used as a more sophisticated depth control when using pooled gain (note that `max_depth`
#' still applies on top of this heuristic).
#' @param missing_action How to handle missing data at both fitting and prediction time. Options are
#' \itemize{
#'   \item `"divide"` (for the single-variable model only, recommended), which will follow both branches and combine
#'   the result with the weight given by the fraction of the data that went to each branch when fitting the model.
#'   \item `"impute"`, which will assign observations to the branch with the most observations in the single-variable model,
#'   or fill in missing values with the median of each column of the sample from which the split was made in the extended
#'   model (recommended for it).
#'   \item `"fail"`, which will assume there are no missing values and will trigger undefined behavior if it encounters any.
#' }
#' In the extended model, infinite values will be treated as missing.
#' Passing `"fail"` will produce faster fitting and prediction
#' times along with decreased model object sizes. Models from references [1], [2], [3], [4] correspond to `"fail"` here.
#' @param new_categ_action What to do after splitting a categorical feature when new data that reaches that split has categories that
#' the sub-sample from which the split was done did not have. Options are
#' \itemize{
#'   \item `"weighted"` (for the single-variable model only, recommended), which will follow both branches and combine
#'   the result with weight given by the fraction of the data that went to each branch when fitting the model.
#'   \item `"impute"` (for the extended model only, recommended) which will assign them the median value for that column
#'   that was added to the linear combination of features.
#'   \item `"smallest"`, which in the single-variable case will assign all observations with unseen categories in the split
#'   to the branch that had fewer observations when fitting the model, and in the extended case will assign them the coefficient
#'   of the least common category.
#'   \item `"random"`, which will assing a branch (coefficient in the extended model) at random for
#'   each category beforehand, even if no observations had that category when fitting the model.
#'   Note that this can produce biased results when deciding splits by a gain criterion.
#'   
#'   Important: under this option, if the model is fitted to a `data.frame`, when calling `predict`
#'   on new data which contains new factor levels (unseen in the data to which the model was fitted),
#'   they will be added to the model's state on-the-fly. This means that, if calling `predict` on data
#'   which has new categories, there might be inconsistencies in the results if predictions are done in
#'   parallel or if passing the same data in batches or with different row orders.
#' }
#' Ignored when passing `categ_split_type` = `"single_categ"`.
#' @param categ_split_type Whether to split categorical features by assigning sub-sets of them to each branch (by passing `"subset"` there),
#' or by assigning a single category to a branch and the rest to the other branch (by passing `"single_categ"` here). For the extended model,
#' whether to give each category a coefficient (`"subset"`), or only one while the rest get zero (`"single_categ"`).
#' @param all_perm When doing categorical variable splits by pooled gain with `ndim=1` (regular model),
#' whether to consider all possible permutations of variables to assign to each branch or not. If `FALSE`,
#' will sort the categories by their frequency and make a grouping in this sorted order. Note that the
#' number of combinations evaluated (if `TRUE`) is the factorial of the number of present categories in
#' a given column (minus 2). For averaged gain, the best split is always to put the second most-frequent
#' category in a separate branch, so not evaluating all  permutations (passing `FALSE`) will make it
#' possible to select other splits that respect the sorted frequency order.
#' Ignored when not using categorical variables or not doing splits by pooled gain or using `ndim>1`.
#' @param coef_by_prop In the extended model, whether to sort the randomly-generated coefficients for categories
#' according to their relative frequency in the tree node. This might provide better results when using
#' categorical variables with too many categories, but is not recommended, and not reflective of
#' real "categorical-ness". Ignored for the regular model (`ndim=1`) and/or when not using categorical
#' variables.
#' @param recode_categ Whether to re-encode categorical variables even in case they are already passed
#' as factors. This is recommended as it will eliminate potentially redundant categorical levels if
#' they have no observations, but if the categorical variables are already of type `factor` with only
#' the levels that are present, it can be skipped for slightly faster fitting times. You'll likely
#' want to pass `FALSE` here if merging several models into one through \link{isotree.append.trees}.
#' @param weights_as_sample_prob If passing `sample_weights` argument, whether to consider those weights as row
#' sampling weights (i.e. the higher the weights, the more likely the observation will end up included
#' in each tree sub-sample), or as distribution density weights (i.e. putting a weight of two is the same
#' as if the row appeared twice, thus higher weight makes it less of an outlier). Note that sampling weight
#' is only used when sub-sampling data for each tree, which is not the default in this implementation.
#' @param sample_with_replacement Whether to sample rows with replacement or not (not recommended).
#' Note that distance calculations, if desired, don't work when there are duplicate rows.
#' @param penalize_range Whether to penalize (add -1 to the terminal depth) observations at prediction time that have a value
#' of the chosen split variable (linear combination in extended model) that falls outside of a pre-determined
#' reasonable range in the data being split (given by `2 * range` in data and centered around the split point),
#' as proposed in reference [4] and implemented in the authors' original code in reference [5]. Not used in single-variable model
#' when splitting by categorical variables.
#' 
#' It's recommended to turn this off for faster predictions on sparse CSC matrices.
#' 
#' Note that this can make a very large difference in the results when using `prob_pick_pooled_gain`.
#' 
#' Be aware that this option can make the distribution of outlier scores a bit different
#' (i.e. not centered around 0.5)
#' @param standardize_data Whether to standardize the features at each node before creating alinear combination of them as suggested
#' in [4]. This is ignored when using `ndim=1`.
#' @param weigh_by_kurtosis Whether to weigh each column according to the kurtosis obtained in the sub-sample that is selected
#' for each tree as briefly proposed in reference [1]. Note that this is only done at the beginning of each tree
#' sample, so if not using sub-samples, it's better to pass column weights calculated externally. For
#' categorical columns, will calculate expected kurtosis if the column was converted to numerical by
#' assigning to each category a random number `~ Unif(0, 1)`.
#' 
#' Note that when using sparse matrices, the calculation of kurtosis will rely on a procedure that
#' uses sums of squares and higher-power numbers, which has less numerical precision than the
#' calculation used for dense inputs, and as such, the results might differ slightly.
#' 
#' Using this option makes the model more likely to pick the columns that have anomalous values
#' when viewed as a 1-d distribution, and can bring a large improvement in some datasets.
#' @param coefs For the extended model, whether to sample random coefficients according to a normal distribution `~ N(0, 1)`
#' (as proposed in reference [4]) or according to a uniform distribution `~ Unif(-1, +1)` as proposed in reference [3].
#' Ignored for the single-variable model. Note that, for categorical variables, the coefficients will be sampled ~ N (0,1)
#' regardless - in order for both types of variables to have transformations in similar ranges (which will tend
#' to boost the importance of categorical variables), pass `"uniform"` here.
#' @param assume_full_distr When calculating pairwise distances (see reference [8]), whether to assume that the fitted model represents
#' a full population distribution (will use a standardizing criterion assuming infinite sample as in reference [6],
#' and the results of the similarity between two points at prediction time will not depend on the
#' prescence of any third point that is similar to them, but will differ more compared to the pairwise
#' distances between points from which the model was fit). If passing `FALSE`, will calculate pairwise distances
#' as if the new observations at prediction time were added to the sample to which each tree was fit, which
#' will make the distances between two points potentially vary according to other newly introduced points.
#' This will not be assumed when the distances are calculated as the model is being fit (see documentation
#' for parameter `output_dist`).
#' @param build_imputer Whether to construct missing-value imputers so that later this same model could be used to impute
#' missing values of new (or the same) observations. Be aware that this will significantly increase the memory
#' requirements and serialized object sizes. Note that this is not related to 'missing_action' as missing values
#' inside the model are treated differently and follow their own imputation or division strategy.
#' @param output_imputations Whether to output imputed missing values for `data`. Passing `TRUE` here will force
#' `build_imputer` to `TRUE`. Note that, for sparse matrix inputs, even though the output will be sparse, it will
#' generate a dense representation of each row with missing values.
#' 
#' This is not supported when using sub-sampling, and if sub-sampling is specified, will override it
#' using the full number of rows.
#' @param min_imp_obs Minimum number of observations with which an imputation value can be produced. Ignored if passing
#' `build_imputer` = `FALSE`.
#' @param depth_imp How to weight observations according to their depth when used for imputing missing values. Passing
#' `"higher"` will weigh observations higher the further down the tree (away from the root node) the
#' terminal node is, while `"lower"` will do the opposite, and `"same"` will not modify the weights according
#' to node depth in the tree. Implemented for testing purposes and not recommended to change
#' from the default. Ignored when passing `build_imputer` = `FALSE`.
#' @param weigh_imp_rows How to weight node sizes when used for imputing missing values. Passing `"inverse"` will weigh
#' a node inversely proportional to the number of observations that end up there, while `"prop"`
#' will weight them heavier the more observations there are, and `"flat"` will weigh all nodes the same
#' in this regard regardless of how many observations end up there. Implemented for testing purposes
#' and not recommended to change from the default. Ignored when passing `build_imputer` = `FALSE`.
#' @param output_score Whether to output outlierness scores for the input data, which will be calculated as
#' the model is being fit and it's thus faster. Cannot be done when using sub-samples of the data for each tree
#' (in such case will later need to call the `predict` function on the same data). If using `penalize_range`, the
#' results from this might differet a bit from those of `predict` called after.
#' 
#' This is not supported when using sub-sampling, and if sub-sampling is specified, will override it
#' using the full number of rows.
#' @param output_dist Whether to output pairwise distances for the input data, which will be calculated as
#' the model is being fit and it's thus faster. Cannot be done when using sub-samples of the data for each tree
#' (in such case will later need to call the `predict` function on the same data). If using `penalize_range`, the
#' results from this might differ a bit from those of `predict` called after.
#' 
#' This is not supported when using sub-sampling, and if sub-sampling is specified, will override it
#' using the full number of rows.
#' @param square_dist If passing `output_dist` = `TRUE`, whether to return a full square matrix or
#' just the upper-triangular part, in which the entry for pair (i,j) with 1 <= i < j <= n is located at position
#' p(i, j) = ((i - 1) * (n - i/2) + j - i).
#' @param sample_weights Sample observation weights for each row of `data`, with higher weights indicating either higher sampling
#' probability (i.e. the observation has a larger effect on the fitted model, if using sub-samples), or
#' distribution density (i.e. if the weight is two, it has the same effect of including the same data
#' point twice), according to parameter `weights_as_sample_prob`. Not supported when calculating pairwise
#' distances while the model is being fit (done by passing `output_dist` = `TRUE`).
#' 
#' If `data` is a `data.frame` and the variable passed here matches to the name of a column in `data`
#' (with or without enclosing `sample_weights` in quotes), it will assume the weights are to be
#' taken as that column name.
#' @param column_weights Sampling weights for each column in `data`. Ignored when picking columns by deterministic criterion.
#' If passing `NULL`, each column will have a uniform weight. Cannot be used when weighting by kurtosis.
#' Note that, if passing a data.frame with both numeric and categorical columns, the column names must
#' not be repeated, otherwise the column weights passed here will not end up matching. If passing a `data.frame`
#' to `data`, will assume the column order is the same as in there, regardless of whether the entries passed to
#' `column_weights` are named or not.
#' @param seed Seed that will be used for random number generation.
#' @param nthreads Number of parallel threads to use. If passing a negative number, will use
#' the maximum number of available threads in the system. Note that, the more threads,
#' the more memory will be allocated, even if the thread does not end up being used.
#' Be aware that most of the operations are bound by memory bandwidth, which means that
#' adding more threads will not result in a linear speed-up. For some types of data
#' (e.g. large sparse matrices with small sample sizes), adding more threads might result
#' in only a very modest speed up (e.g. 1.5x faster with 4x more threads),
#' even if all threads look fully utilized.
#' @return If passing `output_score` = `FALSE`, `output_dist` = `FALSE`, and `output_imputations` = `FALSE` (the defaults),
#' will output an `isolation_forest` object from which `predict` method can then be called on new data.
#' 
#' If passing `TRUE` to any of the former options, will output a list with entries:
#' \itemize{
#'   \item `model`: the `isolation_forest` object from which new predictions can be made.
#'   \item `scores`: a vector with the outlier score for each inpuit observation (if passing `output_score` = `TRUE`).
#'   \item `dist`: the distances (either a 1-d vector with the upper-triangular part or a square matrix), if
#'   passing `output_dist` = `TRUE`.
#'   \item `imputed`: the input data with missing values imputed according to the model (if passing `output_imputations` = `TRUE`).
#' }
#' @seealso \link{predict.isolation_forest},  \link{isotree.add.tree} \link{isotree.restore.handle}
#' @references \enumerate{
#' \item Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
#' \item Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation-based anomaly detection." ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
#' \item Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner. "Extended Isolation Forest." arXiv preprint arXiv:1811.02141 (2018).
#' \item Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "On detecting clustered anomalies using SCiForest." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
#' \item \url{https://sourceforge.net/projects/iforest/}
#' \item \url{https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree}
#' \item Quinlan, J. Ross. "C4. 5: programs for machine learning." Elsevier, 2014.
#' \item Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
#' \item Cortes, David. "Imputing missing values with unsupervised random trees." arXiv preprint arXiv:1911.06646 (2019).
#' \item \url{https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottom}
#' \item Cortes, David. "Revisiting randomized choices in isolation forests." arXiv preprint arXiv:2110.13402 (2021).
#' }
#' @examples
#' ### Example 1: detect an obvious outlier
#' ### (Random data from a standard normal distribution)
#' library(isotree)
#' set.seed(1)
#' m <- 100
#' n <- 2
#' X <- matrix(rnorm(m * n), nrow = m)
#' 
#' ### Will now add obvious outlier point (3, 3) to the data
#' X <- rbind(X, c(3, 3))
#' 
#' ### Fit a small isolation forest model
#' iso <- isolation.forest(X, ntrees = 10, nthreads = 1)
#' 
#' ### Check which row has the highest outlier score
#' pred <- predict(iso, X)
#' cat("Point with highest outlier score: ",
#'     X[which.max(pred), ], "\n")
#' 
#' 
#' ### Example 2: plotting outlier regions
#' ### This example shows predicted outlier score in a small
#' ### grid, with a model fit to a bi-modal distribution. As can
#' ### be seen, the extended model is able to detect high
#' ### outlierness outside of both regions, without having false
#' ### ghost regions of low-outlierness where there isn't any data
#' library(isotree)
#' oldpar <- par(mfrow = c(2, 2), mar = c(2.5,2.2,2,2.5))
#' 
#' ### Randomly-generated data from different distributions
#' set.seed(1)
#' group1 <- data.frame(x = rnorm(1000, -1, .4),
#'     y = rnorm(1000, -1, .2))
#' group2 <- data.frame(x = rnorm(1000, +1, .2),
#'     y = rnorm(1000, +1, .4))
#' X = rbind(group1, group2)
#' 
#' ### Add an obvious outlier which is within the 1d ranges
#' ### (As an interesting test, remove and see what happens,
#' ###  or check how its score changes when using sub-sampling)
#' X = rbind(X, c(-1, 1))
#' 
#' ### Produce heatmaps
#' pts = seq(-3, 3, .1)
#' space_d <- expand.grid(x = pts, y = pts)
#' plot.space <- function(Z, ttl) {
#'     image(pts, pts, matrix(Z, nrow = length(pts)),
#'       col = rev(heat.colors(50)),
#'       main = ttl, cex.main = 1.4,
#'       xlim = c(-3, 3), ylim = c(-3, 3),
#'       xlab = "", ylab = "")
#'     par(new = TRUE)
#'     plot(X, type = "p", xlim = c(-3, 3), ylim = c(-3, 3),
#'      col = "#0000801A",
#'      axes = FALSE, main = "",
#'      xlab = "", ylab = "")
#' }
#' 
#' ### Now try out different variations of the model
#' 
#' ### Single-variable model
#' iso_simple = isolation.forest(
#'     X, ndim=1,
#'     ntrees=100,
#'     nthreads=1,
#'     penalize_range=FALSE,
#'     prob_pick_pooled_gain=0,
#'     prob_pick_avg_gain=0)
#' Z1 <- predict(iso_simple, space_d)
#' plot.space(Z1, "Isolation Forest")
#' 
#' ### Extended model
#' iso_ext = isolation.forest(
#'      X, ndim=2,
#'      ntrees=100,
#'      nthreads=1,
#'      penalize_range=FALSE,
#'      prob_pick_pooled_gain=0,
#'      prob_pick_avg_gain=0)
#' Z2 <- predict(iso_ext, space_d)
#' plot.space(Z2, "Extended Isolation Forest")
#' 
#' ### SCiForest
#' iso_sci = isolation.forest(
#'      X, ndim=2,
#'      ntrees=100,
#'      nthreads=1,
#'      penalize_range=TRUE,
#'      prob_pick_pooled_gain=0,
#'      prob_pick_avg_gain=1)
#' Z3 <- predict(iso_sci, space_d)
#' plot.space(Z3, "SCiForest")
#'      
#' ### Fair-cut forest
#' iso_fcf = isolation.forest(
#'      X, ndim=2,
#'      ntrees=100,
#'      nthreads=1,
#'      penalize_range=FALSE,
#'      prob_pick_pooled_gain=1,
#'      prob_pick_avg_gain=0)
#' Z4 <- predict(iso_fcf, space_d)
#' plot.space(Z4, "Fair-Cut Forest")
#' par(oldpar)
#' 
#' ### (As another interesting variation, try setting
#' ###  'penalize_range=TRUE' for the last model)
#' 
#' ### Example 3: calculating pairwise distances,
#' ### with a short validation against euclidean dist.
#' library(isotree)
#' 
#' ### Generate random data with 3 dimensions
#' set.seed(1)
#' m <- 100
#' n <- 3
#' X <- matrix(rnorm(m * n), nrow=m, ncol=n)
#' 
#' ### Fit isolation forest model
#' iso <- isolation.forest(X, ntrees=100, nthreads=1)
#' 
#' ### Calculate distances with the model
#' D_iso <- predict(iso, X, type = "dist")
#' 
#' ### Check that it correlates with euclidean distance
#' D_euc <- dist(X, method = "euclidean")
#' 
#' cat(sprintf("Correlation with euclidean distance: %f\n",
#'     cor(D_euc, D_iso)))
#' ### (Note that euclidean distance will never take
#' ###  any correlations between variables into account,
#' ###  which the isolation forest model can do)
#' 
#' \donttest{
#' ### Example 4: imputing missing values
#' ### (requires package MASS)
#' library(isotree)
#' 
#' ### Generate random data, set some values as NA
#' if (require("MASS")) {
#'   set.seed(1)
#'   S <- crossprod(matrix(rnorm(5 * 5), nrow = 5))
#'   mu <- rnorm(5)
#'   X <- MASS::mvrnorm(1000, mu, S)
#'   X_na <- X
#'   values_NA <- matrix(runif(1000 * 5) < .15, nrow = 1000)
#'   X_na[values_NA] = NA
#'   
#'   ### Impute missing values with model
#'   iso <- isolation.forest(X_na,
#'       build_imputer = TRUE,
#'       prob_pick_pooled_gain = 1,
#'       ntry = 10,
#'       nthreads = 1)
#'   X_imputed <- predict(iso, X_na, type = "impute")
#'   cat(sprintf("MSE for imputed values w/model: %f\n",
#'       mean((X[values_NA] - X_imputed[values_NA])^2)))
#'     
#'   ### Compare against simple mean imputation
#'   X_means <- apply(X, 2, mean)
#'   X_imp_mean <- X_na
#'   for (cl in 1:5)
#'       X_imp_mean[values_NA[,cl], cl] <- X_means[cl]
#'   cat(sprintf("MSE for imputed values w/means: %f\n",
#'       mean((X[values_NA] - X_imp_mean[values_NA])^2)))
#' }
#' }
#' 
#' \donttest{
#' #### A more interesting example
#' #### (requires package outliertree)
#' 
#' ### Compare outliers returned by these different methods,
#' ### and see why some of the outliers returned by the
#' ### isolation forest could be flagged as outliers
#' if (require("outliertree")) {
#'   hypothyroid <- outliertree::hypothyroid
#'   
#'   iso <- isolation.forest(hypothyroid, nthreads=1)
#'   pred_iso <- predict(iso, hypothyroid)
#'   otree <- outliertree::outlier.tree(
#'       hypothyroid,
#'       z_outlier = 6,
#'       pct_outliers = 0.02,
#'       outliers_print = 20,
#'       nthreads = 1)
#'   
#'   ### Now compare against the top
#'   ### outliers from isolation forest
#'   head(hypothyroid[order(-pred_iso), ], 20)
#' }
#' }
#' @export
isolation.forest <- function(data,
                             sample_size = min(NROW(data), 10000L), ntrees = 500, ndim = min(3, NCOL(data)),
                             ntry = 1, categ_cols = NULL,
                             max_depth = ceiling(log2(sample_size)),
                             ncols_per_tree = NCOL(data),
                             prob_pick_pooled_gain = 0.0, prob_pick_avg_gain = 0.0,
                             min_gain = 0, missing_action = ifelse(ndim > 1, "impute", "divide"),
                             new_categ_action = ifelse(ndim > 1, "impute", "weighted"),
                             categ_split_type = "subset", all_perm = FALSE,
                             coef_by_prop = FALSE, recode_categ = FALSE,
                             weights_as_sample_prob = TRUE, sample_with_replacement = FALSE,
                             penalize_range = FALSE, standardize_data = TRUE, weigh_by_kurtosis = FALSE,
                             coefs = "normal", assume_full_distr = TRUE,
                             build_imputer = FALSE, output_imputations = FALSE, min_imp_obs = 3,
                             depth_imp = "higher", weigh_imp_rows = "inverse",
                             output_score = FALSE, output_dist = FALSE, square_dist = FALSE,
                             sample_weights = NULL, column_weights = NULL,
                             seed = 1, nthreads = parallel::detectCores()) {
    ### validate inputs
    if (NROW(data) < 3L)
        stop("Input data has too few rows.")
    if (is.null(sample_size) || output_score || output_dist || output_imputations) {
        if (!is.null(sample_size) && sample_size != NROW(data))
            warning("'sample_size' is set to the maximum when producing scores while fitting a model.")
        sample_size <- NROW(data)
    }
    if (NROW(sample_size) != 1)
        stop("'sample_size' must be a single integer or fraction.")
    if (!is.na(sample_size) && sample_size > 0 && sample_size <= 1)
        sample_size <- as.integer(ceiling(sample_size * NROW(data)))
    if (!is.na(sample_size) && sample_size < 2)
        stop("'sample_size' is too small (less than 2 rows).")
    if (is.null(ncols_per_tree))
        ncols_per_tree <- NCOL(data)
    if (NROW(ncols_per_tree) != 1)
        stop("'ncols_per_tree' must be an integer or fraction.")
    if (!is.na(ncols_per_tree) && (ncols_per_tree > 0) && (ncols_per_tree <= 1))
        ncols_per_tree <- as.integer(ceiling(ncols_per_tree * NCOL(data)))
    if (!is.na(sample_size) && sample_size > NROW(data)) {
        warning("'sample_size' is larger than the number of rows in 'data', will be decreased.")
        sample_size <- NROW(data)
    }
    if (!is.na(ncols_per_tree) && ncols_per_tree > NCOL(data)) {
        warning("'ncols_per_tree' is larger than the number of columns in 'data', will be decreased.")
        ncols_per_tree <- NCOL(data)
    }
    if (is.null(ndim))
        ndim <- NCOL(data)
    ndim <- as.integer(ndim)
    if (NROW(ndim) == 1L && !is.na(ndim) && ndim > NCOL(data)) {
        warning("'ndim' is larger than the number of columns in 'data', will be decreased.")
        ndim <- NCOL(data)
    }

    check.pos.int(ntrees,          "ntrees")
    check.pos.int(ndim,            "ndim")
    check.pos.int(ntry,            "ntry")
    check.pos.int(min_imp_obs,     "min_imp_obs")
    check.pos.int(seed,            "seed")
    check.pos.int(ncols_per_tree,  "ncols_per_tree")
    
    allowed_missing_action    <-  c("divide",       "impute",   "fail")
    allowed_new_categ_action  <-  c("weighted",     "smallest", "random", "impute")
    allowed_categ_split_type  <-  c("single_categ", "subset")
    allowed_coefs             <-  c("normal",       "uniform")
    allowed_depth_imp         <-  c("lower",        "higher",   "same")
    allowed_weigh_imp_rows    <-  c("inverse",      "prop",     "flat")

    max_depth  <-  check.max.depth(max_depth)
    
    check.str.option(missing_action,    "missing_action",    allowed_missing_action)
    check.str.option(new_categ_action,  "new_categ_action",  allowed_new_categ_action)
    check.str.option(categ_split_type,  "categ_split_type",  allowed_categ_split_type)
    check.str.option(coefs,             "coefs",             allowed_coefs)
    check.str.option(depth_imp,         "depth_imp",         allowed_depth_imp)
    check.str.option(weigh_imp_rows,    "weigh_imp_rows",    allowed_weigh_imp_rows)
    
    check.is.prob(prob_pick_avg_gain,      "prob_pick_avg_gain")
    check.is.prob(prob_pick_pooled_gain,   "prob_pick_pooled_gain")
    
    check.is.bool(all_perm,                 "all_perm")
    check.is.bool(recode_categ,             "recode_categ")
    check.is.bool(coef_by_prop,             "coef_by_prop")
    check.is.bool(weights_as_sample_prob,   "weights_as_sample_prob")
    check.is.bool(sample_with_replacement,  "sample_with_replacement")
    check.is.bool(penalize_range,           "penalize_range")
    check.is.bool(standardize_data,         "standardize_data")
    check.is.bool(weigh_by_kurtosis,        "weigh_by_kurtosis")
    check.is.bool(assume_full_distr,        "assume_full_distr")
    check.is.bool(output_score,             "output_score")
    check.is.bool(output_dist,              "output_dist")
    check.is.bool(square_dist,              "square_dist")
    check.is.bool(build_imputer,            "build_imputer")
    check.is.bool(output_imputations,       "output_imputations")
    
    s <- prob_pick_avg_gain + prob_pick_pooled_gain
    if (s > 1) {
        warning("Split type probabilities sum to more than 1, will standardize them")
        prob_pick_avg_gain      <- as.numeric(prob_pick_avg_gain)     /  s
        prob_pick_pooled_gain   <- as.numeric(prob_pick_pooled_gain)  /  s
    }
    
    if (is.null(min_gain) || NROW(min_gain) > 1 || is.na(min_gain) || min_gain < 0)
        stop("'min_gain' must be a decimal non-negative number.")

    if ((ndim == 1) && (sample_size == NROW(data)) && (prob_pick_avg_gain >= 1 || prob_pick_pooled_gain >= 1) && !sample_with_replacement) {
        warning(paste0("Passed parameters for deterministic single-variable splits ",
                       "with no sub-sampling. ",
                       "Every tree fitted will end up doing exactly the same splits. ",
                       "It's recommended to set 'prob_pick_avg_gain' < 1, 'prob_pick_pooled_gain' < 1, ",
                       "or to use the extended model (ndim > 1)."))
    }
    
    if (ndim == 1) {
        if (categ_split_type != "single_categ" && new_categ_action == "impute")
            stop("'new_categ_action' = 'impute' not supported in single-variable model.")
    } else {
        if (missing_action == "divide")
            stop("'missing_action' = 'divide' not supported in extended model.")
        if (categ_split_type != "single_categ" && new_categ_action == "weighted")
            stop("'new_categ_action' = 'weighted' not supported in extended model.")
    }
    
    nthreads <- check.nthreads(nthreads)

    categ_cols <- check.categ.cols(categ_cols, data)

    if (is.data.frame(data)) {
        subst_sample_weights <- head(as.character(substitute(sample_weights)), 1L)
        if (NROW(subst_sample_weights) && subst_sample_weights %in% names(data)) {
            sample_weights <- subst_sample_weights
        }
        if (!is.null(sample_weights) && is.character(sample_weights) && (sample_weights %in% names(data))) {
            temp <- data[[sample_weights]]
            data <- data[, setdiff(names(data), sample_weights)]
            sample_weights <- temp
            if (!NCOL(data))
                stop("'data' has no non-weight columns.")
        }
    }
    
    if (!is.null(sample_weights)) check.is.1d(sample_weights, "sample_weights")
    if (!is.null(column_weights)) check.is.1d(column_weights, "column_weights")
    
    if (!is.null(sample_weights) && (sample_size == NROW(data)) && weights_as_sample_prob)
        stop("Sampling weights are only supported when using sub-samples for each tree.")
    
    if (weigh_by_kurtosis & !is.null(column_weights))
        stop("Cannot pass column weights when weighting columns by kurtosis.")
    
    if ((output_score || output_dist || output_imputations) & (sample_size != NROW(data)))
        stop("Cannot calculate scores/distances/imputations when sub-sampling data ('sample_size').")
    
    if ((output_score || output_dist) & sample_with_replacement)
        stop("Cannot calculate scores/distances when sampling data with replacement.")
    
    if (output_dist & !is.null(sample_weights))
        stop("Sample weights not supported when calculating distances while the model is being fit.")
    
    if (output_imputations) build_imputer <- TRUE
    
    if (build_imputer && missing_action == "fail")
        stop("Cannot impute missing values when passing 'missing_action' = 'fail'.")
    
    if (output_imputations && NROW(intersect(class(data), c("dgCMatrix", "matrix.csc"))))
        warning(paste0("Imputing missing values from CSC/dgCMatrix matrix on-the-fly can be very slow, ",
                       "it's recommended if possible to fit the model first and then pass the ",
                       "same matrix as CSR/dgRMatrix to 'predict'."))
    
    if (output_imputations && !is.null(sample_weights) && !weights_as_sample_prob)
        stop(paste0("Cannot impute missing values on-the-fly when using sample weights",
                    " as distribution density. Must first fit model and then impute values."))

    if (nthreads > 1L && !R_has_openmp()) {
        msg <- paste0("Attempting to use more than 1 thread, but ",
                      "package was compiled without OpenMP support.")
        if (tolower(Sys.info()[["sysname"]]) == "darwin")
            msg <- paste0(msg, " See https://mac.r-project.org/openmp/")
        warning(msg)
    }

    if ((prob_pick_pooled_gain > 0 || prob_pick_avg_gain > 0) && ndim == 1L) {
        if (ntry > NCOL(data)) {
            warning("Passed 'ntry' larger than number of columns, will decrease it.")
            ntry <- NCOL(data)
        }
    }
    
    ### cast all parameters
    if (!is.null(sample_weights)) {
        sample_weights <- as.numeric(sample_weights)
    } else {
        sample_weights <- numeric()
    }
    if (!is.null(column_weights)) {
        column_weights <- as.numeric(column_weights)
    } else {
        column_weights <- numeric()
    }
    
    sample_size     <-  as.integer(sample_size)
    ntrees          <-  deepcopy_int(as.integer(ntrees))
    ndim            <-  as.integer(ndim)
    ntry            <-  as.integer(ntry)
    max_depth       <-  as.integer(max_depth)
    min_imp_obs     <-  as.integer(min_imp_obs)
    seed            <-  as.integer(seed)
    nthreads        <-  as.integer(nthreads)
    ncols_per_tree  <-  as.integer(ncols_per_tree)
    
    prob_pick_avg_gain       <-  as.numeric(prob_pick_avg_gain)
    prob_pick_pooled_gain    <-  as.numeric(prob_pick_pooled_gain)
    min_gain                 <-  as.numeric(min_gain)
    
    all_perm                 <-  as.logical(all_perm)
    coef_by_prop             <-  as.logical(coef_by_prop)
    weights_as_sample_prob   <-  as.logical(weights_as_sample_prob)
    sample_with_replacement  <-  as.logical(sample_with_replacement)
    penalize_range           <-  as.logical(penalize_range)
    standardize_data         <-  as.logical(standardize_data)
    weigh_by_kurtosis        <-  as.logical(weigh_by_kurtosis)
    assume_full_distr        <-  as.logical(assume_full_distr)
    
    ### split column types
    pdata <- process.data(data, sample_weights, column_weights, recode_categ, categ_cols)

    ### extra check for invalid combinations with categorical data
    if (ndim == 1L && build_imputer &&
        ((new_categ_action == "weighted" && NROW(pdata$X_cat)) || missing_action == "divide")
    ) {
        if (categ_split_type != "single_categ" && new_categ_action == "weighted") {
            stop("Cannot build imputer with 'ndim=1' + 'new_categ_action=weighted'.")
        } else if (missing_action == "divide") {
            stop("Cannot build imputer with 'ndim=1' + 'missing_action=divide'.")
        }
    }
    
    ### extra check for potential integer overflow
    if (all_perm && (ndim == 1) &&
        prob_pick_pooled_gain &&
        NROW(pdata$cat_levs)
        ) {
        max_categ <- max(sapply(pdata$cat_levs, NROW))
        if (factorial(max_categ) > 2 * .Machine$integer.max)
            stop(paste0("Number of permutations for categorical variables is larger than ",
                        "maximum representable integer. Try using 'all_perm=FALSE'."))
    }
    
    ### fit the model
    cpp_outputs <- fit_model(pdata$X_num, pdata$X_cat, unname(pdata$ncat),
                             pdata$Xc, pdata$Xc_ind, pdata$Xc_indptr,
                             pdata$sample_weights, pdata$column_weights,
                             pdata$nrows, pdata$ncols_num, pdata$ncols_cat, ndim, ntry,
                             coefs, coef_by_prop, sample_with_replacement, weights_as_sample_prob,
                             sample_size, ntrees,  max_depth, ncols_per_tree, FALSE,
                             penalize_range, standardize_data, output_dist, TRUE, square_dist,
                             output_score, TRUE, weigh_by_kurtosis,
                             prob_pick_pooled_gain, prob_pick_avg_gain, min_gain,
                             categ_split_type, new_categ_action,
                             missing_action, all_perm,
                             build_imputer, output_imputations, min_imp_obs,
                             depth_imp, weigh_imp_rows,
                             seed, nthreads)
    
    if (cpp_outputs$err)
        stop("Procedure was interrupted.")
    
    has_int_overflow = (
        !NROW(cpp_outputs$serialized_obj) ||
        (build_imputer && !is.null(cpp_outputs$imputer_ser) && !NROW(cpp_outputs$imputer_ser))
    )
    if (has_int_overflow)
        stop("Resulting model is too large for R to handle.")
    
    ### pack the outputs
    this <- list(
        params  =  list(
            sample_size = sample_size, ntrees = ntrees, ndim = ndim,
            ntry = ntry, max_depth = max_depth,
            ncols_per_tree = ncols_per_tree,
            prob_pick_avg_gain = prob_pick_avg_gain,
            prob_pick_pooled_gain = prob_pick_pooled_gain,
            min_gain = min_gain, missing_action = missing_action,
            new_categ_action = new_categ_action,
            categ_split_type = categ_split_type,
            all_perm = all_perm, coef_by_prop = coef_by_prop,
            weights_as_sample_prob = weights_as_sample_prob,
            sample_with_replacement = sample_with_replacement,
            penalize_range = penalize_range,
            standardize_data = standardize_data,
            weigh_by_kurtosis = weigh_by_kurtosis,
            coefs = coefs, assume_full_distr = assume_full_distr,
            build_imputer = build_imputer, min_imp_obs = min_imp_obs,
            depth_imp = depth_imp, weigh_imp_rows = weigh_imp_rows
        ),
        metadata  = list(
            ncols_num  =  pdata$ncols_num,
            ncols_cat  =  pdata$ncols_cat,
            cols_num   =  pdata$cols_num,
            cols_cat   =  pdata$cols_cat,
            cat_levs   =  pdata$cat_levs,
            categ_cols =  pdata$categ_cols,
            categ_max  =  pdata$categ_max
        ),
        random_seed  =  seed,
        nthreads     =  nthreads,
        cpp_obj      =  list(
            ptr         =  cpp_outputs$model_ptr,
            serialized  =  cpp_outputs$serialized_obj,
            imp_ptr     =  cpp_outputs$imputer_ptr,
            imp_ser     =  cpp_outputs$imputer_ser
        )
    )
    
    class(this) <- "isolation_forest"
    if (!output_score && !output_dist && !output_imputations) {
        return(this)
    } else {
        if (inherits(data, c("data.frame", "matrix", "dgCMatrix", "dgRMatrix"))) {
            rnames <- row.names(data)
        } else {
            rnames <- NULL
        }

        outp <- list(model    =  this,
                     scores   =  NULL,
                     dist     =  NULL,
                     imputed  =  NULL)
        if (output_score) {
            outp$scores <- cpp_outputs$depths
            if (NROW(rnames)) names(outp$scores) <- rnames
        }
        if (output_dist) {
            if (square_dist) {
                outp$dist  <-  cpp_outputs$dmat
                if (NROW(rnames)) {
                    row.names(outp$dist) <- rnames
                    colnames(outp$dist)  <- rnames
                }
            } else {
                outp$dist  <-  cpp_outputs$tmat
                attr_D <- attributes(outp$dist)
                attr_D$Size    <-  pdata$nrows
                attr_D$Diag    <-  FALSE
                attr_D$Upper   <-  FALSE
                attr_D$method  <-  "sep_dist"
                attr_D$call    <-  match.call()
                attr_D$class   <-  "dist"
                if (NROW(rnames))
                    attr_D$Labels <- as.character(rnames)
                attributes(outp$dist) <- attr_D
            }
        }
        if (output_imputations) {
            outp$imputed   <-  reconstruct.from.imp(cpp_outputs$imputed_num,
                                                    cpp_outputs$imputed_cat,
                                                    data, this, pdata)
        }
        return(outp)
    }
}

#' @title Predict method for Isolation Forest
#' @param object An Isolation Forest object as returned by \link{isolation.forest}.
#' @param newdata A `data.frame`, `data.table`, `tibble`, `matrix`, or sparse matrix (from package `Matrix` or `SparseM`,
#' CSC/dgCMatrix supported for distance and outlierness, CSR/dgRMatrix supported for outlierness and imputations)
#' for which to predict outlierness, distance, or imputations of missing values.
#' 
#' If `newdata` is sparse and one wants to obtain the outlier score or average depth or tree
#' numbers, it's highly recommended to pass it in CSC (`dgCMatrix`) format as it will be much faster
#' when the number of trees or rows is large.
#' @param type Type of prediction to output. Options are:
#' \itemize{
#'   \item `"score"` for the standardized outlier score, where values closer to 1 indicate more outlierness, while values
#'   closer to 0.5 indicate average outlierness, and close to 0 more averageness (harder to isolate).
#'   \item `"avg_depth"` for  the non-standardized average isolation depth.
#'   \item `"dist"` for approximate pairwise or between-points distances (must pass more than 1 row) - these are
#'   standardized in the same way as outlierness, values closer to zero indicate nearer points,
#'   closer to one further away points, and closer to 0.5 average distance.
#'   \item `"avg_sep"` for the non-standardized average separation depth.
#'   \item `"tree_num"` for the terminal node number for each tree - if choosing this option,
#'   will return a list containing both the average isolation depth and the terminal node numbers, under entries
#'   `avg_depth` and `tree_num`, respectively.
#'   \item `"tree_depths"` for the non-standardized isolation depth or expected isolation depth for each tree
#'   (note that they will not include range penalties from `penalize_range=TRUE`).
#'   \item `"impute"` for imputation of missing values in `newdata`.
#' }
#' @param square_mat When passing `type` = `"dist` or `"avg_sep"` with no `refdata`, whether to return a
#' full square matrix (returned as a numeric `matrix` object) or
#' just its upper-triangular part (returned as a `dist` object and compatible with functions such as `hclust`),
#' in which the entry for pair (i,j) with 1 <= i < j <= n is located at position
#' p(i, j) = ((i - 1) * (n - i/2) + j - i).
#' Ignored when not predicting distance/separation or when passing `refdata`.
#' @param refdata If passing this and calculating distance or average separation depth, will calculate distances
#' between each point in `newdata` and each point in `refdata`, outputing a matrix in which points in `newdata`
#' correspond to rows and points in `refdata` correspond to columns. Must be of the same type as `newdata` (e.g.
#' `data.frame`, `matrix`, `dgCMatrix`, etc.). If this is not passed, and type is `"dist"`
#' or `"avg_sep"`, will calculate pairwise distances/separation between the points in `newdata`.
#' @param ... Not used.
#' @return The requested prediction type, which can be: \itemize{
#' \item A numeric vector with one entry per row in `newdata` (for output types `"score"`, `"avg_depth"`).
#' \item A list with entries `avg_depth` (numeric vector)
#' and `tree_num` (integer matrix indicating the terminal node number under each tree for each
#' observation, with trees as columns), for output type `"tree_num"`.
#' \item A numeric matrix with rows matching to those in `newdata` and one column per tree in the
#' model, for output type `"tree_depths"`.
#' \item A numeric square matrix or `dist` object containing a vector with the upper triangular
#' part of a square matrix
#' (for output types `"dist"`, `"avg_sep"`, with no `refdata`).
#' \item A numeric matrix with points in `newdata` as rows and points in `refdata` as columns
#' (for output types `"dist"`, `"avg_sep"`, with `refdata`).
#' \item The same type as the input `newdata` (for output type `"impute"`).}
#' @section Model serving considerations:
#' If the model was built with `nthreads>1`, this prediction function will
#' use OpenMP for parallelization. In a linux setup, one usually has GNU's "gomp" as OpenMP as backend, which
#' will hang when used in a forked process - for example, if one tries to call this prediction function from
#' `RestRserve`, which uses process forking for parallelization, it will cause the whole application to freeze;
#' and if using kubernetes on top of a different backend such as plumber, might cause it to run slower than
#' needed or to hang too. A potential fix in these cases is to set the number of threads to 1 in the object
#' (e.g. `model$nthreads <- 1L`), or to use a different version of this library compiled without OpenMP
#' (requires manually altering the `Makevars` file), or to use a non-GNU OpenMP backend. This should not
#' be an issue when using this library normally in e.g. an RStudio session.
#' 
#' In order to make model objects serializable (i.e. usable with `save`, `saveRDS`, and similar), these model
#' objects keep serialized raw bytes from which their underlying heap-allocated C++ object (which does not
#' survive serializations) can be reconstructed. For model serving, one would usually want to drop these
#' serialized bytes after having loaded a model through `readRDS` or similar (note that reconstructing the
#' C++ object will first require calling \link{isotree.restore.handle}, which is done automatically when
#' calling `predict` and similar), as they can increase memory usage by a large amount. These redundant raw bytes
#' can be dropped as follows: `model$cpp_obj$serialized <- NULL` (and an additional
#' `model$cpp_obj$imp_ser <- NULL` when using `build_imputer=TRUE`). After that, one might want to force garbage
#' collection through `gc()`.
#' @details The standardized outlier score is calculated according to the original paper's formula:
#' \eqn{  2^{ - \frac{\bar{d}}{c(n)}  }  }{2^(-avg(depth)/c(nobs))}, where
#' \eqn{\bar{d}}{avg(depth)} is the average depth under each tree at which an observation
#' becomes isolated (a remainder is extrapolated if the actual terminal node is not isolated),
#' and \eqn{c(n)}{c(nobs)} is the expected isolation depth if observations were uniformly random
#' (see references under \link{isolation.forest} for details). The actual calculation
#' of \eqn{c(n)}{c(nobs)} however differs from the paper as this package uses more exact procedures
#' for calculation of harmonic numbers.
#' 
#' The distribution of outlier scores should be centered around 0.5, unless using non-random splits (parameters
#' `prob_pick_avg_gain`, `prob_pick_pooled_gain`)
#' and/or range penalizations, or having distributions which are too skewed.
#' 
#' The more threads that are set for the model, the higher the memory requirement will be as each
#' thread will allocate an array with one entry per row (outlierness) or combination (distance).
#' 
#' Outlierness predictions for sparse data will be much slower than for dense data. Not recommended to pass
#' sparse matrices unless they are too big to fit in memory.
#' 
#' Note that after loading a serialized object from `isolation.forest` through `readRDS` or `load`,
#' it will only de-serialize the underlying C++ object upon running `predict`, `print`, or `summary`, so the
#' first run will  be slower, while subsequent runs will be faster as the C++ object will already be in-memory.
#' 
#' In order to save memory when fitting and serializing models, the functionality for outputting
#' terminal node numbers will generate index mappings on the fly for all tree nodes, even if passing only
#' 1 row, so it's only recommended for batch predictions.
#' 
#' The outlier scores/depth predict functionality is optimized for making predictions on one or a
#' few rows at a time - for making large batches of predictions, it might be faster to use the
#' option `output_score=TRUE` in `isolation.forest`.
#' 
#' When making predictions on CSC matrices with many rows using multiple threads, there
#' can be small differences between runs due to roundoff error.
#' 
#' When imputing missing values, the input may contain new columns (i.e. not present when the model was fitted),
#' which will be output as-is.
#' @seealso \link{isolation.forest} \link{isotree.restore.handle}
#' @export predict.isolation_forest
#' @export
predict.isolation_forest <- function(object, newdata, type="score", square_mat=FALSE, refdata=NULL, ...) {
    isotree.restore.handle(object)
    
    allowed_type <- c("score", "avg_depth", "dist", "avg_sep", "tree_num", "tree_depths", "impute")
    check.str.option(type, "type", allowed_type)
    check.is.bool(square_mat)
    if (!NROW(newdata)) stop("'newdata' must be a data.frame, matrix, or sparse matrix.")
    if ((object$metadata$ncols_cat > 0 && is.null(object$metadata$categ_cols)) && NROW(intersect(class(newdata), get.types.spmat(TRUE, TRUE, TRUE)))) {
        stop("Cannot pass sparse inputs if the model was fit to categorical variables in a data.frame.")
    }
    if ((type %in% c("tree_num", "tree_depths")) && (object$params$ndim == 1L)) {
        if ((object$metadata$ncols_cat > 0) &&
            (object$params$categ_split_type != "single_categ") &&
            (object$params$new_categ_action == "weighted")
        ) {
            stop("Cannot output tree numbers/depths when using 'new_categ_action' = 'weighted'.")
        }
        if (object$params$missing_action == "divide")
            stop("Cannot output tree numbers/depths when using 'missing_action' = 'divide'.")
    }
    if (inherits(newdata, "numeric") && is.null(dim(newdata))) {
        newdata <- matrix(newdata, nrow=1)
    }
    
    if (type %in% "impute" && (is.null(object$params$build_imputer) || !(object$params$build_imputer)))
        stop("Cannot impute missing values with model that was built with 'build_imputer=FALSE'.")
    
    if (is.null(refdata) || !(type %in% c("dist", "avg_sep"))) {
        nobs_group1 <- 0L
    } else {
        nobs_group1 <- NROW(newdata)
        newdata     <- rbind(newdata, refdata)
    }
    
    pdata <- process.data.new(newdata, object$metadata,
                              !(type %in% c("dist", "avg_sep")),
                              type != "impute", type == "impute",
                              ((object$params$new_categ_action  == "impute" &&
                                object$params$missing_action == "impute")
                                ||
                               (object$params$new_categ_action == "weighted" &&
                                object$params$categ_split_type != "single_categ" &&
                                object$params$missing_action == "divide")))

    if (object$params$new_categ_action == "random" && NROW(pdata$X_cat) &&
        NROW(object$metadata$cat_levs) && NROW(pdata$cat_levs)
    ) {
        set.list.elt(object$metadata, "cat_levs", pdata$cat_levs)
    }

    square_mat   <-  as.logical(square_mat)
    score_array  <-  numeric()
    dist_tmat    <-  numeric()
    dist_dmat    <-  matrix(numeric(), nrow=0, ncol=0)
    dist_rmat    <-  matrix(numeric(), nrow=0, ncol=0)
    tree_num     <-  get_null_int_mat()
    tree_depths  <-  matrix(numeric(), nrow=0, ncol=0)

    if (inherits(newdata, c("data.frame", "matrix", "dgCMatrix", "dgRMatrix"))) {
        rnames <- row.names(newdata)
    } else {
        rnames <- NULL
    }
    
    if (type %in% c("dist", "avg_sep")) {
        if (NROW(newdata) == 1L) stop("Need more than 1 data point for distance predictions.")
        if (is.null(refdata)) {
            dist_tmat <- get_empty_tmat(pdata$nrows)
            if (square_mat) {
                dist_dmat <- matrix(0, nrow=pdata$nrows, ncol=pdata$nrows)
                if (NROW(rnames)) {
                    row.names(dist_dmat) <- rnames
                    colnames(dist_dmat)  <- rnames
                }
            }
        } else {
            dist_rmat <- matrix(0, nrow=pdata$nrows-nobs_group1, ncol=nobs_group1)
            if (NROW(rnames)) {
                colnames(dist_rmat)   <-  rnames[1:nobs_group1]
                row.names(dist_rmat)  <-  rnames[seq(nobs_group1+1L, NROW(rnames))]
            }
        }
    } else {
        score_array <- numeric(pdata$nrows)
        if (NROW(rnames)) names(score_array) <- rnames

        if (type == "tree_num") {
            tree_num <- get_empty_int_mat(pdata$nrows, get_ntrees(object$cpp_obj$ptr, object$params$ndim > 1))
            if (NROW(rnames)) row.names(tree_num) <- rnames
        } else if (type == "tree_depths") {
            tree_depths <- matrix(0., ncol=pdata$nrows, nrow=get_ntrees(object$cpp_obj$ptr, object$params$ndim > 1))
            if (NROW(rnames)) colnames(tree_depths) <- rnames
        }
    }
    
    if (type %in% c("score", "avg_depth", "tree_num", "tree_depths")) {
        predict_iso(object$cpp_obj$ptr, object$params$ndim > 1,
                    score_array, tree_num, tree_depths,
                    pdata$X_num, pdata$X_cat,
                    pdata$Xc, pdata$Xc_ind, pdata$Xc_indptr,
                    pdata$Xr, pdata$Xr_ind, pdata$Xr_indptr,
                    pdata$nrows, object$nthreads, type == "score")
        if (type == "tree_num") {
            return(list(avg_depth = score_array, tree_num = tree_num+1L))
        } else if (type == "tree_depths") {
            return(t(tree_depths))
        } else {
            return(score_array)
        }
    } else if (type %in% c("dist", "avg_sep")) {
        dist_iso(object$cpp_obj$ptr, dist_tmat, dist_dmat, dist_rmat,
                 object$params$ndim > 1L,
                 pdata$X_num, pdata$X_cat,
                 pdata$Xc, pdata$Xc_ind, pdata$Xc_indptr,
                 pdata$nrows, object$nthreads, object$params$assume_full_distr,
                 type == "dist", square_mat, nobs_group1)
        if (!is.null(refdata))
            return(t(dist_rmat))
        else if (square_mat)
            return(dist_dmat)
        else {
            attr_D <- attributes(dist_tmat)
            attr_D$Size    <-  pdata$nrows
            attr_D$Diag    <-  FALSE
            attr_D$Upper   <-  FALSE
            attr_D$method  <-  ifelse(type == "dist", "sep_dist", "avg_sep")
            attr_D$call    <-  match.call()
            attr_D$class   <-  "dist"
            if (NROW(rnames))
                attr_D$Labels <- as.character(rnames)
            attributes(dist_tmat) <- attr_D
            return(dist_tmat)
        }
    } else if (type == "impute") {
        imp <- impute_iso(object$cpp_obj$ptr, object$cpp_obj$imp_ptr, object$params$ndim > 1,
                          pdata$X_num, pdata$X_cat,
                          pdata$Xr, pdata$Xr_ind, pdata$Xr_indptr,
                          pdata$nrows, object$nthreads)
        return(reconstruct.from.imp(imp$X_num,
                                    imp$X_cat,
                                    newdata, object,
                                    pdata))
    } else {
        stop("Unexpected error. Please open an issue in GitHub explaining what you were doing.")
    }
}


#' @title Print summary information from Isolation Forest model
#' @description Displays the most general characteristics of an isolation forest model (same as `summary`).
#' @param x An Isolation Forest model as produced by function `isolation.forest`.
#' @param ... Not used.
#' @details Note that after loading a serialized object from `isolation.forest` through `readRDS` or `load`,
#' it will only de-serialize the underlying C++ object upon running `predict`, `print`, or `summary`,
#' so the first run will be slower, while subsequent runs will be faster as the C++ object will already be in-memory.
#' @return The same model that was passed as input.
#' @seealso \link{isolation.forest}
#' @export print.isolation_forest
#' @export
print.isolation_forest <- function(x, ...) {
    isotree.restore.handle(x)
    
    if (x$params$ndim > 1) cat("Extended ")
    cat("Isolation Forest model")
    if (x$params$prob_pick_avg_gain + x$params$prob_pick_pooled_gain > 0) {
        cat(" (using guided splits)")
    }
    cat("\n")
    if (x$params$ndim > 1) cat(sprintf("Splitting by %d variables at a time\n", x$params$ndim))
    cat(sprintf("Consisting of %d trees\n", x$params$ntrees))
    if (x$metadata$ncols_num  > 0)  cat(sprintf("Numeric columns: %d\n",     x$metadata$ncols_num))
    if (x$metadata$ncols_cat  > 0)  cat(sprintf("Categorical columns: %d\n", x$metadata$ncols_cat))
    if (NROW(x$cpp_obj$serialized)) {
        bytes <- length(x$cpp_obj$serialized) + NROW(x$cpp_obj$imp_ser)
        if (bytes > 1024^3) {
            cat(sprintf("Size: %.2f GiB\n", bytes/1024^3))
        } else if (bytes > 1024^2) {
            cat(sprintf("Size: %.2f MiB\n", bytes/1024^2))
        } else {
            cat(sprintf("Size: %.2f KiB\n", bytes/1024))
        }
    }
    return(invisible(x))
}


#' @title Print summary information from Isolation Forest model
#' @description Displays the most general characteristics of an isolation forest model (same as `print`).
#' @param object An Isolation Forest model as produced by function `isolation.forest`.
#' @param ... Not used.
#' @details Note that after loading a serialized object from `isolation.forest` through `readRDS` or `load`,
#' it will only de-serialize the underlying C++ object upon running `predict`, `print`, or `summary`,
#' so the first run will be slower, while subsequent runs will be faster as the C++ object will already be in-memory.
#' @return No return value.
#' @seealso \link{isolation.forest}
#' @export summary.isolation_forest
#' @export
summary.isolation_forest <- function(object, ...) {
    print.isolation_forest(object)
}

#' @title Add additional (single) tree to isolation forest model
#' @description Adds a single tree fit to the full (non-subsampled) data passed here. Must
#' have the same columns as previously-fitted data. Categorical columns, if any,
#' may have new categories.
#' @param model An Isolation Forest object as returned by \link{isolation.forest}, to which an additional tree will be added.
#' 
#' \bold{This object will be modified in-place}.
#' @param data A `data.frame`, `data.table`, `tibble`, `matrix`, or sparse matrix (from package `Matrix` or `SparseM`, CSC format)
#' to which to fit the new tree.
#' @param sample_weights Sample observation weights for each row of 'X', with higher weights indicating
#' distribution density (i.e. if the weight is two, it has the same effect of including the same data
#' point twice). If not `NULL`, model must have been built with `weights_as_sample_prob` = `FALSE`.
#' @param column_weights Sampling weights for each column in `data`. Ignored when picking columns by deterministic criterion.
#' If passing `NULL`, each column will have a uniform weight. Cannot be used when weighting by kurtosis.
#' @return The same `model` object now modified, as invisible.
#' @details If constructing trees with different sample sizes, the outlier scores will not be centered around
#' 0.5 and might have a very skewed distribution. The standardizing constant for the scores will be
#' taken according to the sample size passed in the model construction argument.
#' 
#' Be aware that, if an out-of-memory error occurs, the resulting object might be rendered unusable
#' (might crash when calling certain functions).
#' 
#' For safety purposes, the model object can be deep copied (including the underlying C++ object)
#' through function \link{isotree.deep.copy} before undergoing an in-place modification like this.
#' @seealso \link{isolation.forest} \link{isotree.restore.handle}
#' @export
isotree.add.tree <- function(model, data, sample_weights = NULL, column_weights = NULL) {
    
    if (!is.null(sample_weights) && model$weights_as_sample_prob)
        stop("Cannot use sampling weights with 'partial_fit'.")
    if (!is.null(column_weights) && model$weigh_by_kurtosis)
        stop("Cannot pass column weights when weighting columns by kurtosis.")
    if (typeof(model$params$ntrees) != "integer")
        stop("'model' has invalid structure.")
    if (is.na(model$params$ntrees) || model$params$ntrees >= .Machine$integer.max)
        stop("Resulting object would exceed number of trees limit.")
    
    isotree.restore.handle(model)
    
    
    if (!is.null(sample_weights))
        sample_weights  <- as.numeric(sample_weights)
    else
        sample_weights  <- numeric()
    if (!is.null(column_weights))
        column_weights  <- as.numeric(column_weights)
    else
        column_weights  <- numeric()
    if (NROW(sample_weights) && NROW(sample_weights) != NROW(data))
        stop(sprintf("'sample_weights' has different number of rows than data (%d vs. %d).",
                     NROW(data), NROW(sample_weights)))
    if (NROW(column_weights)  && NCOL(data) != NROW(column_weights))
        stop(sprintf("'column_weights' has different dimension than number of columns in data (%d vs. %d).",
                     NCOL(data), NROW(column_weights)))
    
    pdata <- process.data.new(data, model$metadata, FALSE)

    if (NROW(pdata$X_cat) && NROW(model$metadata$cat_levs) && NROW(pdata$cat_levs)) {
        set.list.elt(model$metadata, "cat_levs", pdata$cat_levs)
    }

    if (model$metadata$ncols_cat)
        ncat  <-  sapply(model$metadata$cat_levs, NROW)
    else
        ncat  <-  integer()

    if (NROW(pdata$X_cat) && !NROW(ncat)) {
        ncat <- apply(matrix(pdata$X_cat, nrow=pdata$nrows), 2, max)
        ncat <- pmax(ncat, integer(length(ncat)))
        ncat <- as.integer(ncat)
    }

    serialized <- model$cpp_obj$serialized
    if (!NROW(serialized))
    
    serialized <- raw()
    imp_ser <- raw()
    if (NROW(model$cpp_obj$serialized))
        serialized <- model$cpp_obj$serialized
    if (NROW(model$cpp_obj$imp_ser))
        imp_ser    <- model$cpp_obj$imp_ser

    fit_tree(model$cpp_obj$ptr, serialized, imp_ser,
             pdata$X_num, pdata$X_cat, unname(ncat),
             pdata$Xc, pdata$Xc_ind, pdata$Xc_indptr,
             sample_weights, column_weights,
             pdata$nrows, model$metadata$ncols_num, model$metadata$ncols_cat,
             model$params$ndim, model$params$ntry,
             model$params$coefs, model$params$coef_by_prop,
             model$params$max_depth, model$params$ncols_per_tree,
             FALSE, model$params$penalize_range, model$params$standardize_data,
             model$params$weigh_by_kurtosis,
             model$params$prob_pick_pooled_gain, model$params$prob_pick_avg_gain,
             model$params$min_gain,
             model$params$categ_split_type, model$params$new_categ_action,
             model$params$missing_action, model$params$build_imputer,
             model$params$min_imp_obs, model$cpp_obj$imp_ptr,
             model$params$depth_imp, model$params$weigh_imp_rows,
             model$params$all_perm, model$random_seed + (model$params$ntrees-1L),
             model$cpp_obj, model$params)
    
    return(invisible(model))
}

#' @title Unpack isolation forest model after de-serializing
#' @description After persisting an isolation forest model object through `saveRDS`, `save`, or restarting a session, the
#' underlying C++ objects that constitute the isolation forest model and which live only on the C++ heap memory are not saved along,
#' thus not restored after loading a saved model through `readRDS` or `load`.
#' 
#' The model object however keeps serialized versions of the C++ objects as raw bytes, from which the C++ objects can be
#' reconstructed, and are done so automatically after calling `predict`, `print`, `summary`, or `isotree.add.tree` on the
#' freshly-loaded object from `readRDS` or `load`.
#' 
#' This function allows to automatically de-serialize the object ("complete" or "restore" the
#' handle) without having to call any function that would do extra processing.
#' It is an equivalent to XGBoost's `xgb.Booster.complete` and CatBoost's
#' `catboost.restore_handle` functions.
#' @details If using this function to de-serialize a model in a production system, one might
#' want to delete the serialized bytes inside the object afterwards in order to free up memory.
#' These are under `model$cpp_obj$serialized` (plus `model$cpp_obj$imp_ser` if building with imputer)
#' - e.g.: `model$cpp_obj$serialized = NULL; model$cpp_obj$imp_ser = NULL; gc()`.
#' @param model An Isolation Forest object as returned by `isolation.forest`, which has been just loaded from a disk
#' file through `readRDS`, `load`, or a session restart.
#' @return The same model object that was passed as input. Object is modified in-place
#' however, so it does not need to be re-assigned.
#' @examples 
#' ### Warning: this example will generate a temporary .Rds
#' ### file in your temp folder, and will then delete it
#' library(isotree)
#' set.seed(1)
#' X <- matrix(rnorm(100), nrow = 20)
#' iso <- isolation.forest(X, ntrees=10, nthreads=1)
#' temp_file <- file.path(tempdir(), "iso.Rds")
#' saveRDS(iso, temp_file)
#' iso2 <- readRDS(temp_file)
#' file.remove(temp_file)
#' 
#' cat("Model pointer after loading is this: \n")
#' print(iso2$cpp_obj$ptr)
#' 
#' ### now unpack it
#' isotree.restore.handle(iso2)
#' 
#' cat("Model pointer after unpacking is this: \n")
#' print(iso2$cpp_obj$ptr)
#' @export
isotree.restore.handle <- function(model)  {
    if (!inherits(model, "isolation_forest"))
        stop("'model' must be an isolation forest model object as output by function 'isolation.forest'.")
    
    if (check_null_ptr_model(model$cpp_obj$ptr)) {
        if (!NROW(model$cpp_obj$serialized))
            stop("'model' is missing serialized raw bytes. Cannot restore handle.")

        if (model$params$ndim == 1)
            set.list.elt(model$cpp_obj, "ptr", deserialize_IsoForest(model$cpp_obj$serialized))
        else
            set.list.elt(model$cpp_obj, "ptr", deserialize_ExtIsoForest(model$cpp_obj$serialized))
    }

    if (model$params$build_imputer && check_null_ptr_model(model$cpp_obj$imp_ptr)) {
        if (!NROW(model$cpp_obj$imp_ser))
            stop("'model' is missing serialized raw bytes. Cannot restore handle.")

        set.list.elt(model$cpp_obj, "imp_ptr", deserialize_Imputer(model$cpp_obj$imp_ser))
    }
    
    return(invisible(model))
}

#' @title Get Number of Nodes per Tree
#' @param model An Isolation Forest model as produced by function `isolation.forest`.
#' @return A list with entries `"total"` and `"terminal"`, both of which are integer vectors
#' with length equal to the number of trees. `"total"` contains the total number of nodes that
#' each tree has, while `"terminal"` contains the number of terminal nodes per tree.
#' @export
isotree.get.num.nodes <- function(model)  {
    isotree.restore.handle(model)
    return(get_n_nodes(model$cpp_obj$ptr, model$params$ndim > 1, model$nthreads))
}

#' @title Append isolation trees from one model into another
#' @description This function is intended for merging models \bold{that use the same hyperparameters} but
#' were fitted to different subsets of data.
#' 
#' In order for this to work, both models must have been fit to data in the same format - 
#' that is, same number of columns, same order of the columns, and same column types, although
#' not necessarily same object classes (e.g. can mix `base::matrix` and `Matrix::dgCMatrix`).
#' 
#' If the data has categorical variables, the models should have been built with parameter
#' `recode_categ=FALSE` in the call to \link{isolation.forest},
#' and the categorical columns passed as type `factor` with the same `levels` -
#' otherwise different models might be using different encodings for each categorical column,
#' which will not be preserved as only the trees will be appended without any associated metadata.
#' 
#' Note that this function will not perform any checks on the inputs, and passing two incompatible
#' models (e.g. fit to different numbers of columns) will result in wrong results and
#' potentially crashing the R process when using the resulting object.
#' 
#' Also be aware that the first input will be modified in-place.
#' @param model An Isolation Forest model (as returned by function \link{isolation.forest})
#' to which trees from `other` (another Isolation Forest model) will be appended into.
#' 
#' \bold{Will be modified in-place}, and on exit will contain the resulting merged model.
#' @param other Another Isolation Forest model, from which trees will be appended into
#' `model`. It will not be modified during the call to this function.
#' @return The same input `model` object, now with the new trees appended, returned as invisible.
#' @details Be aware that, if an out-of-memory error occurs, the resulting object might be rendered unusable
#' (might crash when calling certain functions).
#' 
#' For safety purposes, the model object can be deep copied (including the underlying C++ object)
#' through function \link{isotree.deep.copy} before undergoing an in-place modification like this.
#' @examples 
#' library(isotree)
#' 
#' ### Generate two random sets of data
#' m <- 100
#' n <- 2
#' set.seed(1)
#' X1 <- matrix(rnorm(m*n), nrow=m)
#' X2 <- matrix(rnorm(m*n), nrow=m)
#' 
#' ### Fit a model to each dataset
#' iso1 <- isolation.forest(X1, ntrees=3, nthreads=1)
#' iso2 <- isolation.forest(X2, ntrees=2, nthreads=1)
#' 
#' ### Check the terminal nodes for some observations
#' nodes1 <- predict(iso1, head(X1, 3), type="tree_num")
#' nodes2 <- predict(iso2, head(X1, 3), type="tree_num")
#' 
#' ### Append the trees from 'iso2' into 'iso1'
#' iso1 <- isotree.append.trees(iso1, iso2)
#' 
#' ### Check that it predicts the same as the two models
#' nodes.comb <- predict(iso1, head(X1, 3), type="tree_num")
#' nodes.comb$tree_num == cbind(nodes1$tree_num, nodes2$tree_num)
#' 
#' ### The new predicted scores will be a weighted average
#' ### (Be aware that, due to round-off, it will not match with '==')
#' nodes.comb$avg_depth
#' (3*nodes1$avg_depth + 2*nodes2$avg_depth) / 5
#' @export
isotree.append.trees <- function(model, other) {
    if (!inherits(model, "isolation_forest") || !inherits(other, "isolation_forest")) {
        stop("'model' and 'other' must be isolation forest models.")
    }
    if ((model$params$ndim == 1) != (other$params$ndim == 1)) {
        stop("Cannot mix extended and regular isolation forest models (ndim=1).")
    }
    if (model$metadata$ncols_cat) {
        warning("Merging models with categorical features might give wrong results.")
    }
    if ((typeof(model$params$ntrees) != "integer") || (typeof(other$params$ntree) != "integer"))
        stop("One of the objects has invalid structure.")
    if (is.na(model$params$ntrees + other$params$ntrees) || (model$params$ntrees + other$params$ntrees) > .Machine$integer.max)
        stop("Resulting object would exceed number of trees limit.")
    
    serialized <- raw()
    imp_ser <- raw()
    if (NROW(model$cpp_obj$serialized))
        serialized <- model$cpp_obj$serialized
    if (NROW(model$cpp_obj$imp_ser))
        imp_ser    <- model$cpp_obj$imp_ser

    append_trees_from_other(model$cpp_obj$ptr,      other$cpp_obj$ptr,
                            model$cpp_obj$imp_ptr,  other$cpp_obj$imp_ptr,
                            model$params$ndim > 1,
                            serialized, imp_ser,
                            model$cpp_obj, model$params)
    return(invisible(model))
}

#' @title Export Isolation Forest model
#' @description Save Isolation Forest model to a serialized file along with its
#' metadata, in order to be used in the Python or the C++ versions of this package.
#' 
#' This function is not suggested to be used for passing models to and from R -
#' in such case, one can use `saveRDS` and `readRDS` instead, although the function
#' still works correctly for serializing objects between R sessions.
#' 
#' Note that, if the model was fitted to a `data.frame`, the column names must be
#' something exportable as JSON, and must be something that Python's Pandas could
#' use as column names (e.g. strings/character).
#' 
#' Can optionally generate a JSON file with metadata such as the column names and the
#' levels of categorical variables, which can be inspected visually in order to detect
#' potential issues (e.g. character encoding) or to make sure that the columns are of
#' the right types.
#' 
#' Requires the `jsonlite` package in order to work.
#' @details The metadata file, if produced, will contain, among other things, the encoding that was used for
#' categorical columns - this is under `data_info.cat_levels`, as an array of arrays by column,
#' with the first entry for each column corresponding to category 0, second to category 1,
#' and so on (the C++ version takes them as integers). When passing `categ_cols`, there
#' will be no encoding but it will save the maximum category integer and the column
#' numbers instead of names.
#' 
#' The serialized file can be used in the C++ version by reading it as a binary file
#' and de-serializing its contents using the C++ function 'deserialize_combined'
#' (recommended to use 'inspect_serialized_object' beforehand).
#' 
#' Be aware that this function will write raw bytes from memory as-is without compression,
#' so the file sizes can end up being much larger than when using `saveRDS`.
#' 
#' The metadata is not used in the C++ version, but is necessary for the R and Python versions.
#' 
#' Note that the model treats boolean/logical variables as categorical. Thus, if the model was fit
#' to a `data.frame` with boolean columns, when importing this model into C++, they need to be
#' encoded in the same order - e.g. the model might encode `TRUE` as zero and `FALSE`
#' as one - you need to look at the metadata for this.
#' 
#' The files produced by this function will be compatible between:\itemize{
#' \item Different operating systems.
#' \item Different compilers.
#' \item Different Python/R versions.
#' \item Systems with different 'size_t' width (e.g. 32-bit and 64-bit),
#' as long as the file was produced on a system that was either 32-bit or 64-bit,
#' and as long as each saved value fits within the range of the machine's 'size_t' type.
#' \item Systems with different 'int' width,
#' as long as the file was produced on a system that was 16-bit, 32-bit, or 64-bit,
#' and as long as each saved value fits within the range of the machine's int type.
#' \item Systems with different bit endianness (e.g. x86 and PPC64 in non-le mode).
#' \item Versions of this package from 0.3.0 onwards, \bold{but only forwards compatible}
#' (e.g. a model saved with versions 0.3.0 to 0.3.5 can be loaded under version
#' 0.3.6, but not the other way around, and attempting to do so will cause crashes
#' and memory curruptions without an informative error message). \bold{This last point applies
#' also to models saved through save, saveRDS, qsave, and similar}. Note that loading a
#' model produced by an earlier version of the library might be slightly slower.
#' }
#' 
#' But will not be compatible between:\itemize{
#' \item Systems with different floating point numeric representations
#' (e.g. standard IEEE754 vs. a base-10 system).
#' \item Versions of this package earlier than 0.3.0.
#' }
#' This pretty much guarantees that a given file can be serialized and de-serialized
#' in the same machine in which it was built, regardless of how the library was compiled.
#' 
#' Reading a serialized model that was produced in a platform with different
#' characteristics (e.g. 32-bit vs. 64-bit) will be much slower.
#' 
#' On Windows, if compiling this library with a compiler other than MSVC or MINGW,
#' (not currently supported by CRAN's build systems at the moment of writing)
#' there might be issues exporting models larger than 2GB.
#' 
#' In non-windows systems, if the file name contains non-ascii characters, the file name
#' must be in the system's native encoding. In windows, file names with non-ascii
#' characters are supported as long as the package is compiled with GCC5 or newer.
#' 
#' Note that, while `readRDS` and `load` will not make any changes to the serialized format
#' of the objects, reading a serialized model from a file will forcibly re-serialize,
#' using the system's own setup (e.g. 32-bit vs. 64-bit, endianness, etc.), and as such
#' can be used to convert formats.
#' @param model An Isolation Forest model as returned by function \link{isolation.forest}.
#' @param file File path where to save the model. File connections are not accepted, only
#' file paths
#' @param add_metadata_file Whether to generate a JSON file with metadata, which will have
#' the same name as the model but will end in '.metadata'. This file is not used by the
#' de-serialization function, it's only meant to be inspected manually, since such contents
#' will already be written in the produced model file.
#' @return The same `model` object that was passed as input, as invisible.
#' @seealso \link{isotree.import.model} \link{isotree.restore.handle}
#' @export
isotree.export.model <- function(model, file, add_metadata_file=FALSE) {
    if (!inherits(model, "isolation_forest"))
        stop("This function is only available for isolation forest objects as returned from 'isolation.forest'.")

    file <- path.expand(file)
    metadata <- export.metadata(model)

    if (add_metadata_file) {
        file.metadata <- paste0(file, ".metadata")
        jsonlite::write_json(metadata, file.metadata,
                             pretty=TRUE, auto_unbox=TRUE)
    }

    # https://github.com/jeroen/jsonlite/issues/366
    metadata <- jsonlite::toJSON(metadata, pretty=FALSE, auto_unbox=TRUE)
    metadata <- enc2utf8(metadata)
    metadata <- charToRaw(metadata)

    serialized <- raw()
    imp_ser <- raw()
    if (NROW(model$cpp_obj$serialized))
        serialized <- model$cpp_obj$serialized
    if (NROW(model$cpp_obj$imp_ser))
        imp_ser    <- model$cpp_obj$imp_ser
    serialize_to_file(
        serialized,
        imp_ser,
        model$params$ndim > 1,
        metadata,
        file
    )
    return(invisible(model))
}

#' @title Load an Isolation Forest model exported from Python
#' @description Loads a serialized Isolation Forest model as produced and exported
#' by the Python version of this package. Note that the metadata must be something
#' importable in R - e.g. column names must be valid for R (numbers are valid for
#' Python's pandas, but not for R, for example).
#' 
#' It's recommended to generate a '.metadata' file (passing `add_metada_file=TRUE`) and
#' to visually inspect said file in any case.
#' 
#' This function is not meant to be used for passing models to and from R -
#' in such case, one can use `saveRDS` and `readRDS` instead as they will
#' likely result in smaller file sizes (although this function will still
#' work correctly for serialization within R).
#' @param file Path to the saved isolation forest model.
#' Must be a file path, not a file connection,
#' and the character encoding should correspond to the system's native encoding.
#' @details If the model was fit to a `DataFrame` using Pandas' own Boolean types,
#' take a look at the metadata to check if these columns will be taken as booleans
#' (R logicals) or as categoricals with string values `"True"` and `"False"`.
#' 
#' See the documentation for \link{isotree.export.model} for details about compatibility
#' of the generated files across different machines and versions.
#' 
#' If using this function to de-serialize a model in a production system, one might
#' want to delete the serialized bytes inside the object afterwards in order to free up memory.
#' These are under `model$cpp_obj$serialized` (plus `model$cpp_obj$imp_ser` if building with imputer)
#' - e.g.: `model$cpp_obj$serialized = NULL; model$cpp_obj$imp_ser = NULL; gc()`.
#' @return An isolation forest model, as if it had been constructed through
#' \link{isolation.forest}.
#' @seealso \link{isotree.export.model} \link{isotree.restore.handle}
#' @export
isotree.import.model <- function(file) {
    if (!file.exists(file)) stop("'file' does not exist.")
    res <- deserialize_from_file(file)
    metadata <- rawToChar(res$metadata)
    Encoding(metadata) <- "UTF-8"
    metadata <- enc2native(metadata)
    metadata <- jsonlite::fromJSON(metadata,
                                   simplifyVector = TRUE,
                                   simplifyDataFrame = FALSE,
                                   simplifyMatrix = FALSE)
    this <- take.metadata(metadata)
    this$cpp_obj$ptr         <-  res$ptr
    this$cpp_obj$serialized  <-  res$serialized
    this$cpp_obj$imp_ser     <-  res$imp_ser
    this$cpp_obj$imp_ptr     <-  res$imp_ptr
    class(this) <- "isolation_forest"
    return(this)
}

#' @title Generate SQL statements from Isolation Forest model
#' @description Generate SQL statements - either separately per tree (the default),
#' for a single tree if needed (if passing `tree`), or for all trees
#' concatenated together (if passing `table_from`). Can also be made
#' to output terminal node numbers (numeration starting at one).
#' 
#' Some important considerations:\itemize{
#' \item Making predictions through SQL is much less efficient than from the model
#' itself, as each terminal node will have to check all of the conditions
#' that lead to it instead of passing observations down a tree.
#' \item If constructed with the default arguments, the model will not perform any
#' sub-sampling, which can lead to very big trees. If it was fit to a large
#' dataset, the generated SQL might consist of gigabytes of text, and might
#' lay well beyond the character limit of commands accepted by SQL vendors.
#' \item The generated SQL statements will not include range penalizations, thus
#' predictions might differ from calls to `predict` when using
#' `penalize_range=TRUE`.
#' \item The generated SQL statements will only include handling of missing values
#' when using `missing_action="impute"`. When using the single-variable
#' model with categorical variables + subset splits, the rule buckets might be
#' incomplete due to not including categories that were not present in a given
#' node - this last point can be avoided by using `new_categ_action="smallest"`,
#' `new_categ_action="random"`, or `missing_action="impute"` (in the latter
#' case will treat them as missing, but the `predict` function might treat
#' them differently).
#' \item The resulting statements will include all the tree conditions as-is,
#' with no simplification. Thus, there might be lots of redundant conditions
#' in a given terminal node (e.g. "X > 2" and "X > 1", the second of which is
#' redundant).
#' }
#' @param model An Isolation Forest object as returned by \link{isolation.forest}.
#' @param enclose With which symbols to enclose the column names in the select statement
#' so as to make them SQL compatible in case they include characters like dots.
#' Options are:\itemize{
#' \item `"doublequotes"`, which will enclose them as `"column_name"` - this will
#' work for e.g. PostgreSQL.
#' \item `"squarebraces"`, which will enclose them as `[column_name]` - this will
#' work for e.g. SQL Server.
#' \item `"none"`, which will output the column names as-is (e.g. `column_name`)
#' }
#' @param output_tree_num Whether to make the statements return the terminal node number
#' instead of the isolation depth. The numeration will start at one.
#' @param tree Tree for which to generate SQL statements. If passed, will generate
#' the statements only for that single tree. If passing `NULL`, will
#' generate statements for all trees in the model.
#' @param table_from If passing this, will generate a single select statement for the
#' outlier score from all trees, selecting the data from the table
#' name passed here. In this case, will always output the outlier
#' score, regardless of what is passed under `output_tree_num`.
#' @param select_as Alias to give to the generated outlier score in the select statement.
#' Ignored when not passing `table_from`.
#' @param column_names Column names to use for the \bold{numeric} columns.
#' If not passed and the model was fit to a `data.frame`, will use the column
#' names from that `data.frame`, which can be found under `model$metadata$cols_num`.
#' If not passing it and the model was fit to data in a format other than
#' `data.frame`, the columns will be named `column_N` in the resulting
#' SQL statement. Note that the names will be taken verbatim - this function will
#' not do any checks for whether they constitute valid SQL or not, and will not
#' escape characters such as double quotation marks.
#' @param column_names_categ Column names to use for the \bold{categorical} columns.
#' If not passed, will use the column names from the `data.frame` to which the
#' model was fit. These can be found under `model$metadata$cols_cat`.
#' @return \itemize{
#' \item If passing neither `tree` nor `table_from`, will return a list
#' of `character` objects, containing at each entry the SQL statement
#' for the corresponding tree.
#' \item If passing `tree`, will return a single `character` object with
#' the SQL statement representing that tree.
#' \item If passing `table_from`, will return a single `character` object with
#' the full SQL select statement for the outlier score, selecting the columns
#' from the table name passed under `table_from`.
#' }
#' @examples 
#' library(isotree)
#' data(iris)
#' set.seed(1)
#' iso <- isolation.forest(iris, ntrees=2, sample_size=16, ndim=1, nthreads=1)
#' sql_forest <- isotree.to.sql(iso, table_from="my_iris_table")
#' cat(sql_forest)
#' @export
isotree.to.sql <- function(model, enclose="doublequotes", output_tree_num = FALSE, tree = NULL,
                           table_from = NULL, select_as = "outlier_score",
                           column_names = NULL, column_names_categ = NULL) {
    isotree.restore.handle(model)
    
    allowed_enclose <- c("doublequotes", "squarebraces", "none")
    if (NROW(enclose) != 1L)
        stop("'enclose' must be a character variable.")
    if (!(enclose %in% allowed_enclose))
        stop(sprintf("'enclose' must be one of the following: %s",
                     paste(allowed_enclose, sep=",")))
    
    if (NROW(output_tree_num) != 1L)
        stop("'output_tree_num' must be a single logical/boolean.")
    output_tree_num <- as.logical(output_tree_num)
    if (is.na(output_tree_num))
        stop("Invalid 'output_tree_num'.")
    
    single_tree <- !is.null(tree)
    if (single_tree) {
        if (NROW(tree) != 1L)
            stop("'tree' must be a single integer.")
        tree <- as.integer(tree)
        if (is.na(tree) || (tree < 1L) || (tree > get_ntrees(model$cpp_obj$ptr, model$params$ndim > 1)))
            stop("Invalid tree number.")
    } else {
        tree <- 0L
    }
    
    if (!is.null(table_from)) {
        if ((NROW(table_from) != 1L) || !is.character(table_from))
            stop("'table_from' must be a single character variable.")
        if (is.na(table_from))
            stop("Invalid 'table_from'.")
        
        if ((NROW(select_as) != 1L) || !is.character(select_as))
            stop("'select_as' must be a single character variable.")
        if (is.na(select_as))
            stop("Invalid 'select_as'.")
        
        single_tree <- FALSE
        tree <- 0L
        output_tree_num <- FALSE
    }
    
    if (model$metadata$ncols_num) {
        if (!is.null(column_names)) {
            if (NROW(column_names) != model$metadata$ncols_num)
                stop(sprintf("'column_names' must have length %d", model$metadata$ncols_num))
            if (!is.character(column_names))
                stop("'column_names' must be a character vector.")
            cols_num <- column_names
        } else {
            if (NROW(model$metadata$cols_num)) {
                cols_num <- model$metadata$cols_num
            } else {
                cols_num <- paste0("column_", seq(1L, model$metadata$ncols_num))
            }
        }
    } else {
        cols_num <- character()
    }
    
    if (model$metadata$ncols_cat) {
        if (!is.null(column_names_categ)) {
            if (NROW(column_names_categ) != model$metadata$ncols_cat)
                stop(sprintf("'column_names_categ' must have length %d", model$metadata$ncols_cat))
            if (!is.character(column_names_categ))
                stop("'column_names_categ' must be a character vector.")
            cols_cat <- column_names_categ
        } else {
            cols_cat <- model$metadata$cols_cat
        }
        
        cat_levels <- model$metadata$cat_levs
    } else {
        cols_cat <- character()
        cat_levels <- list()
    }
    
    if (enclose != "none") {
        enclose_left <- ifelse(enclose == "doublequotes", '"', '[')
        enclose_right <- ifelse(enclose == "doublequotes", '"', ']')
        if (NROW(cols_num))
            cols_num <- paste0(enclose_left, cols_num, enclose_right)
        if (NROW(cols_cat))
            cols_cat <- paste0(enclose_left, cols_cat, enclose_right)
    }
    
    is_extended <- model$params$ndim > 1L
    
    if (is.null(table_from)) {
        out <- model_to_sql(model$cpp_obj$ptr, is_extended,
                            cols_num, cols_cat, cat_levels,
                            output_tree_num, single_tree, tree,
                            model$nthreads)
        if (single_tree) {
            return(out[[1L]])
        } else {
            return(out)
        }
    } else {
        return(model_to_sql_with_select_from(model$cpp_obj$ptr, is_extended,
                                             cols_num, cols_cat, cat_levels,
                                             table_from,
                                             select_as,
                                             model$nthreads))
    }
}

#' @title Deep-Copy an Isolation Forest Model Object
#' @details Generates a deep copy of a model object, including the C++ objects inside it.
#' This function is only meaningful if one intends to call a function that modifies the
#' internal C++ objects - currently, the only such function are \link{isotree.add.tree}
#' and \link{isotree.append.trees} - as otherwise R's objects follow a copy-on-write logic.
#' @param model An `isolation_forest` model object.
#' @return A new `isolation_forest` object, with deep-copied C++ objects.
#' @export
isotree.deep.copy <- function(model) {
    isotree.restore.handle(model)
    new_pointers <- copy_cpp_objects(model$cpp_obj$ptr, model$params$ndim > 1,
                                     model$cpp_obj$imputer_ptr, !is.null(model$cpp_obj$imputer_ptr))
    new_model <- model
    new_model$cpp_obj <- list(
        ptr         =  new_pointers$model_ptr,
        serialized  =  model$cpp_obj$serialized,
        imp_ptr     =  new_pointers$imputer_ptr,
        imp_ser     =  model$cpp_obj$imputer_ser
    )
    new_model$metadata <- list(
        ncols_num  =  model$metadata$ncols_num,
        ncols_cat  =  model$metadata$ncols_cat,
        cols_num   =  model$metadata$cols_num,
        cols_cat   =  model$metadata$cols_cat,
        cat_levs   =  model$metadata$cat_levs,
        categ_cols =  model$metadata$categ_cols,
        categ_max  =  model$metadata$categ_max
    )
    new_model$params$ntrees <- deepcopy_int(model$params$ntrees)
    return(new_model)
}

#' @title Drop Imputer Sub-Object from Isolation Forest Model Object
#' @details Drops the imputer sub-object from an isolation forest model object, if it was fitted with data imputation
#' capabilities. The imputer, if constructed, is likely to be a very heavy object which might
#' not be needed for all purposes.
#' @param model An `isolation_forest` model object.
#' @return The same `model` object, but now with the imputer removed. \bold{Note that `model` is modified in-place
#' in any event}.
#' @export
isotree.drop.imputer <- function(model) {
    if (!inherits(model, "isolation_forest"))
        stop("This function is only available for isolation forest objects as returned from 'isolation.forest'.")
    set.list.elt(model$params, "build_imputer", FALSE)
    on.exit(set.list.elt(model$cpp_obj, "imputer_ser", NULL))
    if (!is.null(model$cpp_obj$imp_ptr))
        set.list.elt(model$cpp_obj, "imp_ptr", drop_imputer(model$cpp_obj$imp_ptr))
    return(model)
}

#' @title Subset trees of a given model
#' @details Creates a new isolation forest model containing only selected trees of a
#' given isolation forest model object.
#' @param model An `isolation_forest` model object.
#' @param trees_take Indices of the trees of `model` to copy over to a new model,
#' as an integer vector.
#' Must be integers with numeration starting at one
#' @return A new isolation forest model object, containing only the subset of trees
#' from this `model` that was specified under `trees_take`.
#' @export
isotree.subset.trees <- function(model, trees_take) {
    isotree.restore.handle(model)
    trees_take <- as.integer(trees_take)
    if (!NROW(trees_take))
        stop("'trees_take' cannot be empty.")
    if (anyNA(trees_take) || min(trees_take) < 1L || max(trees_take) > model$params$ntrees)
        stop("'trees_take' contains invalid indices.")

    ntrees_new <- length(trees_take)
    new_cpp_obj <- subset_trees(
        model$cpp_obj$ptr, model$cpp_obj$imp_ptr,
        model$params$ndim > 1, model$params$build_imputer,
        trees_take
    )
    new_model <- model
    new_model$cpp_obj <- NULL
    new_model$params$ntrees <- ntrees_new
    new_model$params$build_imputer <- as.logical(NROW(new_cpp_obj$imputer_ser))
    new_model$cpp_obj <- list(
        ptr         =  new_cpp_obj$model_ptr,
        serialized  =  new_cpp_obj$serialized_obj,
        imp_ptr     =  new_cpp_obj$imp_ptr,
        imp_ser     =  new_cpp_obj$imputer_ser
    )
    new_model$metadata <- list(
        ncols_num  =  model$metadata$ncols_num,
        ncols_cat  =  model$metadata$ncols_cat,
        cols_num   =  model$metadata$cols_num,
        cols_cat   =  model$metadata$cols_cat,
        cat_levs   =  model$metadata$cat_levs,
        categ_cols =  model$metadata$categ_cols,
        categ_max  =  model$metadata$categ_max
    )
    return(new_model)
}
