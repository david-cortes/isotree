#' @importFrom parallel detectCores
#' @importFrom stats predict
#' @importFrom utils head

#' @title Create Isolation Forest Model
#' @description Isolation Forest is an algorithm originally developed for outlier detection that consists in splitting
#' sub-samples of the data according to some attribute/feature/column at random. The idea is that, the rarer
#' the observation, the more likely it is that a random uniform split on some feature would put outliers alone
#' in one branch, and the fewer splits it will take to isolate an outlier observation like this. The concept
#' is extended to splitting hyperplanes in the extended model (i.e. splitting by more than one column at a time), and to
#' guided (not entirely random) splits in the SCiForest model that aim at isolating outliers faster and
#' finding clustered outliers.
#' 
#' This version adds heuristics to handle missing data and categorical variables. Can be used to aproximate pairwise
#' distances by checking the depth after which two observations become separated, and to approximate densities by fitting
#' trees beyond balanced-tree limit. Offers options to vary between randomized and deterministic splits too.
#' 
#' Note that the default parameters set up for this implementation will not scale to large datasets. In particular,
#' if the amount of data is large, it's avised to want to set a smaller sample size for each tree, and fit fewer of them. As well, the default option for `missing_action` might slow things down significantly.
#' 
#' The model offers many tunable parameters. The most likely candidate to tune is `prob_pick_pooled_gain`, for
#' which higher values tend to result in a better ability to flag outliers in the training data (`df`)
#' at the expense of hindered performance when making predictions on new data (calling function `predict`) and poorer
#' generalizability to inputs with values outside the variables' ranges to which the model was fit
#' (see plots generated from the examples for a better idea of the difference). The next candidate to tune is
#' `prob_pick_avg_gain` (along with `sample_size`), for which high values tend to result in models that are more likely
#' to flag values outside of the variables' ranges and fewer ghost regions, at the expense of fewer flagged outliers
#' in the original data.
#' 
#' @param df Data to which to fit the model. Supported inputs type are:\itemize{
#' \item A `data.frame`, also accepted as `data.table` or `tibble`.
#' \item A `matrix` object from base R.
#' \item A sparse matrix in CSC format, either from package `Matrix` (class `dgCMatrix`) or
#' from package `SparseM` (class `matrix.csc`).
#' }
#' 
#' If passing a `data.frame`, will assume that columns are:
#' \itemize{
#'   \item Numerical, if they are of types `numeric`, `integer`, `Date`, `POSIXct`.
#'   \item Categorical, if they are of type `character`, `factor`, `bool`.
#' }
#' Other column types are not supported. Note that it does not support inputs with a number of rows or number of columns
#' above that of `.Machine$integer.max` (typically 2^31-1).
#' @param sample_weights Sample observation weights for each row of `df`, with higher weights indicating either higher sampling
#' probability (i.e. the observation has a larger effect on the fitted model, if using sub-samples), or
#' distribution density (i.e. if the weight is two, it has the same effect of including the same data
#' point twice), according to parameter `weights_as_sample_prob`. Not supported when calculating pairwise
#' distances while the model is being fit (done by passing `output_dist` = `TRUE`).
#' @param column_weights Sampling weights for each column in `df`. Ignored when picking columns by deterministic criterion.
#' If passing `NULL`, each column will have a uniform weight. Cannot be used when weighting by kurtosis.
#' Note that, if passing a data.frame with both numeric and categorical columns, the column names must
#' not be repeated, otherwise the column weights passed here will not end up matching. If passing a `data.frame`
#' to `df`, will assume the column order is the same as in there, regardless of whether the entries passed to
#' `column_weights` are named or not.
#' @param sample_size Sample size of the data sub-samples with which each binary tree will be built.
#' Recommended value in references [1], [2], [3], [4] is 256, while the default value in the author's code in reference [5] is
#' `NROW(df)` (same as in here).
#' @param ntrees Number of binary trees to build for the model. Recommended value in reference [1] is 100, while the
#' default value in the author's code in reference [5] is 10. In general, the number of trees required for good results
#' is higher when (a) there are many columns, (b) there are categorical variables, (c) categorical variables have many
#' categories, (d) `ndim` is high.
#' @param ndim Number of columns to combine to produce a split. If passing 1, will produce the single-variable model described
#' in references [1] and [2], while if passing values greater than 1, will produce the extended model described in
#' references [3] and [4].
#' Recommended value in reference [4] is 2, while [3] recommends a low value such as 2 or 3. Models with values higher than 1
#' are referred hereafter as the extended model (as in [3]).
#' @param ntry In the extended model with non-random splits, how many random combinations to try for determining the best gain.
#' Only used when deciding splits by gain (see documentation for parameters `prob_pick_avg_gain` and `prob_pick_pooled_gain`).
#' Recommended value in refernece [4] is 10. Ignored for single-variable model.
#' @param max_depth Maximum depth of the binary trees to grow. By default, will limit it to the corresponding
#' depth of a balanced binary tree with number of terminal nodes corresponding to the sub-sample size (the reason
#' being that, if trying to detect outliers, an outlier will only be so if it turns out to be isolated with shorter average
#' depth than usual, which corresponds to a balanced tree depth).  When a terminal node has more than 1 observation,
#' the remaining isolation depth for them is estimated assuming the data and splits are both uniformly random
#' (separation depth follows a similar process with expected value calculated as in reference [6]). Default setting
#' for references [1], [2], [3], [4] is the same as the default here, but it's recommended to pass higher values if
#' using the model for purposes other than outlier detection.
#' @param prob_pick_avg_gain \itemize{
#' \item For the single-variable model (`ndim=1`), this parameter indicates the probability
#' of making each split by choosing a column and split point in that
#' same column as both the column and split point that gives the largest averaged gain (as proposed in [4]) across
#' all available columns and possible splits in each column. Note that this implies evaluating every single column
#' in the sample data when this type of split happens, which will potentially make the model fitting much slower,
#' but has no impact on prediction time. For categorical variables, will take the expected standard deviation that
#' would be gotten if the column were converted to numerical by assigning to each category a random number `~ Unif(0, 1)`
#' and calculate gain with those assumed standard deviations.
#' \item For the extended model, this parameter indicates the probability that the
#' split point in the chosen linear combination of variables will be decided by this averaged gain criterion.
#' }
#' Compared to a pooled average, this tends to result in more cases in which a single observation or very few of them
#' are put into one branch. Recommended to use sub-samples (parameter `sample_size`) when passing this parameter.
#' Note that, since this will created isolated nodes faster, the resulting object will be lighter (use less memory).
#' When splits are not made according to any of `prob_pick_avg_gain`, `prob_pick_pooled_gain`, `prob_split_avg_gain`,
#' `prob_split_pooled_gain`, both the column and the split point are decided at random. Default setting for 
#' references [1], [2], [3] is zero, and default for reference [4] is 1. This is the randomization parameter
#' that can be passed to the author's original code in [5]. Note that, if passing a value of 1 (100\%) with no sub-sampling and using the
#' single-variable model, every single tree will have the exact same splits.
#' @param prob_pick_pooled_gain \itemize{
#' \item For the single-variable model (`ndim=1`), this parameter indicates the probability
#' of making each split by choosing a column and split point in that
#' same column as both the column and split point that gives the largest pooled gain (as used in decision tree
#' classifiers such as C4.5 in [7]) across all available columns and possible splits in each column. Note
#' that this implies evaluating every single column in the sample data when this type of split happens, which
#' will potentially make the model fitting much slower, but has no impact on prediction time. For categorical
#' variables, will use shannon entropy instead (like in [7]).
#' \item For the extended model, this parameter indicates the probability
#' that the split point in the chosen linear combination of variables will be decided by this pooled gain
#' criterion.
#' }
#' Compared to a simple average, this tends to result in more evenly-divided splits and more clustered
#' groups when they are smaller. Recommended to pass higher values when used for imputation of missing values.
#' When used for outlier detection, higher values of this parameter result in models that are able to better flag
#' outliers in the training data, but generalize poorly to outliers in new data and to values of variables
#' outside of the ranges from the training data. Passing small `sample_size` and high values of this parameter will
#' tend to flag too many outliers.
#' Note that, since this makes the trees more even and thus it takes more steps to produce isolated nodes,
#' the resulting object will be heavier (use more memory). When splits are not made according to any of `prob_pick_avg_gain`,
#' `prob_pick_pooled_gain`, `prob_split_avg_gain`, `prob_split_pooled_gain`, both the column and the split point
#' are decided at random. Note that, if passing value = 1 (100\%) with no sub-sampling and using the single-variable model,
#' every single tree will have the exact same splits.
#' @param prob_split_avg_gain Probability of making each split by selecting a column at random and determining the split point as
#' that which gives the highest averaged gain. Not supported for the extended model as the splits are on
#' linear combinations of variables. See the documentation for parameter `prob_pick_avg_gain` for more details.
#' @param prob_split_pooled_gain Probability of making each split by selecting a column at random and determining the split point as
#' that which gives the highest pooled gain. Not supported for the extended model as the splits are on
#' linear combinations of variables. See the documentation for parameter `prob_pick_pooled_gain`` for more details.
#' @param min_gain Minimum gain that a split threshold needs to produce in order to proceed with a split. Only used when the splits
#' are decided by a gain criterion (either pooled or averaged). If the highest possible gain in the evaluated
#' splits at a node is below this  threshold, that node becomes a terminal node.
#' @param missing_action How to handle missing data at both fitting and prediction time. Options are
#' \itemize{
#'   \item `"divide"` (for the single-variable model only, recommended), which will follow both branches and combine
#'   the result with the weight given by the fraction of the data that went to each branch when fitting the model.
#'   \item `"impute"`, which will assign observations to the branch with the most observations in the single-variable model,
#'   or fill in missing values with the median of each column of the sample from which the split was made in the extended
#'   model (recommended for it).
#'   \item `"fail"`, which will assume there are no missing values and will trigger undefined behavior if it encounters any.
#' }
#' In the extended model, infinite values will be treated as missing. Note that passing `"fail"` might
#' crash the R process if there turn out to be missing values, but will otherwise produce faster fitting and prediction
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
#' want to pass `FALSE` here if merging several models into one through \link{append.trees}.
#' @param weights_as_sample_prob If passing `sample_weights` argument, whether to consider those weights as row
#' sampling weights (i.e. the higher the weights, the more likely the observation will end up included
#' in each tree sub-sample), or as distribution density weights (i.e. putting a weight of two is the same
#' as if the row appeared twice, thus higher weight makes it less of an outlier). Note that sampling weight
#' is only used when sub-sampling data for each tree, which is not the default in this implementation.
#' @param sample_with_replacement Whether to sample rows with replacement or not (not recommended).
#' Note that distance calculations, if desired, don't work when there are duplicate rows.
#' @param penalize_range Whether to penalize (add +1 to the terminal depth) observations at prediction time that have a value
#' of the chosen split variable (linear combination in extended model) that falls outside of a pre-determined
#' reasonable range in the data being split (given by `2 * range` in data and centered around the split point),
#' as proposed in reference [4] and implemented in the authors' original code in reference [5]. Not used in single-variable model
#' when splitting by categorical variables.
#' @param weigh_by_kurtosis Whether to weigh each column according to the kurtosis obtained in the sub-sample that is selected
#' for each tree as briefly proposed in reference [1]. Note that this is only done at the beginning of each tree
#' sample, so if not using sub-samples, it's better to pass column weights calculated externally. For
#' categorical columns, will calculate expected kurtosis if the column was converted to numerical by
#' assigning to each category a random number `~ Unif(0, 1)`.
#' @param coefs For the extended model, whether to sample random coefficients according to a normal distribution `~ N(0, 1)`
#' (as proposed in reference [3]) or according to a uniform distribution `~ Unif(-1, +1)` as proposed in reference [4].
#' Ignored for the single-variable model. Note that, for categorical variables, the coefficients will be sampled ~ N (0,1)
#' regardless - in order for both types of variables to have transformations in similar ranges (which will tend
#' to boost the importance of categorical variables), pass ``"uniform"`` here.
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
#' @param output_imputations Whether to output imputed missing values for `df`. Passing `TRUE` here will force
#' `build_imputer` to `TRUE`. Note that, for sparse matrix inputs, even though the output will be sparse, it will
#' generate a dense representation of each row with missing values.
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
#' @param output_dist Whether to output pairwise distances for the input data, which will be calculated as
#' the model is being fit and it's thus faster. Cannot be done when using sub-samples of the data for each tree
#' (in such case will later need to call the `predict` function on the same data). If using `penalize_range`, the
#' results from this might differ a bit from those of `predict` called after.
#' @param square_dist If passing `output_dist` = `TRUE`, whether to return a full square matrix or
#' just the upper-triangular part, in which the entry for pair (i,j) with 1 <= i < j <= n is located at position
#' p(i, j) = ((i - 1) * (n - i/2) + j - i).
#' @param random_seed Seed that will be used to generate random numbers used by the model.
#' @param handle_interrupt Whether to handle interrupt signals in the C++ code. If passing `TRUE`,
#' when it receives an interrupt signal while fitting the model, will halt before the procedure
#' finishes, but this has unintended side effects such as setting the interrupt handle for the rest
#' of the R session to this package's interrupt switch (which will print an error message), and
#' might cause trouble when interrupting the procedure from some REST framework such as plumber.
#' If passing `FALSE`, the C++ code (which fits the model) will not react to interrupt signals,
#' thus interrupting it will not do anything until  the model is fitted and control goes back
#' to R, but there will not be any side effects with respect to interrupt signals.
#' @param nthreads Number of parallel threads to use. If passing a negative number, will use
#' the maximum number of available threads in the system. Note that, the more threads,
#' the more memory will be allocated, even if the thread does not end up being used.
#' @return If passing `output_score` = `FALSE`, `output_dist` = `FALSE`, and `output_imputations` = `FALSE` (the defaults),
#' will output an `isolation_forest` object from which `predict` method can then be called on new data. If passing
#' `TRUE` to any of the former options, will output a list with entries:
#' \itemize{
#'   \item `model`: the `isolation_forest` object from which new predictions can be made.
#'   \item `scores`: a vector with the outlier score for each inpuit observation (if passing `output_score` = `TRUE`).
#'   \item `dist`: the distances (either a 1-d vector with the upper-triangular part or a square matrix), if
#'   passing `output_dist` = `TRUE`.
#'   \item `imputed`: the input data with missing values imputed according to the model (if passing `output_imputations` = `TRUE`).
#' }
#' @seealso \link{predict.isolation_forest},  \link{add.isolation.tree} \link{unpack.isolation.forest}
#' @references \itemize{
#' \item Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
#' \item Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation-based anomaly detection." ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
#' \item Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner. "Extended Isolation Forest." arXiv preprint arXiv:1811.02141 (2018).
#' \item Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "On detecting clustered anomalies using SCiForest." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
#' \item https://sourceforge.net/projects/iforest/
#' \item https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
#' \item Quinlan, J. Ross. "C4. 5: programs for machine learning." Elsevier, 2014.
#' \item Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
#' \item Cortes, David. "Imputing missing values with unsupervised random trees." arXiv preprint arXiv:1911.06646 (2019).
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
#' ### (As an interesting test, remove and see what happens)
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
#' ### Now try ouy different variations of the model
#' 
#' ### Single-variable model
#' iso_simple = isolation.forest(
#'     X, ndim=1,
#'     ntrees=100,
#'     nthreads=1,
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
#'      prob_pick_pooled_gain=1,
#'      prob_pick_avg_gain=0)
#' Z4 <- predict(iso_fcf, space_d)
#' plot.space(Z4, "Fair-Cut Forest")
#' par(oldpar)
#' 
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
#'   S <- matrix(rnorm(5 * 5), nrow = 5)
#'   S <- t(S) %*% S
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
#'       ntry = 10)
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
#' @details When calculating gain, the variables are standardized at each step, so there is no need to center/scale the
#' data beforehand.
#' 
#' When using sparse matrices, calculations such as standard deviations, gain, and kurtosis, will use procedures
#' that rely on calculating sums of squared numbers. This is not a problem if most of the entries are zero and the
#' numbers are small, but if passing dense matrices as sparse and/or the entries in the sparse matrices have values
#' in wildly different orders of magnitude (e.g. 0.0001 and 10000000), the calculations might be incorrect due to loss of
#' numeric precision, and the results might not be as good. For dense matrices it uses more numerically-robust
#' techniques (which would add a large computational overhead in sparse matrices), so it's not a problem to have values
#' with different orders of magnitude.
#' @export
isolation.forest <- function(df, sample_weights = NULL, column_weights = NULL,
                             sample_size = NROW(df), ntrees = 500, ndim = min(3, NCOL(df)),
                             ntry = 3, max_depth = ceiling(log2(sample_size)),
                             prob_pick_avg_gain = 0.0, prob_pick_pooled_gain = 0.0,
                             prob_split_avg_gain = 0.0, prob_split_pooled_gain = 0.0,
                             min_gain = 0, missing_action = ifelse(ndim > 1, "impute", "divide"),
                             new_categ_action = ifelse(ndim > 1, "impute", "weighted"),
                             categ_split_type = "subset", all_perm = FALSE,
                             coef_by_prop = FALSE, recode_categ = TRUE,
                             weights_as_sample_prob = TRUE, sample_with_replacement = FALSE,
                             penalize_range = TRUE, weigh_by_kurtosis = FALSE,
                             coefs = "normal", assume_full_distr = TRUE,
                             build_imputer = FALSE, output_imputations = FALSE, min_imp_obs = 3,
                             depth_imp = "higher", weigh_imp_rows = "inverse",
                             output_score = FALSE, output_dist = FALSE, square_dist = FALSE,
                             random_seed = 1, handle_interrupt = TRUE,
                             nthreads = parallel::detectCores()) {
    ### validate inputs
    if (NROW(sample_size) != 1 || sample_size < 5) { stop("'sample_size' must be an integer >= 5.") }
    check.pos.int(ntrees,       "ntrees")
    check.pos.int(ndim,         "ndim")
    check.pos.int(ntry,         "ntry")
    check.pos.int(max_depth,    "max_depth")
    check.pos.int(min_imp_obs,  "min_imp_obs")
    check.pos.int(random_seed,  "random_seed")
    
    allowed_missing_action    <-  c("divide",       "impute",   "fail")
    allowed_new_categ_action  <-  c("weighted",     "smallest", "random", "impute")
    allowed_categ_split_type  <-  c("single_categ", "subset")
    allowed_coefs             <-  c("normal",       "uniform")
    allowed_depth_imp         <-  c("lower",        "higher",   "same")
    allowed_weigh_imp_rows    <-  c("inverse",      "prop",     "flat")
    
    check.str.option(missing_action,    "missing_action",    allowed_missing_action)
    check.str.option(new_categ_action,  "new_categ_action",  allowed_new_categ_action)
    check.str.option(categ_split_type,  "categ_split_type",  allowed_categ_split_type)
    check.str.option(coefs,             "coefs",             allowed_coefs)
    check.str.option(depth_imp,         "depth_imp",         allowed_depth_imp)
    check.str.option(weigh_imp_rows,    "weigh_imp_rows",    allowed_weigh_imp_rows)
    
    check.is.prob(prob_pick_avg_gain,      "prob_pick_avg_gain")
    check.is.prob(prob_pick_pooled_gain,   "prob_pick_pooled_gain")
    check.is.prob(prob_split_avg_gain,     "prob_split_avg_gain")
    check.is.prob(prob_split_pooled_gain,  "prob_split_pooled_gain")
    
    check.is.bool(all_perm,                 "all_perm")
    check.is.bool(recode_categ,             "recode_categ")
    check.is.bool(coef_by_prop,             "coef_by_prop")
    check.is.bool(weights_as_sample_prob,   "weights_as_sample_prob")
    check.is.bool(sample_with_replacement,  "sample_with_replacement")
    check.is.bool(penalize_range,           "penalize_range")
    check.is.bool(weigh_by_kurtosis,        "weigh_by_kurtosis")
    check.is.bool(assume_full_distr,        "assume_full_distr")
    check.is.bool(output_score,             "output_score")
    check.is.bool(output_dist,              "output_dist")
    check.is.bool(square_dist,              "square_dist")
    check.is.bool(build_imputer,            "build_imputer")
    check.is.bool(output_imputations,       "output_imputations")
    check.is.bool(handle_interrupt,         "handle_interrupt")
    
    s <- prob_pick_avg_gain + prob_pick_pooled_gain + prob_split_avg_gain + prob_split_pooled_gain
    if (s > 1) {
        warning("Split type probabilities sum to more than 1, will standardize them")
        prob_pick_avg_gain      <- as.numeric(prob_pick_avg_gain)     /  s
        prob_pick_pooled_gain   <- as.numeric(prob_pick_pooled_gain)  /  s
        prob_split_avg_gain     <- as.numeric(prob_split_avg_gain)    /  s
        prob_split_pooled_gain  <- as.numeric(prob_split_pooled_gain) /  s
    }
    
    if (is.null(min_gain) || NROW(min_gain) > 1 || is.na(min_gain) || min_gain < 0)
        stop("'min_gain' must be a decimal non-negative number.")

    if ((ndim == 1) && (sample_size == NROW(df)) && (prob_pick_avg_gain >= 1 || prob_pick_pooled_gain >= 1)) {
        warning(paste0("Passed parameters for deterministic single-variable splits ",
                       "with no sub-sampling. ",
                       "Every tree fitted will end up doing exactly the same splits. ",
                       "It's recommended to set 'prob_pick_avg_gain' < 1, 'prob_pick_pooled_gain' < 1, ",
                       "or to use the extended model (ndim > 1)."))
    }
    
    if (ndim == 1) {
        if (new_categ_action == "impute")
            stop("'new_categ_action' = 'impute' not supported in single-variable model.")
    } else {
        if ((prob_split_avg_gain + prob_split_pooled_gain) > 0) {
            stop(paste0("Non-zero values for 'prob_split_avg_gain' ",
                        "and 'prob_split_pooled_gain' not meaningful in ",
                        "extended model."))
        }
        if (missing_action == "divide")
            stop("'missing_action' = 'divide' not supported in extended model.")
        if (new_categ_action == "weighted")
            stop("'new_categ_action' = 'weighted' not supported in extended model.")
    }
    
    if (ndim > NCOL(df))
        stop("'ndim' must be less or equal than the number of columns in 'df'.")
    
    nthreads <- check.nthreads(nthreads)
    if (sample_size > NROW(df)) stop("'sample_size' cannot be greater then the number of rows in 'df'.")
    
    if (!is.null(sample_weights)) check.is.1d(sample_weights, "sample_weights")
    if (!is.null(column_weights)) check.is.1d(column_weights, "column_weights")
    
    if (!is.null(sample_weights) && (sample_size == NROW(df)) && weights_as_sample_prob)
        stop("Sampling weights are only supported when using sub-samples for each tree.")
    
    if (weigh_by_kurtosis & !is.null(column_weights))
        stop("Cannot pass column weights when weighting columns by kurtosis.")
    
    if ((output_score || output_dist || output_imputations) & (sample_size != NROW(df)))
        stop("Cannot calculate scores/distances/imputations when sub-sampling data ('sample_size').")
    
    if ((output_score || output_dist) & sample_with_replacement)
        stop("Cannot calculate scores/distances when sampling data with replacement.")
    
    if (output_dist & !is.null(sample_weights))
        stop("Sample weights not supported when calculating distances while the model is being fit.")
    
    if (output_imputations) build_imputer <- TRUE
    
    if (build_imputer && missing_action == "fail")
        stop("Cannot impute missing values when passing 'missing_action' = 'fail'.")
    
    if (output_imputations && NROW(intersect(class(df), c("dgCMatrix", "matrix.csc"))))
        warning(paste0("Imputing missing values from CSC/dgCMatrix matrix on-the-fly can be very slow, ",
                       "it's recommended if possible to fit the model first and then pass the ",
                       "same matrix as CSR/dgRMatrix to 'predict'."))
    
    if (output_imputations && !is.null(sample_weights) && !weights_as_sample_prob)
        stop(paste0("Cannot impute missing values on-the-fly when using sample weights",
                    " as distribution density. Must first fit model and then impute values."))
    
    ### cast all parameters
    if (!is.null(sample_weights)) {
        sample_weights <- as.numeric(sample_weights)
    } else {
        sample_weights <- get.empty.vector()
    }
    if (!is.null(column_weights)) {
        column_weights <- as.numeric(column_weights)
    } else {
        column_weights <- get.empty.vector()
    }
    
    sample_size  <-  as.integer(sample_size)
    ntrees       <-  as.integer(ntrees)
    ndim         <-  as.integer(ndim)
    ntry         <-  as.integer(ntry)
    max_depth    <-  as.integer(max_depth)
    min_imp_obs  <-  as.integer(min_imp_obs)
    random_seed  <-  as.integer(random_seed)
    nthreads     <-  as.integer(nthreads)
    
    prob_pick_avg_gain       <-  as.numeric(prob_pick_avg_gain)
    prob_pick_pooled_gain    <-  as.numeric(prob_pick_pooled_gain)
    prob_split_avg_gain      <-  as.numeric(prob_split_avg_gain)
    prob_split_pooled_gain   <-  as.numeric(prob_split_pooled_gain)
    min_gain                 <-  as.numeric(min_gain)
    
    all_perm                 <-  as.logical(all_perm)
    coef_by_prop             <-  as.logical(coef_by_prop)
    weights_as_sample_prob   <-  as.logical(weights_as_sample_prob)
    sample_with_replacement  <-  as.logical(sample_with_replacement)
    penalize_range           <-  as.logical(penalize_range)
    weigh_by_kurtosis        <-  as.logical(weigh_by_kurtosis)
    assume_full_distr        <-  as.logical(assume_full_distr)
    
    ### split column types
    pdata <- process.data(df, sample_weights, column_weights, recode_categ)
    
    ### extra check for potential integer overflow
    if (all_perm && (ndim == 1) &&
        (prob_pick_pooled_gain || prob_split_pooled_gain) &&
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
                             sample_size, ntrees,  max_depth, FALSE,
                             penalize_range, output_dist, TRUE, square_dist,
                             output_score, TRUE, weigh_by_kurtosis,
                             prob_pick_avg_gain, prob_split_avg_gain,
                             prob_pick_pooled_gain,  prob_split_pooled_gain, min_gain,
                             categ_split_type, new_categ_action,
                             missing_action, all_perm,
                             build_imputer, output_imputations, min_imp_obs,
                             depth_imp, weigh_imp_rows,
                             random_seed, handle_interrupt, nthreads)
    
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
            prob_pick_avg_gain = prob_pick_avg_gain,
            prob_pick_pooled_gain = prob_pick_pooled_gain,
            prob_split_avg_gain = prob_split_avg_gain,
            prob_split_pooled_gain = prob_split_pooled_gain,
            min_gain = min_gain, missing_action = missing_action,
            new_categ_action = new_categ_action,
            categ_split_type = categ_split_type,
            all_perm = all_perm, coef_by_prop = coef_by_prop,
            weights_as_sample_prob = weights_as_sample_prob,
            sample_with_replacement = sample_with_replacement,
            penalize_range = penalize_range,
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
            cat_levs   =  pdata$cat_levs
            ),
        random_seed  =  random_seed,
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
        outp <- list(model    =  this,
                     scores   =  NULL,
                     dist     =  NULL,
                     imputed  =  NULL)
        if (output_score) outp$scores <- cpp_outputs$depths
        if (output_dist) {
            if (square_dist) {
                outp$dist  <-  cpp_outputs$dmat
            } else {
                outp$dist  <-  cpp_outputs$tmat
            }
        }
        if (output_imputations) {
            outp$imputed   <-  reconstruct.from.imp(cpp_outputs$imputed_num,
                                                    cpp_outputs$imputed_cat,
                                                    df, this, trans_CSC = FALSE)
        }
        return(outp)
    }
}

#' @title Predict method for Isolation Forest
#' @param object An Isolation Forest object as returned by `isolation.forest`.
#' @param newdata A `data.frame`, `data.table`, `tibble`, `matrix`, or sparse matrix (from package `Matrix` or `SparseM`,
#' CSC/dgCMatrix format for distance and outlierness, or CSR/dgRMatrix format for outlierness and imputations)
#' for which to predict outlierness, distance, or imputations of missing values.
#' 
#' Note that when passing `type` = `"impute"` and `newdata` is a sparse matrix, under some situations it might get modified in-place.
#' 
#' Note also that, if using sparse matrices from package `Matrix`, converting to `dgRMatrix` might require using
#' `as(m, "RsparseMatrix")` instead of `dgRMatrix` directly.
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
#'   will return a list containing both the outlier score and the terminal node numbers, under entries
#'   `score` and `tree_num`, respectively.
#'   \item `"impute"` for imputation of missing values in `newdata`.
#' }
#' @param square_mat When passing `type` = `"dist` or `"avg_sep"` with no `refdata`, whether to return a full square matrix or
#' just the upper-triangular part, in which the entry for pair (i,j) with 1 <= i < j <= n is located at position
#' p(i, j) = ((i - 1) * (n - i/2) + j - i).
#' Ignored when not predicting distance/separation or when passing `refdata`.
#' @param refdata If passing this and calculating distance or average separation depth, will calculate distances
#' between each point in `newdata` and each point in `refdata`, outputing a matrix in which points in `newdata`
#' correspond to rows and points in `refdata` correspond to columns. Must be of the same type as `newdata` (e.g.
#' `data.frame`, `matrix`, `dgCMatrix`, etc.). If this is not passed, and type is `"dist"`
#' or `"avg_sep"`, will calculate pairwise distances/separation between the points in `newdata`.
#' @param ... Not used.
#' @return The requested prediction type, which can be: \itemize{
#' \item A vector with one entry per row in `newdata` (for output types `"score"`, `"avg_depth"`, `"tree_num"`).
#' \item A square matrix or vector with the upper triangular part of a square matrix
#' (for output types `"dist"`, `"avg_sep"`, with no `refdata`)
#' \item A matrix with points in `newdata` as rows and points in `refdata` as columns
#' (for output types `"dist"`, `"avg_sep"`, with `refdata`).
#' \item The same type as the input `newdata` (for output type `"impute"`).}
#' @details The more threads that are set for the model, the higher the memory requirement will be as each
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
#' `output_score=TRUE` in `isolation.forest`.
#' @seealso \link{isolation.forest} \link{unpack.isolation.forest}
#' @export
predict.isolation_forest <- function(object, newdata, type="score", square_mat=FALSE, refdata=NULL, ...) {
    if (check_null_ptr_model(object$cpp_obj$ptr)) {
        obj_new <- object$cpp_obj
        if (object$params$ndim == 1)
            ptr_new <- deserialize_IsoForest(object$cpp_obj$serialized)
        else
            ptr_new <- deserialize_ExtIsoForest(object$cpp_obj$serialized)
        obj_new$ptr <- ptr_new
        
        if (object$params$build_imputer) {
            imp_new <- deserialize_Imputer(object$cpp_obj$imp_ser)
            obj_new$imp_ptr <- imp_new
        }
        
        eval.parent(substitute(object$cpp_obj <- obj_new))
        object$cpp_obj <- obj_new
    }
    
    allowed_type <- c("score", "avg_depth", "dist", "avg_sep", "tree_num", "impute")
    check.str.option(type, "type", allowed_type)
    check.is.bool(square_mat)
    if (!NROW(newdata)) stop("'newdata' must be a data.frame, matrix, or sparse matrix.")
    if ((object$metadata$ncols_cat > 0) && NROW(intersect(class(newdata), get.types.spmat(TRUE, TRUE, TRUE)))) {
        stop("Cannot pass sparse inputs if the model was fit to categorical variables.")
    }
    if ("numeric" %in% class(newdata) && is.null(dim(newdata))) {
        newdata <- matrix(newdata, nrow=1)
    }
    if (NCOL(newdata) < (object$metadata$ncols_num + object$metadata$ncols_cat)) {
        if ("dsparseVector" %in% class(newdata)) {
            if (NROW(newdata) != object$metadata$ncols_num)
                stop("'newdata' has different number of columns than the original data.")
        } else {
            stop("'newdata' has fewer columns than the original data.")
        }
    }
    if (type %in% c("dist", "avg_sep")) {
        if (object$params$new_categ_action == "weighted" && object$params$missing_action != "divide") {
            stop(paste0("Cannot predict distances when using ",
                        "'new_categ_action' = 'weighted' ",
                        "if 'missing_action' != 'divide'."))
        }
    }
    if (type %in% "impute" && (is.null(object$params$build_imputer) || !(object$params$build_imputer)))
        stop("Cannot impute missing values with model that was built with 'build_imputer' =  'FALSE'.")
    
    if (is.null(refdata) || !(type %in% c("dist", "avg_sep"))) {
        nobs_group1 <- 0L
    } else {
        nobs_group1 <- NROW(newdata)
        newdata     <- rbind(newdata, refdata)
    }
    
    pdata <- process.data.new(newdata, object$metadata, !(type %in% c("dist", "avg_sep")), type != "impute")
    
    square_mat   <-  as.logical(square_mat)
    score_array  <-  get.empty.vector()
    dist_tmat    <-  get.empty.vector()
    dist_dmat    <-  get.empty.vector()
    dist_rmat    <-  get.empty.vector()
    tree_num     <-  get.empty.int.vector()
    
    if (type %in% c("dist", "avg_sep")) {
        if (NROW(newdata) == 1) stop("Need more than 1 data point for distance predictions.")
        if (is.null(refdata)) {
            dist_tmat <- vector("numeric", (pdata$nrows * (pdata$nrows - 1L)) / 2L)
            if (square_mat) dist_dmat <- vector("numeric", pdata$nrows ^ 2)
        } else {
            dist_rmat <- vector("numeric", nobs_group1 * (pdata$nrows - nobs_group1))
        }
    } else {
        score_array <- vector("numeric", pdata$nrows)
        if (type == "tree_num") tree_num <- vector("integer", pdata$nrows * object$params$ntrees)
    }
    
    if (type %in% c("score", "avg_depth", "tree_num")) {
        predict_iso(object$cpp_obj$ptr, score_array, tree_num, object$params$ndim > 1,
                    pdata$X_num, pdata$X_cat,
                    pdata$Xc, pdata$Xc_ind, pdata$Xc_indptr,
                    pdata$Xr, pdata$Xr_ind, pdata$Xr_indptr,
                    pdata$nrows, object$nthreads, type == "score")
        if (type == "tree_num")
            return(list(score = score_array, tree_num = matrix(tree_num + 1L, nrow = pdata$nrows, ncol = object$params$ntrees)))
        else
            return(score_array)
    } else if (type != "impute") {
        dist_iso(object$cpp_obj$ptr, dist_tmat, dist_dmat, dist_rmat,
                 object$params$ndim > 1,
                 pdata$X_num, pdata$X_cat,
                 pdata$Xc, pdata$Xc_ind, pdata$Xc_indptr,
                 pdata$nrows, object$nthreads, object$params$assume_full_distr,
                 type == "dist", square_mat, nobs_group1)
        if (!is.null(refdata))
            return(t(matrix(dist_rmat, ncol = nobs_group1)))
        else if (square_mat)
            return(matrix(dist_dmat, nrow = pdata$nrows, ncol = pdata$nrows))
        else
            return(dist_tmat)
    } else {
        imp <- impute_iso(object$cpp_obj$ptr, object$cpp_obj$imp_ptr, object$params$ndim > 1,
                          pdata$X_num, pdata$X_cat,
                          pdata$Xr, pdata$Xr_ind, pdata$Xr_indptr,
                          pdata$nrows, object$nthreads)
        return(reconstruct.from.imp(imp$X_num,
                                    imp$X_cat,
                                    newdata, object,
                                    trans_CSC = TRUE))
    }
}


#' @title Print summary information from Isolation Forest model
#' @description Displays the most general characteristics of an isolation forest model (same as `summary`).
#' @param x An Isolation Forest model as produced by function `isolation.forest`.
#' @param ... Not used.
#' @details Note that after loading a serialized object from `isolation.forest` through `readRDS` or `load`,
#' it will only de-serialize the underlying C++ object upon running `predict`, `print`, or `summary`,
#' so the first run will be slower, while subsequent runs will be faster as the C++ object will already be in-memory.
#' @return No return value.
#' @seealso \link{isolation.forest}
#' @export
print.isolation_forest <- function(x, ...) {
    if (check_null_ptr_model(x$cpp_obj$ptr)) {
        obj_new <- x$cpp_obj
        if (x$params$ndim == 1)
            ptr_new <- deserialize_IsoForest(x$cpp_obj$serialized)
        else
            ptr_new <- deserialize_ExtIsoForest(x$cpp_obj$serialized)
        obj_new$ptr <- ptr_new
        
        if (x$params$build_imputer) {
            imp_new <- deserialize_Imputer(x$cpp_obj$imp_ser)
            obj_new$imp_ptr <- imp_new
        }
        
        eval.parent(substitute(x$cpp_obj <- obj_new))
        x$cpp_obj <- obj_new
    }
    
    if (x$params$ndim > 1) cat("Extended ")
    cat("Isolation Forest model")
    if (
        (x$params$prob_pick_avg_gain + x$params$prob_pick_pooled_gain) > 0 ||
        (x$params$ndim == 1 & (x$params$prob_split_avg_gain + x$params$prob_split_pooled_gain) > 0)
    ) {
        cat(" (using guided splits)")
    }
    cat("\n")
    if (x$params$ndim > 1) cat(sprintf("Splitting by %d variables at a time\n", x$params$ndim))
    cat(sprintf("Consisting of %d trees\n", x$params$ntrees))
    if (x$metadata$ncols_num  > 0)  cat(sprintf("Numeric columns: %d\n",     x$metadata$ncols_num))
    if (x$metadata$ncols_cat  > 0)  cat(sprintf("Categorical columns: %d\n", x$metadata$ncols_cat))
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
#' @export
summary.isolation_forest <- function(object, ...) {
    print.isolation_forest(object)
}

#' @title Add additional (single) tree to isolation forest model
#' @description Adds a single tree fit to the full (non-subsampled) data passed here. Must
#' have the same columns as previously-fitted data.
#' @param model An Isolation Forest object as returned by `isolation.forest`, to which an additional tree will be added.
#' The result of this function must be reassigned to `model`, and the old `model` should not be used any further.
#' @param df A `data.frame`, `data.table`, `tibble`, `matrix`, or sparse matrix (from package `Matrix` or `SparseM`, CSC format)
#' to which to fit the new tree.
#' @param sample_weights Sample observation weights for each row of 'X', with higher weights indicating
#' distribution density (i.e. if the weight is two, it has the same effect of including the same data
#' point twice). If not `NULL`, model must have been built with `weights_as_sample_prob` = `FALSE`.
#' @param column_weights Sampling weights for each column in `df`. Ignored when picking columns by deterministic criterion.
#' If passing `NULL`, each column will have a uniform weight. Cannot be used when weighting by kurtosis.
#' @return No return value. The model is modified in-place.
#' @details Important: this function will modify the model object in-place, but this modification will only affect the R
#' object in the environment in which it was called. If trying to use the same model object in e.g. its parent environment,
#' it will lead to issues due to the C++ object being modified but the R object remaining the same, so if this method is used
#' inside a function, make sure to output the newly-modified R object and have it replace the old R object outside the calling
#' function too.
#' @seealso \link{isolation.forest} \link{unpack.isolation.forest}
#' @export
add.isolation.tree <- function(model, df, sample_weights = NULL, column_weights = NULL) {
    
    if (!("isolation_forest" %in% class(model)))
        stop("'model' must be an isolation forest model object as output by function 'isolation.forest'.")
    if (!is.null(sample_weights) && model$weights_as_sample_prob)
        stop("Cannot use sampling weights with 'partial_fit'.")
    if (!is.null(column_weights) && model$weigh_by_kurtosis)
        stop("Cannot pass column weights when weighting columns by kurtosis.")
    
    if (check_null_ptr_model(model$cpp_obj$ptr)) {
        obj_new <- model$cpp_obj
        if (model$params$ndim == 1)
            ptr_new <- deserialize_IsoForest(model$cpp_obj$serialized)
        else
            ptr_new <- deserialize_ExtIsoForest(model$cpp_obj$serialized)
        obj_new$ptr <- ptr_new
        
        if (model$params$build_imputer) {
            imp_new <- deserialize_Imputer(model$cpp_obj$imp_ser)
            obj_new$imp_ptr <- imp_new
        }
        
        ## eval.parent(substitute(model$cpp_obj <- obj_new)) ## this is done after adding the tree
        model$cpp_obj <- obj_new
    }
    
    
    if (!is.null(sample_weights))
        sample_weights  <- as.numeric(sample_weights)
    else
        sample_weights  <- get.empty.vector()
    if (!is.null(column_weights))
        column_weights  <- as.numeric(column_weights)
    else
        column_weights  <- get.empty.vector()
    if (NROW(sample_weights) && NROW(sample_weights) != NROW(df))
        stop(sprintf("'sample_weights' has different number of rows than df (%d vs. %d).",
                     NROW(df), NROW(sample_weights)))
    if (NROW(column_weights)  && NCOL(df) != NROW(column_weights))
        stop(sprintf("'column_weights' has different dimension than number of columns in df (%d vs. %d).",
                     NCOL(df), NROW(column_weights)))
    
    if (model$metadata$ncols_cat)
        ncat  <-  sapply(model$metadata$cat_levs, NROW)
    else
        ncat  <-  get.empty.int.vector()
    
    pdata <- process.data.new(df, model$metadata, FALSE)
    
    model_new <- model
    model_new$cpp_obj$serialized <- fit_tree(model$cpp_obj$ptr, 
                                             pdata$X_num, pdata$X_cat, unname(ncat),
                                             pdata$Xc, pdata$Xc_ind, pdata$Xc_indptr,
                                             sample_weights, column_weights,
                                             pdata$nrows, model$metadata$ncols_num, model$metadata$ncols_cat,
                                             model$params$ndim, model$params$ntry,
                                             model$params$coefs, model$params$coef_by_prop,
                                             model$params$max_depth,
                                             FALSE, model$params$penalize_range,
                                             model$params$weigh_by_kurtosis,
                                             model$params$prob_pick_avg_gain, model$params$prob_split_avg_gain,
                                             model$params$prob_pick_pooled_gain,  model$params$prob_split_pooled_gain,
                                             model$params$min_gain,
                                             model$params$categ_split_type, model$params$new_categ_action,
                                             model$params$missing_action, model$params$build_imputer,
                                             model$params$min_imp_obs, model$cpp_obj$imp_ptr,
                                             model$params$depth_imp, model$params$weigh_imp_rows,
                                             model$params$all_perm, model$random_seed)
    
    model_new$params$ntrees <- model_new$params$ntrees + 1L
    eval.parent(substitute(model <- model_new))
    return(invisible(NULL))
}

#' @title Unpack isolation forest model after de-serializing
#' @description  After persisting an isolation forest model object through `saveRDS`, `save`, or restarting a session, the
#' underlying C++ objects that constitute the isolation forest model and which live only on the C++ heap memory are not saved along,
#' thus not restored after loading a saved model through `readRDS` or `load`.
#' 
#' The model object however keeps serialized versions of the C++ objects as raw bytes, from which the C++ objects can be
#' reconstructed, and are done so automatically after calling `predict`, `print`, `summary`, or `add.isolation.tree` on the
#' freshly-loaded object from `readRDS` or `load`.
#' 
#' But due to R's environments system (as opposed to other systems such as Python which can use pass-by-reference), they will
#' only be re-constructed in the environment that is calling `predict`, `print`, etc. and not in higher-up environments
#' (i.e. if calling `predict` on the object from inside different functions, each function will have to reconstruct the
#' C++ objects independently and they will only live within the function that called `predict`).
#' 
#' This function serves as an environment-level unpacker that will reconstruct the C++ object in the environment in which
#' it is called (i.e. if it's desired to call `predict` from inside multiple functions, use this function before passing the
#' freshly-loaded model object to those other functions, and then they will not need to reconstruct the C++ objects anymore),
#' in the same way as `predict` or `print`, but without producing any outputs or messages.
#' @param model An Isolation Forest object as returned by `isolation.forest`, which has been just loaded from a disk
#' file through `readRDS`, `load`, or a session restart.
#' @return No return value. Object is modified in-place.
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
#' ### will de-serialize inside, but object is short-lived
#' wrap_predict <- function(model, data) {
#'     pred <- predict(model, data)
#'     cat("pointer inside function is this: ")
#'     print(model$cpp_obj$ptr)
#'     return(pred)
#' }
#' temp <- wrap_predict(iso2, X)
#' cat("pointer outside function is this: \n")
#' print(iso2$cpp_obj$ptr) ### pointer to the C++ object
#' 
#' ### now unpack the C++ object beforehand
#' unpack.isolation.forest(iso2)
#' print("after unpacking beforehand")
#' temp <- wrap_predict(iso2, X)
#' cat("pointer outside function is this: \n")
#' print(iso2$cpp_obj$ptr)
#' @export
unpack.isolation.forest <- function(model)  {
    if (!("isolation_forest" %in% class(model)))
        stop("'model' must be an isolation forest model object as output by function 'isolation.forest'.")
    
    if (check_null_ptr_model(model$cpp_obj$ptr)) {
        obj_new <- model$cpp_obj
        if (model$params$ndim == 1)
            ptr_new <- deserialize_IsoForest(model$cpp_obj$serialized)
        else
            ptr_new <- deserialize_ExtIsoForest(model$cpp_obj$serialized)
        obj_new$ptr <- ptr_new
        
        if (model$params$build_imputer) {
            imp_new <- deserialize_Imputer(model$cpp_obj$imp_ser)
            obj_new$imp_ptr <- imp_new
        }
        
        eval.parent(substitute(model$cpp_obj <- obj_new))
        model$cpp_obj <- obj_new
    }
    
    return(invisible(NULL))
}

#' @title Get Number of Nodes per Tree
#' @param model An Isolation Forest model as produced by function `isolation.forest`.
#' @return A list with entries `"total"` and `"terminal"`, both of which are integer vectors
#' with length equal to the number of trees. `"total"` contains the total number of nodes that
#' each tree has, while `"terminal"` contains the number of terminal nodes per tree.
#' @export
get.num.nodes <- function(model)  {
    if (!("isolation_forest" %in% class(model)))
        stop("'model' must be an isolation forest model object as output by function 'isolation.forest'.")

    if (check_null_ptr_model(model$cpp_obj$ptr)) {
        obj_new <- model$cpp_obj
        if (model$params$ndim == 1)
            ptr_new <- deserialize_IsoForest(model$cpp_obj$serialized)
        else
            ptr_new <- deserialize_ExtIsoForest(model$cpp_obj$serialized)
        obj_new$ptr <- ptr_new
        
        if (model$params$build_imputer) {
            imp_new <- deserialize_Imputer(model$cpp_obj$imp_ser)
            obj_new$imp_ptr <- imp_new
        }
        
        eval.parent(substitute(model$cpp_obj <- obj_new))
        model$cpp_obj <- obj_new
    }

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
#' `recode_categ=FALSE` in the call to \link{isolation.forest} (which is \bold{not} the
#' default), and the categorical columns passed as type `factor` with the same `levels` -
#' otherwise different models might be using different encodings for each categorical column,
#' which will not be preserved as only the trees will be appended without any associated metadata.
#' 
#' Note that this function will not perform any checks on the inputs, and passing two incompatible
#' models (e.g. fit to different numbers of columns) will result in wrong results and
#' potentially crashing the R process when using it.
#' 
#' Also be aware that the result \bold{must} be reassigned to the first input, as the first
#' input will no longer work correctly after appending more trees to it.
#' 
#' \bold{Important:} the result of this function must be reassigned to `model` in order for it
#' to work properly - e.g. `model <- append.trees(model, other)`.
#' @param model An Isolation Forest model (as returned by function \link{isolation.forest})
#' to which trees from `other` (another Isolation Forest model) will be appended into.
#' The result of this function must be reassigned to `model`, and the old `model` should
#' not be used any further.
#' @param other Another Isolation Forest model, from which trees will be appended into
#' `model`. It will not be modified during the call to this function.
#' @return The updated `model` object, to which `model` needs to be reassigned
#' (i.e. you need to use it as follows: `model <- append.trees(model, other)`).
#' @details Important: this function will modify the model object in-place, but this modification will only affect the R
#' object in the environment in which it was called. If trying to use the same model object in e.g. its parent environment,
#' it will lead to issues due to the C++ object being modified but the R object remaining the same, so if this method is used
#' inside a function, make sure to output the newly-modified R object and have it replace the old R object outside the calling
#' function too.
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
#' iso1 <- append.trees(iso1, iso2)
#' 
#' ### Check that it predicts the same as the two models
#' nodes.comb <- predict(iso1, head(X1, 3), type="tree_num")
#' nodes.comb$tree_num == cbind(nodes1$tree_num, nodes2$tree_num)
#' 
#' ### The new predicted scores will be a weighted average
#' ### (Be aware that, due to round-off, it will not match with '==')
#' nodes.comb$score
#' (3*nodes1$score + 2*nodes2$score) / 5
#' @export
append.trees <- function(model, other) {
    if (!("isolation_forest" %in% class(model)) || !("isolation_forest" %in% class(other))) {
        stop("'model' and 'other' must be isolation forest models.")
    }
    if ((model$params$ndim == 1) != (other$params$ndim == 1)) {
        stop("Cannot mix extended and regular isolation forest models (ndim=1).")
    }
    if (model$metadata$ncols_cat) {
        warning("Merging models with categorical features might give wrong results.")
    }
    
    serialized <- append_trees_from_other(model$cpp_obj$ptr,      other$cpp_obj$ptr,
                                          model$cpp_obj$imp_ptr,  other$cpp_obj$imp_ptr,
                                          model$params$ndim > 1)
    model$cpp_obj$serialized <- serialized$serialized
    if ("imp_ser" %in% names(model)) {
        model$cpp_obj$imp_ser <- serialized$imp_ser
    }
    
    model$params$ntrees <- model$params$ntrees + other$params$ntrees
    return(model)
}

#' @title Export Isolation Forest model
#' @description Save Isolation Forest model to a serialized file along with its
#' metadata, in order to be used in the Python or the C++ versions of this package.
#' 
#' This function is not meant to be used for passing models to and from R -
#' in such case, you can use `saveRDS` and `readRDS` instead.
#' 
#' Note that, if the model was fitted to a `data.frame`, the column names must be
#' something exportable as JSON, and must be something that Python's Pandas could
#' use as column names (e.g. strings/character).
#' 
#' It is recommended to visually inspect the produced `.metadata` file in any case.
#' @details This function will create 2 files: the serialized model, in binary format,
#' with the name passed in `file`; and a metadata file in JSON format with the same
#' name but ending in `.metadata`. The second file should \bold{NOT} be edited manually,
#' except for the field `nthreads` if desired.
#' 
#' If the model was built with `build_imputer=TRUE`, there will also be a third binary file
#' ending in `.imputer`.
#' 
#' The metadata will contain, among other things, the encoding that was used for
#' categorical columns - this is under `data_info.cat_levels`, as an array of arrays by column,
#' with the first entry for each column corresponding to category 0, second to category 1,
#' and so on (the C++ version takes them as integers). This metadata is written to a JSON file
#' using the `jsonlite` package, which must be installed in order for this to work.
#' 
#' The serialized file can be used in the C++ version by reading it as a binary raw file
#' and de-serializing its contents with the `cereal` library or using the provided C++ functions
#' for de-serialization. If using `ndim=1`, it will be an object of class `IsoForest`, and if
#' using `ndim>1`, will be an object of class `ExtIsoForest`. The imputer file, if produced, will
#' be an object of class `Imputer`.
#' 
#' The metadata is not used in the C++ version, but is necessary for the Python version.
#' 
#' Note that the model treats boolean/logical variables as categorical. Thus, if the model was fit
#' to a `data.frame` with boolean columns, when importing this model into C++, they need to be
#' encoded in the same order - e.g. the model might encode `TRUE` as zero and `FALSE`
#' as one - you need to look at the metadata for this.
#' @param model An Isolation Forest model as returned by function \link{isolation.forest}.
#' @param file File path where to save the model. File connections are not accepted, only
#' file paths
#' @param ... Additional arguments to pass to \link{writeBin} - you might want to pass
#' extra parameters if passing files between different CPU architectures or similar.
#' @return No return value.
#' @seealso \link{load.isotree.model} \link{writeBin} \link{unpack.isolation.forest}
#' @references \url{https://uscilab.github.io/cereal/}
#' @export
export.isotree.model <- function(model, file, ...) {
    if (!("isolation_forest" %in% class(model)))
        stop("This function is only available for isolation forest objects as returned from 'isolation.forest'.")
    metadata <- export.metadata(model)
    file.metadata <- paste0(file, ".metadata")
    jsonlite::write_json(export.metadata(model), file.metadata,
                         pretty=TRUE, auto_unbox=TRUE)
    writeBin(model$cpp_obj$serialized, file, ...)
    if (model$params$build_imputer) {
        file.imp <- paste0(file, ".imputer")
        writeBin(model$cpp_obj$imp_ser, file.imp, ...)
    }
    return(invisible(NULL))
}

#' @title Load an Isolation Forest model exported from Python
#' @description Loads a serialized Isolation Forest model as produced and exported
#' by the Python version of this package. Note that the metadata must be something
#' importable in R - e.g. column names must be valid for R (numbers are not valid names for R).
#' It's recommended to visually inspect the `.metadata` file in any case.
#' 
#' This function is not meant to be used for passing models to and from R -
#' in such case, you can use `saveRDS` and `readRDS` instead.
#' @param file Path to the saved isolation forest model along with its metadata file,
#' and imputer file if produced. Must be a file path, not a file connection.
#' @details Internally, this function uses `readr::read_file_raw` (from the `readr` package)
#' and `jsonlite::fromJSON` (from the `jsonlite` package). Be sure to have those installed
#' and that the files are readable through them.
#' 
#' Note: If the model was fit to a ``DataFrame`` using Pandas' own Boolean types,
#' take a look at the metadata to check if these columns will be taken as booleans
#' (R logicals) or as categoricals with string values `"True"` or `"False"`.
#' @return An isolation forest model, as if it had been constructed through
#' \link{isolation.forest}.
#' @seealso \link{export.isotree.model} \link{unpack.isolation.forest}
#' @export
load.isotree.model <- function(file) {
    if (!file.exists(file)) stop("'file' does not exist.")
    metadata.file <- paste0(file, ".metadata")
    if (!file.exists(metadata.file)) stop("No matching metadata for 'file'.")
    
    metadata <- jsonlite::fromJSON(metadata.file,
                                   simplifyVector = TRUE,
                                   simplifyDataFrame = FALSE,
                                   simplifyMatrix = FALSE)
    
    this <- take.metadata(metadata)
    this$cpp_obj$serialized <- readr::read_file_raw(file)
    if (this$params$ndim == 1)
        this$cpp_obj$ptr <- deserialize_IsoForest(this$cpp_obj$serialized)
    else
        this$cpp_obj$ptr <- deserialize_ExtIsoForest(this$cpp_obj$serialized)
    
    imputer.file <- paste0(file, ".imputer")
    if (file.exists(imputer.file)) {
        this$cpp_obj$imp_ser <- readr::read_file_raw(imputer.file)
        this$cpp_obj$imp_ptr <- deserialize_Imputer(this$cpp_obj$imp_ser)
    }
    
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
#' `penalize_range=TRUE` (which is the default).
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
    if (!("isolation_forest" %in% class(model)))
        stop("'model' must be an isolation forest model.")
    
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
        if (is.na(tree) || (tree < 1L) || (tree > model$params$ntrees))
            stop("Invalid tree number.")
    } else {
        tree <- 0L
    }
    
    if (!is.null(table_from)) {
        if ((NROW(table_from) != 1L) || !("character" %in% class(table_from)))
            stop("'table_from' must be a single character variable.")
        if (is.na(table_from))
            stop("Invalid 'table_from'.")
        
        if ((NROW(select_as) != 1L) || !("character" %in% class(select_as)))
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
            if (!("character" %in% class(column_names)))
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
            if (!("character" %in% class(column_names_categ)))
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
