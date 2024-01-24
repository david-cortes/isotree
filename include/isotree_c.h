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

/***********************************************************************************
    -------------------
    IsoTree C interface
    -------------------

    This is a C wrapper over the "isotree_oop.hpp" interface but with C ABI, so as
    to ensure better compatibility between compilers and make for easier FFI
    bindings.

    In order to achieve both compatibility and ease of use, it uses structs to pass
    arguments but in two different ways, which might be a bit confusing at a fist
    glance:
     - As a plain C struct that is defined in this same header and only used through
       'static inline' functions but allowing easy modification of its members
       through C syntax, thus ensuring that any usage of them is confined to the
       same compilation unit where this header is used and not passed to any shared
       object compiled elsewhere.
     - As opaque void pointers to an underlying C++ struct which is only used
       internally by the shared object, but providing functions to translate to it
       from the C structs which are more user friendly.
    
    Note that this is a more limited interface compared to the non-OOP C++ interface
    from the header 'isotree.hpp' - for example, this interface only allows passing
    data in 'double' and 'int' types, and does not have all the same functionality
    for object serialization, distance calculations, or producing predictions while
    a model is being fitted. If possible, it is recommended to use the 'isotree.hpp'
    interface instead of the C interface or the OOP C++ interface.

    See the 'isotree.hpp' and 'isotree_oop.hpp' headers for documentation about the
    functions and the parameters that they take.

***********************************************************************************/

#ifndef ISOTREE_C_H
#define ISOTREE_C_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifndef ISOTREE_EXPORTED
#   ifdef _WIN32
#       define ISOTREE_EXPORTED __declspec(dllimport)
#   else
#       define ISOTREE_EXPORTED 
#   endif
#endif

#ifdef __cplusplus
extern "C" {
#endif


/* Types used through the package - zero is the suggested value (when appropriate) */
#if !defined(ISOTREE_H) && !defined(ISOTREE_OOP_H)
typedef enum  NewCategAction {Weighted=0,  Smallest=11,    Random=12}  NewCategAction; /* Weighted means Impute in the extended model */
typedef enum  MissingAction  {Divide=21,   Impute=22,      Fail=0}     MissingAction;  /* Divide is only for non-extended model */
typedef enum  CategSplit     {SubSet=0,    SingleCateg=41}             CategSplit;
typedef enum  CoefType       {Uniform=61,  Normal=0}                   CoefType;       /* For extended model */
typedef enum  UseDepthImp    {Lower=71,    Higher=0,       Same=72}    UseDepthImp;    /* For NA imputation */
typedef enum  WeighImpRows   {Inverse=0,   Prop=81,        Flat=82}    WeighImpRows;   /* For NA imputation */
typedef enum  ScoringMetric  {Depth=0,     Density=92,     BoxedDensity=94, BoxedDensity2=96, BoxedRatio=95,
                              AdjDepth=91, AdjDensity=93}              ScoringMetric;
#endif

/* These codes are used to signal error status from functions */
enum IsoTreeExitCodes {IsoTreeSuccess=0, IsoTreeError=1};

typedef uint8_t isotree_bool;
typedef uint8_t NewCategAction_t;
typedef uint8_t MissingAction_t;
typedef uint8_t CategSplit_t;
typedef uint8_t CoefType_t;
typedef uint8_t UseDepthImp_t;
typedef uint8_t WeighImpRows_t;
typedef uint8_t ScoringMetric_t;
typedef int isotree_exit_code;

/*  Parameters to fit the model  */
typedef struct isotree_parameters {
    int nthreads; /* <- May be manually changed at any time, default=-1 */

    uint64_t random_seed; /* default=1 */

    /*  General tree construction parameters  */
    size_t ndim; /* default=1 */
    size_t ntry; /* default=1 */
    CoefType coef_type; /* only for ndim>1, default=Uniform */
    bool   with_replacement; /* default=false */
    bool   weight_as_sample; /* default=true */
    size_t sample_size; /* default=0 */
    size_t ntrees; /* default=500 */
    size_t max_depth; /* default=0 [ignored if 'limit_depth==true'] */
    size_t ncols_per_tree; /* default=0 */
    bool   limit_depth; /* default=true [if 'true' then 'max_depth' is ignored] */
    bool   penalize_range; /* default=false */
    bool   standardize_data; /* default=true */
    bool   fast_bratio; /* default=true, only for 'scoring_metric' with 'Boxed' */
    ScoringMetric scoring_metric; /* default=Depth */
    bool   weigh_by_kurt; /* default=false */
    double prob_pick_by_gain_pl; /* default=0 */
    double prob_pick_by_gain_avg; /* default=0 */
    double prob_pick_by_full_gain; /* default=0 */
    double prob_pick_by_dens; /* default=0 */
    double prob_pick_col_by_range; /* default=0 */
    double prob_pick_col_by_var; /* default=0 */
    double prob_pick_col_by_kurt; /* default=0 */
    double min_gain; /* default=0 */
    MissingAction missing_action; /* default=Impute */

    /*  For categorical variables  */
    CategSplit cat_split_type; /* default=SubSet */
    NewCategAction new_cat_action; /* default=Weighted */
    bool   coef_by_prop; /* default=false */
    bool   all_perm; /* default=false */

    /*  For imputation methods (when using 'build_imputer=true' and calling 'impute')  */
    bool   build_imputer; /* default=false */
    size_t min_imp_obs; /* default=3 */
    UseDepthImp depth_imp; /* default=Higher */
    WeighImpRows weigh_imp_rows; /* default=Inverse */
} isotree_parameters;

/*  General notes about passing data to this library:
     - All data passed to the 'isotree_fit' function must be in column-major
       order (like Fortran). Most of the prediction functions support
       row-major order however, and predictions are typically faster in
       row-major order (with some exceptions). When passing data in row-major
       format, must also pass the leading dimension of the array (typically
       corresponds to the number of column, but may be larger than that when
       passing a subset of a larger array).
     - Data can be passed in different formats. When data in some format
       is not to be passed to some function, the struct here should contain
       a NULL pointer in the non-used members.
     - The library supports numeric and categorical data. Categorical data
       should be passed as integers with numeration starting at 0, with missing
       values encoded as negative integers. If passing categorical data, must
       also pass the number of categories per column ('ncat').
     - Numeric data may be passed in sparse CSC format (in which case the array
       'numeric_data' should be NULL). Some prediction functions may also
       accept sparse data in CSR format. Note that, if passing numeric data
       in sparse format, there should be no further numeric data in dense
       format passed to the same function (i.e. can pass at most one of
       'numeric_data' or 'csc_*', and same for the prediction functions).
     - The structs here ('isotree_input_data' and 'isotree_prediction_data')
       are not used in any function call, they are just laid out for easier
       understanding of the format. */
typedef struct isotree_input_data {
    size_t nrows;
    double *numeric_data;
    size_t ncols_numeric;
    int *categ_data;
    size_t ncols_categ;
    int *ncateg;
    double *csc_values;
    int *csc_indices;
    int *csc_indptr;
    double *row_weights;
    double *column_weights;
} isotree_input_data;

typedef struct isotree_prediction_data {
    size_t nrows;
    bool is_col_major; /* applies to 'numeric_data' and 'categ_data' */
    double *numeric_data;
    size_t ld_numeric; /* this is >= 'ncols_numeric', typically == */
    int *categ_data;
    size_t ld_categ; /* this is >= 'ncols_categ', typically == */
    bool is_csc; /* applies to 'sparse_*' */
    double *sparse_values;
    int *sparse_indices;
    int *sparse_indptr;
} isotree_prediction_data;

typedef void* isotree_parameters_t; /* <- do not confuse with 'isotree_parameters' */
typedef void* isotree_model_t; /* <- it's a pointer to a C++ 'IsolationForest' object instance */


/*  Note: this returns an 'isotree_parameters' type, but the function that fits the
    model requires instead an 'isotree_parameters_t' object. See below for a function
    that converts it. */
static inline isotree_parameters get_default_isotree_parameters()
{
    return (isotree_parameters) {
        -1, 1, 1, 1, Uniform, false, true, 0, 500, 0, 0, true, false,
        true, Depth, true, false, 0., 0., 0., 0., 0., 0., 0., 0., Impute, SubSet, Weighted,
        false, false, false, 3, Higher, Inverse
    };
}

/*  Alternatively, this function returns 'isotree_parameters_t', which can be passed directly
    to the model-fitting function, and can be modified using the less-convenient function
    'set_isotree_parameters'. Note that it is a pointer and thust must be freed after using
    it through the function 'delete_isotree_parameters'.

    If allocation fails, will return a NULL pointer.  */
ISOTREE_EXPORTED
isotree_parameters_t allocate_default_isotree_parameters();

/*  This is the equivalent to 'free' or C++'s 'delete'.  */
ISOTREE_EXPORTED
void delete_isotree_parameters(isotree_parameters_t parameters);


/*  The arguments to this function are all pointers. It works as follows:
     - 'isotree_parameters' is a void pointer to a C++ struct which contains
       parameters to pass to the model fitting function. This is what gets
       modified by this function.
     - If a given pointer is NULL, will not modify the value of the underlying
       parameter in the opaque C++ struct.
     - If a given pointer is *not* NULL, will be de-referenced and its value
       assigned to the corresponding member of the opaque struct pointer.  */
ISOTREE_EXPORTED
void set_isotree_parameters
(
    isotree_parameters_t isotree_parameters,
    int*       nthreads,
    uint64_t*  random_seed,
    size_t*    ndim,
    size_t*    ntry,
    CoefType_t*     coef_type,
    isotree_bool*   with_replacement,
    isotree_bool*   weight_as_sample,
    size_t*    sample_size,
    size_t*    ntrees,
    size_t*    max_depth,
    size_t*    ncols_per_tree,
    isotree_bool*   limit_depth,
    isotree_bool*   penalize_range,
    isotree_bool*   standardize_data,
    ScoringMetric_t* scoring_metric,
    isotree_bool*    fast_bratio,
    isotree_bool*   weigh_by_kurt,
    double*    prob_pick_by_gain_pl,
    double*    prob_pick_by_gain_avg,
    double*    prob_pick_by_full_gain,
    double*    prob_pick_by_dens,
    double*    prob_pick_col_by_range,
    double*    prob_pick_col_by_var,
    double*    prob_pick_col_by_kurt,
    double*    min_gain,
    MissingAction_t*   missing_action,
    CategSplit_t*      cat_split_type,
    NewCategAction_t*  new_cat_action,
    isotree_bool*   coef_by_prop,
    isotree_bool*   all_perm,
    isotree_bool*   build_imputer,
    size_t*    min_imp_obs,
    UseDepthImp_t*   depth_imp,
    WeighImpRows_t*  weigh_imp_rows
);

/*  This function will overwrite the values in all of the non-NULL pointers,
    outputting the current values in the C++ opaque struct. */
ISOTREE_EXPORTED
void get_isotree_parameters
(
    const isotree_parameters_t isotree_parameters,
    int*       nthreads,
    uint64_t*  random_seed,
    size_t*    ndim,
    size_t*    ntry,
    CoefType_t*     coef_type,
    isotree_bool*   with_replacement,
    isotree_bool*   weight_as_sample,
    size_t*    sample_size,
    size_t*    ntrees,
    size_t*    max_depth,
    size_t*    ncols_per_tree,
    isotree_bool*   limit_depth,
    isotree_bool*   penalize_range,
    isotree_bool*   standardize_data,
    ScoringMetric_t scoring_metric,
    isotree_bool*   fast_bratio,
    isotree_bool*   weigh_by_kurt,
    double*    prob_pick_by_gain_pl,
    double*    prob_pick_by_gain_avg,
    double*    prob_pick_by_full_gain,
    double*    prob_pick_by_dens,
    double*    prob_pick_col_by_range,
    double*    prob_pick_col_by_var,
    double*    prob_pick_col_by_kurt,
    double*    min_gain,
    MissingAction_t*   missing_action,
    CategSplit_t*      cat_split_type,
    NewCategAction_t*  new_cat_action,
    isotree_bool*   coef_by_prop,
    isotree_bool*   all_perm,
    isotree_bool*   build_imputer,
    size_t*    min_imp_obs,
    UseDepthImp_t*   depth_imp,
    WeighImpRows_t*  weigh_imp_rows
);

/*  This is a short-hand for an easier-to-use alternative to the function that sets the parameters,
    so as to allow using the more transparent C struct instead. In order to modify values in the C
    struct, it's just enough with reassigning each member and then passing it to this function.

    Note that the output is a pointer and must be freed afterwards through 'delete_isotree_parameters'. */
static inline isotree_parameters_t allocate_isotree_parameters(isotree_parameters parameters)
{
    isotree_parameters_t out = allocate_default_isotree_parameters();
    if (!out) return NULL;
    uint8_t coef_type = parameters.coef_type, with_replacement = parameters.with_replacement,
            weight_as_sample = parameters.weight_as_sample, limit_depth = parameters.limit_depth,
            penalize_range = parameters.penalize_range, standardize_data = parameters.standardize_data,
            scoring_metric = parameters.scoring_metric, fast_bratio = parameters.fast_bratio,
            weigh_by_kurt = parameters.weigh_by_kurt,
            missing_action = parameters.missing_action, cat_split_type = parameters.cat_split_type,
            new_cat_action = parameters.new_cat_action, coef_by_prop = parameters.coef_by_prop,
            all_perm = parameters.all_perm, build_imputer = parameters.build_imputer,
            depth_imp = parameters.depth_imp, weigh_imp_rows = parameters.weigh_imp_rows;
    set_isotree_parameters(
        out, &parameters.nthreads, &parameters.random_seed,
        &parameters.ndim, &parameters.ntry, &coef_type, &with_replacement,
        &weight_as_sample, &parameters.sample_size, &parameters.ntrees, &parameters.max_depth,
        &parameters.ncols_per_tree, &limit_depth, &penalize_range, &standardize_data,
        &scoring_metric, &fast_bratio, &weigh_by_kurt,
        &parameters.prob_pick_by_gain_pl, &parameters.prob_pick_by_gain_avg,
        &parameters.prob_pick_by_full_gain, &parameters.prob_pick_by_dens,
        &parameters.prob_pick_col_by_range, &parameters.prob_pick_col_by_var,
        &parameters.prob_pick_col_by_kurt,
        &parameters.min_gain, &missing_action, &cat_split_type, &new_cat_action, &coef_by_prop,
        &all_perm, &build_imputer, &parameters.min_imp_obs, &depth_imp, &weigh_imp_rows
    );
    return out;
}


/*  This returns a pointer to a struct, must be freed after use through
    the function 'delete_isotree_model'. If anything fails, will print
    an error message to 'stderr' and return a NULL pointer.  */
ISOTREE_EXPORTED
isotree_model_t isotree_fit
(
    const isotree_parameters_t,
    size_t nrows,
    double *numeric_data,
    size_t ncols_numeric,
    int *categ_data,
    size_t ncols_categ,
    int *ncateg,
    double *csc_values,
    int *csc_indices,
    int *csc_indptr,
    double *row_weights,
    double *column_weights
);

ISOTREE_EXPORTED
void delete_isotree_model(isotree_model_t isotree_model);

/*  Here the data can be row-major or column-major. If passing dense data
    (arrays 'numeric_data' and 'categ_data', this is the most usual case),
    it's recommended to pass it in row-major order (the order for them
    must be signaled through 'is_col_major').

    If passing sparse data, it's recommended to do so in CSC format
    (column-major), and to pass the categorical data also in column-major
    order if there is any. Sparse data might alternatively be passed in
    CSR format (must be signaled through 'is_csc').

    'output_scores' must always be passed, while 'output_tree_num' and
    'per_tree_depths' are optional. 'output_scores' is of dimension 'nrows',
    while 'output_tree_num' and 'per_tree_depths' are of dimensions
    'nrows*ntrees' (output format is column-major for 'output_tree_num', but
    row-major for 'per_tree_depths'). Note that 'output_tree_num' and
    'per_tree_depths' are not calculable when using 'ndim==1' plus either
    'missing_action==Divide' or 'new_cat_action==Weighted'.

    Will return 0 if it executes successfully, or 1 if an error happens
    (along with printing a message to 'stderr' if an error is encountered).
    Note that the only possible throwable error that can happen inside
    'isotree_predict' is an out-of-memory condition when passing
    CSC data, or a safety check when passing 'output_tree_num' or
    'per_tree_depths' in cases in which they are not fillable.  */
ISOTREE_EXPORTED
isotree_exit_code isotree_predict
(
    isotree_model_t isotree_model,
    double *output_scores,
    int *output_tree_num,
    double *per_tree_depths,
    isotree_bool standardize_scores,
    size_t nrows,
    isotree_bool is_col_major,
    double *numeric_data,
    size_t ld_numeric,
    int *categ_data,
    size_t ld_categ,
    isotree_bool is_csc,
    double *sparse_values,
    int *sparse_indices,
    int *sparse_indptr
);

/*  Here the data is only supported in column-major order.
     - If passing 'output_triangular=true', then 'output_dist'
       should have length 'nrows*(nrows-1)/2' (which corresponds
       to the upper diagonal of a symmetric square matrix).
     - If passing 'output_triangular=false', then 'output_dist'
       should have length 'nrows^2' (which corresponds to a full
       symmetric square matrix).

    Note that unlike 'isotree_predict', there are more possible
    error-throwing scenarios in this function, such as receiving a
    stop signal (errors are signaled in the same way). */
ISOTREE_EXPORTED
isotree_exit_code isotree_predict_distance
(
    isotree_model_t isotree_model,
    isotree_bool output_triangular,
    isotree_bool as_kernel,
    isotree_bool standardize,
    isotree_bool assume_full_distr,
    double *output_dist, /* <- output goes here */
    size_t nrows,
    double *numeric_data,
    int *categ_data,
    double *csc_values,
    int *csc_indices,
    int *csc_indptr
);

/*  This will replace NAN values in-place. Note that for sparse inputs it
    will impute NANs, not values that are ommited from the sparse format.  */
ISOTREE_EXPORTED
isotree_exit_code isotree_impute
(
    isotree_model_t isotree_model,
    size_t nrows,
    isotree_bool is_col_major, /* applies to 'numeric_data' and 'categ_data' */
    double *numeric_data,
    int *categ_data,
    double *csr_values,
    int *csr_indices,
    int *csr_indptr
);

ISOTREE_EXPORTED
isotree_exit_code isotree_set_reference_points
(
    isotree_model_t isotree_model,
    isotree_bool with_distances,
    size_t nrows,
    isotree_bool is_col_major,
    double *numeric_data,
    size_t ld_numeric,
    int *categ_data,
    size_t ld_categ,
    double *csc_values,
    int *csc_indices,
    int *csc_indptr
);

ISOTREE_EXPORTED
size_t isotree_get_num_reference_points(isotree_model_t isotree_model);

/*  Must call 'isotree_set_reference_points' to make this method available.

    Here 'output_dist' should have dimension [nrows, n_references],
    and will be filled in row-major order.

    This will always take 'assume_full_distr=true'.  */
ISOTREE_EXPORTED
isotree_exit_code isotree_predict_distance_to_ref_points
(
    isotree_model_t isotree_model,
    double *output_dist, /* <- output goes here */
    isotree_bool as_kernel,
    isotree_bool standardize,
    size_t nrows,
    isotree_bool is_col_major,
    double *numeric_data,
    size_t ld_numeric,
    int *categ_data,
    size_t ld_categ,
    double *csc_values,
    int *csc_indices,
    int *csc_indptr
);

/*  Files should be opened in binary mode.  */
ISOTREE_EXPORTED
isotree_exit_code isotree_serialize_to_file(const isotree_model_t isotree_model, FILE *output);

/*  'nthreads' here means 'what value to set 'nthreads' to in the resulting
    object' (which are used for the prediction functions). The de-serialization
    process itself is always single-threaded.

    If there is an error, will print a message to 'stderr' and return a NULL pointer.  */
ISOTREE_EXPORTED
isotree_model_t isotree_deserialize_from_file(FILE *serialized_model, int nthreads);

/*  Can serialize to an array of raw bytes (char*), in which case this function
    will tell how large the array needs to be to hold the serialized model.

    If an error occurs, will print a message to 'stderr' and return 0.  */
ISOTREE_EXPORTED
size_t isotree_serialize_get_raw_size(const isotree_model_t isotree_model);

ISOTREE_EXPORTED
isotree_exit_code isotree_serialize_to_raw(const isotree_model_t isotree_model, char *output);

ISOTREE_EXPORTED
isotree_model_t isotree_deserialize_from_raw(const char *serialized_model, int nthreads);

/*  If passing a negative number, will set to:
      nthreads = max_threads + nthreads + 1
    So passing -1 means using all threads, passing -2 all but 1 thread, and so on.  */
ISOTREE_EXPORTED
isotree_exit_code isotree_set_num_threads(isotree_model_t isotree_model, int nthreads);

/*  If an error occurs (e.g. passing a NULL pointer), will return -INT_MAX.  */
ISOTREE_EXPORTED
int isotree_get_num_threads(const isotree_model_t isotree_model);

/*  If an error occurs (e.g. passing a NULL pointer), will return SIZE_MAX.  */
ISOTREE_EXPORTED
size_t isotree_get_ntrees(const isotree_model_t isotree_model);

ISOTREE_EXPORTED
isotree_exit_code isotree_build_indexer(isotree_model_t isotree_model, const isotree_bool with_distances);

/*  If an error occurs (e.g. passing a NULL pointer), will return NULL.  */
ISOTREE_EXPORTED
isotree_model_t isotree_copy_model(isotree_model_t isotree_model);

#ifdef __cplusplus
}
#endif

#endif /* ifndef ISOTREE_C_H */
