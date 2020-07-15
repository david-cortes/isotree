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
*     Copyright (c) 2019, David Cortes
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
#include <stddef.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <signal.h>
#include <vector>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <utility>
#include <cstdint>
#include <iostream>
#ifndef _FOR_R
    #include <stdio.h> 
#else
    extern "C" {
        #include <R_ext/Print.h>
    }
    #define printf Rprintf
    #define fprintf(f, message) REprintf(message)
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif
#ifdef _ENABLE_CEREAL
    #include <cereal/archives/binary.hpp>
    #include <cereal/types/vector.hpp>
    #include <sstream>
    #include <string>
    #include <fstream>
#endif

/* By default, will use Mersenne-Twister for RNG, but can be switched to something faster */
#ifdef _USE_MERSENNE_TWISTER
    #if SIZE_MAX >= UINT64_MAX /* 64-bit systems or higher */
        #define RNG_engine std::mt19937_64
    #else /* 32-bit systems and non-standard architectures */
        #define RNG_engine std::mt19937
    #endif
#else
    #define RNG_engine std::default_random_engine
#endif

/* Short functions */
#define ix_parent(ix) (((ix) - 1) / 2)  /* integer division takes care of deciding left-right */
#define ix_child(ix)  (2 * (ix) + 1)
/* https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int */
#define pow2(n) ( ((size_t) 1) << (n) )
#define square(x) ((x) * (x))
/* https://stackoverflow.com/questions/2249731/how-do-i-get-bit-by-bit-data-from-an-integer-value-in-c */
#define extract_bit(number, bit) (((number) >> (bit)) & 1)
#ifndef isinf
    #define isinf std::isinf
#endif
#ifndef isnan
    #define isnan std::isnan
#endif
#define is_na_or_inf(x) (isnan(x) || isinf(x))


/* Aliasing for compiler optimizations */
#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER)
    #define restrict __restrict
#else
    #define restrict 
#endif

/* MSVC is stuck with an OpenMP version that's 19 years old at the time of writing and does not support unsigned iterators */
#ifdef _OPENMP
    #if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
        #define size_t_for long
    #else
        #define size_t_for size_t
    #endif
#else
    #define size_t_for size_t
#endif


/*    Apple at some point decided to drop OMP library and headersfrom its compiler distribution
*    and to alias 'gcc' to 'clang', which work differently when given flags they cannot interpret,
*    causing installation issues with pretty much all scientific software due to OMP headers that
*    would normally do nothing. This piece of code is to allow compilation without OMP header. */
#ifndef _OPENMP
    #define omp_get_thread_num() 0
#endif


/* For sparse matrices */
#ifdef _FOR_R
    #define sparse_ix int
#else
    #define sparse_ix size_t
#endif


/* Types used through the package */
typedef enum  NewCategAction {Weighted, Smallest, Random}      NewCategAction; /* Weighted means Impute in the extended model */
typedef enum  MissingAction  {Divide,   Impute,   Fail}        MissingAction;  /* Divide is only for non-extended model */
typedef enum  ColType        {Numeric,  Categorical, NotUsed}  ColType;
typedef enum  CategSplit     {SubSet,   SingleCateg}           CategSplit;
typedef enum  GainCriterion  {Averaged, Pooled,   NoCrit}      Criterion;      /* For guided splits */
typedef enum  CoefType       {Uniform,  Normal}                CoefType;       /* For extended model */
typedef enum  UseDepthImp    {Lower,    Higher,   Same}        UseDepthImp;    /* For NA imputation */
typedef enum  WeighImpRows   {Inverse,  Prop,     Flat}        WeighImpRows;   /* For NA imputation */

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
    ColType  col_type = NotUsed; /* issues with uninitialized values passed to Cereal */
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

    #ifdef _ENABLE_CEREAL
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
            this->col_type,
            this->col_num,
            this->num_split,
            this->cat_split,
            this->chosen_cat,
            this->tree_left,
            this->tree_right,
            this->pct_tree_left,
            this->score,
            this->range_low,
            this->range_high,
            this->remainder
            );
    }
    #endif

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

    #ifdef _ENABLE_CEREAL
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
            this->col_num,
            this->col_type,
            this->coef,
            this->mean,
            this->cat_coef,
            this->chosen_cat,
            this->fill_val,
            this->fill_new,
            this->split_point,
            this->hplane_left,
            this->hplane_right,
            this->score,
            this->range_low,
            this->range_high,
            this->remainder
            );
    }
    #endif

    IsoHPlane() = default;
} IsoHPlane;

/* Note: don't use long doubles in the outside outputs or there will be issues with MINGW in windows */


typedef struct IsoForest {
    std::vector< std::vector<IsoTree> > trees;
    NewCategAction    new_cat_action;
    CategSplit        cat_split_type;
    MissingAction     missing_action;
    double            exp_avg_depth;
    double            exp_avg_sep;
    size_t            orig_sample_size;

    #ifdef _ENABLE_CEREAL
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
            this->trees,
            this->new_cat_action,
            this->cat_split_type,
            this->missing_action,
            this->exp_avg_depth,
            this->exp_avg_sep,
            this->orig_sample_size
            );
    }
    #endif

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

    #ifdef _ENABLE_CEREAL
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
            this->hplanes,
            this->new_cat_action,
            this->cat_split_type,
            this->missing_action,
            this->exp_avg_depth,
            this->exp_avg_sep,
            this->orig_sample_size
            );
    }
    #endif

    ExtIsoForest() = default;
} ExtIsoForest;

typedef struct ImputeNode {
    std::vector<double>  num_sum;
    std::vector<double>  num_weight;
    std::vector<std::vector<double>>  cat_sum;
    std::vector<double>  cat_weight;
    size_t               parent;

    #ifdef _ENABLE_CEREAL
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
            this->num_sum,
            this->num_weight,
            this->cat_sum,
            this->cat_weight,
            this->parent
            );
    }
    #endif
    ImputeNode() = default;

    ImputeNode(size_t parent)
    {
        this->parent = parent;
    }

} ImputeNode; /* this is for each tree node */

typedef struct Imputer {
    size_t               ncols_numeric;
    size_t               ncols_categ;
    std::vector<int>     ncat;
    std::vector<std::vector<ImputeNode>> imputer_tree;
    std::vector<double>  col_means;
    std::vector<int>     col_modes;

    #ifdef _ENABLE_CEREAL
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
            this->ncols_numeric,
            this->ncols_categ,
            this->ncat,
            this->imputer_tree,
            this->col_means,
            this->col_modes
            );
    }
    #endif

    Imputer() = default;

} Imputer;


/* Structs that are only used internally */
typedef struct {
    double*     numeric_data;
    size_t      ncols_numeric;
    int*        categ_data;
    int*        ncat;
    int         max_categ;
    size_t      ncols_categ;
    size_t      nrows;
    size_t      ncols_tot;
    double*     sample_weights;
    bool        weight_as_sample;
    double*     col_weights;
    double*     Xc;           /* only for sparse matrices */
    sparse_ix*  Xc_ind;       /* only for sparse matrices */
    sparse_ix*  Xc_indptr;    /* only for sparse matrices */
    size_t      log2_n;       /* only when using weights for sampling */
    size_t      btree_offset; /* only when using weights for sampling */
    std::vector<double> btree_weights_init;  /* only when using weights for sampling */
    std::vector<char>   has_missing;         /* only used when producing missing imputations on-the-fly */
    size_t              n_missing;           /* only used when producing missing imputations on-the-fly */
} InputData;


typedef struct {
    double*     numeric_data;
    int*        categ_data;
    size_t      nrows;
    double*     Xc;           /* only for sparse matrices */
    sparse_ix*  Xc_ind;       /* only for sparse matrices */
    sparse_ix*  Xc_indptr;    /* only for sparse matrices */
    double*     Xr;           /* only for sparse matrices */
    sparse_ix*  Xr_ind;       /* only for sparse matrices */
    sparse_ix*  Xr_indptr;    /* only for sparse matrices */
} PredictionData;

typedef struct {
    bool      with_replacement;
    size_t    sample_size;
    size_t    ntrees;
    size_t    max_depth;
    bool      penalize_range;
    uint64_t  random_seed;
    bool      weigh_by_kurt;
    double    prob_pick_by_gain_avg;
    double    prob_split_by_gain_avg;
    double    prob_pick_by_gain_pl;
    double    prob_split_by_gain_pl;
    double    min_gain;
    CategSplit      cat_split_type;
    NewCategAction  new_cat_action;
    MissingAction   missing_action;
    bool            all_perm;

    size_t ndim;        /* only for extended model */
    size_t ntry;        /* only for extended model */
    CoefType coef_type; /* only for extended model */
    bool coef_by_prop;  /* only for extended model */

    bool calc_dist;     /* checkbox for calculating distances on-the-fly */
    bool calc_depth;    /* checkbox for calculating depths on-the-fly */
    bool impute_at_fit; /* checkbox for producing imputed missing values on-the-fly */

    UseDepthImp   depth_imp;      /* only when building NA imputer */
    WeighImpRows  weigh_imp_rows; /* only when building NA imputer */
    size_t        min_imp_obs;     /* only when building NA imputer */
} ModelParams;

typedef struct ImputedData {
    std::vector<long double>  num_sum;
    std::vector<long double>  num_weight;
    std::vector<std::vector<long double>> cat_sum;
    std::vector<long double>  cat_weight;
    std::vector<long double>  sp_num_sum;
    std::vector<long double>  sp_num_weight;

    std::vector<size_t>     missing_num;
    std::vector<size_t>     missing_cat;
    std::vector<sparse_ix>  missing_sp;
    size_t                  n_missing_num;
    size_t                  n_missing_cat;
    size_t                  n_missing_sp;

    ImputedData() {};

    ImputedData(InputData &input_data, size_t row);

} ImputedData;

typedef struct {
    std::vector<size_t>  ix_arr;
    std::vector<size_t>  ix_all;
    RNG_engine           rnd_generator;
    std::uniform_int_distribution<size_t>  runif;
    std::uniform_real_distribution<double> rbin;
    size_t               st;
    size_t               end;
    size_t               st_NA;
    size_t               end_NA;
    size_t               split_ix;
    std::unordered_map<size_t, double> weights_map;
    std::vector<double>  weights_arr;    /* when not ignoring NAs and when using weights as density */
    double               xmin;
    double               xmax;
    size_t               npresent;       /* 'npresent' and 'ncols_tried' are used interchangeable and for unrelated things */
    bool                 unsplittable;
    std::vector<bool>    is_repeated;
    std::vector<char>    categs;
    size_t               ncols_tried;    /* 'npresent' and 'ncols_tried' are used interchangeable and for unrelated things */
    int                  ncat_tried;
    std::vector<bool>    cols_possible;
    std::vector<double>  btree_weights;  /* only when using weights for sampling */
    std::discrete_distribution<size_t> col_sampler; /* columns can get eliminated, keep a copy for each thread */

    /* for split criterion */
    std::vector<double>  buffer_dbl;
    std::vector<size_t>  buffer_szt;
    std::vector<char>    buffer_chr;
    double               prob_split_type;
    GainCriterion        criterion;
    double               this_gain;
    double               this_split_point;
    int                  this_categ;
    std::vector<char>    this_split_categ;
    bool                 determine_split;

    /* for the extended model */
    size_t   ntry;
    size_t   ntaken;
    size_t   ntaken_best;
    bool     tried_all;
    size_t   col_chosen;
    ColType  col_type;
    double   ext_sd;
    std::vector<size_t>  cols_shuffled;
    std::vector<double>  comb_val;
    std::vector<size_t>  col_take;
    std::vector<ColType> col_take_type;
    std::vector<double>  ext_offset;
    std::vector<double>  ext_coef;
    std::vector<double>  ext_mean;
    std::vector<double>  ext_fill_val;
    std::vector<double>  ext_fill_new;
    std::vector<int>     chosen_cat;
    std::vector<std::vector<double>>       ext_cat_coef;
    std::uniform_real_distribution<double> coef_unif;
    std::normal_distribution<double>       coef_norm;

    /* for similarity/distance calculations */
    std::vector<double> tmat_sep;

    /* when calculating average depth on-the-fly */
    std::vector<double> row_depths;

    /* when imputing NAs on-the-fly */
    std::vector<ImputedData> impute_vec;
    std::unordered_map<size_t, ImputedData> impute_map;

} WorkerMemory;

typedef struct WorkerForSimilarity {
    std::vector<size_t> ix_arr;
    size_t              st;
    size_t              end;
    std::vector<double> weights_arr;
    std::vector<double> comb_val;
    std::vector<double> tmat_sep;
    std::vector<double> rmat;
    size_t              n_from;
    bool                assume_full_distr; /* doesn't need to have one copy per worker */
} WorkerForSimilarity;

typedef struct {
    size_t  st;
    size_t  st_NA;
    size_t  end_NA;
    size_t  split_ix;
    size_t  end;
    std::vector<size_t> ix_arr;
    std::unordered_map<size_t, double> weights_map;
    std::vector<double> weights_arr;
    std::vector<bool>   cols_possible;
    std::discrete_distribution<size_t> col_sampler;
} RecursionState;

/* Function prototypes */

/* fit_model.cpp */
extern bool interrupt_switch;
int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                double numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                double sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees, size_t max_depth,
                bool   limit_depth, bool penalize_range,
                bool   standardize_dist, double tmat[],
                double output_depths[], bool standardize_depth,
                double col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, int nthreads);
int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
             double numeric_data[],  size_t ncols_numeric,
             int    categ_data[],    size_t ncols_categ,    int ncat[],
             double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
             double sample_weights[], size_t nrows, size_t max_depth,
             bool   limit_depth,   bool penalize_range,
             double col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
             double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, std::vector<ImputeNode> *impute_nodes, size_t min_imp_obs,
             uint64_t random_seed);
void fit_itree(std::vector<IsoTree>    *tree_root,
               std::vector<IsoHPlane>  *hplane_root,
               WorkerMemory             &workspace,
               InputData                &input_data,
               ModelParams              &model_params,
               std::vector<ImputeNode> *impute_nodes,
               size_t                   tree_num);

/* isoforest.cpp */
void split_itree_recursive(std::vector<IsoTree>     &trees,
                           WorkerMemory             &workspace,
                           InputData                &input_data,
                           ModelParams              &model_params,
                           std::vector<ImputeNode> *impute_nodes,
                           size_t                   curr_depth);

/* extended.cpp */
void split_hplane_recursive(std::vector<IsoHPlane>   &hplanes,
                            WorkerMemory             &workspace,
                            InputData                &input_data,
                            ModelParams              &model_params,
                            std::vector<ImputeNode> *impute_nodes,
                            size_t                   curr_depth);
void add_chosen_column(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params,
                       std::vector<bool> &col_is_taken, std::unordered_set<size_t> &col_is_taken_s);
void shrink_to_fit_hplane(IsoHPlane &hplane, bool clear_vectors);
void simplify_hplane(IsoHPlane &hplane, WorkerMemory &workspace, InputData &input_data, ModelParams &model_params);


/* predict.cpp */
void predict_iforest(double numeric_data[], int categ_data[],
                     double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     double Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[]);
void traverse_itree_no_recurse(std::vector<IsoTree>  &tree,
                               IsoForest             &model_outputs,
                               PredictionData        &prediction_data,
                               double                &output_depth,
                               sparse_ix *restrict   tree_num,
                               size_t                row);
double traverse_itree(std::vector<IsoTree>     &tree,
                      IsoForest                &model_outputs,
                      PredictionData           &prediction_data,
                      std::vector<ImputeNode> *impute_nodes,
                      ImputedData             *imputed_data,
                      double                   curr_weight,
                      size_t                   row,
                      sparse_ix *restrict      tree_num,
                      size_t                   curr_lev);
void traverse_hplane_fast(std::vector<IsoHPlane>  &hplane,
                          ExtIsoForest            &model_outputs,
                          PredictionData          &prediction_data,
                          double                  &output_depth,
                          sparse_ix *restrict     tree_num,
                          size_t                  row);
void traverse_hplane(std::vector<IsoHPlane>   &hplane,
                     ExtIsoForest             &model_outputs,
                     PredictionData           &prediction_data,
                     double                   &output_depth,
                     std::vector<ImputeNode> *impute_nodes,
                     ImputedData             *imputed_data,
                     sparse_ix *restrict      tree_num,
                     size_t                   row);
double extract_spC(PredictionData &prediction_data, size_t row, size_t col_num);
double extract_spR(PredictionData &prediction_data, sparse_ix *row_st, sparse_ix *row_end, size_t col_num);
void get_num_nodes(IsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads);
void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads);

/* dist.cpp */
void calc_similarity(double numeric_data[], int categ_data[],
                     double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, int nthreads, bool assume_full_distr, bool standardize_dist,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from);
void traverse_tree_sim(WorkerForSimilarity   &workspace,
                       PredictionData        &prediction_data,
                       IsoForest             &model_outputs,
                       std::vector<IsoTree>  &trees,
                       size_t                curr_tree);
void traverse_hplane_sim(WorkerForSimilarity     &workspace,
                         PredictionData          &prediction_data,
                         ExtIsoForest            &model_outputs,
                         std::vector<IsoHPlane>  &hplanes,
                         size_t                  curr_tree);
void gather_sim_result(std::vector<WorkerForSimilarity> *worker_memory,
                       std::vector<WorkerMemory> *worker_memory_m,
                       PredictionData *prediction_data, InputData *input_data,
                       IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                       double *restrict tmat, double *restrict rmat, size_t n_from,
                       size_t ntrees, bool assume_full_distr,
                       bool standardize_dist, int nthreads);
void initialize_worker_for_sim(WorkerForSimilarity  &workspace,
                               PredictionData       &prediction_data,
                               IsoForest            *model_outputs,
                               ExtIsoForest         *model_outputs_ext,
                               size_t                n_from,
                               bool                  assume_full_distr);

/* impute.cpp */
void impute_missing_values(double numeric_data[], int categ_data[],
                           double Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer);
void initialize_imputer(Imputer &imputer, InputData &input_data, size_t ntrees, int nthreads);
void build_impute_node(ImputeNode &imputer,    WorkerMemory &workspace,
                       InputData  &input_data, ModelParams  &model_params,
                       std::vector<ImputeNode> &imputer_tree,
                       size_t curr_depth, size_t min_imp_obs);
void shrink_impute_node(ImputeNode &imputer);
void drop_nonterminal_imp_node(std::vector<ImputeNode>  &imputer_tree,
                               std::vector<IsoTree>     *trees,
                               std::vector<IsoHPlane>   *hplanes);
void combine_imp_single(ImputedData &imp_addfrom, ImputedData &imp_addto);
void combine_tree_imputations(WorkerMemory &workspace,
                              std::vector<ImputedData> &impute_vec,
                              std::unordered_map<size_t, ImputedData> &impute_map,
                              std::vector<char> &has_missing,
                              int nthreads);
void add_from_impute_node(ImputeNode &imputer, ImputedData &imputed_data, double w);
void add_from_impute_node(ImputeNode &imputer, WorkerMemory &workspace, InputData &input_data);
template <class imp_arr>
void apply_imputation_results(imp_arr    &impute_vec,
                              Imputer    &imputer,
                              InputData  &input_data,
                              int        nthreads);
void apply_imputation_results(std::vector<ImputedData> &impute_vec,
                              std::unordered_map<size_t, ImputedData> &impute_map,
                              Imputer   &imputer,
                              InputData &input_data,
                              int nthreads);
void apply_imputation_results(PredictionData  &prediction_data,
                              ImputedData     &imp,
                              Imputer         &imputer,
                              size_t          row);
void initialize_impute_calc(ImputedData &imp, InputData &input_data, size_t row);
void initialize_impute_calc(ImputedData &imp, PredictionData &prediction_data, Imputer &imputer, size_t row);
void allocate_imp_vec(std::vector<ImputedData> &impute_vec, InputData &input_data, int nthreads);
void allocate_imp_map(std::unordered_map<size_t, ImputedData> &impute_map, InputData &input_data);
void allocate_imp(InputData &input_data,
                  std::vector<ImputedData> &impute_vec,
                  std::unordered_map<size_t, ImputedData> &impute_map,
                  int nthreads);
void check_for_missing(InputData &input_data,
                       std::vector<ImputedData> &impute_vec,
                       std::unordered_map<size_t, ImputedData> &impute_map,
                       int nthreads);
size_t check_for_missing(PredictionData  &prediction_data,
                         Imputer         &imputer,
                         size_t          ix_arr[],
                         int             nthreads);

/* helpers_iforest.cpp */
void decide_column(size_t ncols_numeric, size_t ncols_categ, size_t &col_chosen, ColType &col_type,
                   RNG_engine &rnd_generator, std::uniform_int_distribution<size_t> &runif,
                   std::discrete_distribution<size_t> &col_sampler);
void add_unsplittable_col(WorkerMemory &workspace, IsoTree &tree, InputData &input_data);
void add_unsplittable_col(WorkerMemory &workspace, InputData &input_data);
bool check_is_not_unsplittable_col(WorkerMemory &workspace, IsoTree &tree, InputData &input_data);
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params, IsoTree &tree);
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params);
int choose_cat_from_present(WorkerMemory &workspace, InputData &input_data, size_t col_num);
void update_col_sampler(WorkerMemory &workspace, InputData &input_data);
bool is_col_taken(std::vector<bool> &col_is_taken, std::unordered_set<size_t> &col_is_taken_s,
                  InputData &input_data, size_t col_num, ColType col_type);
void set_col_as_taken(std::vector<bool> &col_is_taken, std::unordered_set<size_t> &col_is_taken_s,
                      InputData &input_data, size_t col_num, ColType col_type);
void add_separation_step(WorkerMemory &workspace, InputData &input_data, double remainder);
void add_remainder_separation_steps(WorkerMemory &workspace, InputData &input_data, long double sum_weight);
void remap_terminal_trees(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                          PredictionData &prediction_data, sparse_ix *restrict tree_num, int nthreads);
void backup_recursion_state(WorkerMemory &workspace, RecursionState &recursion_state);
void restore_recursion_state(WorkerMemory &workspace, RecursionState &recursion_state);


/* utils.cpp */
size_t log2ceil(size_t x);
double harmonic(size_t n);
double harmonic_recursive(double a, double b);
double expected_avg_depth(size_t sample_size);
double expected_avg_depth(long double approx_sample_size);
double expected_separation_depth(size_t n);
double expected_separation_depth_hotstart(double curr, size_t n_curr, size_t n_final);
double expected_separation_depth(long double n);
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n, double counter[], double exp_remainder);
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double *restrict counter, double *restrict weights, double exp_remainder);
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double counter[], std::unordered_map<size_t, double> &weights, double exp_remainder);
void increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end, size_t split_ix, size_t n,
                                     double counter[], double exp_remainder);
void increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end, size_t split_ix, size_t n,
                                     double *restrict counter, double *restrict weights, double exp_remainder);
void tmat_to_dense(double *restrict tmat, double *restrict dmat, size_t n, bool diag_to_one);
double calc_sd_raw(size_t cnt, long double sum, long double sum_sq);
long double calc_sd_raw_l(size_t cnt, long double sum, long double sum_sq);
void build_btree_sampler(std::vector<double> &btree_weights, double *restrict sample_weights,
                         size_t nrows, size_t &log2_n, size_t &btree_offset);
void sample_random_rows(std::vector<size_t> &ix_arr, size_t nrows, bool with_replacement,
                        RNG_engine &rnd_generator, std::vector<size_t> &ix_all,
                        double sample_weights[], std::vector<double> &btree_weights,
                        size_t log2_n, size_t btree_offset, std::vector<bool> &is_repeated);
void weighted_shuffle(size_t *restrict outp, size_t n, double *restrict weights, double *restrict buffer_arr, RNG_engine &rnd_generator);
size_t divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point);
void divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point,
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix);
void divide_subset_split(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                         double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[], double split_point,
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix);
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, char split_categ[],
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix);
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, char split_categ[],
                         int ncat, MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &st_NA, size_t &end_NA, size_t &split_ix);
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, int split_categ,
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix);
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end,
                         MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &st_NA, size_t &end_NA, size_t &split_ix);
void get_range(size_t ix_arr[], double x[], size_t st, size_t end,
               MissingAction missing_action, double &xmin, double &xmax, bool &unsplittable);
void get_range(size_t ix_arr[], size_t st, size_t end, size_t col_num,
               double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
               MissingAction missing_action, double &xmin, double &xmax, bool &unsplittable);
void get_categs(size_t ix_arr[], int x[], size_t st, size_t end, int ncat,
                MissingAction missing_action, char categs[], size_t &npresent, bool &unsplittable);
long double calculate_sum_weights(std::vector<size_t> &ix_arr, size_t st, size_t end, size_t curr_depth,
                                  std::vector<double> &weights_arr, std::unordered_map<size_t, double> &weights_map);
void set_interrup_global_variable(int s);
int return_EXIT_SUCCESS();
int return_EXIT_FAILURE();



size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, double x[]);
size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, size_t col_num, double Xc[], size_t Xc_ind[], size_t Xc_indptr[]);
size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, int x[]);
size_t center_NAs(size_t *restrict ix_arr, size_t st_left, size_t st, size_t curr_pos);
void todense(size_t ix_arr[], size_t st, size_t end,
             size_t col_num, double *restrict Xc, sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             double *restrict buffer_arr);

/* mult.cpp */
void calc_mean_and_sd(size_t ix_arr[], size_t st, size_t end, double *restrict x,
                      MissingAction missing_action, double &x_sd, double &x_mean);
void calc_mean_and_sd(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                      double *restrict Xc, sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                      double &x_sd, double &x_mean);
void add_linear_comb(size_t ix_arr[], size_t st, size_t end, double *restrict res,
                     double *restrict x, double &coef, double x_sd, double x_mean, double &fill_val,
                     MissingAction missing_action, double *restrict buffer_arr,
                     size_t *restrict buffer_NAs, bool first_run);
void add_linear_comb(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, double *restrict res,
                     double *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     double &coef, double x_sd, double x_mean, double &fill_val, MissingAction missing_action,
                     double *restrict buffer_arr, size_t *restrict buffer_NAs, bool first_run);
void add_linear_comb(size_t *restrict ix_arr, size_t st, size_t end, double *restrict res,
                     int x[], int ncat, double *restrict cat_coef, double single_cat_coef, int chosen_cat,
                     double &fill_val, double &fill_new, size_t *restrict buffer_cnt, size_t *restrict buffer_pos,
                     NewCategAction new_cat_action, MissingAction missing_action, CategSplit cat_split_type, bool first_run);

/* crit.cpp */
double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, double x[], MissingAction missing_action);
double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                     double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     MissingAction missing_action);
double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, int x[], int ncat, size_t buffer_cnt[], double buffer_prob[],
                     MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator);
double expected_sd_cat(double p[], size_t n, size_t pos[]);
double expected_sd_cat(size_t counts[], double p[], size_t n, size_t pos[]);
double expected_sd_cat_single(size_t counts[], double p[], size_t n, size_t pos[], size_t cat_exclude, size_t cnt);
double numeric_gain(size_t cnt_left, size_t cnt_right,
                    long double sum_left, long double sum_right,
                    long double sum_sq_left, long double sum_sq_right,
                    double sd_full, long double cnt);
double numeric_gain_no_div(size_t cnt_left, size_t cnt_right,
                           long double sum_left, long double sum_right,
                           long double sum_sq_left, long double sum_sq_right,
                           double sd_full, long double cnt);
double categ_gain(size_t cnt_left, size_t cnt_right,
                  long double s_left, long double s_right,
                  long double base_info, long double cnt);
double eval_guided_crit(double *restrict x, size_t n, GainCriterion criterion, double min_gain,
                        double &split_point, double &xmin, double &xmax);
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, double *restrict x,
                        size_t &split_ix, double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action);
double eval_guided_crit(size_t ix_arr[], size_t st, size_t end,
                        size_t col_num, double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                        double buffer_arr[], size_t buffer_pos[],
                        double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action);
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int ncat,
                        size_t *restrict buffer_cnt, size_t *restrict buffer_pos, double *restrict buffer_prob,
                        int &chosen_cat, char *restrict split_categ, char *restrict buffer_split,
                        GainCriterion criterion, double min_gain, bool all_perm, MissingAction missing_action, CategSplit cat_split_type);

/* merge_models.cpp */
void merge_models(IsoForest*     model,      IsoForest*     other,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                  Imputer*       imputer,    Imputer*       iother);

#ifdef _ENABLE_CEREAL
/* serialize.cpp */
void serialize_isoforest(IsoForest &model, std::ostream &output);
void serialize_isoforest(IsoForest &model, const char *output_file_path);
std::string serialize_isoforest(IsoForest &model);
void deserialize_isoforest(IsoForest &output_obj, std::istream &serialized);
void deserialize_isoforest(IsoForest &output_obj, const char *input_file_path);
void deserialize_isoforest(IsoForest &output_obj, std::string &serialized, bool move_str);
void serialize_ext_isoforest(ExtIsoForest &model, std::ostream &output);
void serialize_ext_isoforest(ExtIsoForest &model, const char *output_file_path);
std::string serialize_ext_isoforest(ExtIsoForest &model);
void deserialize_ext_isoforest(ExtIsoForest &output_obj, std::istream &serialized);
void deserialize_ext_isoforest(ExtIsoForest &output_obj, const char *input_file_path);
void deserialize_ext_isoforest(ExtIsoForest &output_obj, std::string &serialized, bool move_str);
void serialize_imputer(Imputer &imputer, std::ostream &output);
void serialize_imputer(Imputer &imputer, const char *output_file_path);
std::string serialize_imputer(Imputer &imputer);
void deserialize_imputer(Imputer &output_obj, std::istream &serialized);
void deserialize_imputer(Imputer &output_obj, const char *input_file_path);
void deserialize_imputer(Imputer &output_obj, std::string &serialized, bool move_str);
#ifdef _MSC_VER
void serialize_isoforest(IsoForest &model, const wchar_t *output_file_path);
void deserialize_isoforest(IsoForest &output_obj, const wchar_t *input_file_path);
void serialize_ext_isoforest(ExtIsoForest &model, const wchar_t *output_file_path);
void deserialize_ext_isoforest(ExtIsoForest &output_obj, const wchar_t *input_file_path);
void serialize_imputer(Imputer &imputer, const wchar_t *output_file_path);
void deserialize_imputer(Imputer &output_obj, const wchar_t *input_file_path);
#endif /* _MSC_VER */
bool has_msvc();
#endif /* _ENABLE_CEREAL */

/* dealloc.cpp */
void dealloc_IsoForest(IsoForest &model_outputs);
void dealloc_IsoExtForest(ExtIsoForest &model_outputs_ext);
void dealloc_Imputer(Imputer &imputer);
