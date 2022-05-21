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

#ifndef ISOTREE_H
#define ISOTREE_H

/* This is only used for the serialiation format and might not reflect the
   actual version of the library, do not use for anything else. */
#define ISOTREE_VERSION_MAJOR 0
#define ISOTREE_VERSION_MINOR 5
#define ISOTREE_VERSION_PATCH 6

/* For MinGW, needs to be defined before including any headers */
#if (defined(_WIN32) || defined(_WIN64)) && (SIZE_MAX >= UINT64_MAX)
#   if defined(__GNUG__) || defined(__GNUC__)
#       ifndef _FILE_OFFSET_BITS
#           define _FILE_OFFSET_BITS 64
#       endif
#   endif
#endif
#ifdef _MSC_VER
#   define _CRT_SECURE_NO_WARNINGS
#endif


/* Standard headers */
#include <cstddef>
#include <cmath>
#include <climits>
#include <cstring>
#include <cerrno>
#include <vector>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <random>
#include <memory>
#include <utility>
#include <cstdint>
#include <cinttypes>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <string>

#ifdef _FOR_R
    extern "C" {
        #include <R_ext/Print.h>
    }
    #define printf Rprintf
    #define fprintf(f, message) REprintf(message)
#elif defined(_FOR_PYTHON)
    extern "C" void cy_warning(const char *msg);
    #define fprintf(f, message) cy_warning(message)
#else
    #include <cstdio>
    using std::printf;
    using std::fprintf;
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif
#ifdef _FOR_R
    #include <Rcpp.h>
#endif
#include <csignal>
typedef void (*sig_t_)(int);
using std::signal;
using std::raise;

using std::size_t;
using std::memset;
using std::memcpy;

#if defined(__GNUC__) || defined(__clang__)
    #define likely(x) __builtin_expect((bool)(x), true)
    #define unlikely(x) __builtin_expect((bool)(x), false)
#else
    #define likely(x) (x)
    #define unlikely(x) (x)
#endif

#if defined(__GNUC__)  || defined(__clang__) || defined(_MSC_VER)
    #define unexpected_error() throw std::runtime_error(\
        std::string("Unexpected error in ") + \
        std::string(__FILE__) + \
        std::string(":") + \
        std::to_string(__LINE__) + \
        std::string(". Please open an issue in GitHub with this information, indicating the installed version of 'isotree'.\n"))
#else
    #define unexpected_error() throw std::runtime_error("Unexpected error. Please open an issue in GitHub.\n")
#endif

/* By default, will use Xoshiro256++ or Xoshiro128++ for RNG, but can be switched to something faster */
#ifdef _USE_XOSHIRO
    #include "xoshiro.hpp"
    #if SIZE_MAX >= UINT64_MAX /* 64-bit systems or higher */
        #define RNG_engine Xoshiro::Xoshiro256PP
    #else /* 32-bit systems and non-standard architectures */
        #define RNG_engine Xoshiro::Xoshiro128PP
    #endif
    #if defined(DBL_MANT_DIG) && (DBL_MANT_DIG == 53) && (FLT_RADIX == 2)
        using Xoshiro::UniformUnitInterval;
        using Xoshiro::UniformMinusOneToOne;
        using Xoshiro::StandardNormalDistr;
    #else
        #define UniformUnitInterval std::uniform_real_distribution<double>
        #define UniformMinusOneToOne std::uniform_real_distribution<double>
        #define StandardNormalDistr std::normal_distribution<double>
    #endif
#else
    #if defined(_USE_MERSENNE_TWISTER)
        #if SIZE_MAX >= UINT64_MAX /* 64-bit systems or higher */
            #define RNG_engine std::mt19937_64
        #else /* 32-bit systems and non-standard architectures */
            #define RNG_engine std::mt19937
        #endif
    #else
        #define RNG_engine std::default_random_engine
    #endif

    #define UniformUnitInterval std::uniform_real_distribution<double>
    #define UniformMinusOneToOne std::uniform_real_distribution<double>
    #define StandardNormalDistr std::normal_distribution<double>
#endif

/* At the time of writing, this brought a sizeable speed up compared to
   'unordered_map' and 'unordered_set' from both GCC and CLANG.
   But perhaps should consider others in the future, such as this:
   https://github.com/ktprime/emhash  */
#if defined(_USE_ROBIN_MAP)
    #ifndef _USE_SYSTEM_ROBIN
        #include "robinmap/include/tsl/robin_growth_policy.h"
        #include "robinmap/include/tsl/robin_hash.h"
        #include "robinmap/include/tsl/robin_set.h"
        #include "robinmap/include/tsl/robin_map.h"
    #else
        #include "tsl/robin_growth_policy.h"
        #include "tsl/robin_hash.h"
        #include "tsl/robin_set.h"
        #include "tsl/robin_map.h"
    #endif
    #define hashed_set tsl::robin_set
    #define hashed_map tsl::robin_map
#else
    #include <unordered_set>
    #include <unordered_map>
    #define hashed_set std::unordered_set
    #define hashed_map std::unordered_map
#endif

/* Short functions */
/* https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int */
#define pow2(n) ( ((size_t) 1) << (n) )
#define div2(n) ((n) >> 1)
#define mult2(n) ((n) << 1)
#define ix_parent(ix) (div2((ix) - (size_t)1))  /* integer division takes care of deciding left-right */
#define ix_child(ix)  (mult2(ix) + (size_t)1)
#define square(x) ((x) * (x))
#ifndef _FOR_R
    #if defined(__GNUC__) && (__GNUC__ >= 5)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    #elif defined(__clang__) && !defined(_FOR_R)
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wuninitialized"
    #endif
#endif
/* https://stackoverflow.com/questions/2249731/how-do-i-get-bit-by-bit-data-from-an-integer-value-in-c */
#define extract_bit(number, bit) (((number) >> (bit)) & 1)
#ifndef _FOR_R
    #if defined(__GNUC__) && (__GNUC__ >= 5)
        #pragma GCC diagnostic pop
    #elif defined(__clang__)
        #pragma clang diagnostic pop
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wunknown-attributes"
    #endif
#endif
#define is_na_or_inf(x) (std::isnan(x) || std::isinf(x))

/* MSVC doesn't support long doubles, so this avoids unnecessarily increasing library size.
   MinGW supports them but has issues with their computations.
   See https://sourceforge.net/p/mingw-w64/bugs/909/ */
#if defined(_WIN32) && !defined(NO_LONG_DOUBLE)
    #define NO_LONG_DOUBLE
#endif


/* Aliasing for compiler optimizations */
#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(__IBMCPP__) || defined(__ibmxl__) || defined(SUPPORTS_RESTRICT)
    #define restrict __restrict
#else
    #define restrict 
#endif

/* MSVC is stuck with an OpenMP version that's 19 years old at the time of writing and does not support unsigned iterators */
#ifdef _OPENMP
    #if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
        #define size_t_for long long
    #else
        #define size_t_for size_t
    #endif
#else
    #define size_t_for size_t
#endif

#if defined(_FOR_R) || defined(_FOR_PYTHON)
    #define ISOTREE_EXPORTED 
#else
    #if defined(_WIN32)
        #ifdef ISOTREE_COMPILE_TIME
            #define ISOTREE_EXPORTED __declspec(dllexport)
        #else
            #define ISOTREE_EXPORTED __declspec(dllimport)
        #endif
    #else
        #if defined(EXPLICITLTY_EXPORT_SYMBOLS) && defined(ISOTREE_COMPILE_TIME)
            #define ISOTREE_EXPORTED [[gnu::visibility("default")]]
        #else
            #define ISOTREE_EXPORTED 
        #endif
    #endif
#endif


/*   Apple at some point decided to drop OMP library and headers from its compiler distribution
*    and to alias 'gcc' to 'clang', which work differently when given flags they cannot interpret,
*    causing installation issues with pretty much all scientific software due to OMP headers that
*    would normally do nothing. This piece of code is to allow compilation without OMP header. */
#ifndef _OPENMP
    #define omp_get_thread_num() (0)
#endif

/* Some aggregation functions will prefer more precise data types when the data is large */
#define THRESHOLD_LONG_DOUBLE (size_t)1e6

/* Types used through the package */
typedef enum  NewCategAction {Weighted=0,  Smallest=11,    Random=12}  NewCategAction; /* Weighted means Impute in the extended model */
typedef enum  MissingAction  {Divide=21,   Impute=22,      Fail=0}     MissingAction;  /* Divide is only for non-extended model */
typedef enum  ColType        {Numeric=31,  Categorical=32, NotUsed=0}  ColType;
typedef enum  CategSplit     {SubSet=0,    SingleCateg=41}             CategSplit;
typedef enum  CoefType       {Uniform=61,  Normal=0}                   CoefType;       /* For extended model */
typedef enum  UseDepthImp    {Lower=71,    Higher=0,       Same=72}    UseDepthImp;    /* For NA imputation */
typedef enum  WeighImpRows   {Inverse=0,   Prop=81,        Flat=82}    WeighImpRows;   /* For NA imputation */
typedef enum  ScoringMetric  {Depth=0,     Density=92,     BoxedDensity=94, BoxedDensity2=96, BoxedRatio=95,
                              AdjDepth=91, AdjDensity=93}              ScoringMetric;

/* These are only used internally */
typedef enum  ColCriterion   {Uniformly=0, ByRange=1, ByVar=2, ByKurt=3} ColCriterion;   /* For proportional choices */
typedef enum  GainCriterion  {NoCrit=0, Averaged=1, Pooled=2, FullGain=3, DensityCrit=4} Criterion; /* For guided splits */


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
    ColType  col_type = NotUsed; /* issues with uninitialized values when serializing */
    size_t   col_num;
    double   num_split;
    std::vector<signed char> cat_split;
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
    std::vector<double>   fill_new; /* <- when using single categ, coef will be here */

    double   split_point;
    size_t   hplane_left;
    size_t   hplane_right;
    double   score;        /* will not be integer when there are weights or early stop */
    double   range_low  = -HUGE_VAL;
    double   range_high =  HUGE_VAL;
    double   remainder; /* only used for distance/similarity */

    IsoHPlane() = default;
} IsoHPlane;

/* Note: don't use long doubles in the outside outputs or there will be issues with MINGW in windows */


typedef struct IsoForest {
    std::vector< std::vector<IsoTree> > trees;
    NewCategAction    new_cat_action;
    CategSplit        cat_split_type;
    MissingAction     missing_action;
    ScoringMetric     scoring_metric;
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
    ScoringMetric     scoring_metric;
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


/* Structs that are only used internally */
template <class real_t, class sparse_ix>
struct InputData {
    real_t*     numeric_data;
    size_t      ncols_numeric;
    int*        categ_data;
    int*        ncat;
    int         max_categ;
    size_t      ncols_categ;
    size_t      nrows;
    size_t      ncols_tot;
    real_t*     sample_weights;
    bool        weight_as_sample;
    real_t*     col_weights;
    real_t*     Xc;           /* only for sparse matrices */
    sparse_ix*  Xc_ind;       /* only for sparse matrices */
    sparse_ix*  Xc_indptr;    /* only for sparse matrices */
    size_t      log2_n;       /* only when using weights for sampling */
    size_t      btree_offset; /* only when using weights for sampling */
    std::vector<double> btree_weights_init;  /* only when using weights for sampling */
    std::vector<char>   has_missing;         /* only used when producing missing imputations on-the-fly */
    size_t              n_missing;           /* only used when producing missing imputations on-the-fly */
    void*       preinitialized_col_sampler;  /* only when using column weights */
    double*     range_low;   /* only when calculating variable ranges or boxed densities with no sub-sampling */
    double*     range_high;  /* only when calculating variable ranges or boxed densities with no sub-sampling */
    int*        ncat_;       /* only when calculating boxed densities with no sub-sampling */
    std::vector<double> all_kurtoses; /* only when using 'prob_pick_col_by_kurtosis' or mixing 'weigh_by_kurt' with 'prob_pick_col*' with no sub-sampling */

    std::vector<double>  X_row_major; /* created by this library, only used when calculating full gain */
    std::vector<double>  Xr;          /* created by this library, only used when calculating full gain */
    std::vector<size_t>  Xr_ind;      /* created by this library, only used when calculating full gain */
    std::vector<size_t>  Xr_indptr;   /* created by this library, only used when calculating full gain */
};


template <class real_t, class sparse_ix>
struct PredictionData {
    real_t*     numeric_data;
    int*        categ_data;
    size_t      nrows;
    bool        is_col_major;
    size_t      ncols_numeric; /* only required for row-major data */
    size_t      ncols_categ;   /* only required for row-major data */
    real_t*     Xc;            /* only for sparse matrices */
    sparse_ix*  Xc_ind;        /* only for sparse matrices */
    sparse_ix*  Xc_indptr;     /* only for sparse matrices */
    real_t*     Xr;            /* only for sparse matrices */
    sparse_ix*  Xr_ind;        /* only for sparse matrices */
    sparse_ix*  Xr_indptr;     /* only for sparse matrices */
};

typedef struct {
    bool      with_replacement;
    size_t    sample_size;
    size_t    ntrees;
    size_t    ncols_per_tree;
    size_t    max_depth;
    bool      penalize_range;
    bool      standardize_data;
    uint64_t  random_seed;
    bool      weigh_by_kurt;
    double    prob_pick_by_gain_avg;
    double    prob_pick_by_gain_pl;
    double    prob_pick_by_full_gain;
    double    prob_pick_by_dens;
    double    prob_pick_col_by_range;
    double    prob_pick_col_by_var;
    double    prob_pick_col_by_kurt;
    double    min_gain;
    CategSplit      cat_split_type;
    NewCategAction  new_cat_action;
    MissingAction   missing_action;
    ScoringMetric   scoring_metric;
    bool            fast_bratio;
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
    size_t        min_imp_obs;    /* only when building NA imputer */
} ModelParams;

template <class sparse_ix, class ldouble_safe>
struct ImputedData {
    std::vector<ldouble_safe>  num_sum;
    std::vector<ldouble_safe>  num_weight;
    std::vector<std::vector<ldouble_safe>> cat_sum;
    std::vector<ldouble_safe>  cat_weight;
    std::vector<ldouble_safe>  sp_num_sum;
    std::vector<ldouble_safe>  sp_num_weight;

    std::vector<size_t>     missing_num;
    std::vector<size_t>     missing_cat;
    std::vector<sparse_ix>  missing_sp;
    size_t                  n_missing_num;
    size_t                  n_missing_cat;
    size_t                  n_missing_sp;

    ImputedData() {};

    template <class InputData>
    ImputedData(InputData &input_data, size_t row)
    {
        initialize_impute_calc(*this, input_data, row);
    }

};

/*  This class provides efficient methods for sampling columns at random,
    given that at a given node a column might no longer be splittable,
    and when that happens, it also makes it non-splittable in any children
    node from there onwards. The idea is to provide efficient methods for
    passing the state from a parent node to a left node and then restore
    the state before going for the right node.
    It can be used in 3 modes:
    - As a uniform sampler with replacement.
    - As a weighted sampler with replacement.
    - As an array that keeps track of which columns are still splittable. */
template <class ldouble_safe>
class ColumnSampler
{
public:
    std::vector<size_t> col_indices;
    std::vector<double> tree_weights;
    size_t curr_pos;
    size_t curr_col;
    size_t last_given;
    size_t n_cols;
    size_t tree_levels;
    size_t offset;
    size_t n_dropped;
    template <class real_t>
    void initialize(real_t weights[], size_t n_cols);
    void initialize(size_t n_cols);
    void drop_weights();
    void leave_m_cols(size_t m, RNG_engine &rnd_generator);
    bool sample_col(size_t &col, RNG_engine &rnd_generator);
    void prepare_full_pass();        /* when passing through all columns */
    bool sample_col(size_t &col); /* when passing through all columns */
    void drop_col(size_t col, size_t nobs_left);
    void drop_col(size_t col);
    void drop_from_tail(size_t col);
    void shuffle_remainder(RNG_engine &rnd_generator);
    bool has_weights();
    size_t get_remaining_cols();
    void get_array_remaining_cols(std::vector<size_t> &restrict cols);
    template <class other_t>
    ColumnSampler& operator=(const ColumnSampler<other_t> &other);
    ColumnSampler() = default;
};

template <class ldouble_safe, class real_t>
class DensityCalculator
{
public:
    std::vector<ldouble_safe> multipliers;
    double xmin;
    double xmax;
    std::vector<size_t> counts;
    int n_present;
    int n_left;
    std::vector<double> box_low;
    std::vector<double> box_high;
    std::vector<double> queue_box;
    bool fast_bratio;
    std::vector<ldouble_safe> ranges;
    std::vector<int> ncat;
    std::vector<int> queue_ncat;
    std::vector<int> ncat_orig;
    std::vector<double> vals_ext_box;
    std::vector<double> queue_ext_box;
    
    void initialize(size_t max_depth, int max_categ, bool reserve_counts, ScoringMetric scoring_metric);
    template <class InputData>
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    void initialize_bdens(const InputData &input_data,
                          const ModelParams &model_params,
                          std::vector<size_t> &ix_arr,
                          ColumnSampler<ldouble_safe> &col_sampler);
    template <class InputData>
    void initialize_bdens_ext(const InputData &input_data,
                              const ModelParams &model_params,
                              std::vector<size_t> &ix_arr,
                              ColumnSampler<ldouble_safe> &col_sampler,
                              bool col_sampler_is_fresh);
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    void push_density(double xmin, double xmax, double split_point);
    void push_density(size_t counts[], int ncat);
    void push_density(int n_left, int n_present);
    void push_density(int n_present);
    void push_density();
    void push_adj(double xmin, double xmax, double split_point, double pct_tree_left, ScoringMetric scoring_metric);
    void push_adj(signed char *restrict categ_present, size_t *restrict counts, int ncat, ScoringMetric scoring_metric);
    void push_adj(size_t *restrict counts, int ncat, int chosen_cat, ScoringMetric scoring_metric);
    void push_adj(double pct_tree_left, ScoringMetric scoring_metric);
    void push_bdens(double split_point, size_t col);
    void push_bdens(int ncat_branch_left, size_t col);
    void push_bdens(const std::vector<signed char> &cat_split, size_t col);
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    void push_bdens_fast_route(double split_point, size_t col);
    void push_bdens_internal(double split_point, size_t col);
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    void push_bdens_fast_route(int ncat_branch_left, size_t col);
    void push_bdens_internal(int ncat_branch_left, size_t col);
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    void push_bdens_fast_route(const std::vector<signed char> &cat_split, size_t col);
    void push_bdens_internal(const std::vector<signed char> &cat_split, size_t col);
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    void push_bdens_ext(const IsoHPlane &hplane, const ModelParams &model_params);
    void pop();
    void pop_right();
    void pop_bdens(size_t col);
    void pop_bdens_right(size_t col);
    void pop_bdens_cat(size_t col);
    void pop_bdens_cat_right(size_t col);
    void pop_bdens_fast_route(size_t col);
    void pop_bdens_internal(size_t col);
    void pop_bdens_right_fast_route(size_t col);
    void pop_bdens_right_internal(size_t col);
    void pop_bdens_cat_fast_route(size_t col);
    void pop_bdens_cat_internal(size_t col);
    void pop_bdens_cat_right_fast_route(size_t col);
    void pop_bdens_cat_right_internal(size_t col);
    void pop_bdens_ext();
    void pop_bdens_ext_right();
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double calc_density(ldouble_safe remainder, size_t sample_size);
    ldouble_safe calc_adj_depth();
    double calc_adj_density();
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    ldouble_safe calc_bratio_log();
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    ldouble_safe calc_bratio_inv_log();
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double calc_bratio();
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double calc_bdens(ldouble_safe remainder, size_t sample_size);
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double calc_bdens2(ldouble_safe remainder, size_t sample_size);
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    ldouble_safe calc_bratio_log_ext();
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double calc_bratio_ext();
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double calc_bdens_ext(ldouble_safe remainder, size_t sample_size);
    void save_range(double xmin, double xmax);
    void restore_range(double &restrict xmin, double &restrict xmax);
    void save_counts(size_t *restrict cat_counts, int ncat);
    void save_n_present_and_left(signed char *restrict split_left, int ncat);
    void save_n_present(size_t *restrict cat_counts, int ncat);
};

template <class ldouble_safe, class real_t>
class SingleNodeColumnSampler
{
public:
    double *restrict weights_orig;
    std::vector<bool> inifinite_weights;
    ldouble_safe cumw;
    size_t n_inf;
    size_t *restrict col_indices;
    size_t curr_pos;
    bool using_tree;

    bool backup_weights;
    std::vector<double> weights_own;
    size_t n_left;

    std::vector<double> tree_weights;
    size_t offset;
    size_t tree_levels;
    std::vector<double> used_weights;
    std::vector<size_t> mapped_indices;
    std::vector<size_t> mapped_inf_indices;

    bool initialize(
        double *restrict weights,
        std::vector<size_t> *col_indices,
        size_t curr_pos,
        size_t n_sample,
        bool backup_weights
    );

    bool sample_col(size_t &col_chosen, RNG_engine &rnd_generator);

    void backup(SingleNodeColumnSampler<ldouble_safe, real_t> &other, size_t ncols_tot);

    void restore(const SingleNodeColumnSampler<ldouble_safe, real_t> &other);
};

template <class ImputedData, class ldouble_safe, class real_t>
struct WorkerMemory {
    std::vector<size_t>  ix_arr;
    std::vector<size_t>  ix_all;
    RNG_engine           rnd_generator;
    UniformUnitInterval  rbin;
    size_t               st;
    size_t               end;
    size_t               st_NA;
    size_t               end_NA;
    size_t               split_ix;
    hashed_map<size_t, double> weights_map;
    std::vector<double>  weights_arr;     /* when not ignoring NAs and when using weights as dty */
    bool                 changed_weights; /* when using 'missing_action'='Divide' or density weights */
    double               xmin;
    double               xmax;
    size_t               npresent;        /* 'npresent' and 'ncols_tried' are used interchangeable and for unrelated things */
    bool                 unsplittable;
    std::vector<bool>    is_repeated;
    std::vector<signed char> categs;
    size_t               ncols_tried;     /* 'npresent' and 'ncols_tried' are used interchangeable and for unrelated things */
    int                  ncat_tried;
    std::vector<double>  btree_weights;   /* only when using weights for sampling */
    ColumnSampler<ldouble_safe> col_sampler; /* columns can get eliminated, keep a copy for each thread */
    SingleNodeColumnSampler<ldouble_safe, real_t> node_col_sampler;
    SingleNodeColumnSampler<ldouble_safe, real_t> node_col_sampler_backup;

    /* for split criterion */
    std::vector<double>  buffer_dbl;
    std::vector<size_t>  buffer_szt;
    std::vector<signed char> buffer_chr;
    double               prob_split_type;
    ColCriterion         col_criterion;
    GainCriterion        criterion;
    double               this_gain;
    double               this_split_point;
    int                  this_categ;
    std::vector<signed char> this_split_categ;
    bool                 determine_split;
    std::vector<double>  imputed_x_buffer;
    double               saved_xmedian;
    double               best_xmedian;
    int                  saved_cat_mode;
    int                  best_cat_mode;
    std::vector<size_t>  col_indices;    /* only for full gain calculation */

    /* for weighted column choices */
    std::vector<double>  node_col_weights;
    std::vector<double>  saved_stat1;
    std::vector<double>  saved_stat2;
    bool                 has_saved_stats;
    double*              tree_kurtoses;  /* only when mixing 'weight_by_kurt' with 'prob_pick_col*' */

    /* for the extended model */
    size_t   ntry;
    size_t   ntaken;
    size_t   ntaken_best;
    size_t   ntried;
    bool     try_all;
    size_t   col_chosen; /* also used as placeholder in the single-variable model */
    ColType  col_type;
    double   ext_sd;
    std::vector<double>  comb_val;
    std::vector<size_t>  col_take;
    std::vector<ColType> col_take_type;
    std::vector<double>  ext_offset;
    std::vector<double>  ext_coef;
    std::vector<double>  ext_mean;
    std::vector<double>  ext_fill_val;
    std::vector<double>  ext_fill_new;
    std::vector<int>     chosen_cat;
    std::vector<std::vector<double>> ext_cat_coef;
    UniformMinusOneToOne coef_unif;
    StandardNormalDistr  coef_norm;
    std::vector<double> sample_weights; /* when using weights and split criterion */

    /* for similarity/distance calculations */
    std::vector<double> tmat_sep;

    /* when calculating average depth on-the-fly */
    std::vector<double> row_depths;

    /* when imputing NAs on-the-fly */
    std::vector<ImputedData> impute_vec;
    hashed_map<size_t, ImputedData> impute_map;

    /* for non-depth scoring metric */
    DensityCalculator<ldouble_safe, real_t> density_calculator;
};

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

typedef struct WorkerForPredictCSC {
    std::vector<size_t> ix_arr;
    size_t              st;
    size_t              end;
    std::vector<double> comb_val;
    std::vector<double> weights_arr;
    std::vector<double> depths;
} WorkerForPredictCSC;

class RecursionState {
public:
    size_t  st;
    size_t  st_NA;
    size_t  end_NA;
    size_t  split_ix;
    size_t  end;
    size_t  sampler_pos;
    size_t  n_dropped;
    bool    changed_weights;
    bool    full_state;
    std::vector<size_t> ix_arr;
    std::vector<bool>   cols_possible;
    std::vector<double> col_sampler_weights;
    std::unique_ptr<double[]> weights_arr;

    RecursionState() = default;
    template <class WorkerMemory>
    RecursionState(WorkerMemory &workspace, bool full_state);
    template <class WorkerMemory>
    void restore_state(WorkerMemory &workspace);
};

/* Function prototypes */

/* fit_model.cpp */
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
                uint64_t random_seed, int nthreads);
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
                uint64_t random_seed, bool use_long_double, int nthreads);
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
             uint64_t random_seed, bool use_long_double);
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
             uint64_t random_seed);
template <class InputData, class WorkerMemory, class ldouble_safe>
void fit_itree(std::vector<IsoTree>    *tree_root,
               std::vector<IsoHPlane>  *hplane_root,
               WorkerMemory             &workspace,
               InputData                &input_data,
               ModelParams              &model_params,
               std::vector<ImputeNode> *impute_nodes,
               size_t                   tree_num);

/* isoforest.cpp */
template <class InputData, class WorkerMemory, class ldouble_safe>
void split_itree_recursive(std::vector<IsoTree>     &trees,
                           WorkerMemory             &workspace,
                           InputData                &input_data,
                           ModelParams              &model_params,
                           std::vector<ImputeNode> *impute_nodes,
                           size_t                   curr_depth);

/* extended.cpp */
template <class InputData, class WorkerMemory, class ldouble_safe>
void split_hplane_recursive(std::vector<IsoHPlane>   &hplanes,
                            WorkerMemory             &workspace,
                            InputData                &input_data,
                            ModelParams              &model_params,
                            std::vector<ImputeNode> *impute_nodes,
                            size_t                   curr_depth);
template <class InputData, class WorkerMemory, class ldouble_safe>
void add_chosen_column(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params,
                       std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s);
void shrink_to_fit_hplane(IsoHPlane &hplane, bool clear_vectors);
template <class InputData, class WorkerMemory>
void simplify_hplane(IsoHPlane &hplane, WorkerMemory &workspace, InputData &input_data, ModelParams &model_params);


/* predict.cpp */
template <class real_t, class sparse_ix>
#ifndef _FOR_R
[[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno"), gnu::hot]]
#endif
void predict_iforest(real_t *restrict numeric_data, int *restrict categ_data,
                     bool is_col_major, size_t ld_numeric, size_t ld_categ,
                     real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     real_t *restrict Xr, sparse_ix *restrict Xr_ind, sparse_ix *restrict Xr_indptr,
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double *restrict output_depths,   sparse_ix *restrict tree_num,
                     double *restrict per_tree_depths,
                     TreesIndexer *indexer);
template <class real_t, class sparse_ix>
[[gnu::hot]]
void traverse_itree_fast(std::vector<IsoTree>  &tree,
                         IsoForest             &model_outputs,
                         real_t *restrict      row_numeric_data,
                         double &restrict      output_depth,
                         sparse_ix *restrict   tree_num,
                         double *restrict      tree_depth,
                         size_t                row) noexcept;
template <class PredictionData, class sparse_ix>
[[gnu::hot]]
void traverse_itree_no_recurse(std::vector<IsoTree>  &tree,
                               IsoForest             &model_outputs,
                               PredictionData        &prediction_data,
                               double &restrict      output_depth,
                               sparse_ix *restrict   tree_num,
                               double *restrict      tree_depth,
                               size_t                row) noexcept;
template <class PredictionData, class sparse_ix, class ImputedData>
[[gnu::hot]]
double traverse_itree(std::vector<IsoTree>     &tree,
                      IsoForest                &model_outputs,
                      PredictionData           &prediction_data,
                      std::vector<ImputeNode> *impute_nodes,
                      ImputedData             *imputed_data,
                      double                   curr_weight,
                      size_t                   row,
                      sparse_ix *restrict      tree_num,
                      double *restrict         tree_depth,
                      size_t                   curr_lev) noexcept;
template <class PredictionData, class sparse_ix>
[[gnu::hot]]
void traverse_hplane_fast_colmajor(std::vector<IsoHPlane>  &hplane,
                                   ExtIsoForest            &model_outputs,
                                   PredictionData          &prediction_data,
                                   double &restrict        output_depth,
                                   sparse_ix *restrict     tree_num,
                                   double *restrict        tree_depth,
                                   size_t                  row) noexcept;
template <class real_t, class sparse_ix>
[[gnu::hot]]
void traverse_hplane_fast_rowmajor(std::vector<IsoHPlane>  &hplane,
                                   ExtIsoForest            &model_outputs,
                                   real_t *restrict        row_numeric_data,
                                   double &restrict        output_depth,
                                   sparse_ix *restrict     tree_num,
                                   double *restrict        tree_depth,
                                   size_t                  row) noexcept;
template <class PredictionData, class sparse_ix, class ImputedData>
[[gnu::hot]]
void traverse_hplane(std::vector<IsoHPlane>   &hplane,
                     ExtIsoForest             &model_outputs,
                     PredictionData           &prediction_data,
                     double &restrict         output_depth,
                     std::vector<ImputeNode> *impute_nodes,
                     ImputedData             *imputed_data,
                     sparse_ix *restrict      tree_num,
                     double *restrict         tree_depth,
                     size_t                   row) noexcept;
template <class real_t, class sparse_ix>
void batched_csc_predict(PredictionData<real_t, sparse_ix> &prediction_data, int nthreads,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double *restrict output_depths,   sparse_ix *restrict tree_num,
                         double *restrict per_tree_depths);
template <class PredictionData, class sparse_ix>
void traverse_itree_csc(WorkerForPredictCSC   &workspace,
                        std::vector<IsoTree>  &trees,
                        IsoForest             &model_outputs,
                        PredictionData        &prediction_data,
                        sparse_ix *restrict   tree_num,
                        double *restrict      per_tree_depths,
                        size_t                curr_tree,
                        bool                  has_range_penalty);
template <class PredictionData, class sparse_ix>
void traverse_hplane_csc(WorkerForPredictCSC      &workspace,
                         std::vector<IsoHPlane>   &hplanes,
                         ExtIsoForest             &model_outputs,
                         PredictionData           &prediction_data,
                         sparse_ix *restrict      tree_num,
                         double *restrict         per_tree_depths,
                         size_t                   curr_tree,
                         bool                     has_range_penalty);
template <class PredictionData>
void add_csc_range_penalty(WorkerForPredictCSC  &workspace,
                           PredictionData       &prediction_data,
                           double *restrict     weights_arr,
                           size_t               col_num,
                           double               range_low,
                           double               range_high);
template <class PredictionData>
double extract_spC(PredictionData &prediction_data, size_t row, size_t col_num) noexcept;
template <class PredictionData, class sparse_ix>
static inline double extract_spR(PredictionData &prediction_data, sparse_ix *row_st, sparse_ix *row_end, size_t col_num, size_t lb, size_t ub) noexcept;
template <class PredictionData, class sparse_ix>
double extract_spR(PredictionData &prediction_data, sparse_ix *row_st, sparse_ix *row_end, size_t col_num) noexcept;
template <class sparse_ix>
void get_num_nodes(IsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads) noexcept;
template <class sparse_ix>
void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads) noexcept;

/* dist.cpp */
template <class real_t, class sparse_ix>
void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, bool use_long_double, int nthreads,
                     bool assume_full_distr, bool standardize_dist, bool as_kernel,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from, bool use_indexed_references,
                     TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ);
template <class real_t, class sparse_ix, class ldouble_safe>
void calc_similarity_internal(
                     real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, int nthreads,
                     bool assume_full_distr, bool standardize_dist, bool as_kernel,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from, bool use_indexed_references,
                     TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ);
template <class PredictionData, class ldouble_safe>
void traverse_tree_sim(WorkerForSimilarity   &workspace,
                       PredictionData        &prediction_data,
                       IsoForest             &model_outputs,
                       std::vector<IsoTree>  &trees,
                       size_t                curr_tree,
                       const bool            as_kernel);
template <class PredictionData, class ldouble_safe>
void traverse_hplane_sim(WorkerForSimilarity     &workspace,
                         PredictionData          &prediction_data,
                         ExtIsoForest            &model_outputs,
                         std::vector<IsoHPlane>  &hplanes,
                         size_t                  curr_tree,
                         const bool              as_kernel);
template <class PredictionData, class InputData, class WorkerMemory>
#ifndef _FOR_R
[[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
void gather_sim_result(std::vector<WorkerForSimilarity> *worker_memory,
                       std::vector<WorkerMemory> *worker_memory_m,
                       PredictionData *prediction_data, InputData *input_data,
                       IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                       double *restrict tmat, double *restrict rmat, size_t n_from,
                       size_t ntrees, bool assume_full_distr,
                       bool standardize_dist, bool as_kernel, int nthreads);
template <class PredictionData>
void initialize_worker_for_sim(WorkerForSimilarity  &workspace,
                               PredictionData       &prediction_data,
                               IsoForest            *model_outputs,
                               ExtIsoForest         *model_outputs_ext,
                               size_t                n_from,
                               bool                  assume_full_distr);
template <class real_t, class sparse_ix>
#ifndef _FOR_R
[[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
void calc_similarity_from_indexer
(
    real_t *restrict numeric_data, int *restrict categ_data,
    real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
    size_t nrows, int nthreads, bool assume_full_distr, bool standardize_dist,
    IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
    double *restrict tmat, double *restrict rmat, size_t n_from,
    TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ
);
template <class real_t, class sparse_ix>
#ifndef _FOR_R
[[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
void calc_similarity_from_indexer_with_references
(
    real_t *restrict numeric_data, int *restrict categ_data,
    real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
    size_t nrows, int nthreads, bool standardize_dist,
    IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
    double *restrict rmat,
    TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ
);
template <class real_t, class sparse_ix>
void kernel_to_references(TreesIndexer &indexer,
                          IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                          real_t *restrict numeric_data, int *restrict categ_data,
                          real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          size_t nrows, int nthreads,
                          double *restrict rmat,
                          bool standardize);

/* impute.cpp */
template <class real_t, class sparse_ix>
void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, bool use_long_double, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer);
template <class real_t, class sparse_ix, class ldouble_safe>
void impute_missing_values_internal(
                           real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer);
template <class InputData, class ldouble_safe>
void initialize_imputer(Imputer &imputer, InputData &input_data, size_t ntrees, int nthreads);
template <class InputData, class WorkerMemory, class ldouble_safe>
void build_impute_node(ImputeNode &imputer,    WorkerMemory &workspace,
                       InputData  &input_data, ModelParams  &model_params,
                       std::vector<ImputeNode> &imputer_tree,
                       size_t curr_depth, size_t min_imp_obs);
void shrink_impute_node(ImputeNode &imputer);
void drop_nonterminal_imp_node(std::vector<ImputeNode>  &imputer_tree,
                               std::vector<IsoTree>     *trees,
                               std::vector<IsoHPlane>   *hplanes);
template <class ImputedData>
void combine_imp_single(ImputedData &restrict imp_addfrom, ImputedData &restrict imp_addto);
template <class ImputedData, class WorkerMemory>
void combine_tree_imputations(WorkerMemory &workspace,
                              std::vector<ImputedData> &impute_vec,
                              hashed_map<size_t, ImputedData> &impute_map,
                              std::vector<char> &has_missing,
                              int nthreads);
template <class ImputedData>
void add_from_impute_node(ImputeNode &imputer, ImputedData &imputed_data, double w);
template <class InputData, class WorkerMemory>
void add_from_impute_node(ImputeNode &imputer, WorkerMemory &workspace, InputData &input_data);
template <class imp_arr, class InputData>
void apply_imputation_results(imp_arr    &impute_vec,
                              Imputer    &imputer,
                              InputData  &input_data,
                              int        nthreads);
template <class ImputedData, class InputData>
void apply_imputation_results(std::vector<ImputedData> &impute_vec,
                              hashed_map<size_t, ImputedData> &impute_map,
                              Imputer   &imputer,
                              InputData &input_data,
                              int nthreads);
template <class PredictionData, class ImputedData>
void apply_imputation_results(PredictionData  &prediction_data,
                              ImputedData     &imp,
                              Imputer         &imputer,
                              size_t          row);
template <class ImputedData, class InputData>
void initialize_impute_calc(ImputedData &imp, InputData &input_data, size_t row);
template <class ImputedData, class PredictionData>
void initialize_impute_calc(ImputedData &imp, PredictionData &prediction_data, Imputer &imputer, size_t row);
template <class ImputedData, class InputData>
void allocate_imp_vec(std::vector<ImputedData> &impute_vec, InputData &input_data, int nthreads);
template <class ImputedData, class InputData>
void allocate_imp_map(hashed_map<size_t, ImputedData> &impute_map, InputData &input_data);
template <class ImputedData, class InputData>
void allocate_imp(InputData &input_data,
                  std::vector<ImputedData> &impute_vec,
                  hashed_map<size_t, ImputedData> &impute_map,
                  int nthreads);
template <class ImputedData, class InputData>
void check_for_missing(InputData &input_data,
                       std::vector<ImputedData> &impute_vec,
                       hashed_map<size_t, ImputedData> &impute_map,
                       int nthreads);
template <class PredictionData>
size_t check_for_missing(PredictionData  &prediction_data,
                         Imputer         &imputer,
                         size_t          ix_arr[],
                         int             nthreads);

/* helpers_iforest.cpp */
static inline size_t get_ntrees(const IsoForest &model)
{
    return model.trees.size();
}

static inline size_t get_ntrees(const ExtIsoForest &model)
{
    return model.hplanes.size();
}

static inline size_t get_ntrees(const Imputer &model)
{
    return model.imputer_tree.size();
}

static inline size_t get_ntrees(const TreesIndexer &model)
{
    return model.indices.size();
}
template <class InputData, class WorkerMemory>
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params, IsoTree &tree);
template <class InputData, class WorkerMemory>
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params);
template <class InputData, class WorkerMemory>
void get_split_range_v2(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params);
template <class InputData, class WorkerMemory>
int choose_cat_from_present(WorkerMemory &workspace, InputData &input_data, size_t col_num);
bool is_col_taken(std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s,
                  size_t col_num);
template <class InputData>
void set_col_as_taken(std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s,
                      InputData &input_data, size_t col_num, ColType col_type);
template <class InputData>
void set_col_as_taken(std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s,
                      InputData &input_data, size_t col_num);
template <class InputData, class WorkerMemory>
void add_separation_step(WorkerMemory &workspace, InputData &input_data, double remainder);
template <class InputData, class WorkerMemory, class ldouble_safe>
void add_remainder_separation_steps(WorkerMemory &workspace, InputData &input_data, ldouble_safe sum_weight);
template <class PredictionData, class sparse_ix>
void remap_terminal_trees(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                          PredictionData &prediction_data, sparse_ix *restrict tree_num, int nthreads);
template <class InputData, class ldouble_safe>
std::vector<double> calc_kurtosis_all_data(InputData &input_data, ModelParams &model_params, RNG_engine &rnd_generator);
template <class InputData, class WorkerMemory>
void calc_ranges_all_cols(InputData &input_data, WorkerMemory &workspace, ModelParams &model_params,
                          double *restrict ranges, double *restrict saved_xmin, double *restrict saved_xmax);
template <class InputData, class WorkerMemory, class ldouble_safe>
void calc_var_all_cols(InputData &input_data, WorkerMemory &workspace, ModelParams &model_params,
                       double *restrict variances, double *restrict saved_xmin, double *restrict saved_xmax,
                       double *restrict saved_means, double *restrict saved_sds);
template <class InputData, class WorkerMemory, class ldouble_safe>
void calc_kurt_all_cols(InputData &input_data, WorkerMemory &workspace, ModelParams &model_params,
                        double *restrict kurtosis, double *restrict saved_xmin, double *restrict saved_xmax);
bool is_boxed_metric(const ScoringMetric scoring_metric);


/* utils.cpp */
#define ix_comb_(i, j, n, ncomb) (  ((ncomb)  + ((j) - (i))) - (size_t)1 - div2(((n) - (i)) * ((n) - (i) - (size_t)1))  )
#define ix_comb(i, j, n, ncomb) (  ((i) < (j))? ix_comb_(i, j, n, ncomb) : ix_comb_(j, i, n, ncomb)  )
#define calc_ncomb(n) (((n) % 2) == 0)? (div2(n) * ((n)-(size_t)1)) : ((n) * div2((n)-(size_t)1))
size_t log2ceil(size_t x);
#ifndef _FOR_R
[[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
double digamma(double x);
template <class ldouble_safe>
#ifndef _FOR_R
[[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
double harmonic(size_t n);
double harmonic_recursive(double a, double b);
template <class ldouble_safe>
double expected_avg_depth(size_t sample_size);
template <class ldouble_safe>
#ifndef _FOR_R
[[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
double expected_avg_depth(ldouble_safe approx_sample_size);
double expected_separation_depth(size_t n);
double expected_separation_depth_hotstart(double curr, size_t n_curr, size_t n_final);
template <class ldouble_safe>
double expected_separation_depth(ldouble_safe n);
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n, double counter[], double exp_remainder);
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double *restrict counter, double *restrict weights, double exp_remainder);
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double counter[], hashed_map<size_t, double> &weights, double exp_remainder);
void increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end, size_t split_ix, size_t n,
                                     double counter[], double exp_remainder);
void increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end, size_t split_ix, size_t n,
                                     double *restrict counter, double *restrict weights, double exp_remainder);
void tmat_to_dense(double *restrict tmat, double *restrict dmat, size_t n, double fill_diag);
template <class real_t=double>
void build_btree_sampler(std::vector<double> &btree_weights, real_t *restrict sample_weights,
                         size_t nrows, size_t &restrict log2_n, size_t &restrict btree_offset);
template <class real_t=double, class ldouble_safe>
void sample_random_rows(std::vector<size_t> &restrict ix_arr, size_t nrows, bool with_replacement,
                        RNG_engine &rnd_generator, std::vector<size_t> &restrict ix_all,
                        real_t *restrict sample_weights, std::vector<double> &restrict btree_weights,
                        size_t log2_n, size_t btree_offset, std::vector<bool> &is_repeated);
template <class real_t=double>
void weighted_shuffle(size_t *restrict outp, size_t n, real_t *restrict weights, double *restrict buffer_arr, RNG_engine &rnd_generator);
double sample_random_uniform(double xmin, double xmax, RNG_engine &rng) noexcept;
size_t divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point) noexcept;
template <class real_t=double>
void divide_subset_split(size_t *restrict ix_arr, real_t x[], size_t st, size_t end, double split_point,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept;
template <class real_t, class sparse_ix>
void divide_subset_split(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                         real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr, double split_point,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept;
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end, signed char split_categ[],
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept;
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end, signed char split_categ[],
                         int ncat, MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept;
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end, int split_categ,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept;
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end,
                         MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept;
template <class real_t=double>
void get_range(size_t ix_arr[], real_t *restrict x, size_t st, size_t end,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable) noexcept;
template <class real_t>
void get_range(real_t *restrict x, size_t n,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable) noexcept;
template <class real_t, class sparse_ix>
void get_range(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
               real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
               MissingAction missing_action, double &restrict xmin_, double &restrict xmax_, bool &unsplittable) noexcept;
template <class real_t, class sparse_ix>
void get_range(size_t col_num, size_t nrows,
               real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable) noexcept;
void get_categs(size_t *restrict ix_arr, int x[], size_t st, size_t end, int ncat,
                MissingAction missing_action, signed char categs[], size_t &restrict npresent, bool &unsplittable) noexcept;
template <class real_t>
bool check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end, real_t x[], MissingAction missing_action);
bool check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end, int x[], MissingAction missing_action);
template <class real_t, class sparse_ix>
bool check_more_than_two_unique_values(size_t *restrict ix_arr, size_t st, size_t end, size_t col,
                                       sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_ind, real_t *restrict Xc,
                                       MissingAction missing_action);
template <class real_t, class sparse_ix>
bool check_more_than_two_unique_values(size_t nrows, size_t col,
                                       sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_ind, real_t *restrict Xc,
                                       MissingAction missing_action);
void count_categs(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat, size_t *restrict counts);
int count_ncateg_in_col(const int x[], const size_t n, const int ncat, unsigned char buffer[]);
template <class ldouble_safe>
ldouble_safe calculate_sum_weights(std::vector<size_t> &ix_arr, size_t st, size_t end, size_t curr_depth,
                                   std::vector<double> &weights_arr, hashed_map<size_t, double> &weights_map);
extern bool interrupt_switch;
extern bool signal_is_locked;
void set_interrup_global_variable(int s);
#ifdef _FOR_PYTHON
bool cy_check_interrupt_switch();
void cy_tick_off_interrupt_switch();
#endif
class SignalSwitcher
{
public:
    sig_t_ old_sig;
    bool is_active;
    SignalSwitcher();
    ~SignalSwitcher();
    void restore_handle();
};
void check_interrupt_switch(SignalSwitcher &ss);
bool has_long_double();
int return_EXIT_SUCCESS();
int return_EXIT_FAILURE();



template <class real_t=double>
size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, real_t x[]);
template <class real_t, class sparse_ix>
size_t move_NAs_to_front(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr);
size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, int x[]);
size_t center_NAs(size_t ix_arr[], size_t st_left, size_t st, size_t curr_pos);
template <class real_t>
void fill_NAs_with_median(size_t *restrict ix_arr, size_t st_orig, size_t st, size_t end, real_t *restrict x,
                          double *restrict buffer_imputed_x, double *restrict xmedian);
template <class real_t, class sparse_ix>
void todense(size_t *restrict ix_arr, size_t st, size_t end,
             size_t col_num, real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
             double *restrict buffer_arr);
template <class real_t>
void colmajor_to_rowmajor(real_t *restrict X, size_t nrows, size_t ncols, std::vector<double> &X_row_major);
template <class real_t, class sparse_ix>
void colmajor_to_rowmajor(real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                          size_t nrows, size_t ncols,
                          std::vector<double> &Xr, std::vector<size_t> &Xr_ind, std::vector<size_t> &Xr_indptr);
template <class sparse_ix=size_t>
bool check_indices_are_sorted(sparse_ix indices[], size_t n);
template <class real_t, class sparse_ix>
void sort_csc_indices(real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr, size_t ncols_numeric);

/* mult.cpp */
template <class real_t, class real_t_>
void calc_mean_and_sd_t(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x,
                        MissingAction missing_action, double &restrict x_sd, double &restrict x_mean);
template <class real_t_, class ldouble_safe>
void calc_mean_and_sd(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x,
                      MissingAction missing_action, double &restrict x_sd, double &restrict x_mean);
template <class real_t_>
double calc_mean_only(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x);
template <class real_t_, class mapping, class ldouble_safe>
void calc_mean_and_sd_weighted(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x, mapping &restrict w,
                               MissingAction missing_action, double &restrict x_sd, double &restrict x_mean);
template <class real_t_, class mapping>
double calc_mean_only_weighted(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x, mapping &restrict w);
template <class real_t_, class sparse_ix, class ldouble_safe>
void calc_mean_and_sd(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                      real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                      double &restrict x_sd, double &restrict x_mean);
template <class real_t_, class sparse_ix, class ldouble_safe>
double calc_mean_only(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                      real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr);
template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
void calc_mean_and_sd_weighted(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                               real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                               double &restrict x_sd, double &restrict x_mean, mapping &restrict w);
template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
double calc_mean_only_weighted(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                               real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                               mapping &restrict w);
template <class real_t_>
void add_linear_comb(size_t ix_arr[], size_t st, size_t end, double *restrict res,
                     real_t_ *restrict x, double &restrict coef, double x_sd, double x_mean, double &restrict fill_val,
                     MissingAction missing_action, double *restrict buffer_arr,
                     size_t *restrict buffer_NAs, bool first_run);
template <class real_t_, class mapping, class ldouble_safe>
void add_linear_comb_weighted(size_t ix_arr[], size_t st, size_t end, double *restrict res,
                              real_t_ *restrict x, double &restrict coef, double x_sd, double x_mean, double &restrict fill_val,
                              MissingAction missing_action, double *restrict buffer_arr,
                              size_t *restrict buffer_NAs, bool first_run, mapping &restrict w);
template <class real_t_, class sparse_ix>
void add_linear_comb(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, double *restrict res,
                     real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     double &restrict coef, double x_sd, double x_mean, double &restrict fill_val, MissingAction missing_action,
                     double *restrict buffer_arr, size_t *restrict buffer_NAs, bool first_run);
template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
void add_linear_comb_weighted(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, double *restrict res,
                              real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                              double &restrict coef, double x_sd, double x_mean, double &restrict fill_val, MissingAction missing_action,
                              double *restrict buffer_arr, size_t *restrict buffer_NAs, bool first_run, mapping &restrict w);
template <class mapping>
void add_linear_comb_weighted(size_t *restrict ix_arr, size_t st, size_t end, double *restrict res,
                              int x[], int ncat, double *restrict cat_coef, double single_cat_coef, int chosen_cat,
                              double &restrict fill_val, double &restrict fill_new, size_t *restrict buffer_pos,
                              NewCategAction new_cat_action, MissingAction missing_action, CategSplit cat_split_type,
                              bool first_run, mapping &restrict w);
template <class ldouble_safe>
void add_linear_comb(size_t *restrict ix_arr, size_t st, size_t end, double *restrict res,
                     int x[], int ncat, double *restrict cat_coef, double single_cat_coef, int chosen_cat,
                     double &restrict fill_val, double &restrict fill_new, size_t *restrict buffer_cnt, size_t *restrict buffer_pos,
                     NewCategAction new_cat_action, MissingAction missing_action, CategSplit cat_split_type, bool first_run);
template <class mapping, class ldouble_safe>
void add_linear_comb_weighted(size_t *restrict ix_arr, size_t st, size_t end, double *restrict res,
                              int x[], int ncat, double *restrict cat_coef, double single_cat_coef, int chosen_cat,
                              double &restrict fill_val, double &restrict fill_new, size_t *restrict buffer_pos,
                              NewCategAction new_cat_action, MissingAction missing_action, CategSplit cat_split_type,
                              bool first_run, mapping &restrict w);

/* crit.cpp */
template <class real_t, class ldouble_safe>
double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, real_t x[], MissingAction missing_action);
template <class real_t, class ldouble_safe>
double calc_kurtosis(real_t x[], size_t n, MissingAction missing_action);
template <class real_t, class mapping, class ldouble_safe>
double calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, real_t x[],
                              MissingAction missing_action, mapping &restrict w);
template <class real_t, class ldouble_safe>
double calc_kurtosis_weighted(real_t *restrict x, size_t n_, MissingAction missing_action, real_t *restrict w);
template <class real_t, class sparse_ix, class ldouble_safe>
double calc_kurtosis(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                     real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     MissingAction missing_action);
template <class real_t, class sparse_ix, class ldouble_safe>
double calc_kurtosis(size_t col_num, size_t nrows,
                     real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     MissingAction missing_action);
template <class real_t, class sparse_ix, class mapping, class ldouble_safe>
double calc_kurtosis_weighted(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                              real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                              MissingAction missing_action, mapping &restrict w);
template <class real_t, class sparse_ix, class ldouble_safe>
double calc_kurtosis_weighted(size_t col_num, size_t nrows,
                              real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                              MissingAction missing_action, real_t *restrict w);
template <class ldouble_safe>
double calc_kurtosis_internal(size_t cnt, int x[], int ncat, size_t buffer_cnt[], double buffer_prob[],
                              MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator);
template <class ldouble_safe>
double calc_kurtosis(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat, size_t *restrict buffer_cnt, double buffer_prob[],
                     MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator);
template <class ldouble_safe>
double calc_kurtosis(size_t nrows, int x[], int ncat, size_t buffer_cnt[], double buffer_prob[],
                     MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator);
template <class mapping, class ldouble_safe>
double calc_kurtosis_weighted_internal(std::vector<ldouble_safe> &buffer_cnt, int x[], int ncat,
                                       double buffer_prob[], MissingAction missing_action, CategSplit cat_split_type,
                                       RNG_engine &rnd_generator, mapping &restrict w);
template <class mapping, class ldouble_safe>
double calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, int x[], int ncat, double buffer_prob[],
                              MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator,
                              mapping &restrict w);
template <class real_t, class ldouble_safe>
double calc_kurtosis_weighted(size_t nrows, int x[], int ncat, double *restrict buffer_prob,
                              MissingAction missing_action, CategSplit cat_split_type,
                              RNG_engine &rnd_generator, real_t *restrict w);
template <class int_t, class ldouble_safe>
double expected_sd_cat(double p[], size_t n, int_t pos[]);
template <class number, class int_t, class ldouble_safe>
double expected_sd_cat(number *restrict counts, double *restrict p, size_t n, int_t *restrict pos);
template <class number, class int_t, class ldouble_safe>
double expected_sd_cat_single(number *restrict counts, double *restrict p, size_t n, int_t *restrict pos, size_t cat_exclude, number cnt);
template <class number, class int_t, class ldouble_safe>
double expected_sd_cat_internal(int ncat, number *restrict buffer_cnt, ldouble_safe cnt_l,
                                int_t *restrict buffer_pos, double *restrict buffer_prob);
template <class int_t, class ldouble_safe>
double expected_sd_cat(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat,
                       MissingAction missing_action,
                       size_t *restrict buffer_cnt, int_t *restrict buffer_pos, double buffer_prob[]);
template <class mapping, class int_t, class ldouble_safe>
double expected_sd_cat_weighted(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat,
                                MissingAction missing_action, mapping &restrict w,
                                double *restrict buffer_cnt, int_t *restrict buffer_pos, double *restrict buffer_prob);
template <class number, class ldouble_safe>
double categ_gain(number cnt_left, number cnt_right,
                  ldouble_safe s_left, ldouble_safe s_right,
                  ldouble_safe base_info, ldouble_safe cnt);
template <class real_t, class real_t_>
double find_split_rel_gain_t(real_t_ *restrict x, size_t n, double &restrict split_point);
template <class real_t_, class ldouble_safe>
double find_split_rel_gain(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &restrict split_point, size_t &restrict split_ix);
template <class real_t, class real_t_>
double find_split_rel_gain_t(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &split_point, size_t &restrict split_ix);
template <class real_t_, class ldouble_safe>
double find_split_rel_gain(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double &split_point, size_t &split_ix);
template <class real_t, class real_t_, class mapping>
double find_split_rel_gain_weighted_t(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &split_point, size_t &restrict split_ix, mapping &restrict w);
template <class real_t_, class mapping, class ldouble_safe>
double find_split_rel_gain_weighted(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &restrict split_point, size_t &restrict split_ix, mapping &restrict w);
template <class real_t, class real_t_=double>
real_t calc_sd_right_to_left(real_t_ *restrict x, size_t n, double *restrict sd_arr);
template <class real_t_, class ldouble_safe>
ldouble_safe calc_sd_right_to_left_weighted(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                                           double *restrict w, ldouble_safe &cumw, size_t *restrict sorted_ix);
template <class real_t, class real_t_>
real_t calc_sd_right_to_left(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr);
template <class real_t_, class mapping, class ldouble_safe>
ldouble_safe calc_sd_right_to_left_weighted(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end,
                                           double *restrict sd_arr, mapping &restrict w, ldouble_safe &cumw);
template <class real_t, class real_t_>
double find_split_std_gain_t(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                             GainCriterion criterion, double min_gain, double &restrict split_point);
template <class real_t_, class ldouble_safe>
double find_split_std_gain(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                           GainCriterion criterion, double min_gain, double &restrict split_point);
template <class real_t, class ldouble_safe>
double find_split_std_gain_weighted(real_t *restrict x, size_t n, double *restrict sd_arr,
                                    GainCriterion criterion, double min_gain, double &restrict split_point,
                                    double *restrict w, size_t *restrict sorted_ix);
template <class real_t, class real_t_>
double find_split_std_gain_t(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                             GainCriterion criterion, double min_gain, double &restrict split_point, size_t &restrict split_ix);
template <class real_t_, class ldouble_safe>
double find_split_std_gain(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                           GainCriterion criterion, double min_gain, double &restrict split_point, size_t &restrict split_ix);
template <class real_t, class mapping, class ldouble_safe>
double find_split_std_gain_weighted(real_t *restrict x, real_t xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                                    GainCriterion criterion, double min_gain, double &restrict split_point, size_t &restrict split_ix, mapping &restrict w);
template <class real_t, class ldouble_safe>
double find_split_full_gain(real_t *restrict x, size_t st, size_t end, size_t *restrict ix_arr,
                            size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                            double *restrict X_row_major, size_t ncols,
                            double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                            double *restrict buffer_sum_left, double *restrict buffer_sum_tot,
                            size_t &restrict split_ix, double &restrict split_point,
                            bool x_uses_ix_arr);
template <class real_t, class mapping, class ldouble_safe>
double find_split_full_gain_weighted(real_t *restrict x, size_t st, size_t end, size_t *restrict ix_arr,
                                     size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                     double *restrict X_row_major, size_t ncols,
                                     double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                                     double *restrict buffer_sum_left, double *restrict buffer_sum_tot,
                                     size_t &restrict split_ix, double &restrict split_point,
                                     bool x_uses_ix_arr,
                                     mapping &restrict w);
template <class real_t_, class real_t>
double find_split_dens_shortform_t(real_t *restrict x, size_t n, double &restrict split_point);
template <class real_t, class ldouble_safe>
double find_split_dens_shortform(real_t *restrict x, size_t n, double &restrict split_point);
template <class real_t_, class real_t, class mapping>
double find_split_dens_shortform_weighted_t(real_t *restrict x, size_t n, double &restrict split_point, mapping &restrict w, size_t *restrict buffer_indices);
template <class real_t, class mapping, class ldouble_safe>
double find_split_dens_shortform_weighted(real_t *restrict x, size_t n, double &restrict split_point, mapping &restrict w, size_t *restrict buffer_indices);
template <class real_t>
double find_split_dens_shortform(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                 double &restrict split_point, size_t &restrict split_ix);
template <class real_t, class mapping>
double find_split_dens_shortform_weighted(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                          double &restrict split_point, size_t &restrict split_ix, mapping &restrict w);
template <class real_t, class ldouble_safe>
double find_split_dens_longform(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                double &restrict split_point, size_t &restrict split_ix);
template <class real_t, class mapping, class ldouble_safe>
double find_split_dens_longform_weighted(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                         double &restrict split_point, size_t &restrict split_ix, mapping &restrict w);
template <class real_t, class ldouble_safe>
double find_split_dens(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                       double &restrict split_point, size_t &restrict split_ix);
template <class real_t, class mapping, class ldouble_safe>
double find_split_dens_weighted(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                double &restrict split_point, size_t &restrict split_ix, mapping &restrict w);
template <class int_t, class ldouble_safe>
double find_split_dens_longform(int *restrict x, int ncat, size_t *restrict ix_arr, size_t st, size_t end,
                                CategSplit cat_split_type, MissingAction missing_action,
                                int &restrict chosen_cat, signed char *restrict split_categ, int *restrict saved_cat_mode,
                                size_t *restrict buffer_cnt, int_t *restrict buffer_indices);
template <class mapping, class int_t, class ldouble_safe>
double find_split_dens_longform_weighted(int *restrict x, int ncat, size_t *restrict ix_arr, size_t st, size_t end,
                                         CategSplit cat_split_type, MissingAction missing_action,
                                         int &restrict chosen_cat, signed char *restrict split_categ, int *restrict saved_cat_mode,
                                         int_t *restrict buffer_indices, mapping &restrict w);
template <class ldouble_safe>
double eval_guided_crit(double *restrict x, size_t n, GainCriterion criterion,
                        double min_gain, bool as_relative_gain, double *restrict buffer_sd,
                        double &restrict split_point, double &restrict xmin, double &restrict xmax,
                        size_t *restrict ix_arr_plus_st,
                        size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                        double *restrict X_row_major, size_t ncols,
                        double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr);
template <class ldouble_safe>
double eval_guided_crit_weighted(double *restrict x, size_t n, GainCriterion criterion,
                                 double min_gain, bool as_relative_gain, double *restrict buffer_sd,
                                 double &restrict split_point, double &restrict xmin, double &restrict xmax,
                                 double *restrict w, size_t *restrict buffer_indices,
                                 size_t *restrict ix_arr_plus_st,
                                 size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                 double *restrict X_row_major, size_t ncols,
                                 double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr);
template <class real_t_, class ldouble_safe>
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, real_t_ *restrict x,
                        double *restrict buffer_sd, bool as_relative_gain,
                        double *restrict buffer_imputed_x, double *restrict saved_xmedian,
                        size_t &split_ix, double &restrict split_point, double &restrict xmin, double &restrict xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action,
                        size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                        double *restrict X_row_major, size_t ncols,
                        double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr);
template <class real_t_, class mapping, class ldouble_safe>
double eval_guided_crit_weighted(size_t *restrict ix_arr, size_t st, size_t end, real_t_ *restrict x,
                                 double *restrict buffer_sd, bool as_relative_gain,
                                 double *restrict buffer_imputed_x, double *restrict saved_xmedian,
                                 size_t &split_ix, double &restrict split_point, double &restrict xmin, double &restrict xmax,
                                 GainCriterion criterion, double min_gain, MissingAction missing_action,
                                 size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                 double *restrict X_row_major, size_t ncols,
                                 double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                                 mapping &restrict w);
template <class real_t_, class sparse_ix, class ldouble_safe>
double eval_guided_crit(size_t ix_arr[], size_t st, size_t end,
                        size_t col_num, real_t_ Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                        double buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
                        double *restrict saved_xmedian,
                        double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action,
                        size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                        double *restrict X_row_major, size_t ncols,
                        double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr);
template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
double eval_guided_crit_weighted(size_t ix_arr[], size_t st, size_t end,
                                 size_t col_num, real_t_ Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                                 double buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
                                 double *restrict saved_xmedian,
                                 double &restrict split_point, double &restrict xmin, double &restrict xmax,
                                 GainCriterion criterion, double min_gain, MissingAction missing_action,
                                 size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                 double *restrict X_row_major, size_t ncols,
                                 double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                                 mapping &restrict w);
template <class ldouble_safe>
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int ncat,
                        int *restrict saved_cat_mode,
                        size_t *restrict buffer_cnt, size_t *restrict buffer_pos, double *restrict buffer_prob,
                        int &restrict chosen_cat, signed char *restrict split_categ, signed char *restrict buffer_split,
                        GainCriterion criterion, double min_gain, bool all_perm,
                        MissingAction missing_action, CategSplit cat_split_type);
template <class mapping, class ldouble_safe>
double eval_guided_crit_weighted(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int ncat,
                                 int *restrict saved_cat_mode,
                                 size_t *restrict buffer_pos, double *restrict buffer_prob,
                                 int &restrict chosen_cat, signed char *restrict split_categ, signed char *restrict buffer_split,
                                 GainCriterion criterion, double min_gain, bool all_perm,
                                 MissingAction missing_action, CategSplit cat_split_type,
                                 mapping &restrict w);

/* indexer.cpp */
template <class Tree>
void build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings, size_t &n_terminal, const std::vector<Tree> &tree);
void build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings, size_t &n_terminal, const std::vector<IsoTree> &tree);
void build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings, size_t &n_terminal, const std::vector<IsoHPlane> &tree);
template <class Model>
void build_terminal_node_mappings(TreesIndexer &indexer, const Model &model);
template <class Node>
void build_dindex_recursive
(
    const size_t curr_node,
    const size_t n_terminal, const size_t ncomb,
    const size_t st, const size_t end,
    std::vector<size_t> &restrict node_indices, /* array with all terminal indices in 'tree' */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    size_t curr_depth,
    const std::vector<Node> &tree
);
template <class Node>
void build_dindex
(
    std::vector<size_t> &restrict node_indices, /* empty, but correctly sized */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    const size_t n_terminal,
    const std::vector<Node> &tree
);
void build_dindex
(
    std::vector<size_t> &restrict node_indices, /* empty, but correctly sized */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    const size_t n_terminal,
    const std::vector<IsoTree> &tree
);
void build_dindex
(
    std::vector<size_t> &restrict node_indices, /* empty, but correctly sized */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    const size_t n_terminal,
    const std::vector<IsoHPlane> &tree
);
template <class Model>
void build_distance_mappings(TreesIndexer &indexer, const Model &model, int nthreads);
template <class Model>
void build_tree_indices(TreesIndexer &indexer, const Model &model, int nthreads, const bool with_distances);
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
ISOTREE_EXPORTED
size_t get_number_of_reference_points(const TreesIndexer &indexer) noexcept;
void build_ref_node(SingleTreeIndex &node);

/* ref_indexer.hpp */
template <class Model, class real_t, class sparse_ix>
void set_reference_points(TreesIndexer &indexer, Model &model, const bool with_distances,
                          real_t *restrict numeric_data, int *restrict categ_data,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                          real_t *restrict Xr, sparse_ix *restrict Xr_ind, sparse_ix *restrict Xr_indptr,
                          size_t nrows, int nthreads);
template <class real_t, class sparse_ix>
void set_reference_points(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
                          const bool with_distances,
                          real_t *restrict numeric_data, int *restrict categ_data,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                          real_t *restrict Xr, sparse_ix *restrict Xr_ind, sparse_ix *restrict Xr_indptr,
                          size_t nrows, int nthreads);

/* merge_models.cpp */
ISOTREE_EXPORTED
void merge_models(IsoForest*     model,      IsoForest*     other,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                  Imputer*       imputer,    Imputer*       iother,
                  TreesIndexer*  indexer,    TreesIndexer*  ind_other);

/* subset_models.cpp */
ISOTREE_EXPORTED
void subset_model(IsoForest*     model,      IsoForest*     model_new,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                  Imputer*       imputer,    Imputer*       imputer_new,
                  TreesIndexer*  indexer,    TreesIndexer*  indexer_new,
                  size_t *trees_take, size_t ntrees_take);

/* serialize.cpp */
[[noreturn]]
void throw_errno();
[[noreturn]]
void throw_ferror(FILE *file);
[[noreturn]]
void throw_feoferr();
class FileHandle
{
public:
    FILE *handle = NULL;
    FileHandle(const char *fname, const char *mode)
    {
        this->handle = std::fopen(fname, mode);
        if (!(this->handle))
            throw_errno();
    }
    ~FileHandle()
    {
        if (this->handle) {
            int err = std::fclose(this->handle);
            if (err)
                fprintf(stderr, "Error: could not close file.\n");
        }
        this->handle = NULL;
    }
};

#if defined(_WIN32) && (defined(_MSC_VER) || defined(__GNUC__))
    #define WCHAR_T_FUNS
    #include <stdio.h>
    class WFileHandle
    {
    public:
        FILE *handle = NULL;
        WFileHandle(const wchar_t *fname, const wchar_t *mode)
        {
            this->handle = _wfopen(fname, mode);
            if (!(this->handle))
                throw_errno();
        }
        ~WFileHandle()
        {
            if (this->handle) {
                int err = std::fclose(this->handle);
                if (err)
                    fprintf(stderr, "Error: could not close file.\n");
            }
            this->handle = NULL;
        }
    };
#endif
ISOTREE_EXPORTED
bool has_wchar_t_file_serializers() noexcept;
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
void serialize_IsoForest_ToFile(const IsoForest &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void serialize_IsoForest_ToFile(const IsoForest &model, const wchar_t *fname);
#endif
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, const char *in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, const std::string &in);
ISOTREE_EXPORTED
void deserialize_IsoForest_FromFile(IsoForest &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void deserialize_IsoForest_FromFile(IsoForest &model, const wchar_t *fname);
#endif
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, char *out);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, FILE *out);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_ExtIsoForest(const ExtIsoForest &model);
ISOTREE_EXPORTED
void serialize_ExtIsoForest_ToFile(const ExtIsoForest &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void serialize_ExtIsoForest_ToFile(const ExtIsoForest &model, const wchar_t *fname);
#endif
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const char *in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const std::string &in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest_FromFile(ExtIsoForest &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void deserialize_ExtIsoForest_FromFile(ExtIsoForest &model, const wchar_t *fname);
#endif
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, char *out);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, FILE *out);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_Imputer(const Imputer &model);
ISOTREE_EXPORTED
void serialize_Imputer_ToFile(const Imputer &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void serialize_Imputer_ToFile(const Imputer &model, const wchar_t *fname);
#endif
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, const char *in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, const std::string &in);
ISOTREE_EXPORTED
void deserialize_Imputer_FromFile(Imputer &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void deserialize_Imputer_FromFile(Imputer &model, const wchar_t *fname);
#endif
ISOTREE_EXPORTED
void serialize_Indexer(const TreesIndexer &model, char *out);
ISOTREE_EXPORTED
void serialize_Indexer(const TreesIndexer &model, FILE *out);
ISOTREE_EXPORTED
void serialize_Indexer(const TreesIndexer &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_Indexer(const TreesIndexer &model);
ISOTREE_EXPORTED
void serialize_Indexer_ToFile(const TreesIndexer &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void serialize_Indexer_ToFile(const TreesIndexer &model, const wchar_t *fname);
#endif
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, const char *in);
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_Indexer(TreesIndexer &model, const std::string &in);
ISOTREE_EXPORTED
void deserialize_Indexer_FromFile(TreesIndexer &model, const char *fname);
#ifdef WCHAR_T_FUNS
ISOTREE_EXPORTED
void deserialize_Indexer_FromFile(TreesIndexer &model, const wchar_t *fname);
#endif
void serialize_isotree(const IsoForest &model, char *out);
void serialize_isotree(const ExtIsoForest &model, char *out);
void serialize_isotree(const Imputer &model, char *out);
void serialize_isotree(const TreesIndexer &model, char *out);
void deserialize_isotree(IsoForest &model, const char *in);
void deserialize_isotree(ExtIsoForest &model, const char *in);
void deserialize_isotree(Imputer &model, const char *in);
void deserialize_isotree(TreesIndexer &model, const char *in);
void incremental_serialize_isotree(const IsoForest &model, char *old_bytes_reallocated);
void incremental_serialize_isotree(const ExtIsoForest &model, char *old_bytes_reallocated);
void incremental_serialize_isotree(const Imputer &model, char *old_bytes_reallocated);
void incremental_serialize_isotree(const TreesIndexer &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_Indexer(const TreesIndexer &model, std::string &old_bytes);
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
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const IsoForest &model, const char *serialized_bytes);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const ExtIsoForest &model, const char *serialized_bytes);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const Imputer &model, const char *serialized_bytes);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const TreesIndexer &model, const char *serialized_bytes);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const IsoForest &model, size_t old_ntrees) noexcept;
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const ExtIsoForest &model, size_t old_ntrees) noexcept;
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const Imputer &model, size_t old_ntrees) noexcept;
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const TreesIndexer &model, size_t old_ntrees) noexcept;
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_Indexer(const TreesIndexer &model, char *old_bytes_reallocated);
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
bool check_model_has_range_penalty(const IsoForest &model) noexcept;
bool check_model_has_range_penalty(const ExtIsoForest &model) noexcept;
void add_range_penalty(IsoForest &model) noexcept;
void add_range_penalty(ExtIsoForest &model) noexcept;
void add_range_penalty(Imputer &model) noexcept;
void add_range_penalty(TreesIndexer &model) noexcept;

/* sql.cpp */
ISOTREE_EXPORTED
std::vector<std::string> generate_sql(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                      std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                      std::vector<std::vector<std::string>> &categ_levels,
                                      bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                      int nthreads);
ISOTREE_EXPORTED
std::string generate_sql_with_select_from(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                          std::string &table_from, std::string &select_as,
                                          std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                          std::vector<std::vector<std::string>> &categ_levels,
                                          bool index1, int nthreads);
void generate_tree_rules(std::vector<IsoTree> *trees, std::vector<IsoHPlane> *hplanes, bool output_score,
                         size_t curr_ix, bool index1, std::string &prev_cond, std::vector<std::string> &node_rules,
                         std::vector<std::string> &conditions_left, std::vector<std::string> &conditions_right,
                         const IsoForest *model_outputs, const ExtIsoForest *model_outputs_ext);
void extract_cond_isotree(IsoForest &model, IsoTree &tree,
                          std::string &cond_left, std::string &cond_right,
                          std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                          std::vector<std::vector<std::string>> &categ_levels);
void extract_cond_ext_isotree(ExtIsoForest &model, IsoHPlane &hplane,
                              std::string &cond_left, std::string &cond_right,
                              std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                              std::vector<std::vector<std::string>> &categ_levels);

#ifndef _FOR_R
    #if defined(__clang__)
        #pragma clang diagnostic pop
    #endif
#endif
#endif /* ISOTREE_H */
