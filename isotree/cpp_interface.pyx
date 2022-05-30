#cython: language_level=3

#     Isolation forests and variations thereof, with adjustments for incorporation
#     of categorical variables and missing values.
#     Writen for C++11 standard and aimed at being used in R and Python.
#     
#     This library is based on the following works:
#     [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
#         "Isolation forest."
#         2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
#     [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
#         "Isolation-based anomaly detection."
#         ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
#     [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner.
#         "Extended Isolation Forest."
#         arXiv preprint arXiv:1811.02141 (2018).
#     [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
#         "On detecting clustered anomalies using SCiForest."
#         Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
#     [5] https://sourceforge.net/projects/iforest/
#     [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
#     [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
#     [8] Cortes, David.
#         "Distance approximation using Isolation Forests."
#         arXiv preprint arXiv:1910.12362 (2019).
#     [9] Cortes, David.
#         "Imputing missing values with unsupervised random trees."
#         arXiv preprint arXiv:1911.06646 (2019).
#     [10] https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottomF
#     [11] Cortes, David.
#          "Revisiting randomized choices in isolation forests."
#          arXiv preprint arXiv:2110.13402 (2021).
#     [12] Guha, Sudipto, et al.
#          "Robust random cut forest based anomaly detection on streams."
#          International conference on machine learning. PMLR, 2016.
#     [13] Cortes, David.
#          "Isolation forests: looking beyond tree depth."
#          arXiv preprint arXiv:2111.11639 (2021).
#     [14] Ting, Kai Ming, Yue Zhu, and Zhi-Hua Zhou.
#          "Isolation kernel and its effect on SVM"
#          Proceedings of the 24th ACM SIGKDD
#          International Conference on Knowledge Discovery & Data Mining. 2018.
# 
#     BSD 2-Clause License
#     Copyright (c) 2019-2022, David Cortes
#     All rights reserved.
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import  numpy as np
cimport numpy as np
import pandas as pd
from scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr
from libcpp cimport bool as bool_t ###don't confuse it with Python bool
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string
from libc.string cimport memcpy
from libc.stdio cimport FILE, fopen, fclose
from cython cimport boundscheck, nonecheck, wraparound
import ctypes
import os
import warnings

cdef public void cy_warning(const char *msg) nogil:
    with gil:
        warnings.warn((<bytes>msg).decode())

cdef extern from "headers_joined.hpp":

    bool_t cy_check_interrupt_switch()
    void cy_tick_off_interrupt_switch()

    ctypedef enum NewCategAction:
        Weighted = 0
        Smallest = 11
        Random = 12

    ctypedef enum MissingAction:
        Divide = 21
        Impute = 22
        Fail = 0

    ctypedef enum ColType:
        Numeric = 31
        Categorical = 32
        NotUsed = 0

    ctypedef enum CategSplit:
        SubSet = 0
        SingleCateg = 41

    ctypedef enum GainCriterion:
        Averaged = 51
        Pooled = 52
        NoCrit = 0

    ctypedef enum CoefType:
        Uniform = 61
        Normal = 0

    ctypedef enum UseDepthImp:
        Lower = 71
        Higher = 0
        Same = 72

    ctypedef enum WeighImpRows:
        Inverse = 0
        Prop = 81
        Flat = 82

    ctypedef enum ScoringMetric:
        Depth = 0
        Density = 92
        BoxedDensity = 94
        BoxedDensity2 = 96
        BoxedRatio = 95
        AdjDepth = 91
        AdjDensity = 93

    ctypedef struct IsoTree:
        ColType       col_type
        size_t        col_num
        double        num_split
        vector[signed char]  cat_split
        int           chosen_cat
        size_t        tree_left
        size_t        tree_right
        double        pct_tree_left
        double        score
        double        range_low
        double        range_high
        double        remainder

    ctypedef struct IsoForest:
        vector[vector[IsoTree]] trees
        NewCategAction   new_cat_action
        CategSplit       cat_split_type
        MissingAction    missing_action
        ScoringMetric    scoring_metric
        double           exp_avg_depth
        double           exp_avg_sep
        size_t           orig_sample_size
        bool_t           has_range_penalty

    ctypedef struct IsoHPlane:
        vector[size_t]    col_num
        vector[ColType]   col_type
        vector[double]    coef
        vector[double]    mean
        vector[vector[double]] cat_coef
        vector[int]       chosen_cat
        vector[double]    fill_val
        vector[double]    fill_new

        double   split_point
        size_t   hplane_left
        size_t   hplane_right
        double   score
        double   range_low
        double   range_high
        double   remainder

    ctypedef struct ExtIsoForest:
        vector[vector[IsoHPlane]] hplanes
        NewCategAction    new_cat_action
        CategSplit        cat_split_type
        MissingAction     missing_action
        ScoringMetric     scoring_metric
        double            exp_avg_depth
        double            exp_avg_sep
        size_t            orig_sample_size
        bool_t            has_range_penalty

    ctypedef struct ImputeNode:
        vector[double]          num_sum
        vector[double]          num_weight
        vector[vector[double]]  cat_sum
        vector[double]          cat_weight
        size_t                  parent

    ctypedef struct Imputer:
        size_t          ncols_numeric
        size_t          ncols_categ
        vector[int]     ncat
        vector[vector[ImputeNode]] imputer_tree
        vector[double]  col_means
        vector[int]     col_modes

    ctypedef struct SingleTreeIndex:
        vector[size_t] terminal_node_mappings
        vector[double] node_distances
        vector[double] node_depths
        vector[size_t] reference_points
        vector[size_t] reference_indptr
        vector[size_t] reference_mapping
        size_t n_terminal

    ctypedef struct TreesIndexer:
        vector[SingleTreeIndex] indices


    void tmat_to_dense(double *tmat, double *dmat, size_t n, double fill_diag)

    void merge_models(IsoForest*     model,      IsoForest*     other,
                      ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                      Imputer*       imputer,    Imputer*       iother,
                      TreesIndexer*  indexer,    TreesIndexer*  ind_other) nogil except +

    void subset_model(IsoForest*     model,      IsoForest*     model_new,
                      ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                      Imputer*       imputer,    Imputer*       imputer_new,
                      TreesIndexer*  indexer,    TreesIndexer*  indexer_new,
                      size_t *trees_take, size_t ntrees_take) nogil except +

    vector[cpp_string] generate_sql(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                    vector[cpp_string] &numeric_colnames, vector[cpp_string] &categ_colnames,
                                    vector[vector[cpp_string]] &categ_levels,
                                    bool_t output_tree_num, bool_t index1, bool_t single_tree, size_t tree_num,
                                    int nthreads) nogil except +
    cpp_string generate_sql_with_select_from(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                             cpp_string &table_from, cpp_string &select_as,
                                             vector[cpp_string] &numeric_colnames, vector[cpp_string] &categ_colnames,
                                             vector[vector[cpp_string]] &categ_levels,
                                             bool_t index1, int nthreads) nogil except +

    bool_t has_wchar_t_file_serializers()

    void inspect_serialized_object(
        const char *serialized_bytes,
        bool_t &is_isotree_model,
        bool_t &is_compatible,
        bool_t &has_combined_objects,
        bool_t &has_IsoForest,
        bool_t &has_ExtIsoForest,
        bool_t &has_Imputer,
        bool_t &has_Indexer,
        bool_t &has_metadata,
        size_t &size_metadata) nogil except +

    void inspect_serialized_object(
        FILE *serialized_bytes,
        bool_t &is_isotree_model,
        bool_t &is_compatible,
        bool_t &has_combined_objects,
        bool_t &has_IsoForest,
        bool_t &has_ExtIsoForest,
        bool_t &has_Imputer,
        bool_t &has_Indexer,
        bool_t &has_metadata,
        size_t &size_metadata) nogil except +

    size_t determine_serialized_size_combined(
        const IsoForest *model,
        const ExtIsoForest *model_ext,
        const Imputer *imputer,
        const TreesIndexer *indexer,
        const size_t size_optional_metadata) nogil except +

    void serialize_combined(
        const IsoForest *model,
        const ExtIsoForest *model_ext,
        const Imputer *imputer,
        const TreesIndexer *indexer,
        const char *optional_metadata,
        const size_t size_optional_metadata,
        char *out) nogil except +

    void serialize_combined(
        const IsoForest *model,
        const ExtIsoForest *model_ext,
        const Imputer *imputer,
        const TreesIndexer *indexer,
        const char *optional_metadata,
        const size_t size_optional_metadata,
        FILE *out) nogil except +

    void deserialize_combined(
        const char *inp,
        IsoForest *model,
        ExtIsoForest *model_ext,
        Imputer *imputer,
        TreesIndexer *indexer,
        char *optional_metadata) nogil except +

    void deserialize_combined(
        FILE* inp,
        IsoForest *model,
        ExtIsoForest *model_ext,
        Imputer *imputer,
        TreesIndexer *indexer,
        char *optional_metadata) nogil except +

    int return_EXIT_SUCCESS()
    int return_EXIT_FAILURE()


    int fit_iforest[real_t_, sparse_ix_](
                    IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                    real_t_ *numeric_data,  size_t ncols_numeric,
                    int    *categ_data,    size_t ncols_categ,    int *ncat,
                    real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
                    size_t ndim, size_t ntry, CoefType coef_type, bool_t coef_by_prop,
                    real_t_ *sample_weights, bool_t with_replacement, bool_t weight_as_sample,
                    size_t nrows, size_t sample_size, size_t ntrees,
                    size_t max_depth,   size_t ncols_per_tree,
                    bool_t limit_depth, bool_t penalize_range, bool_t standardize_data,
                    ScoringMetric scoring_metric, bool_t fast_bratio,
                    bool_t standardize_dist, double *tmat,
                    double *output_depths, bool_t standardize_depth,
                    real_t_ *col_weights, bool_t weigh_by_kurt,
                    double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                    double prob_pick_by_full_gain, double prob_pick_by_dens,
                    double prob_pick_col_by_range, double prob_pick_col_by_var,
                    double prob_pick_col_by_kurt,
                    double min_gain, MissingAction missing_action,
                    CategSplit cat_split_type, NewCategAction new_cat_action,
                    bool_t all_perm, Imputer *imputer, size_t min_imp_obs,
                    UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool_t impute_at_fit,
                    uint64_t random_seed, bool_t use_long_double, int nthreads) nogil except +

    void predict_iforest[real_t_, sparse_ix_](
                         real_t_ *numeric_data, int *categ_data,
                         bool_t is_col_major, size_t ncols_numeric, size_t ncols_categ,
                         real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
                         real_t_ *Xr, sparse_ix_ *Xr_ind, sparse_ix_ *Xr_indptr,
                         size_t nrows, int nthreads, bool_t standardize,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double *output_depths, sparse_ix_ *tree_num,
                         double *per_tree_depths,
                         TreesIndexer *indexer) nogil except +

    void get_num_nodes[sparse_ix_](IsoForest &model_outputs, sparse_ix_ *n_nodes, sparse_ix_ *n_terminal, int nthreads) except +

    void get_num_nodes[sparse_ix_](ExtIsoForest &model_outputs, sparse_ix_ *n_nodes, sparse_ix_ *n_terminal, int nthreads) except +

    void calc_similarity[real_t_, sparse_ix_](
                         real_t_ numeric_data[], int categ_data[],
                         real_t_ Xc[], sparse_ix_ Xc_ind[], sparse_ix_ Xc_indptr[],
                         size_t nrows, bool_t use_long_double, int nthreads,
                         bool_t assume_full_distr, bool_t standardize_dist, bool_t as_kernel,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double tmat[], double rmat[], size_t n_from, bool_t use_indexed_references,
                         TreesIndexer *indexer, bool_t is_col_major, size_t ld_numeric, size_t ld_categ) nogil except +

    void impute_missing_values[real_t_, sparse_ix_](
                               real_t_ *numeric_data, int *categ_data, bool_t is_col_major,
                               real_t_ *Xr, sparse_ix_ *Xr_ind, sparse_ix_ *Xr_indptr,
                               size_t nrows, bool_t use_long_double, int nthreads,
                               IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                               Imputer &imputer) nogil except +

    int add_tree[real_t_, sparse_ix_](
                 IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                 real_t_ *numeric_data,  size_t ncols_numeric,
                 int    *categ_data,    size_t ncols_categ,    int *ncat,
                 real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
                 size_t ndim, size_t ntry, CoefType coef_type, bool_t coef_by_prop,
                 real_t_ *sample_weights,
                 size_t nrows, size_t max_depth, size_t ncols_per_tree,
                 bool_t   limit_depth,  bool_t penalize_range, bool_t standardize_data,
                 bool_t   fast_bratio,
                 real_t_ *col_weights, bool_t weigh_by_kurt,
                 double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                 double prob_pick_by_full_gain, double prob_pick_by_dens,
                 double prob_pick_col_by_range, double prob_pick_col_by_var,
                 double prob_pick_col_by_kurt,
                 double min_gain, MissingAction missing_action,
                 CategSplit cat_split_type, NewCategAction new_cat_action,
                 UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
                 bool_t  all_perm, Imputer *imputer, size_t min_imp_obs,
                 TreesIndexer *indexer,
                 real_t_ ref_numeric_data[], int ref_categ_data[],
                 bool_t ref_is_col_major, size_t ref_ld_numeric, size_t ref_ld_categ,
                 real_t_ ref_Xc[], sparse_ix_ ref_Xc_ind[], sparse_ix_ ref_Xc_indptr[],
                 uint64_t random_seed, bool_t use_long_double) nogil except +

    void set_reference_points[real_t_, sparse_ix_](
                              IsoForest *model_outputs, ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
                              const bool_t with_distances,
                              real_t_ *numeric_data, int *categ_data,
                              bool_t is_col_major, size_t ld_numeric, size_t ld_categ,
                              real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
                              real_t_ *Xr, sparse_ix_ *Xr_ind, sparse_ix_ *Xr_indptr,
                              size_t nrows, int nthreads) nogil except +

    void build_tree_indices(TreesIndexer &indexer, const IsoForest &model, int nthreads, const bool_t with_distances) nogil except +

    void build_tree_indices(TreesIndexer &indexer, const ExtIsoForest &model, int nthreads, const bool_t with_distances) nogil except +


cdef extern from "python_helpers.hpp":
    model_t deepcopy_obj[model_t](model_t obj) except +

    IsoForest get_IsoForest() except +
    ExtIsoForest get_ExtIsoForest() except +
    Imputer get_Imputer() except +
    TreesIndexer get_Indexer() except +

    void dealloc_IsoForest(IsoForest &model_outputs) except +
    void dealloc_IsoExtForest(ExtIsoForest &model_outputs_ext) except +
    void dealloc_Imputer(Imputer &imputer) except +
    void dealloc_Indexer(TreesIndexer &indexer) except +

    bool_t get_has_openmp() except +

    size_t py_strerrorlen_s() except +
    void copy_errno_msg(char *inp) except +

cdef extern from "other_helpers.hpp":
    void sort_csc_indices[real_t_, sparse_ix_](real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr, size_t ncols_numeric) nogil except +

    void reconstruct_csr_sliced[real_t_, sparse_ix_](
        real_t_ *orig_Xr, sparse_ix_ *orig_Xr_indptr,
        real_t_ *rec_Xr, sparse_ix_ *rec_Xr_indptr,
        size_t nrows
    ) nogil except +

    void reconstruct_csr_with_categ[real_t_, sparse_ix_, size_t_](
        real_t_ *orig_Xr, sparse_ix_ *orig_Xr_ind, sparse_ix_ *orig_Xr_indptr,
        real_t_ *rec_Xr, sparse_ix_ *rec_Xr_ind, sparse_ix_ *rec_Xr_indptr,
        int *rec_X_cat, bool_t is_col_major,
        size_t_ *cols_numeric, size_t_ *cols_categ,
        size_t nrows, size_t ncols, size_t ncols_numeric, size_t ncols_categ
    ) nogil except +

IF UNAME_SYSNAME != "Windows":
    cdef FILE* cy_fopen(str fname, bool_t read):
        cdef bytes fname_py = fname.encode()
        cdef char* fname_c = fname_py
        cdef char* mode
        if (read):
            mode = b"rb"
        else:
            mode = b"wb"
        cdef FILE *out = fopen(fname_c, mode)
        return out
ELSE:
    from libc.stddef cimport wchar_t
    cdef extern from "stdio.h":
        FILE *_wfopen(const wchar_t *filename, const wchar_t *mode)
    cdef FILE* cy_fopen(str fname, bool_t read):
        cdef Py_UNICODE *fname_c = fname
        cdef str mode
        if (read):
            mode = "rb"
        else:
            mode = "wb"

        cdef Py_UNICODE *mode_ptr = mode
        cdef FILE *out = _wfopen(<wchar_t*>fname_c, <wchar_t*>mode_ptr)
        return out

ctypedef fused sparse_ix:
    int
    np.int64_t
    size_t

ctypedef fused real_t:
    float
    double

cdef double* get_ptr_dbl_vec(np.ndarray[double, ndim = 1] a):
    return &a[0]

cdef int* get_ptr_int_vec(np.ndarray[int, ndim = 1] a):
    return &a[0]

cdef size_t* get_ptr_szt_vec(np.ndarray[size_t, ndim = 1] a):
    return &a[0]

cdef float* get_ptr_float_vec(np.ndarray[float, ndim = 1] a):
    return &a[0]

cdef np.int64_t* get_ptr_int64_vec(np.ndarray[np.int64_t, ndim = 1] a):
    return &a[0]

cdef double* get_ptr_dbl_mat(np.ndarray[double, ndim = 2] a):
    return &a[0, 0]

cdef int* get_ptr_int_mat(np.ndarray[int, ndim = 2] a):
    return &a[0, 0]

cdef float* get_ptr_float_mat(np.ndarray[float, ndim = 2] a):
    return &a[0, 0]

def _get_has_openmp():
    return get_has_openmp()

def _sort_csc_indices(Xcsc):
    cdef size_t ncols_numeric = Xcsc.shape[1] if isspmatrix_csc(Xcsc) else Xcsc.shape[0]
    if (Xcsc.indptr.shape[0] > 1) and (Xcsc.data.shape[0] > 1):
        if Xcsc.data.dtype == ctypes.c_double:
            if Xcsc.indices.dtype == ctypes.c_int:
                sort_csc_indices(get_ptr_dbl_vec(Xcsc.data), get_ptr_int_vec(Xcsc.indices), get_ptr_int_vec(Xcsc.indptr), ncols_numeric)
            elif Xcsc.indices.dtype == np.int64:
                sort_csc_indices(get_ptr_dbl_vec(Xcsc.data), get_ptr_int64_vec(Xcsc.indices), get_ptr_int64_vec(Xcsc.indptr), ncols_numeric)
            else:
                sort_csc_indices(get_ptr_dbl_vec(Xcsc.data), get_ptr_szt_vec(Xcsc.indices), get_ptr_szt_vec(Xcsc.indptr), ncols_numeric)
        elif Xcsc.data.dtype == ctypes.c_float:
            if Xcsc.indices.dtype == ctypes.c_int:
                sort_csc_indices(get_ptr_float_vec(Xcsc.data), get_ptr_int_vec(Xcsc.indices), get_ptr_int_vec(Xcsc.indptr), ncols_numeric)
            elif Xcsc.indices.dtype == np.int64:
                sort_csc_indices(get_ptr_float_vec(Xcsc.data), get_ptr_int64_vec(Xcsc.indices), get_ptr_int64_vec(Xcsc.indptr), ncols_numeric)
            else:
                sort_csc_indices(get_ptr_float_vec(Xcsc.data), get_ptr_szt_vec(Xcsc.indices), get_ptr_szt_vec(Xcsc.indptr), ncols_numeric)
        else:
            raise ValueError("Invalid dtype for 'X'.")


def _reconstruct_csr_sliced(
        np.ndarray[real_t, ndim=1] orig_Xr,
        np.ndarray[sparse_ix, ndim=1] orig_Xr_indptr,
        np.ndarray[real_t, ndim=1] rec_Xr,
        np.ndarray[sparse_ix, ndim=1] rec_Xr_indptr,
        size_t nrows
    ):
    cdef real_t *ptr_orig_Xr = NULL
    cdef real_t *ptr_rec_Xr = NULL
    if orig_Xr.shape[0]:
        ptr_orig_Xr = &orig_Xr[0]
    if rec_Xr.shape[0]:
        ptr_rec_Xr = &rec_Xr[0]
    reconstruct_csr_sliced(
        ptr_orig_Xr, &orig_Xr_indptr[0],
        ptr_rec_Xr, &rec_Xr_indptr[0],
        nrows
    )

def _reconstruct_csr_with_categ(
        np.ndarray[real_t, ndim=1] orig_Xr,
        np.ndarray[sparse_ix, ndim=1] orig_Xr_ind,
        np.ndarray[sparse_ix, ndim=1] orig_Xr_indptr,
        np.ndarray[real_t, ndim=1] rec_Xr,
        np.ndarray[sparse_ix, ndim=1] rec_Xr_ind,
        np.ndarray[sparse_ix, ndim=1] rec_Xr_indptr,
        np.ndarray[int, ndim=2] X_cat,
        np.ndarray[size_t, ndim=1] cols_numeric,
        np.ndarray[size_t, ndim=1] cols_categ,
        size_t nrows, size_t ncols,
        bool_t is_col_major
    ):
    
    cdef int *ptr_X_cat = NULL
    if (X_cat.shape[0] > 0) and (X_cat.shape[1] > 0):
        ptr_X_cat = &X_cat[0,0]
    cdef real_t *ptr_orig_Xr = NULL
    cdef real_t *ptr_rec_Xr = NULL
    cdef sparse_ix *ptr_orig_Xr_ind = NULL
    cdef sparse_ix *ptr_rec_Xr_ind = NULL
    if orig_Xr.shape[0]:
        ptr_orig_Xr = &orig_Xr[0]
        ptr_orig_Xr_ind = &orig_Xr_ind[0]
    if rec_Xr.shape[0]:
        ptr_rec_Xr = &rec_Xr[0]
        ptr_rec_Xr_ind = &rec_Xr_ind[0]

    cdef size_t *ptr_cols_numeric = NULL
    cdef size_t *ptr_cols_categ = NULL
    if cols_numeric.shape[0]:
        ptr_cols_numeric = &cols_numeric[0]
    if cols_categ.shape[0]:
        ptr_cols_categ = &cols_categ[0]

    reconstruct_csr_with_categ(
        ptr_orig_Xr, ptr_orig_Xr_ind, &orig_Xr_indptr[0],
        ptr_rec_Xr, ptr_rec_Xr_ind, &rec_Xr_indptr[0],
        ptr_X_cat, is_col_major,
        ptr_cols_numeric, ptr_cols_categ,
        nrows, ncols, cols_numeric.shape[0], cols_categ.shape[0]
    )

def get_list_left_categories(vector[signed char] &cat_split):
    cdef size_t ix
    cdef size_t n_left = 0
    for ix in range(cat_split.size()):
        n_left += cat_split[ix] == <signed char>1
    cdef np.ndarray[int, ndim=1] categs_ = np.empty(n_left, dtype=ctypes.c_int)
    cdef int *categs = &categs_[0]
    cdef size_t n_used = 0
    cdef int iix
    for iix in range(<int>(cat_split.size())):
        if cat_split[iix] == <signed char>1:
            categs[n_used] = iix
            n_used += 1
    return list(cat_split)

cdef class isoforest_cpp_obj:
    cdef IsoForest     isoforest
    cdef ExtIsoForest  ext_isoforest
    cdef Imputer       imputer
    cdef TreesIndexer  indexer

    def __dealloc__(self):
        dealloc_IsoForest(self.isoforest)
        dealloc_IsoExtForest(self.ext_isoforest)
        dealloc_Imputer(self.imputer)
        dealloc_Indexer(self.indexer)

    def __getstate__(self):
        cdef IsoForest *ptr_IsoForest = NULL
        cdef ExtIsoForest *ptr_ExtIsoForest = NULL
        cdef Imputer *ptr_Imputer = NULL
        cdef TreesIndexer *ptr_Indexer = NULL

        if not self.isoforest.trees.empty():
            ptr_IsoForest = &self.isoforest
        else:
            ptr_ExtIsoForest = &self.ext_isoforest
        if not self.imputer.imputer_tree.empty():
            ptr_Imputer = &self.imputer
        if not self.indexer.indices.empty():
            ptr_Indexer = &self.indexer

        cdef size_t size_ser = determine_serialized_size_combined(
            ptr_IsoForest,
            ptr_ExtIsoForest,
            ptr_Imputer,
            ptr_Indexer,
            <size_t>0
        )
        cdef bytes serialized = bytes(size_ser)
        serialize_combined(
            ptr_IsoForest,
            ptr_ExtIsoForest,
            ptr_Imputer,
            ptr_Indexer,
            <const char*>NULL,
            <size_t>0,
            serialized
        )
        return serialized

    def __setstate__(self, bytes state):
        deserialize_combined(
            state,
            &self.isoforest,
            &self.ext_isoforest,
            &self.imputer,
            &self.indexer,
            <char*>NULL
        )

    def __deepcopy__(self, memo):
        other = isoforest_cpp_obj()
        other.isoforest = deepcopy_obj(self.isoforest)
        other.ext_isoforest = deepcopy_obj(self.ext_isoforest)
        other.imputer = deepcopy_obj(self.imputer)
        other.indexer = deepcopy_obj(self.indexer)
        return other

    def drop_imputer(self):
        dealloc_Imputer(self.imputer)
        self.imputer = get_Imputer()

    def drop_indexer(self):
        dealloc_Indexer(self.indexer)
        self.indexer = get_Indexer()

    def drop_reference_points(self):
        cdef size_t tree, ntrees
        ntrees = self.indexer.indices.size()
        if (ntrees > 0) and (self.indexer.indices.front().reference_points.empty()):
            return None
        for tree in range(ntrees):
            self.indexer.indices[tree].reference_points.clear()
            self.indexer.indices[tree].reference_indptr.clear()
            self.indexer.indices[tree].reference_mapping.clear()

    def get_cpp_obj(self, is_extended):
        if is_extended:
            return self.ext_isoforest
        else:
            return self.isoforest

    def get_imputer(self):
        return self.imputer

    def get_indexer(self):
        return self.indexer

    def fit_model(self,
                  np.ndarray[real_t, ndim=1] placeholder_real_t,
                  np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
                  X_num, X_cat, ncat, sample_weights, col_weights,
                  size_t nrows, size_t ncols_numeric, size_t ncols_categ,
                  size_t ndim, size_t ntry, coef_type, bool_t coef_by_prop,
                  bool_t with_replacement, bool_t weight_as_sample,
                  size_t sample_size, size_t ntrees,
                  size_t max_depth,   size_t ncols_per_tree,
                  bool_t limit_depth, bool_t penalize_range, bool_t standardize_data,
                  scoring_metric, bool_t fast_bratio,
                  bool_t calc_dist, bool_t standardize_dist, bool_t sq_dist,
                  bool_t calc_depth, bool_t standardize_depth,
                  bool_t weigh_by_kurt,
                  double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                  double prob_pick_by_full_gain, double prob_pick_by_dens,
                  double prob_pick_col_by_range, double prob_pick_col_by_var,
                  double prob_pick_col_by_kurt,
                  double min_gain, missing_action, cat_split_type, new_cat_action,
                  bool_t build_imputer, size_t min_imp_obs,
                  depth_imp, weigh_imp_rows, bool_t impute_at_fit,
                  bool_t all_perm, uint64_t random_seed, bool_t use_long_double,
                  int nthreads):
        cdef real_t*     numeric_data_ptr    =  NULL
        cdef int*        categ_data_ptr      =  NULL
        cdef int*        ncat_ptr            =  NULL
        cdef real_t*     Xc_ptr              =  NULL
        cdef sparse_ix*  Xc_ind_ptr          =  NULL
        cdef sparse_ix*  Xc_indptr_ptr       =  NULL
        cdef real_t*     sample_weights_ptr  =  NULL
        cdef real_t*     col_weights_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                if real_t is float:
                    numeric_data_ptr  =  get_ptr_float_mat(X_num)
                else:
                    if X_num.dtype != ctypes.c_double:
                        X_num = X_num.astype(ctypes.c_double)
                    numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
            else:
                if real_t is float:
                    Xc_ptr         =  get_ptr_float_vec(X_num.data)
                else:
                    if X_num.data.dtype != ctypes.c_double:
                        X_num.data = X_num.data.astype(ctypes.c_double)
                    Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                if sparse_ix is int:
                    Xc_ind_ptr     =  get_ptr_int_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int_vec(X_num.indptr)
                elif sparse_ix is np.int64_t:
                    Xc_ind_ptr     =  get_ptr_int64_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int64_vec(X_num.indptr)
                else:
                    if X_num.indices.dtype != ctypes.c_size_t:
                        X_num.indices = X_num.indices.astype(ctypes.c_size_t)
                    if X_num.indptr.dtype != ctypes.c_size_t:
                        X_num.indptr = X_num.indptr.astype(ctypes.c_size_t)
                    Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            categ_data_ptr     =  get_ptr_int_mat(X_cat)
            ncat_ptr           =  get_ptr_int_vec(ncat)
        if sample_weights is not None:
            if real_t is float:
                sample_weights_ptr =  get_ptr_float_vec(sample_weights)
            else:
                if sample_weights.dtype != ctypes.c_double:
                    sample_weights = sample_weights.astype(ctypes.c_double)
                sample_weights_ptr =  get_ptr_dbl_vec(sample_weights)
        if col_weights is not None:
            if real_t is float:
                col_weights_ptr    =  get_ptr_float_vec(col_weights)
            else:
                if col_weights.dtype != ctypes.c_double:
                    col_weights = col_weights.astype(ctypes.c_double)
                col_weights_ptr    =  get_ptr_dbl_vec(col_weights)

        cdef CoefType        coef_type_C       =  Normal
        cdef CategSplit      cat_split_type_C  =  SubSet
        cdef NewCategAction  new_cat_action_C  =  Weighted
        cdef MissingAction   missing_action_C  =  Divide
        cdef UseDepthImp     depth_imp_C       =  Same
        cdef WeighImpRows    weigh_imp_rows_C  =  Flat
        cdef ScoringMetric   scoring_metric_C  =  Depth

        if coef_type == "uniform":
            coef_type_C       =  Uniform
        if cat_split_type == "single_categ":
            cat_split_type_C  =  SingleCateg
        if new_cat_action == "smallest":
            new_cat_action_C  =  Smallest
        elif new_cat_action == "random":
            new_cat_action_C  =  Random
        if missing_action == "impute":
            missing_action_C  =  Impute
        elif missing_action == "fail":
            missing_action_C  =  Fail
        if depth_imp == "lower":
            depth_imp_C       =  Lower
        elif depth_imp == "higher":
            depth_imp_C       =  Higher
        if weigh_imp_rows == "inverse":
            weigh_imp_rows_C  =  Inverse
        elif weigh_imp_rows == "prop":
            weigh_imp_rows_C  =  Prop
        if scoring_metric == "adj_depth":
            scoring_metric_C  =  AdjDepth
        elif scoring_metric == "density":
            scoring_metric_C  =  Density
        elif scoring_metric == "adj_density":
            scoring_metric_C  =  AdjDensity
        elif scoring_metric == "boxed_density":
            scoring_metric_C  =  BoxedDensity
        elif scoring_metric == "boxed_density2":
            scoring_metric_C  =  BoxedDensity2
        elif scoring_metric == "boxed_ratio":
            scoring_metric_C  =  BoxedRatio

        cdef np.ndarray[double, ndim = 1]  tmat    =  np.empty(0, dtype = ctypes.c_double)
        cdef np.ndarray[double, ndim = 2]  dmat    =  np.empty((0, 0), dtype = ctypes.c_double)
        cdef np.ndarray[double, ndim = 1]  depths  =  np.empty(0, dtype = ctypes.c_double)
        cdef double*  tmat_ptr    =  NULL
        cdef double*  dmat_ptr    =  NULL
        cdef double*  depths_ptr  =  NULL

        if calc_dist:
            tmat      =  np.zeros(int((nrows * (nrows - <size_t>1)) / <size_t>2), dtype = ctypes.c_double)
            tmat_ptr  =  &tmat[0]
            if sq_dist:
                dmat      =  np.zeros((nrows, nrows), dtype = ctypes.c_double)
                dmat_ptr  =  &dmat[0, 0]
        if calc_depth:
            depths      =  np.zeros(nrows, dtype = ctypes.c_double)
            depths_ptr  =  &depths[0]

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        cdef Imputer*       imputer_ptr    =  NULL

        dealloc_IsoForest(self.isoforest)
        dealloc_IsoExtForest(self.ext_isoforest)
        dealloc_Imputer(self.imputer)
        if ndim == 1:
            self.isoforest      =  get_IsoForest()
            model_ptr           =  &self.isoforest
        else:
            self.ext_isoforest  =  get_ExtIsoForest()
            ext_model_ptr       =  &self.ext_isoforest

        if build_imputer:
            self.imputer = get_Imputer()
            imputer_ptr  = &self.imputer

        cdef int ret_val = 0

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            ret_val = \
            fit_iforest(model_ptr, ext_model_ptr,
                        numeric_data_ptr,  ncols_numeric,
                        categ_data_ptr,    ncols_categ,    ncat_ptr,
                        Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                        ndim, ntry, coef_type_C, coef_by_prop,
                        sample_weights_ptr, with_replacement, weight_as_sample,
                        nrows, sample_size, ntrees,
                        max_depth,   ncols_per_tree,
                        limit_depth, penalize_range, standardize_data,
                        scoring_metric_C, fast_bratio,
                        standardize_dist, tmat_ptr,
                        depths_ptr, standardize_depth,
                        col_weights_ptr, weigh_by_kurt,
                        prob_pick_by_gain_pl, prob_pick_by_gain_avg,
                        prob_pick_by_full_gain, prob_pick_by_dens,
                        prob_pick_col_by_range, prob_pick_col_by_var,
                        prob_pick_col_by_kurt,
                        min_gain, missing_action_C,
                        cat_split_type_C, new_cat_action_C,
                        all_perm, imputer_ptr, min_imp_obs,
                        depth_imp_C, weigh_imp_rows_C, impute_at_fit,
                        random_seed, use_long_double, nthreads)

        if cy_check_interrupt_switch():
            cy_tick_off_interrupt_switch()
            raise InterruptedError("Error: procedure was interrupted.")

        if ret_val == return_EXIT_FAILURE():
            raise ValueError("Error: something went wrong. Procedure failed.")

        if (calc_dist) and (sq_dist):
            tmat_to_dense(tmat_ptr, dmat_ptr, nrows, 0. if standardize_dist else np.inf)

        return depths, tmat, dmat, X_num, X_cat

    def fit_tree(self,
                 np.ndarray[real_t, ndim=1] placeholder_real_t,
                 np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
                 X_num, X_cat, ncat, sample_weights, col_weights,
                 size_t nrows, size_t ncols_numeric, size_t ncols_categ,
                 size_t ndim, size_t ntry, coef_type, bool_t coef_by_prop,
                 size_t max_depth,   size_t ncols_per_tree,
                 bool_t limit_depth, bool_t penalize_range, bool_t standardize_data,
                 bool_t fast_bratio, bool_t weigh_by_kurt,
                 double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                 double prob_pick_by_full_gain, double prob_pick_by_dens,
                 double prob_pick_col_by_range, double prob_pick_col_by_var,
                 double prob_pick_col_by_kurt,
                 double min_gain, missing_action, cat_split_type, new_cat_action,
                 bool_t build_imputer, size_t min_imp_obs,
                 depth_imp, weigh_imp_rows,
                 bool_t all_perm,
                 ref_X_num, ref_X_cat,
                 uint64_t random_seed, bool_t use_long_double):
        cdef real_t*     numeric_data_ptr    =  NULL
        cdef int*        categ_data_ptr      =  NULL
        cdef int*        ncat_ptr            =  NULL
        cdef real_t*     Xc_ptr              =  NULL
        cdef sparse_ix*  Xc_ind_ptr          =  NULL
        cdef sparse_ix*  Xc_indptr_ptr       =  NULL
        cdef real_t*     sample_weights_ptr  =  NULL
        cdef real_t*     col_weights_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                if real_t is float:
                    numeric_data_ptr  =  get_ptr_float_mat(X_num)
                else:
                    if X_num.dtype != ctypes.c_double:
                        X_num = X_num.astype(ctypes.c_double)
                    numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
            else:
                if real_t is float:
                    Xc_ptr         =  get_ptr_float_vec(X_num.data)
                else:
                    if X_num.data.dtype != ctypes.c_double:
                        X_num.data = X_num.data.astype(ctypes.c_double)
                    Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                if sparse_ix is int:
                    Xc_ind_ptr     =  get_ptr_int_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int_vec(X_num.indptr)
                elif sparse_ix is np.int64_t:
                    Xc_ind_ptr     =  get_ptr_int64_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int64_vec(X_num.indptr)
                else:
                    if X_num.indices.dtype != ctypes.c_size_t:
                        X_num.indices = X_num.indices.astype(ctypes.c_size_t)
                    if X_num.indptr.dtype != ctypes.c_size_t:
                        X_num.indptr = X_num.indptr.astype(ctypes.c_size_t)
                    Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            if ncat.dtype != ctypes.c_int:
                ncat = ncat.astype(ctypes.c_int)
            categ_data_ptr     =  get_ptr_int_mat(X_cat)
            ncat_ptr           =  get_ptr_int_vec(ncat)
        if sample_weights is not None:
            if real_t is float:
                sample_weights_ptr  =  get_ptr_float_vec(sample_weights)
            else:
                if sample_weights.dtype != ctypes.c_double:
                    sample_weights  = sample_weights.astype(ctypes.c_double)
                sample_weights_ptr  =  get_ptr_dbl_vec(sample_weights)
        if col_weights is not None:
            if real_t is float:
                col_weights_ptr     =  get_ptr_float_vec(col_weights)
            else:
                if col_weights.dtype != ctypes.c_double:
                    col_weights = col_weights.astype(ctypes.c_double)
                col_weights_ptr =  get_ptr_dbl_vec(col_weights)


        cdef real_t*     ref_numeric_data_ptr    =  NULL
        cdef int*        ref_categ_data_ptr      =  NULL
        cdef real_t*     ref_Xc_ptr              =  NULL
        cdef sparse_ix*  ref_Xc_ind_ptr          =  NULL
        cdef sparse_ix*  ref_Xc_indptr_ptr       =  NULL
        cdef bool_t      ref_is_col_major    =  True
        cdef size_t      ref_ncols_numeric   =  0
        cdef size_t      ref_ncols_categ     =  0

        if ref_X_num is not None:
            if not issparse(ref_X_num):
                if real_t is float:
                    ref_numeric_data_ptr  =  get_ptr_float_mat(ref_X_num)
                else:
                    if ref_X_num.dtype != ctypes.c_double:
                        ref_X_num = ref_X_num.astype(ctypes.c_double)
                    ref_numeric_data_ptr  =  get_ptr_dbl_mat(ref_X_num)
                ref_ncols_numeric      =  ref_X_num.shape[1]
                ref_is_col_major       =  np.isfortran(ref_X_num)
            else:
                if real_t is float:
                    ref_Xc_ptr         =  get_ptr_float_vec(ref_X_num.data)
                else:
                    if ref_X_num.data.dtype != ctypes.c_double:
                        ref_X_num.data = ref_X_num.data.astype(ctypes.c_double)
                    ref_Xc_ptr         =  get_ptr_dbl_vec(ref_X_num.data)
                if sparse_ix is int:
                    ref_Xc_ind_ptr     =  get_ptr_int_vec(ref_X_num.indices)
                    ref_Xc_indptr_ptr  =  get_ptr_int_vec(ref_X_num.indptr)
                elif sparse_ix is np.int64_t:
                    ref_Xc_ind_ptr     =  get_ptr_int64_vec(ref_X_num.indices)
                    ref_Xc_indptr_ptr  =  get_ptr_int64_vec(ref_X_num.indptr)
                else:
                    if ref_X_num.indices.dtype != ctypes.c_size_t:
                        ref_X_num.indices = ref_X_num.indices.astype(ctypes.c_size_t)
                    if ref_X_num.indptr.dtype != ctypes.c_size_t:
                        ref_X_num.indptr = ref_X_num.indptr.astype(ctypes.c_size_t)
                    ref_Xc_ind_ptr     =  get_ptr_szt_vec(ref_X_num.indices)
                    ref_Xc_indptr_ptr  =  get_ptr_szt_vec(ref_X_num.indptr)
        if ref_X_cat is not None:
            ref_categ_data_ptr     =  get_ptr_int_mat(ref_X_cat)
            ref_ncols_categ        =  ref_X_cat.shape[1]
            ref_is_col_major       =  np.isfortran(ref_X_cat)

        cdef CoefType        coef_type_C       =  Normal
        cdef CategSplit      cat_split_type_C  =  SubSet
        cdef NewCategAction  new_cat_action_C  =  Weighted
        cdef MissingAction   missing_action_C  =  Divide
        cdef UseDepthImp     depth_imp_C       =  Same
        cdef WeighImpRows    weigh_imp_rows_C  =  Flat
        

        if coef_type == "uniform":
            coef_type_C       =  Uniform
        if cat_split_type == "single_categ":
            cat_split_type_C  =  SingleCateg
        if new_cat_action == "smallest":
            new_cat_action_C  =  Smallest
        elif new_cat_action == "random":
            new_cat_action_C  =  Random
        if missing_action == "impute":
            missing_action_C  =  Impute
        elif missing_action == "fail":
            missing_action_C  =  Fail
        if depth_imp == "lower":
            depth_imp_C       =  Lower
        elif depth_imp == "higher":
            depth_imp_C       =  Higher
        if weigh_imp_rows == "inverse":
            weigh_imp_rows_C  =  Inverse
        elif weigh_imp_rows == "prop":
            weigh_imp_rows_C  =  Prop

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if ndim == 1:
            model_ptr           =  &self.isoforest
        else:
            ext_model_ptr       =  &self.ext_isoforest

        cdef Imputer *imputer_ptr = NULL
        if build_imputer:
            imputer_ptr = &self.imputer

        cdef TreesIndexer *indexer_ptr = NULL
        if not self.indexer.indices.empty():
            indexer_ptr = &self.indexer

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            add_tree(model_ptr, ext_model_ptr,
                     numeric_data_ptr,  ncols_numeric,
                     categ_data_ptr,    ncols_categ,    ncat_ptr,
                     Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                     ndim, ntry, coef_type_C, coef_by_prop,
                     sample_weights_ptr,
                     nrows, max_depth, ncols_per_tree,
                     limit_depth,  penalize_range, standardize_data, fast_bratio,
                     col_weights_ptr, weigh_by_kurt,
                     prob_pick_by_gain_pl, prob_pick_by_gain_avg,
                     prob_pick_by_full_gain, prob_pick_by_dens,
                     prob_pick_col_by_range, prob_pick_col_by_var,
                     prob_pick_col_by_kurt,
                     min_gain, missing_action_C,
                     cat_split_type_C, new_cat_action_C,
                     depth_imp_C, weigh_imp_rows_C,
                     all_perm, imputer_ptr, min_imp_obs,
                     indexer_ptr,
                     ref_numeric_data_ptr, ref_categ_data_ptr,
                     ref_is_col_major, ref_ncols_numeric, ref_ncols_categ,
                     ref_Xc_ptr, ref_Xc_ind_ptr, ref_Xc_indptr_ptr,
                     random_seed, use_long_double)

    def predict(self,
                np.ndarray[real_t, ndim=1] placeholder_real_t,
                np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
                X_num, X_cat, is_extended,
                size_t nrows, int nthreads, bool_t standardize,
                bool_t output_tree_num, bool_t output_per_tree_depths):

        cdef real_t*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef real_t*     Xc_ptr            =  NULL
        cdef sparse_ix*  Xc_ind_ptr        =  NULL
        cdef sparse_ix*  Xc_indptr_ptr     =  NULL
        cdef real_t*     Xr_ptr            =  NULL
        cdef sparse_ix*  Xr_ind_ptr        =  NULL
        cdef sparse_ix*  Xr_indptr_ptr     =  NULL

        cdef bool_t is_col_major    =  True
        cdef size_t ncols_numeric   =  0
        cdef size_t ncols_categ     =  0

        if X_num is not None:
            if not issparse(X_num):
                if real_t is float:
                    numeric_data_ptr   =  get_ptr_float_mat(X_num)
                else:
                    if X_num.dtype != ctypes.c_double:
                        X_num = X_num.astype(ctypes.c_double)
                    numeric_data_ptr   =  get_ptr_dbl_mat(X_num)
                ncols_numeric      =  X_num.shape[1]
                is_col_major       =  np.isfortran(X_num)
            else:
                if isspmatrix_csc(X_num):
                    if X_num.data.shape[0]:
                        if real_t is float:
                            Xc_ptr         =  get_ptr_float_vec(X_num.data)
                        else:
                            if X_num.data.dtype != ctypes.c_double:
                                X_num.data = X_num.data.astype(ctypes.c_double)
                            Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                    if sparse_ix is int:
                        if X_num.indices.shape[0]:
                            Xc_ind_ptr     =  get_ptr_int_vec(X_num.indices)
                        Xc_indptr_ptr      =  get_ptr_int_vec(X_num.indptr)
                    elif sparse_ix is np.int64_t:
                        if X_num.indices.shape[0]:
                            Xc_ind_ptr     =  get_ptr_int64_vec(X_num.indices)
                        Xc_indptr_ptr      =  get_ptr_int64_vec(X_num.indptr)
                    else:
                        if X_num.indices.shape[0]:
                            if X_num.indices.dtype != ctypes.c_size_t:
                                X_num.indices = X_num.indices.astype(ctypes.c_size_t)
                            Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                        if X_num.indptr.dtype != ctypes.c_size_t:
                            X_num.indptr = X_num.indptr.astype(ctypes.c_size_t)
                        Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
                else:
                    if X_num.data.shape[0]:
                        if real_t is float:
                            Xr_ptr         =  get_ptr_float_vec(X_num.data)
                        else:
                            if X_num.data.dtype != ctypes.c_double:
                                X_num.data = X_num.data.astype(ctypes.c_double)
                            Xr_ptr         =  get_ptr_dbl_vec(X_num.data)
                    if sparse_ix is int:
                        if X_num.indices.shape[0]:
                            Xr_ind_ptr     =  get_ptr_int_vec(X_num.indices)
                        Xr_indptr_ptr      =  get_ptr_int_vec(X_num.indptr)
                    elif sparse_ix is np.int64_t:
                        if X_num.indices.shape[0]:
                            Xr_ind_ptr     =  get_ptr_int64_vec(X_num.indices)
                        Xr_indptr_ptr      =  get_ptr_int64_vec(X_num.indptr)
                    else:
                        if X_num.indices.shape[0]:
                            if X_num.indices.dtype != ctypes.c_size_t:
                                X_num.indices = X_num.indices.astype(ctypes.c_size_t)
                            Xr_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                        if X_num.indptr.dtype != ctypes.c_size_t:
                            X_num.indptr = X_num.indptr.astype(ctypes.c_size_t)
                        Xr_indptr_ptr      =  get_ptr_szt_vec(X_num.indptr)

        if X_cat is not None:
            categ_data_ptr    =  get_ptr_int_mat(X_cat)
            ncols_categ       =  X_cat.shape[1]
            is_col_major      =  np.isfortran(X_cat)

        cdef np.ndarray[double, ndim = 1]    depths       =  np.zeros(nrows, dtype = ctypes.c_double)
        cdef np.ndarray[double, ndim = 2]    tree_depths  =  np.empty((0, 0), dtype = ctypes.c_double)
        cdef np.ndarray[sparse_ix, ndim = 2] tree_num     =  np.empty((0, 0), order = 'F',
                                                                      dtype = placeholder_sparse_ix.dtype)
        cdef double* depths_ptr       =  &depths[0]
        cdef double* tree_depths_ptr  =  NULL
        cdef sparse_ix* tree_num_ptr  =  NULL

        if output_tree_num or output_per_tree_depths:
            if is_extended:
                sz = self.ext_isoforest.hplanes.size()
            else:
                sz = self.isoforest.trees.size()
            if output_tree_num:
                tree_num         =  np.empty((nrows, sz), dtype = placeholder_sparse_ix.dtype, order = 'F')
                tree_num_ptr     =  &tree_num[0, 0]
            if output_per_tree_depths:
                tree_depths      =  np.empty((nrows, sz), dtype = ctypes.c_double, order = 'C')
                tree_depths_ptr  =  &tree_depths[0, 0]

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest

        cdef TreesIndexer*  indexer_ptr = NULL
        if not self.indexer.indices.empty():
            indexer_ptr = &self.indexer
        
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            predict_iforest(numeric_data_ptr, categ_data_ptr,
                            is_col_major, ncols_numeric, ncols_categ,
                            Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                            Xr_ptr, Xr_ind_ptr, Xr_indptr_ptr,
                            nrows, nthreads, standardize,
                            model_ptr, ext_model_ptr,
                            depths_ptr, tree_num_ptr, tree_depths_ptr,
                            indexer_ptr)

        return depths, tree_num, tree_depths


    def dist(self,
             np.ndarray[real_t, ndim=1] placeholder_real_t,
             np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
             X_num, X_cat, is_extended,
             size_t nrows, bool_t use_long_double, int nthreads,
             bool_t assume_full_distr,
             bool_t standardize_dist,    bool_t sq_dist,
             size_t n_from, bool_t use_reference_points,
             bool_t as_kernel):

        cdef real_t*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef real_t*     Xc_ptr            =  NULL
        cdef sparse_ix*  Xc_ind_ptr        =  NULL
        cdef sparse_ix*  Xc_indptr_ptr     =  NULL

        cdef bool_t is_col_major    =  True
        cdef size_t ncols_numeric   =  0
        cdef size_t ncols_categ     =  0

        cdef bool_t avoid_row_major = (
            as_kernel and
            (not use_reference_points or
             self.indexer.indices.empty() or
             self.indexer.indices.front().reference_points.empty())
        )

        if X_num is not None:
            if not issparse(X_num):
                if avoid_row_major and not np.isfortran(X_num):
                    if real_t is float:
                        X_num = np.asfortranarray(X_num, dtype=ctypes.c_float)
                    else:
                        X_num = np.asfortranarray(X_num, dtype=ctypes.c_double)

                if real_t is float:
                    numeric_data_ptr  =  get_ptr_float_mat(X_num)
                else:
                    if X_num.dtype != ctypes.c_double:
                        X_num = X_num.astype(ctypes.c_double)
                    numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
                ncols_numeric  =  X_num.shape[1]
                is_col_major   =  np.isfortran(X_num)
            else:
                if X_num.data.shape[0]:
                    if real_t is float:
                        Xc_ptr         =  get_ptr_float_vec(X_num.data)
                    else:
                        if X_num.data.dtype != ctypes.c_double:
                            X_num.data = X_num.data.astype(ctypes.c_double)
                        Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                if sparse_ix is int:
                    if X_num.indices.shape[0]:
                        Xc_ind_ptr =  get_ptr_int_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int_vec(X_num.indptr)
                elif sparse_ix is np.int64_t:
                    if X_num.indices.shape[0]:
                        Xc_ind_ptr =  get_ptr_int64_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int64_vec(X_num.indptr)
                else:
                    if X_num.indices.shape[0]:
                        if X_num.indices.dtype != ctypes.c_size_t:
                            X_num.indices = X_num.indices.astype(ctypes.c_size_t)
                        Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                    if X_num.indptr.dtype != ctypes.c_size_t:
                        X_num.indptr = X_num.indptr.astype(ctypes.c_size_t)
                    Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            if avoid_row_major and not np.isfortran(X_cat):
                X_cat = np.asfortranarray(X_cat, dtype=ctypes.c_int)

            categ_data_ptr     =  get_ptr_int_mat(X_cat)
            ncols_categ        =  X_cat.shape[1]
            is_col_major       =  np.isfortran(X_cat)

        cdef np.ndarray[double, ndim = 1]  tmat    =  np.empty(0, dtype = ctypes.c_double)
        cdef np.ndarray[double, ndim = 2]  dmat    =  np.empty((0, 0), dtype = ctypes.c_double)
        cdef np.ndarray[double, ndim = 2]  rmat    =  np.empty((0, 0), dtype = ctypes.c_double)
        cdef double*  tmat_ptr    =  NULL
        cdef double*  dmat_ptr    =  NULL
        cdef double*  rmat_ptr    =  NULL

        if n_from != 0:
            rmat = np.zeros((n_from, nrows - n_from), dtype = ctypes.c_double)
            rmat_ptr = &rmat[0, 0]
        elif (
              use_reference_points and
              not self.indexer.indices.empty() and
              not self.indexer.indices.front().reference_points.empty() and
              (as_kernel or not self.indexer.indices.front().node_distances.empty())
        ):
            rmat = np.zeros((nrows, self.indexer.indices.front().reference_points.size()), dtype = ctypes.c_double)
            rmat_ptr = &rmat[0, 0]
        else:
            tmat      =  np.zeros(int((nrows * (nrows - 1)) / 2), dtype = ctypes.c_double)
            tmat_ptr  =  &tmat[0]
            if sq_dist:
                dmat      =  np.zeros((nrows, nrows), dtype = ctypes.c_double)
                dmat_ptr  =  &dmat[0, 0]

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest

        cdef TreesIndexer*  indexer_ptr = NULL
        if not self.indexer.indices.empty():
            indexer_ptr = &self.indexer
        
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            calc_similarity(numeric_data_ptr, categ_data_ptr,
                            Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                            nrows, use_long_double, nthreads,
                            assume_full_distr, standardize_dist, as_kernel,
                            model_ptr, ext_model_ptr,
                            tmat_ptr, rmat_ptr, n_from, use_reference_points,
                            indexer_ptr, is_col_major, ncols_numeric, ncols_categ)

        if cy_check_interrupt_switch():
            cy_tick_off_interrupt_switch()
            raise InterruptedError("Error: procedure was interrupted.")

        cdef double diag_filler

        if (sq_dist) and (n_from == 0) and (not rmat.shape[1]):
            if as_kernel:
                if standardize_dist:
                    diag_filler = 1
                else:
                    diag_filler = max(self.isoforest.trees.size(), self.ext_isoforest.hplanes.size())
            else:
                if standardize_dist:
                    diag_filler = 0
                else:
                    diag_filler = np.inf
            tmat_to_dense(tmat_ptr, dmat_ptr, nrows, diag_filler)

        return tmat, dmat, rmat

    def impute(self,
               np.ndarray[real_t, ndim=1] placeholder_real_t,
               np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
               X_num, X_cat, bool_t is_extended, size_t nrows,
               bool_t use_long_double, int nthreads):
        cdef real_t*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef real_t*     Xr_ptr            =  NULL
        cdef sparse_ix*  Xr_ind_ptr        =  NULL
        cdef sparse_ix*  Xr_indptr_ptr     =  NULL
        cdef bool_t      is_col_major      =  True

        if X_num is not None:
            if not issparse(X_num):
                if real_t is float:
                    numeric_data_ptr  =  get_ptr_float_mat(X_num)
                else:
                    if X_num.dtype != ctypes.c_double:
                        X_num = X_num.astype(ctypes.c_double)
                    numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
                is_col_major      =  np.isfortran(X_num)
            else:
                if X_num.data.shape[0]:
                    if real_t is float:
                        Xr_ptr         =  get_ptr_float_vec(X_num.data)
                    else:
                        if X_num.data.dtype != ctypes.c_double:
                            X_num.data = X_num.data.astype(ctypes.c_double)
                        Xr_ptr         =  get_ptr_dbl_vec(X_num.data)
                if sparse_ix is int:
                    if X_num.indices.shape[0]:
                        Xr_ind_ptr =  get_ptr_int_vec(X_num.indices)
                    Xr_indptr_ptr  =  get_ptr_int_vec(X_num.indptr)
                elif sparse_ix is np.int64_t:
                    if X_num.indices.shape[0]:
                        Xr_ind_ptr =  get_ptr_int64_vec(X_num.indices)
                    Xr_indptr_ptr  =  get_ptr_int64_vec(X_num.indptr)
                else:
                    if X_num.indices.shape[0]:
                        if X_num.indices.dtype != ctypes.c_size_t:
                            X_num.indices = X_num.indices.astype(ctypes.c_size_t)
                        Xr_ind_ptr =  get_ptr_szt_vec(X_num.indices)
                    if X_num.indptr.dtype != ctypes.c_size_t:
                        X_num.indptr = X_num.indptr.astype(ctypes.c_size_t)
                    Xr_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            categ_data_ptr     =  get_ptr_int_mat(X_cat)
            is_col_major       =  np.isfortran(X_cat)

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            impute_missing_values(numeric_data_ptr, categ_data_ptr, is_col_major,
                                  Xr_ptr, Xr_ind_ptr, Xr_indptr_ptr,
                                  nrows, use_long_double, nthreads,
                                  model_ptr, ext_model_ptr,
                                  self.imputer)

        return X_num, X_cat

    def get_n_nodes(self, bool_t is_extended, int nthreads):
        cdef size_t ntrees
        if not is_extended:
            ntrees = self.isoforest.trees.size()
        else:
            ntrees = self.ext_isoforest.hplanes.size()
        cdef np.ndarray[size_t, ndim=1] n_nodes    = np.empty(ntrees, dtype=ctypes.c_size_t)
        cdef np.ndarray[size_t, ndim=1] n_terminal = np.empty(ntrees, dtype=ctypes.c_size_t)
        if not is_extended:
            get_num_nodes(self.isoforest, &n_nodes[0], &n_terminal[0], nthreads)
        else:
            get_num_nodes(self.ext_isoforest, &n_nodes[0], &n_terminal[0], nthreads)
        return n_nodes, n_terminal

    def append_trees_from_other(self, isoforest_cpp_obj other, bool_t is_extended):
        cdef IsoForest *ptr_model = NULL
        cdef IsoForest *ptr_other = NULL
        cdef ExtIsoForest *ptr_ext_model = NULL
        cdef ExtIsoForest *ptr_ext_other = NULL
        cdef Imputer *ptr_imp = NULL
        cdef Imputer *prt_iother = NULL
        cdef TreesIndexer *ptr_indexer = NULL
        cdef TreesIndexer *prt_ind_other = NULL

        if is_extended:
            ptr_ext_model = &self.ext_isoforest
            ptr_ext_other = &other.ext_isoforest
        else:
            ptr_model = &self.isoforest
            ptr_other = &other.isoforest

        if not self.imputer.imputer_tree.empty():
            ptr_imp = &self.imputer
        if not other.imputer.imputer_tree.empty():
            prt_iother = &other.imputer

        if not self.indexer.indices.empty():
            ptr_indexer = &self.indexer
        if not other.indexer.indices.empty():
            prt_ind_other = &other.indexer

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            merge_models(ptr_model, ptr_other,
                         ptr_ext_model, ptr_ext_other,
                         ptr_imp, prt_iother,
                         ptr_indexer, prt_ind_other)

    def serialize_obj(self, str fpath, bytes metadata, bool_t is_extended=False, bool_t has_imputer=False):
        cdef FILE* file_ptr = cy_fopen(fpath, read=False)
        cdef size_t errmgs_len = 0
        cdef bytes errstr = bytes(0)
        if not file_ptr:
            errmgs_len = py_strerrorlen_s()
            errstr = bytes(errmgs_len)
            copy_errno_msg(errstr)
            raise ValueError(errstr.decode("utf-8"))
        
        cdef IsoForest *ptr_model = NULL
        cdef ExtIsoForest *ptr_ext_model = NULL
        cdef Imputer *ptr_imputer = NULL
        cdef TreesIndexer *ptr_indexer = NULL
        if not is_extended:
            ptr_model = &self.isoforest
        else:
            ptr_ext_model = &self.ext_isoforest
        if has_imputer:
            ptr_imputer = &self.imputer
        if not self.indexer.indices.empty():
            ptr_indexer = &self.indexer
        try:
            serialize_combined(
                ptr_model,
                ptr_ext_model,
                ptr_imputer,
                ptr_indexer,
                metadata,
                len(metadata),
                file_ptr
            )
        finally:
            fclose(file_ptr)

    def deserialize_obj(self, str fpath):
        cdef bool_t is_isotree_model = 0
        cdef bool_t is_compatible = 0
        cdef bool_t has_combined_objects = 0
        cdef bool_t has_IsoForest = 0
        cdef bool_t has_ExtIsoForest = 0
        cdef bool_t has_Imputer = 0
        cdef bool_t has_Indexer = 0
        cdef bool_t has_metadata = 0
        cdef size_t size_metadata = 0

        cdef bytes metadata
        cdef char *ptr_metadata

        cdef IsoForest *ptr_model = &self.isoforest
        cdef ExtIsoForest *ptr_ext_model = &self.ext_isoforest
        cdef Imputer *ptr_imputer = &self.imputer
        cdef TreesIndexer *ptr_indexer = &self.indexer

        cdef FILE* file_ptr = cy_fopen(fpath, read=True)
        cdef size_t errmgs_len = 0
        cdef bytes errstr = bytes(0)
        if not file_ptr:
            errmgs_len = py_strerrorlen_s()
            errstr = bytes(errmgs_len)
            copy_errno_msg(errstr)
            raise ValueError(errstr.decode("utf-8"))

        try:
            inspect_serialized_object(
                file_ptr,
                is_isotree_model,
                is_compatible,
                has_combined_objects,
                has_IsoForest,
                has_ExtIsoForest,
                has_Imputer,
                has_Indexer,
                has_metadata,
                size_metadata
            )

            if (not is_isotree_model or not has_combined_objects):
                raise ValueError("Input file is not a serialized isotree model.");
            if (not is_compatible):
                raise ValueError("Model file format is incompatible.");
            if (not size_metadata):
                raise ValueError("Input file does not contain metadata.");

            metadata = bytes(size_metadata)
            ptr_metadata = metadata

            deserialize_combined(
                file_ptr,
                ptr_model,
                ptr_ext_model,
                ptr_imputer,
                ptr_indexer,
                ptr_metadata
            )

            return metadata.decode('utf-8').encode()
        
        finally:
            fclose(file_ptr)

    def generate_sql(self, is_extended,
                     vector[cpp_string] numeric_colnames,
                     vector[cpp_string] categ_colnames,
                     vector[vector[cpp_string]] categ_levels,
                     bool_t output_tree_num=False,
                     bool_t single_tree=False, size_t tree_num=0,
                     int nthreads=1):
        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest

        cdef vector[cpp_string] res
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            res = generate_sql(model_ptr, ext_model_ptr,
                               numeric_colnames, categ_colnames, categ_levels,
                               output_tree_num, 0, single_tree, tree_num, nthreads)
        return res

    def subset_model(self, np.ndarray[size_t, ndim=1] trees_take, bool_t is_extended, bool_t has_imputer):
        cdef isoforest_cpp_obj new_obj = isoforest_cpp_obj()
        cdef IsoForest* model = NULL
        cdef ExtIsoForest* ext_model = NULL
        cdef Imputer* imputer = NULL
        cdef TreesIndexer* indexer = NULL
        if not is_extended:
            model = &self.isoforest
        else:
            ext_model = &self.ext_isoforest
        if has_imputer:
            imputer = &self.imputer
        if not self.indexer.indices.empty():
            indexer = &self.indexer
        subset_model(model,      &new_obj.isoforest,
                     ext_model,  &new_obj.ext_isoforest,
                     imputer,    &new_obj.imputer,
                     indexer,    &new_obj.indexer,
                     &trees_take[0], trees_take.shape[0])
        return new_obj

    def get_node(self, size_t tree, size_t node, double[:] out):
        # if self.isoforest.trees[tree][node].score < 0:
        if self.isoforest.trees[tree][node].tree_left != 0:

            out[1] = <double>self.isoforest.trees[tree][node].col_num
            out[3] = <double>(self.isoforest.trees[tree][node].pct_tree_left >= .5)
            out[4] = <double>self.isoforest.trees[tree][node].tree_left
            out[5] = <double>self.isoforest.trees[tree][node].tree_right

            if self.isoforest.trees[tree][node].col_type == Numeric:
                out[0] = 0
                out[2] = self.isoforest.trees[tree][node].num_split
            else:
                out[0] = -1
                if self.isoforest.cat_split_type == SingleCateg:
                    return [self.isoforest.trees[tree][node].chosen_cat]
                else:
                    if self.isoforest.trees[tree][node].cat_split.size():
                        return get_list_left_categories(self.isoforest.trees[tree][node].cat_split)
                    else:
                        return [0]
        else:
            out[0] = 1
            if ((self.isoforest.scoring_metric != Density) and
                (self.isoforest.scoring_metric != BoxedDensity) and
                (self.isoforest.scoring_metric != BoxedDensity2)
            ):
                out[1] = self.isoforest.trees[tree][node].score
            else:
                out[1] = -self.isoforest.trees[tree][node].score

        return None

    def get_expected_isolation_depth(self):
        return self.isoforest.exp_avg_depth

    def build_tree_indices(self, bool_t is_extended, bool_t with_distances, int nthreads):
        if not is_extended:
            build_tree_indices(self.indexer, self.isoforest, nthreads, with_distances)
        else:
            build_tree_indices(self.indexer, self.ext_isoforest, nthreads, with_distances)

    def has_indexer(self):
        return not self.indexer.indices.empty()

    def has_indexer_with_distances(self):
        if not self.has_indexer():
            return False
        return not self.indexer.indices.front().node_distances.empty()

    def set_reference_points(self,
                             np.ndarray[real_t, ndim=1] placeholder_real_t,
                             np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
                             X_num, X_cat, is_extended,
                             size_t nrows, int nthreads, bool_t with_distances):

        cdef real_t*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef real_t*     Xc_ptr            =  NULL
        cdef sparse_ix*  Xc_ind_ptr        =  NULL
        cdef sparse_ix*  Xc_indptr_ptr     =  NULL

        cdef bool_t is_col_major    =  True
        cdef size_t ncols_numeric   =  0
        cdef size_t ncols_categ     =  0

        if X_num is not None:
            if not issparse(X_num):
                if real_t is float:
                    numeric_data_ptr  =  get_ptr_float_mat(X_num)
                else:
                    if X_num.dtype != ctypes.c_double:
                        X_num = X_num.astype(ctypes.c_double)
                    numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
                ncols_numeric  =  X_num.shape[1]
                is_col_major   =  np.isfortran(X_num)
            else:
                if X_num.data.shape[0]:
                    if real_t is float:
                        Xc_ptr         =  get_ptr_float_vec(X_num.data)
                    else:
                        if X_num.data.dtype != ctypes.c_double:
                            X_num.data = X_num.data.astype(ctypes.c_double)
                        Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                if sparse_ix is int:
                    if X_num.indices.shape[0]:
                        Xc_ind_ptr =  get_ptr_int_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int_vec(X_num.indptr)
                elif sparse_ix is np.int64_t:
                    if X_num.indices.shape[0]:
                        Xc_ind_ptr =  get_ptr_int64_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_int64_vec(X_num.indptr)
                else:
                    if X_num.indices.shape[0]:
                        if X_num.indices.dtype != ctypes.c_size_t:
                            X_num.indices = X_num.indices.astype(ctypes.c_size_t)
                        Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                    if X_num.indptr.dtype != ctypes.c_size_t:
                        X_num.indptr = X_num.indptr.astype(ctypes.c_size_t)
                    Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            categ_data_ptr     =  get_ptr_int_mat(X_cat)
            ncols_categ        =  X_cat.shape[1]
            is_col_major       =  np.isfortran(X_cat)


        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest
        cdef TreesIndexer*  indexer_ptr = &self.indexer

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            set_reference_points(model_ptr, ext_model_ptr, indexer_ptr,
                                 with_distances,
                                 numeric_data_ptr, categ_data_ptr,
                                 is_col_major, ncols_numeric, ncols_categ,
                                 Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                                 <real_t*>NULL, <sparse_ix*>NULL, <sparse_ix*>NULL,
                                 nrows, nthreads)

        if cy_check_interrupt_switch():
            cy_tick_off_interrupt_switch()
            raise InterruptedError("Error: procedure was interrupted.")

    def has_reference_points(self):
        if self.indexer.indices.empty():
            return False
        if self.indexer.indices.front().reference_points.empty():
            return False
        else:
            return True

    def get_n_reference_points(self):
        if self.indexer.indices.empty():
            return 0
        return self.indexer.indices.front().reference_points.size()

