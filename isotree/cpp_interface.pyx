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
#     [8] Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
#     [9] Cortes, David. "Imputing missing values with unsupervised random trees." arXiv preprint arXiv:1911.06646 (2019).
# 
#     BSD 2-Clause License
#     Copyright (c) 2019-2021, David Cortes
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


cdef extern from "model_joined.hpp":

    bool_t cy_check_interrupt_switch()
    void cy_tick_off_interrupt_switch()

    ctypedef enum NewCategAction:
        Weighted
        Smallest
        Random

    ctypedef enum MissingAction:
        Divide
        Impute
        Fail

    ctypedef enum ColType:
        Numeric
        Categorical
        NotUsed

    ctypedef enum CategSplit:
        SubSet
        SingleCateg

    ctypedef enum GainCriterion:
        Averaged
        Pooled
        NoCrit

    ctypedef enum CoefType:
        Uniform
        Normal

    ctypedef enum UseDepthImp:
        Lower
        Higher
        Same

    ctypedef enum WeighImpRows:
        Inverse
        Prop
        Flat

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
        double           exp_avg_depth
        double           exp_avg_sep
        size_t           orig_sample_size

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
        double            exp_avg_depth
        double            exp_avg_sep
        size_t            orig_sample_size

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


    void tmat_to_dense(double *tmat, double *dmat, size_t n, bool_t diag_to_one)

    void merge_models(IsoForest*     model,      IsoForest*     other,
                      ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                      Imputer*       imputer,    Imputer*       iother) nogil except +

    void subset_model(IsoForest*     model,      IsoForest*     model_new,
                      ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                      Imputer*       imputer,    Imputer*       imputer_new,
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
        bool_t &has_metadata,
        size_t &size_metadata) nogil except +

    size_t determine_serialized_size_combined(
        const IsoForest *model,
        const ExtIsoForest *model_ext,
        const Imputer *imputer,
        const size_t size_optional_metadata) nogil except +

    void serialize_combined(
        const IsoForest *model,
        const ExtIsoForest *model_ext,
        const Imputer *imputer,
        const char *optional_metadata,
        const size_t size_optional_metadata,
        char *out) nogil except +

    void serialize_combined(
        const IsoForest *model,
        const ExtIsoForest *model_ext,
        const Imputer *imputer,
        const char *optional_metadata,
        const size_t size_optional_metadata,
        FILE *out) nogil except +

    void deserialize_combined(
        const char *inp,
        IsoForest *model,
        ExtIsoForest *model_ext,
        Imputer *imputer,
        char *optional_metadata) nogil except +

    void deserialize_combined(
        FILE* inp,
        IsoForest *model,
        ExtIsoForest *model_ext,
        Imputer *imputer,
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
                    bool_t limit_depth, bool_t penalize_range,
                    bool_t standardize_dist, double *tmat,
                    double *output_depths, bool_t standardize_depth,
                    real_t_ *col_weights, bool_t weigh_by_kurt,
                    double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                    double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                    double min_gain, MissingAction missing_action,
                    CategSplit cat_split_type, NewCategAction new_cat_action,
                    bool_t all_perm, Imputer *imputer, size_t min_imp_obs,
                    UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool_t impute_at_fit,
                    uint64_t random_seed, int nthreads) nogil except +

    void predict_iforest[real_t_, sparse_ix_](
                         real_t_ *numeric_data, int *categ_data,
                         bool_t is_col_major, size_t ncols_numeric, size_t ncols_categ,
                         real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
                         real_t_ *Xr, sparse_ix_ *Xr_ind, sparse_ix_ *Xr_indptr,
                         size_t nrows, int nthreads, bool_t standardize,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double *output_depths, sparse_ix_ *tree_num) nogil except +

    void get_num_nodes[sparse_ix_](IsoForest &model_outputs, sparse_ix_ *n_nodes, sparse_ix_ *n_terminal, int nthreads) except +

    void get_num_nodes[sparse_ix_](ExtIsoForest &model_outputs, sparse_ix_ *n_nodes, sparse_ix_ *n_terminal, int nthreads) except +

    void calc_similarity[real_t_, sparse_ix_](
                         real_t_ numeric_data[], int categ_data[],
                         real_t_ Xc[], sparse_ix_ Xc_ind[], sparse_ix_ Xc_indptr[],
                         size_t nrows, int nthreads, bool_t assume_full_distr, bool_t standardize_dist,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double tmat[], double rmat[], size_t n_from) nogil except +

    void impute_missing_values[real_t_, sparse_ix_](
                               real_t_ *numeric_data, int *categ_data, bool_t is_col_major,
                               real_t_ *Xr, sparse_ix_ *Xr_ind, sparse_ix_ *Xr_indptr,
                               size_t nrows, int nthreads,
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
                 bool_t   limit_depth,  bool_t penalize_range,
                 real_t_ *col_weights, bool_t weigh_by_kurt,
                 double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                 double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                 double min_gain, MissingAction missing_action,
                 CategSplit cat_split_type, NewCategAction new_cat_action,
                 UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
                 bool_t  all_perm, Imputer *imputer, size_t min_imp_obs,
                 uint64_t random_seed) nogil except +


cdef extern from "python_helpers.hpp":
    model_t deepcopy_obj[model_t](model_t obj) except +

    IsoForest get_IsoForest() except +
    ExtIsoForest get_ExtIsoForest() except +
    Imputer get_Imputer() except +

    void dealloc_IsoForest(IsoForest &model_outputs) except +
    void dealloc_IsoExtForest(ExtIsoForest &model_outputs_ext) except +
    void dealloc_Imputer(Imputer &imputer) except +

    bool_t get_has_openmp() except +

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


cdef class isoforest_cpp_obj:
    cdef IsoForest     isoforest
    cdef ExtIsoForest  ext_isoforest
    cdef Imputer       imputer

    def __dealloc__(self):
        dealloc_IsoForest(self.isoforest)
        dealloc_IsoExtForest(self.ext_isoforest)
        dealloc_Imputer(self.imputer)

    def __getstate__(self):
        cdef IsoForest *ptr_IsoForest = NULL
        cdef ExtIsoForest *ptr_ExtIsoForest = NULL
        cdef Imputer *ptr_Imputer = NULL

        if self.isoforest.trees.size():
            ptr_IsoForest = &self.isoforest
        else:
            ptr_ExtIsoForest = &self.ext_isoforest
        if self.imputer.imputer_tree.size():
            ptr_Imputer = &self.imputer

        cdef size_t size_ser = determine_serialized_size_combined(
            ptr_IsoForest,
            ptr_ExtIsoForest,
            ptr_Imputer,
            <size_t>0
        )
        cdef bytes serialized = bytes(size_ser)
        serialize_combined(
            ptr_IsoForest,
            ptr_ExtIsoForest,
            ptr_Imputer,
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
            <char*>NULL
        )

    def deepcopy(self):
        other = isoforest_cpp_obj()
        other.isoforest = deepcopy_obj(self.isoforest)
        other.ext_isoforest = deepcopy_obj(self.ext_isoforest)
        other.imputer = deepcopy_obj(self.imputer)
        return other

    def drop_imputer(self):
        dealloc_Imputer(self.imputer)
        self.imputer = get_Imputer()

    def get_cpp_obj(self, is_extended):
        if is_extended:
            return self.ext_isoforest
        else:
            return self.isoforest

    def get_imputer(self):
        return self.imputer

    def fit_model(self,
                  np.ndarray[real_t, ndim=1] placeholder_real_t,
                  np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
                  X_num, X_cat, ncat, sample_weights, col_weights,
                  size_t nrows, size_t ncols_numeric, size_t ncols_categ,
                  size_t ndim, size_t ntry, coef_type, bool_t coef_by_prop,
                  bool_t with_replacement, bool_t weight_as_sample,
                  size_t sample_size, size_t ntrees,
                  size_t max_depth,   size_t ncols_per_tree,
                  bool_t limit_depth, bool_t penalize_range,
                  bool_t calc_dist, bool_t standardize_dist, bool_t sq_dist,
                  bool_t calc_depth, bool_t standardize_depth,
                  bool_t weigh_by_kurt,
                  double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                  double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                  double min_gain, missing_action, cat_split_type, new_cat_action,
                  bool_t build_imputer, size_t min_imp_obs,
                  depth_imp, weigh_imp_rows, bool_t impute_at_fit,
                  bool_t all_perm, uint64_t random_seed,
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
                dmat      =  np.zeros((nrows, nrows), dtype = ctypes.c_double, order = 'F')
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
                        limit_depth, penalize_range,
                        standardize_dist, tmat_ptr,
                        depths_ptr, standardize_depth,
                        col_weights_ptr, weigh_by_kurt,
                        prob_pick_by_gain_avg, prob_split_by_gain_avg,
                        prob_pick_by_gain_pl,  prob_split_by_gain_pl,
                        min_gain, missing_action_C,
                        cat_split_type_C, new_cat_action_C,
                        all_perm, imputer_ptr, min_imp_obs,
                        depth_imp_C, weigh_imp_rows_C, impute_at_fit,
                        random_seed, nthreads)

        if cy_check_interrupt_switch():
            cy_tick_off_interrupt_switch()
            raise InterruptedError("Error: procedure was interrupted.")

        if ret_val == return_EXIT_FAILURE():
            raise ValueError("Error: something went wrong. Procedure failed.")

        if (calc_dist) and (sq_dist):
            tmat_to_dense(tmat_ptr, dmat_ptr, nrows, <bool_t>(not standardize_dist))

        return depths, tmat, dmat, X_num, X_cat

    def fit_tree(self,
                 np.ndarray[real_t, ndim=1] placeholder_real_t,
                 np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
                 X_num, X_cat, ncat, sample_weights, col_weights,
                 size_t nrows, size_t ncols_numeric, size_t ncols_categ,
                 size_t ndim, size_t ntry, coef_type, bool_t coef_by_prop,
                 size_t max_depth,   size_t ncols_per_tree,
                 bool_t limit_depth, bool_t penalize_range,
                 bool_t weigh_by_kurt,
                 double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                 double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                 double min_gain, missing_action, cat_split_type, new_cat_action,
                 bool_t build_imputer, size_t min_imp_obs,
                 depth_imp, weigh_imp_rows,
                 bool_t all_perm, uint64_t random_seed):
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

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            add_tree(model_ptr, ext_model_ptr,
                     numeric_data_ptr,  ncols_numeric,
                     categ_data_ptr,    ncols_categ,    ncat_ptr,
                     Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                     ndim, ntry, coef_type_C, coef_by_prop,
                     sample_weights_ptr,
                     nrows, max_depth, ncols_per_tree,
                     limit_depth,  penalize_range,
                     col_weights_ptr, weigh_by_kurt,
                     prob_pick_by_gain_avg, prob_split_by_gain_avg,
                     prob_pick_by_gain_pl,  prob_split_by_gain_pl,
                     min_gain, missing_action_C,
                     cat_split_type_C, new_cat_action_C,
                     depth_imp_C, weigh_imp_rows_C,
                     all_perm, imputer_ptr, min_imp_obs, random_seed)

    def predict(self,
                np.ndarray[real_t, ndim=1] placeholder_real_t,
                np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
                X_num, X_cat, is_extended,
                size_t nrows, int nthreads, bool_t standardize, bool_t output_tree_num):

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

        cdef np.ndarray[double, ndim = 1]    depths    =  np.zeros(nrows, dtype = ctypes.c_double)
        cdef np.ndarray[sparse_ix, ndim = 2] tree_num  =  np.empty((0, 0), order = 'F', dtype = placeholder_sparse_ix.dtype)
        cdef double* depths_ptr       =  &depths[0]
        cdef sparse_ix* tree_num_ptr  =  NULL

        if output_tree_num:
            if is_extended:
                sz = self.ext_isoforest.hplanes.size()
            else:
                sz = self.isoforest.trees.size()
            tree_num      =  np.empty((nrows, sz), dtype = placeholder_sparse_ix.dtype, order = 'F')
            tree_num_ptr  =  &tree_num[0, 0]

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest
        
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            predict_iforest(numeric_data_ptr, categ_data_ptr,
                            is_col_major, ncols_numeric, ncols_categ,
                            Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                            Xr_ptr, Xr_ind_ptr, Xr_indptr_ptr,
                            nrows, nthreads, standardize,
                            model_ptr, ext_model_ptr,
                            depths_ptr, tree_num_ptr)

        return depths, tree_num


    def dist(self,
             np.ndarray[real_t, ndim=1] placeholder_real_t,
             np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
             X_num, X_cat, is_extended,
             size_t nrows, int nthreads, bool_t assume_full_distr,
             bool_t standardize_dist,    bool_t sq_dist,
             size_t n_from):

        cdef real_t*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef real_t*     Xc_ptr            =  NULL
        cdef sparse_ix*  Xc_ind_ptr        =  NULL
        cdef sparse_ix*  Xc_indptr_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                if real_t is float:
                    numeric_data_ptr  =  get_ptr_float_mat(X_num)
                else:
                    if X_num.dtype != ctypes.c_double:
                        X_num = X_num.astype(ctypes.c_double)
                    numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
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

        cdef np.ndarray[double, ndim = 1]  tmat    =  np.empty(0, dtype = ctypes.c_double)
        cdef np.ndarray[double, ndim = 2]  dmat    =  np.empty((0, 0), dtype = ctypes.c_double)
        cdef np.ndarray[double, ndim = 2]  rmat    =  np.empty((0, 0), dtype = ctypes.c_double)
        cdef double*  tmat_ptr    =  NULL
        cdef double*  dmat_ptr    =  NULL
        cdef double*  rmat_ptr    =  NULL

        if n_from == 0:
            tmat      =  np.zeros(int((nrows * (nrows - 1)) / 2), dtype = ctypes.c_double)
            tmat_ptr  =  &tmat[0]
            if sq_dist:
                dmat      =  np.zeros((nrows, nrows), dtype = ctypes.c_double, order = 'F')
                dmat_ptr  =  &dmat[0, 0]
        else:
            rmat = np.zeros((n_from, nrows - n_from), dtype = ctypes.c_double)
            rmat_ptr = &rmat[0, 0]

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest
        
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            calc_similarity(numeric_data_ptr, categ_data_ptr,
                            Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                            nrows, nthreads, assume_full_distr, standardize_dist,
                            model_ptr, ext_model_ptr,
                            tmat_ptr, rmat_ptr, n_from)

        if cy_check_interrupt_switch():
            cy_tick_off_interrupt_switch()
            raise InterruptedError("Error: procedure was interrupted.")

        if (sq_dist) and (n_from == 0):
            tmat_to_dense(tmat_ptr, dmat_ptr, nrows, <bool_t>(not standardize_dist))

        return tmat, dmat, rmat

    def impute(self,
               np.ndarray[real_t, ndim=1] placeholder_real_t,
               np.ndarray[sparse_ix, ndim=1] placeholder_sparse_ix,
               X_num, X_cat, bool_t is_extended, size_t nrows, int nthreads):
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
                                  nrows, nthreads,
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

        if is_extended:
            ptr_ext_model = &self.ext_isoforest
            ptr_ext_other = &other.ext_isoforest
        else:
            ptr_model = &self.isoforest
            ptr_other = &other.isoforest

        if self.imputer.imputer_tree.size():
            ptr_imp = &self.imputer
        if other.imputer.imputer_tree.size():
            prt_iother = &other.imputer

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            merge_models(ptr_model, ptr_other,
                         ptr_ext_model, ptr_ext_other,
                         ptr_imp, prt_iother)

    def serialize_obj(self, str fpath, bytes metadata, bool_t is_extended=False, bool_t has_imputer=False):
        cdef FILE* file_ptr = cy_fopen(fpath, read=False)
        
        cdef IsoForest *ptr_model = NULL
        cdef ExtIsoForest *ptr_ext_model = NULL
        cdef Imputer *ptr_imputer = NULL
        if not is_extended:
            ptr_model = &self.isoforest
        else:
            ptr_ext_model = &self.ext_isoforest
        if has_imputer:
            ptr_imputer = &self.imputer
        try:
            serialize_combined(
                ptr_model,
                ptr_ext_model,
                ptr_imputer,
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
        cdef bool_t has_metadata = 0
        cdef size_t size_metadata = 0

        cdef bytes metadata
        cdef char *ptr_metadata

        cdef IsoForest *ptr_model = &self.isoforest
        cdef ExtIsoForest *ptr_ext_model = &self.ext_isoforest
        cdef Imputer *ptr_imputer = &self.imputer

        cdef FILE* file_ptr = cy_fopen(fpath, read=True)

        try:
            inspect_serialized_object(
                file_ptr,
                is_isotree_model,
                is_compatible,
                has_combined_objects,
                has_IsoForest,
                has_ExtIsoForest,
                has_Imputer,
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
        if not is_extended:
            model = &self.isoforest
        else:
            ext_model = &self.ext_isoforest
        if has_imputer:
            imputer = &self.imputer
        subset_model(model,      &new_obj.isoforest,
                     ext_model,  &new_obj.ext_isoforest,
                     imputer,    &new_obj.imputer,
                     &trees_take[0], trees_take.shape[0])
        return new_obj
