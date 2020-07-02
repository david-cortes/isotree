#cython: auto_pickle=True

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
#     Copyright (c) 2019, David Cortes
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
import ctypes
import os

cdef extern from "isotree.hpp":
    ctypedef size_t sparse_ix

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
        vector[char]  cat_split
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


    int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                    double *numeric_data,  size_t ncols_numeric,
                    int    *categ_data,    size_t ncols_categ,    int *ncat,
                    double *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
                    size_t ndim, size_t ntry, CoefType coef_type, bool_t coef_by_prop,
                    double *sample_weights, bool_t with_replacement, bool_t weight_as_sample,
                    size_t nrows, size_t sample_size, size_t ntrees, size_t max_depth,
                    bool_t limit_depth, bool_t penalize_range,
                    bool_t standardize_dist, double *tmat,
                    double *output_depths, bool_t standardize_depth,
                    double *col_weights, bool_t weigh_by_kurt,
                    double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                    double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                    double min_gain, MissingAction missing_action,
                    CategSplit cat_split_type, NewCategAction new_cat_action,
                    bool_t all_perm, Imputer *imputer, size_t min_imp_obs,
                    UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool_t impute_at_fit,
                    uint64_t random_seed, int nthreads)

    void predict_iforest(double *numeric_data, int *categ_data,
                         double *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
                         double *Xr, sparse_ix *Xr_ind, sparse_ix *Xr_indptr,
                         size_t nrows, int nthreads, bool_t standardize,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double *output_depths, size_t *tree_num)

    void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads)

    void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads)

    void tmat_to_dense(double *tmat, double *dmat, size_t n, bool_t diag_to_one)

    void calc_similarity(double numeric_data[], int categ_data[],
                         double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                         size_t nrows, int nthreads, bool_t assume_full_distr, bool_t standardize_dist,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double tmat[], double rmat[], size_t n_from)

    void impute_missing_values(double *numeric_data, int *categ_data,
                               double *Xr, sparse_ix *Xr_ind, sparse_ix *Xr_indptr,
                               size_t nrows, int nthreads,
                               IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                               Imputer &imputer)

    int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                 double *numeric_data,  size_t ncols_numeric,
                 int    *categ_data,    size_t ncols_categ,    int *ncat,
                 double *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
                 size_t ndim, size_t ntry, CoefType coef_type, bool_t coef_by_prop,
                 double *sample_weights,
                 size_t nrows, size_t max_depth,
                 bool_t   limit_depth,  bool_t penalize_range,
                 double *col_weights, bool_t weigh_by_kurt,
                 double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                 double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                 double min_gain, MissingAction missing_action,
                 CategSplit cat_split_type, NewCategAction new_cat_action,
                 UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
                 bool_t  all_perm, vector[ImputeNode] *impute_nodes, size_t min_imp_obs,
                 uint64_t random_seed)

    void merge_models(IsoForest*     model,      IsoForest*     other,
                      ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                      Imputer*       imputer,    Imputer*       iother)

    void serialize_isoforest(IsoForest &model, const char *output_file_path)
    cpp_string serialize_isoforest(IsoForest &model)
    void deserialize_isoforest(IsoForest &output, const char *input_file_path)
    void deserialize_isoforest(IsoForest &output, cpp_string &serialized, bool_t move_str)
    void serialize_ext_isoforest(ExtIsoForest &model, const char *output_file_path)
    cpp_string serialize_ext_isoforest(ExtIsoForest &model)
    void deserialize_ext_isoforest(ExtIsoForest &output, const char *input_file_path)
    void deserialize_ext_isoforest(ExtIsoForest &output, cpp_string &serialized, bool_t move_str)
    void serialize_imputer(Imputer &imputer, const char *output_file_path)
    cpp_string serialize_imputer(Imputer &imputer)
    void deserialize_imputer(Imputer &output, const char *input_file_path)
    void deserialize_imputer(Imputer &output, cpp_string &serialized, bool_t move_str)
    bool_t has_msvc()

    void dealloc_IsoForest(IsoForest &model_outputs)
    void dealloc_IsoExtForest(ExtIsoForest &model_outputs_ext)
    void dealloc_Imputer(Imputer &imputer)

    int return_EXIT_SUCCESS()
    int return_EXIT_FAILURE()


cdef double* get_ptr_dbl_vec(np.ndarray[double, ndim = 1] a):
    return &a[0]

cdef int* get_ptr_int_vec(np.ndarray[int, ndim = 1] a):
    return &a[0]

cdef size_t* get_ptr_szt_vec(np.ndarray[size_t, ndim = 1] a):
    return &a[0]

cdef double* get_ptr_dbl_mat(np.ndarray[double, ndim = 2] a):
    return &a[0, 0]

cdef int* get_ptr_int_mat(np.ndarray[int, ndim = 2] a):
    return &a[0, 0]


# @cython.auto_pickle(True)
cdef class isoforest_cpp_obj:
    cdef IsoForest     isoforest
    cdef ExtIsoForest  ext_isoforest
    cdef Imputer       imputer

    def __init__(self):
        pass

    def __dealloc__(self):
        dealloc_IsoForest(self.isoforest)
        dealloc_IsoExtForest(self.ext_isoforest)
        dealloc_Imputer(self.imputer)

    def get_cpp_obj(self, is_extended):
        if is_extended:
            return self.ext_isoforest
        else:
            return self.isoforest

    def get_imputer(self):
        return self.imputer

    def fit_model(self, X_num, X_cat, ncat, sample_weights, col_weights,
                  size_t nrows, size_t ncols_numeric, size_t ncols_categ,
                  size_t ndim, size_t ntry, coef_type, bool_t coef_by_prop,
                  bool_t with_replacement, bool_t weight_as_sample,
                  size_t sample_size, size_t ntrees, size_t max_depth,
                  bool_t limit_depth, bool_t penalize_range,
                  bool_t calc_dist, bool_t standardize_dist, bool_t sq_dist,
                  bool_t calc_depth, bool_t standardize_depth,
                  bool_t weigh_by_kurt,
                  double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                  double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                  double min_gain, missing_action, cat_split_type, new_cat_action,
                  bool_t build_imputer, size_t min_imp_obs,
                  depth_imp, weigh_imp_rows, bool_t impute_at_fit,
                  bool_t all_perm, uint64_t random_seed, int nthreads):
        cdef double*     numeric_data_ptr    =  NULL
        cdef int*        categ_data_ptr      =  NULL
        cdef int*        ncat_ptr            =  NULL
        cdef double*     Xc_ptr              =  NULL
        cdef sparse_ix*  Xc_ind_ptr          =  NULL
        cdef sparse_ix*  Xc_indptr_ptr       =  NULL
        cdef double*     sample_weights_ptr  =  NULL
        cdef double*     col_weights_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
            else:
                Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            categ_data_ptr     =  get_ptr_int_mat(X_cat)
            ncat_ptr           =  get_ptr_int_vec(ncat)
        if sample_weights is not None:
            sample_weights_ptr =  get_ptr_dbl_vec(sample_weights)
        if col_weights is not None:
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
            tmat      =  np.zeros(int((nrows * (nrows - 1)) / 2), dtype = ctypes.c_double)
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
            self.isoforest      =  IsoForest()
            model_ptr           =  &self.isoforest
        else:
            self.ext_isoforest  =  ExtIsoForest()
            ext_model_ptr       =  &self.ext_isoforest

        if build_imputer:
            self.imputer = Imputer()
            imputer_ptr  = &self.imputer

        cdef int ret_val = 0

        ret_val = \
        fit_iforest(model_ptr, ext_model_ptr,
                    numeric_data_ptr,  ncols_numeric,
                    categ_data_ptr,    ncols_categ,    ncat_ptr,
                    Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                    ndim, ntry, coef_type_C, coef_by_prop,
                    sample_weights_ptr, with_replacement, weight_as_sample,
                    nrows, sample_size, ntrees, max_depth,
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

        if ret_val == return_EXIT_FAILURE():
            raise KeyboardInterrupt("Error: procedure was interrupted.")

        if (calc_dist) and (sq_dist):
            tmat_to_dense(tmat_ptr, dmat_ptr, nrows, <bool_t>(not standardize_dist))

        return depths, tmat, dmat, X_num, X_cat

    def fit_tree(self, X_num, X_cat, ncat, sample_weights, col_weights,
                 size_t nrows, size_t ncols_numeric, size_t ncols_categ,
                 size_t ndim, size_t ntry, coef_type, bool_t coef_by_prop,
                 size_t max_depth, bool_t limit_depth, bool_t penalize_range,
                 bool_t weigh_by_kurt,
                 double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                 double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                 double min_gain, missing_action, cat_split_type, new_cat_action,
                 bool_t build_imputer, size_t min_imp_obs,
                 depth_imp, weigh_imp_rows,
                 bool_t all_perm, uint64_t random_seed):
        cdef double*     numeric_data_ptr    =  NULL
        cdef int*        categ_data_ptr      =  NULL
        cdef int*        ncat_ptr            =  NULL
        cdef double*     Xc_ptr              =  NULL
        cdef sparse_ix*  Xc_ind_ptr          =  NULL
        cdef sparse_ix*  Xc_indptr_ptr       =  NULL
        cdef double*     sample_weights_ptr  =  NULL
        cdef double*     col_weights_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
            else:
                Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            categ_data_ptr     =  get_ptr_int_mat(X_cat)
            ncat_ptr           =  get_ptr_int_vec(ncat)
        if sample_weights is not None:
            sample_weights_ptr  =  get_ptr_dbl_vec(sample_weights)
        if col_weights is not None:
            col_weights_ptr     =  get_ptr_dbl_vec(col_weights)

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

        cdef vector[ImputeNode] *imputer_tree_ptr = NULL
        if build_imputer:
            self.imputer.imputer_tree.push_back(vector[ImputeNode]()) ### emplace back doesn't work in cython
            imputer_tree_ptr = &self.imputer.imputer_tree.back()

        add_tree(model_ptr, ext_model_ptr,
                 numeric_data_ptr,  ncols_numeric,
                 categ_data_ptr,    ncols_categ,    ncat_ptr,
                 Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                 ndim, ntry, coef_type_C, coef_by_prop,
                 sample_weights_ptr,
                 nrows, max_depth,
                 limit_depth,  penalize_range,
                 col_weights_ptr, weigh_by_kurt,
                 prob_pick_by_gain_avg, prob_split_by_gain_avg,
                 prob_pick_by_gain_pl,  prob_split_by_gain_pl,
                 min_gain, missing_action_C,
                 cat_split_type_C, new_cat_action_C,
                 depth_imp_C, weigh_imp_rows_C,
                 all_perm, imputer_tree_ptr, min_imp_obs, random_seed)

    def predict(self, X_num, X_cat, is_extended,
                size_t nrows, int nthreads, bool_t standardize, bool_t output_tree_num):

        cdef double*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef double*     Xc_ptr            =  NULL
        cdef sparse_ix*  Xc_ind_ptr        =  NULL
        cdef sparse_ix*  Xc_indptr_ptr     =  NULL
        cdef double*     Xr_ptr            =  NULL
        cdef sparse_ix*  Xr_ind_ptr        =  NULL
        cdef sparse_ix*  Xr_indptr_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                numeric_data_ptr   =  get_ptr_dbl_mat(X_num)
            else:
                if isspmatrix_csc(X_num):
                    Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                    Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                    Xc_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
                else:
                    Xr_ptr         =  get_ptr_dbl_vec(X_num.data)
                    Xr_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                    Xr_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)

        if X_cat is not None:
            categ_data_ptr    =  get_ptr_int_mat(X_cat)

        cdef np.ndarray[double, ndim = 1] depths    =  np.zeros(nrows, dtype = ctypes.c_double)
        cdef np.ndarray[size_t, ndim = 2] tree_num  =  np.empty((0, 0), dtype = ctypes.c_size_t, order = 'F')
        cdef double* depths_ptr    =  &depths[0]
        cdef size_t* tree_num_ptr  =  NULL

        if output_tree_num:
            if is_extended:
                sz = self.ext_isoforest.hplanes.size()
            else:
                sz = self.isoforest.trees.size()
            tree_num      =  np.empty((nrows, sz), dtype = ctypes.c_size_t, order = 'F')
            tree_num_ptr  =  &tree_num[0, 0]

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest
        
        predict_iforest(numeric_data_ptr, categ_data_ptr,
                        Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                        Xr_ptr, Xr_ind_ptr, Xr_indptr_ptr,
                        nrows, nthreads, standardize,
                        model_ptr, ext_model_ptr,
                        depths_ptr, tree_num_ptr)

        return depths, tree_num


    def dist(self, X_num, X_cat, is_extended,
             size_t nrows, int nthreads, bool_t assume_full_distr,
             bool_t standardize_dist,    bool_t sq_dist,
             size_t n_from):

        cdef double*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef double*     Xc_ptr            =  NULL
        cdef sparse_ix*  Xc_ind_ptr        =  NULL
        cdef sparse_ix*  Xc_indptr_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
            else:
                Xc_ptr         =  get_ptr_dbl_vec(X_num.data)
                Xc_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
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
        
        calc_similarity(numeric_data_ptr, categ_data_ptr,
                        Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                        nrows, nthreads, assume_full_distr, standardize_dist,
                        model_ptr, ext_model_ptr,
                        tmat_ptr, rmat_ptr, n_from)

        if (sq_dist) and (n_from == 0):
            tmat_to_dense(tmat_ptr, dmat_ptr, nrows, <bool_t>(not standardize_dist))

        return tmat, dmat, rmat

    def impute(self, X_num, X_cat, bool_t is_extended, size_t nrows, int nthreads):
        cdef double*     numeric_data_ptr  =  NULL
        cdef int*        categ_data_ptr    =  NULL
        cdef double*     Xr_ptr            =  NULL
        cdef sparse_ix*  Xr_ind_ptr        =  NULL
        cdef sparse_ix*  Xr_indptr_ptr     =  NULL

        if X_num is not None:
            if not issparse(X_num):
                numeric_data_ptr  =  get_ptr_dbl_mat(X_num)
            else:
                Xr_ptr         =  get_ptr_dbl_vec(X_num.data)
                Xr_ind_ptr     =  get_ptr_szt_vec(X_num.indices)
                Xr_indptr_ptr  =  get_ptr_szt_vec(X_num.indptr)
        if X_cat is not None:
            categ_data_ptr     =  get_ptr_int_mat(X_cat)

        cdef IsoForest*     model_ptr      =  NULL
        cdef ExtIsoForest*  ext_model_ptr  =  NULL
        if not is_extended:
            model_ptr      =  &self.isoforest
        else:
            ext_model_ptr  =  &self.ext_isoforest

        impute_missing_values(numeric_data_ptr, categ_data_ptr,
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
        cdef np.ndarray[sparse_ix, ndim=1] n_nodes    = np.empty(ntrees, dtype=ctypes.c_size_t)
        cdef np.ndarray[sparse_ix, ndim=1] n_terminal = np.empty(ntrees, dtype=ctypes.c_size_t)
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

        merge_models(ptr_model, ptr_other,
                     ptr_ext_model, ptr_ext_other,
                     ptr_imp, prt_iother)

    def serialize_obj(self, fpath, use_cpp=False, is_extended=False, has_imputer=False):
        fpath_imputer = fpath + ".imputer"
        cdef char *fpath_as_char = NULL
        cdef char *fpath_imp_as_char = NULL
        cdef cpp_string obj_as_string

        if use_cpp:
            py_byte_string = fpath.encode()
            fpath_as_char = py_byte_string
            if is_extended:
                serialize_ext_isoforest(self.ext_isoforest, fpath_as_char)
            else:
                serialize_isoforest(self.isoforest, fpath_as_char)
            if has_imputer:
                imp_bytes = fpath_imputer.encode()
                fpath_imp_as_char = imp_bytes
                serialize_imputer(self.imputer, fpath_imp_as_char)
        else:
            if is_extended:
                obj_as_string = serialize_ext_isoforest(self.ext_isoforest)
            else:
                obj_as_string = serialize_isoforest(self.isoforest)
            with open(fpath, "wb") as of:
                of.write(obj_as_string)
            if has_imputer:
                obj_as_string.clear()
                obj_as_string = serialize_imputer(self.imputer)
                with open(fpath_imputer, "wb") as of:
                    of.write(obj_as_string)

    def deserialize_obj(self, fpath, is_extended=False, use_cpp=False):
        fpath_imputer = fpath + ".imputer"
        has_imputer = os.path.isfile(fpath_imputer)
        cdef char *fpath_as_char = NULL
        cdef char *fpath_imp_as_char = NULL
        cdef cpp_string obj_as_string
        cdef char *ptr_to_bytes = NULL
        cdef size_t n_bytes = 0

        if use_cpp:
            py_byte_string = fpath.encode()
            fpath_as_char = py_byte_string
            if is_extended:
                deserialize_ext_isoforest(self.ext_isoforest, fpath_as_char)
            else:
                deserialize_isoforest(self.isoforest, fpath_as_char)
            if has_imputer:
                imp_bytes = fpath_imputer.encode()
                fpath_imp_as_char = imp_bytes
                deserialize_imputer(self.imputer, fpath_imp_as_char)
        else:
            with open(fpath, "rb") as ff:
                model_bytes = ff.read()
            n_bytes = len(model_bytes)
            ptr_to_bytes = model_bytes
            obj_as_string = cpp_string(ptr_to_bytes, n_bytes)
            del model_bytes
            if is_extended:
                deserialize_ext_isoforest(self.ext_isoforest, obj_as_string, 1)
            else:
                deserialize_isoforest(self.isoforest, obj_as_string, 1)
            if has_imputer:
                obj_as_string.clear()
                with open(fpath_imputer, "rb"):
                    model_bytes = ff.read()
                n_bytes = len(model_bytes)
                ptr_to_bytes = model_bytes
                obj_as_string = cpp_string(ptr_to_bytes, n_bytes)
                del model_bytes
                deserialize_imputer(self.imputer, obj_as_string, 1)
