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
#ifdef _FOR_R

#include <Rcpp.h>
#include <Rcpp/unwindProtect.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(unwindProtect)]]
#include <Rinternals.h>

#ifndef _FOR_R
#define FOR_R
#endif

/* This is the package's header */
#include "isotree.hpp"

/* Library is templated, base R comes with only these 2 types though */
#include "headers_joined.hpp"
#define real_t double
#define sparse_ix int
#include "instantiate_template_headers.hpp"

/* For imputing CSR matrices with differing columns from input */
#include "other_helpers.hpp"

/*  Note: the R version calls the 'sort_csc_indices' templated function,
    so it's not enough to just include 'isotree_exportable.hpp' and let
    the templates be instantiated elsewhere. */

#define throw_mem_err() Rcpp::stop("Error: insufficient memory. Try smaller sample sizes and fewer trees.\n")

SEXP alloc_RawVec(void *data)
{
    size_t vecsize = *(size_t*)data;
    if (unlikely(vecsize > (size_t)std::numeric_limits<R_xlen_t>::max()))
        Rcpp::stop("Object is too big for R to handle.");
    return Rcpp::RawVector((R_xlen_t)vecsize);
}

SEXP safe_copy_vec(void *data)
{
    std::vector<double> *vec = (std::vector<double>*)data;
    return Rcpp::NumericVector(vec->begin(), vec->end());
}

SEXP safe_copy_intvec(void *data)
{
    std::vector<int> *vec = (std::vector<int>*)data;
    return Rcpp::IntegerVector(vec->begin(), vec->end());
}

SEXP safe_int_matrix(void *dims)
{
    size_t *dims_ = (size_t*)dims;
    size_t nrows = dims_[0];
    size_t ncols = dims_[1];
    return Rcpp::IntegerMatrix(nrows, ncols);
}

template <class Model>
SEXP safe_XPtr(void *model_ptr)
{
    return Rcpp::XPtr<Model>((Model*)model_ptr, true);
}

SEXP safe_errlist(void *ignored)
{
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::LogicalVector::create(1));
}

SEXP safe_FALSE(void *ignored)
{
    return Rcpp::LogicalVector::create(0);
}

Rcpp::RawVector resize_vec(Rcpp::RawVector inp, size_t new_size)
{
    Rcpp::RawVector out = Rcpp::unwindProtect(alloc_RawVec, (void*)&new_size);
    memcpy(RAW(out), RAW(inp), std::min((size_t)inp.size(), new_size));
    return out;
}

/* for model serialization and re-usage in R */
/* https://stackoverflow.com/questions/18474292/how-to-handle-c-internal-data-structure-in-r-in-order-to-allow-save-load */
/* this extra comment below the link is a workaround for Rcpp issue 675 in GitHub, do not remove it */
template <class Model>
Rcpp::RawVector serialize_cpp_obj(const Model *model_outputs)
{
    size_t serialized_size = determine_serialized_size(*model_outputs);
    if (unlikely(!serialized_size))
        Rcpp::stop("Unexpected error.");
    if (unlikely(serialized_size > (size_t)std::numeric_limits<R_xlen_t>::max()))
        Rcpp::stop("Resulting model is too large for R to handle.");
    Rcpp::RawVector out = Rcpp::unwindProtect(alloc_RawVec, (void*)&serialized_size);
    char *out_ = (char*)RAW(out);
    serialize_isotree(*model_outputs, out_);
    return out;
}

template <class Model>
SEXP deserialize_cpp_obj(Rcpp::RawVector src)
{
    if (unlikely(!src.size()))
        Rcpp::stop("Unexpected error.");
    std::unique_ptr<Model> out(new Model());
    const char *inp = (const char*)RAW(src);
    deserialize_isotree(*out, inp);
    SEXP out_ = Rcpp::unwindProtect(safe_XPtr<Model>, out.get());
    out.release();
    return out_;
}

// [[Rcpp::export(rng = false)]]
SEXP deserialize_IsoForest(Rcpp::RawVector src)
{
    return deserialize_cpp_obj<IsoForest>(src);
}

// [[Rcpp::export(rng = false)]]
SEXP deserialize_ExtIsoForest(Rcpp::RawVector src)
{
    return deserialize_cpp_obj<ExtIsoForest>(src);
}

// [[Rcpp::export(rng = false)]]
SEXP deserialize_Imputer(Rcpp::RawVector src)
{
    return deserialize_cpp_obj<Imputer>(src);
}

// [[Rcpp::export(rng = false)]]
SEXP deserialize_Indexer(Rcpp::RawVector src)
{
    return deserialize_cpp_obj<TreesIndexer>(src);
}

// [[Rcpp::export(rng = false)]]
Rcpp::LogicalVector check_null_ptr_model(SEXP ptr_model)
{
    return Rcpp::LogicalVector(R_ExternalPtrAddr(ptr_model) == NULL);
}

double* set_R_nan_as_C_nan(double *x, size_t n, std::vector<double> &v, int nthreads)
{
    v.assign(x, x + n);
    for (size_t i = 0; i < n; i++)
        if (unlikely(isnan(v[i]))) v[i] = NAN;
    return v.data();
}

double* set_R_nan_as_C_nan(double *x, size_t n, Rcpp::NumericVector &v, int nthreads)
{
    v = Rcpp::NumericVector(x, x + n);
    for (size_t i = 0; i < n; i++)
        if (unlikely(isnan(v[i]))) v[i] = NAN;
    return REAL(v);
}

double* set_R_nan_as_C_nan(double *x, size_t n, int nthreads)
{
    for (size_t i = 0; i < n; i++)
        if (unlikely(isnan(x[i]))) x[i] = NAN;
    return x;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List fit_model(Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat, Rcpp::IntegerVector ncat,
                     Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
                     Rcpp::NumericVector sample_weights, Rcpp::NumericVector col_weights,
                     size_t nrows, size_t ncols_numeric, size_t ncols_categ, size_t ndim, size_t ntry,
                     Rcpp::CharacterVector coef_type, bool coef_by_prop, bool with_replacement, bool weight_as_sample,
                     size_t sample_size, size_t ntrees,  size_t max_depth, size_t ncols_per_tree, bool limit_depth,
                     bool penalize_range, bool standardize_data,
                     Rcpp::CharacterVector scoring_metric, bool fast_bratio,
                     bool calc_dist, bool standardize_dist, bool sq_dist,
                     bool calc_depth, bool standardize_depth, bool weigh_by_kurt,
                     double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                     double prob_pick_by_full_gain, double prob_pick_by_dens,
                     double prob_pick_col_by_range, double prob_pick_col_by_var,
                     double prob_pick_col_by_kurt, double min_gain,
                     Rcpp::CharacterVector cat_split_type, Rcpp::CharacterVector new_cat_action,
                     Rcpp::CharacterVector missing_action, bool all_perm,
                     bool build_imputer, bool output_imputations, size_t min_imp_obs,
                     Rcpp::CharacterVector depth_imp, Rcpp::CharacterVector weigh_imp_rows,
                     int random_seed, int nthreads)
{
    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    int*        ncat_ptr            =  NULL;
    double*     Xc_ptr              =  NULL;
    int*        Xc_ind_ptr          =  NULL;
    int*        Xc_indptr_ptr       =  NULL;
    double*     sample_weights_ptr  =  NULL;
    double*     col_weights_ptr     =  NULL;
    Rcpp::NumericVector Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr = REAL(X_num);
        if (Rcpp::as<std::string>(missing_action) != "fail")
            numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, nrows * ncols_numeric, Xcpp, nthreads);
    }

    if (X_cat.size())
    {
        categ_data_ptr  =  INTEGER(X_cat);
        ncat_ptr        =  INTEGER(ncat);
    }

    if (Xc.size())
    {
        Xc_ptr          =  REAL(Xc);
        Xc_ind_ptr      =  INTEGER(Xc_ind);
        Xc_indptr_ptr   =  INTEGER(Xc_indptr);
        if (Rcpp::as<std::string>(missing_action) != "fail")
            Xc_ptr = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, nthreads);
    }

    if (sample_weights.size())
    {
        sample_weights_ptr  =  REAL(sample_weights);
    }

    if (col_weights.size())
    {
        col_weights_ptr     =  REAL(col_weights);
    }

    CoefType        coef_type_C       =  Normal;
    CategSplit      cat_split_type_C  =  SubSet;
    NewCategAction  new_cat_action_C  =  Weighted;
    MissingAction   missing_action_C  =  Divide;
    UseDepthImp     depth_imp_C       =  Higher;
    WeighImpRows    weigh_imp_rows_C  =  Inverse;
    ScoringMetric   scoring_metric_C  =  Depth;

    if (Rcpp::as<std::string>(coef_type) == "uniform")
    {
        coef_type_C       =  Uniform;
    }
    if (Rcpp::as<std::string>(cat_split_type) == "single_categ")
    {
        cat_split_type_C  =  SingleCateg;
    }
    if (Rcpp::as<std::string>(new_cat_action) == "smallest")
    {
        new_cat_action_C  =  Smallest;
    }
    else if (Rcpp::as<std::string>(new_cat_action) == "random")
    {
        new_cat_action_C  =  Random;
    }
    if (Rcpp::as<std::string>(missing_action) == "impute")
    {
        missing_action_C  =  Impute;
    }
    else if (Rcpp::as<std::string>(missing_action) == "fail")
    {
        missing_action_C  =  Fail;
    }
    if (Rcpp::as<std::string>(depth_imp) == "lower")
    {
        depth_imp_C       =  Lower;
    }
    else if (Rcpp::as<std::string>(depth_imp) == "same")
    {
        depth_imp_C       =  Same;
    }
    if (Rcpp::as<std::string>(weigh_imp_rows) == "prop")
    {
        weigh_imp_rows_C  =  Prop;
    }
    else if (Rcpp::as<std::string>(weigh_imp_rows) == "flat")
    {
        weigh_imp_rows_C  =  Flat;
    }
    if (Rcpp::as<std::string>(scoring_metric) == "adj_depth")
    {
        scoring_metric_C  =  AdjDepth;
    }
    else if (Rcpp::as<std::string>(scoring_metric) == "density")
    {
        scoring_metric_C  =  Density;
    }
    else if (Rcpp::as<std::string>(scoring_metric) == "adj_density")
    {
        scoring_metric_C  =  AdjDensity;
    }
    else if (Rcpp::as<std::string>(scoring_metric) == "boxed_density")
    {
        scoring_metric_C  =  BoxedDensity;
    }
    else if (Rcpp::as<std::string>(scoring_metric) == "boxed_density2")
    {
        scoring_metric_C  =  BoxedDensity2;
    }
    else if (Rcpp::as<std::string>(scoring_metric) == "boxed_ratio")
    {
        scoring_metric_C  =  BoxedRatio;
    }

    Rcpp::NumericVector  tmat    =  Rcpp::NumericVector();
    Rcpp::NumericMatrix  dmat    =  Rcpp::NumericMatrix();
    Rcpp::NumericVector  depths  =  Rcpp::NumericVector();
    double*  tmat_ptr    =  NULL;
    double*  dmat_ptr    =  NULL;
    double*  depths_ptr  =  NULL;

    if (calc_dist)
    {
        tmat      =  Rcpp::NumericVector(calc_ncomb(nrows));
        tmat_ptr  =  REAL(tmat);
        if (sq_dist)
        {
            dmat      =  Rcpp::NumericMatrix(nrows, nrows);
            dmat_ptr  =  REAL(dmat);
        }
    }

    if (calc_depth)
    {
        depths      =  Rcpp::NumericVector(nrows);
        depths_ptr  =  REAL(depths);
    }

    Rcpp::List outp = Rcpp::List::create(
            Rcpp::_["depths"]    = depths,
            Rcpp::_["tmat"]      = tmat,
            Rcpp::_["dmat"]      = dmat,
            Rcpp::_["ptr"] = R_NilValue,
            Rcpp::_["serialized"] = R_NilValue,
            Rcpp::_["imp_ptr"]    = R_NilValue,
            Rcpp::_["imp_ser"]    = R_NilValue,
            Rcpp::_["imputed_num"]    = R_NilValue,
            Rcpp::_["imputed_cat"]    = R_NilValue,
            Rcpp::_["err"] = Rcpp::LogicalVector::create(1)
    );

    std::unique_ptr<IsoForest>     model_ptr(nullptr);
    std::unique_ptr<ExtIsoForest>  ext_model_ptr(nullptr);
    std::unique_ptr<Imputer>       imputer_ptr(nullptr);

    if (ndim == 1)
        model_ptr      =  std::unique_ptr<IsoForest>(new IsoForest());
    else
        ext_model_ptr  =  std::unique_ptr<ExtIsoForest>(new ExtIsoForest());

    if (build_imputer)
        imputer_ptr    =  std::unique_ptr<Imputer>(new Imputer());

    int ret_val;
    try {
    ret_val = 
    fit_iforest(model_ptr.get(), ext_model_ptr.get(),
                numeric_data_ptr,  ncols_numeric,
                categ_data_ptr,    ncols_categ,    ncat_ptr,
                Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                ndim, ntry, coef_type_C, coef_by_prop,
                sample_weights_ptr, with_replacement, weight_as_sample,
                nrows, sample_size, ntrees, max_depth, ncols_per_tree,
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
                all_perm, imputer_ptr.get(), min_imp_obs,
                depth_imp_C, weigh_imp_rows_C, output_imputations,
                (uint64_t) random_seed, nthreads);
    }
    catch (std::bad_alloc &e) {
        throw_mem_err();
    }
    Rcpp::checkUserInterrupt();

    if (ret_val == EXIT_FAILURE)
    {
        return Rcpp::unwindProtect(safe_errlist, nullptr);
    }

    if (calc_dist && sq_dist)
        tmat_to_dense(tmat_ptr, dmat_ptr, nrows, standardize_dist? 0. : std::numeric_limits<double>::infinity());

    bool serialization_failed = false;
    Rcpp::RawVector serialized_obj;
    try {
        if (ndim == 1)
            serialized_obj  =  serialize_cpp_obj(model_ptr.get());
        else
            serialized_obj  =  serialize_cpp_obj(ext_model_ptr.get());
    }
    catch (std::bad_alloc &e) {
        throw_mem_err();
    }
    if (unlikely(!serialized_obj.size())) serialization_failed = true;
    if (unlikely(serialization_failed)) {
        if (ndim == 1)
            model_ptr.reset();
        else
            ext_model_ptr.reset();
    }

    if (!serialization_failed)
    {
        outp["serialized"] = serialized_obj;
        if (ndim == 1) {
            outp["ptr"]   =  Rcpp::unwindProtect(safe_XPtr<IsoForest>, model_ptr.get());
            model_ptr.release();
        }
        else {
            outp["ptr"]   =  Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, ext_model_ptr.get());
            ext_model_ptr.release();
        }
    } else
        outp["ptr"] = R_NilValue;

    if (build_imputer && !serialization_failed)
    {
        try {
            outp["imp_ser"] =  serialize_cpp_obj(imputer_ptr.get());
        }
        catch (std::bad_alloc &e) {
            throw_mem_err();
        }
        if (!Rf_xlength(outp["imp_ser"]))
        {
            serialization_failed = true;
            imputer_ptr.reset();
            if (ndim == 1)
                model_ptr.reset();
            else
                ext_model_ptr.reset();
            outp["imp_ptr"]  =  R_NilValue;
            outp["ptr"]    =  R_NilValue;
        } else {
            outp["imp_ptr"] =  Rcpp::unwindProtect(safe_XPtr<Imputer>, imputer_ptr.get());
            imputer_ptr.release();
        }
    }

    if (output_imputations && !serialization_failed)
    {
        outp["imputed_num"] = Xcpp;
        outp["imputed_cat"] = X_cat;
    }

    outp["err"] = Rcpp::unwindProtect(safe_FALSE, nullptr);
    return outp;
}

// [[Rcpp::export(rng = false)]]
void fit_tree(SEXP model_R_ptr, Rcpp::RawVector serialized_obj, Rcpp::RawVector serialized_imputer,
              SEXP indexer_R_ptr, Rcpp::RawVector serialized_indexer,
              Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat, Rcpp::IntegerVector ncat,
              Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
              Rcpp::NumericVector sample_weights, Rcpp::NumericVector col_weights,
              size_t nrows, size_t ncols_numeric, size_t ncols_categ,
              size_t ndim, size_t ntry, Rcpp::CharacterVector coef_type, bool coef_by_prop,
              size_t max_depth, size_t ncols_per_tree, bool limit_depth, bool penalize_range,
              bool standardize_data, bool fast_bratio, bool weigh_by_kurt,
              double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
              double prob_pick_by_full_gain, double prob_pick_by_dens,
              double prob_pick_col_by_range, double prob_pick_col_by_var,
              double prob_pick_col_by_kurt, double min_gain,
              Rcpp::CharacterVector cat_split_type, Rcpp::CharacterVector new_cat_action,
              Rcpp::CharacterVector missing_action, bool build_imputer, size_t min_imp_obs, SEXP imp_R_ptr,
              Rcpp::CharacterVector depth_imp, Rcpp::CharacterVector weigh_imp_rows,
              bool all_perm,
              Rcpp::NumericVector ref_X_num, Rcpp::IntegerVector ref_X_cat,
              Rcpp::NumericVector ref_Xc, Rcpp::IntegerVector ref_Xc_ind, Rcpp::IntegerVector ref_Xc_indptr,
              uint64_t random_seed,
              Rcpp::List &model_cpp_obj_update, Rcpp::List &model_params_update)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["serialized"] = R_NilValue,
        Rcpp::_["imp_ser"] = R_NilValue,
        Rcpp::_["ind_ser"] = R_NilValue
    );

    Rcpp::IntegerVector ntrees_plus1 = Rcpp::IntegerVector::create(Rf_asInteger(model_params_update["ntrees"]) + 1);

    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    int*        ncat_ptr            =  NULL;
    double*     Xc_ptr              =  NULL;
    int*        Xc_ind_ptr          =  NULL;
    int*        Xc_indptr_ptr       =  NULL;
    double*     sample_weights_ptr  =  NULL;
    double*     col_weights_ptr     =  NULL;
    Rcpp::NumericVector Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr = REAL(X_num);
        if (Rcpp::as<std::string>(missing_action) != "fail")
            numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, nrows * ncols_numeric, Xcpp, 1);
    }

    if (X_cat.size())
    {
        categ_data_ptr  =  INTEGER(X_cat);
        ncat_ptr        =  INTEGER(ncat);
    }

    if (Xc.size())
    {
        Xc_ptr         =  REAL(Xc);
        Xc_ind_ptr     =  INTEGER(Xc_ind);
        Xc_indptr_ptr  =  INTEGER(Xc_indptr);
        if (Rcpp::as<std::string>(missing_action) != "fail")
            Xc_ptr = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, 1);
    }

    double*     ref_numeric_data_ptr    =  NULL;
    int*        ref_categ_data_ptr      =  NULL;
    double*     ref_Xc_ptr              =  NULL;
    int*        ref_Xc_ind_ptr          =  NULL;
    int*        ref_Xc_indptr_ptr       =  NULL;
    Rcpp::NumericVector ref_Xcpp;
    if (ref_X_num.size())
    {
        ref_numeric_data_ptr = REAL(ref_X_num);
        if (Rcpp::as<std::string>(missing_action) != "fail")
            ref_numeric_data_ptr = set_R_nan_as_C_nan(ref_numeric_data_ptr, ref_X_num.size(), ref_Xcpp, 1);
    }

    if (ref_X_cat.size())
    {
        ref_categ_data_ptr  =  INTEGER(ref_X_cat);
    }

    if (ref_Xc.size())
    {
        ref_Xc_ptr         =  REAL(ref_Xc);
        ref_Xc_ind_ptr     =  INTEGER(ref_Xc_ind);
        ref_Xc_indptr_ptr  =  INTEGER(ref_Xc_indptr);
        if (Rcpp::as<std::string>(missing_action) != "fail")
            ref_Xc_ptr = set_R_nan_as_C_nan(ref_Xc_ptr, ref_Xc.size(), ref_Xcpp, 1);
    }

    if (sample_weights.size())
    {
        sample_weights_ptr  =  REAL(sample_weights);
    }

    if (col_weights.size())
    {
        col_weights_ptr     =  REAL(col_weights);
    }

    CoefType        coef_type_C       =  Normal;
    CategSplit      cat_split_type_C  =  SubSet;
    NewCategAction  new_cat_action_C  =  Weighted;
    MissingAction   missing_action_C  =  Divide;
    UseDepthImp     depth_imp_C       =  Higher;
    WeighImpRows    weigh_imp_rows_C  =  Inverse;

    if (Rcpp::as<std::string>(coef_type) == "uniform")
    {
        coef_type_C       =  Uniform;
    }
    if (Rcpp::as<std::string>(cat_split_type) == "single_categ")
    {
        cat_split_type_C  =  SingleCateg;
    }
    if (Rcpp::as<std::string>(new_cat_action) == "smallest")
    {
        new_cat_action_C  =  Smallest;
    }
    else if (Rcpp::as<std::string>(new_cat_action) == "random")
    {
        new_cat_action_C  =  Random;
    }
    if (Rcpp::as<std::string>(missing_action) == "impute")
    {
        missing_action_C  =  Impute;
    }
    else if (Rcpp::as<std::string>(missing_action) == "fail")
    {
        missing_action_C  =  Fail;
    }
    if (Rcpp::as<std::string>(depth_imp) == "lower")
    {
        depth_imp_C       =  Lower;
    }
    else if (Rcpp::as<std::string>(depth_imp) == "same")
    {
        depth_imp_C       =  Same;
    }
    if (Rcpp::as<std::string>(weigh_imp_rows) == "prop")
    {
        weigh_imp_rows_C  =  Prop;
    }
    else if (Rcpp::as<std::string>(weigh_imp_rows) == "flat")
    {
        weigh_imp_rows_C  =  Flat;
    }
    

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    Imputer*       imputer_ptr    =  NULL;
    TreesIndexer*  indexer_ptr    =  NULL;
    if (ndim == 1)
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));

    if (build_imputer)
        imputer_ptr = static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));

    if (!Rf_isNull(indexer_R_ptr) && R_ExternalPtrAddr(indexer_R_ptr) != NULL)
        indexer_ptr = static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (indexer_ptr != NULL && indexer_ptr->indices.empty())
        indexer_ptr = NULL;

    size_t old_ntrees = (ndim == 1)? (model_ptr->trees.size()) : (ext_model_ptr->hplanes.size());

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
             depth_imp_C, weigh_imp_rows_C, all_perm,
             imputer_ptr, min_imp_obs,
             indexer_ptr,
             ref_numeric_data_ptr, ref_categ_data_ptr,
             true, (size_t)0, (size_t)0,
             ref_Xc_ptr, ref_Xc_ind_ptr, ref_Xc_indptr_ptr,
             (uint64_t)random_seed);
    
    Rcpp::RawVector new_serialized, new_imp_serialized, new_ind_serialized;
    size_t new_size;
    try
    {
        if (ndim == 1)
        {
            if (serialized_obj.size() &&
                check_can_undergo_incremental_serialization(*model_ptr, (char*)RAW(serialized_obj)))
            {
                try {
                    new_size = serialized_obj.size()
                                + determine_serialized_size_additional_trees(*model_ptr, old_ntrees);
                    new_serialized = resize_vec(serialized_obj, new_size);
                    char *temp = (char*)RAW(new_serialized);
                    incremental_serialize_isotree(*model_ptr, temp);
                    out["serialized"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_singlevar;
                }
            }

            else {
                serialize_anew_singlevar:
                out["serialized"] = serialize_cpp_obj(model_ptr);
            }
        }

        else
        {
            if (serialized_obj.size() &&
                check_can_undergo_incremental_serialization(*ext_model_ptr, (char*)RAW(serialized_obj)))
            {
                try {
                    new_size = serialized_obj.size()
                                + determine_serialized_size_additional_trees(*ext_model_ptr, old_ntrees);
                    new_serialized = resize_vec(serialized_obj, new_size);
                    char *temp = (char*)RAW(new_serialized);
                    incremental_serialize_isotree(*ext_model_ptr, temp);
                    out["serialized"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ext;
                }
            }

            else {
                serialize_anew_ext:
                out["serialized"] = serialize_cpp_obj(ext_model_ptr);
            }
        }

        if (imputer_ptr != NULL)
        {
            if (serialized_imputer.size() &&
                check_can_undergo_incremental_serialization(*imputer_ptr, (char*)RAW(serialized_imputer)))
            {
                try {
                    new_size = serialized_imputer.size()
                                + determine_serialized_size_additional_trees(*imputer_ptr, old_ntrees);
                    new_imp_serialized = resize_vec(serialized_imputer, new_size);
                    char *temp = (char*)RAW(new_imp_serialized);
                    incremental_serialize_isotree(*imputer_ptr, temp);
                    out["imp_ser"] = new_imp_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_imp;
                }
            }

            else {
                serialize_anew_imp:
                out["imp_ser"] = serialize_cpp_obj(imputer_ptr);
            }
        }

        if (indexer_ptr != NULL)
        {
            if (serialized_indexer.size() &&
                check_can_undergo_incremental_serialization(*indexer_ptr, (char*)RAW(serialized_indexer)))
            {
                try {
                    new_size = serialized_indexer.size()
                                + determine_serialized_size_additional_trees(*indexer_ptr, old_ntrees);
                    new_ind_serialized = resize_vec(serialized_indexer, new_size);
                    char *temp = (char*)RAW(new_ind_serialized);
                    incremental_serialize_isotree(*indexer_ptr, temp);
                    out["ind_ser"] = new_ind_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ind;
                }
            }

            else {
                serialize_anew_ind:
                out["ind_ser"] = serialize_cpp_obj(indexer_ptr);
            }
        }
    }

    catch (...)
    {
        if (ndim == 1)
            model_ptr->trees.resize(old_ntrees);
        else
            ext_model_ptr->hplanes.resize(old_ntrees);
        if (build_imputer)
            imputer_ptr->imputer_tree.resize(old_ntrees);
        if (indexer_ptr != NULL)
            indexer_ptr->indices.resize(old_ntrees);
        throw;
    }

    model_cpp_obj_update["serialized"] = out["serialized"];
    if (build_imputer)
        model_cpp_obj_update["imp_ser"] = out["imp_ser"];
    if (indexer_ptr != NULL)
        model_cpp_obj_update["ind_ser"] = out["ind_ser"];
    model_params_update["ntrees"] = ntrees_plus1;
}

// [[Rcpp::export(rng = false)]]
void predict_iso(SEXP model_R_ptr, bool is_extended,
                 SEXP indexer_R_ptr,
                 Rcpp::NumericVector outp, Rcpp::IntegerMatrix tree_num, Rcpp::NumericMatrix tree_depths,
                 Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat,
                 Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
                 Rcpp::NumericVector Xr, Rcpp::IntegerVector Xr_ind, Rcpp::IntegerVector Xr_indptr,
                 size_t nrows, int nthreads, bool standardize)
{
    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    double*     Xc_ptr              =  NULL;
    int*        Xc_ind_ptr          =  NULL;
    int*        Xc_indptr_ptr       =  NULL;
    double*     Xr_ptr              =  NULL;
    int*        Xr_ind_ptr          =  NULL;
    int*        Xr_indptr_ptr       =  NULL;
    Rcpp::NumericVector Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr  =  REAL(X_num);
    }

    if (X_cat.size())
    {
        categ_data_ptr    =  INTEGER(X_cat);
    }

    if (Xc_indptr.size())
    {
        Xc_ptr         =  REAL(Xc);
        Xc_ind_ptr     =  INTEGER(Xc_ind);
        Xc_indptr_ptr  =  INTEGER(Xc_indptr);
    }

    if (Xr_indptr.size())
    {
        Xr_ptr         =  REAL(Xr);
        Xr_ind_ptr     =  INTEGER(Xr_ind);
        Xr_indptr_ptr  =  INTEGER(Xr_indptr);
    }

    double *depths_ptr       =  REAL(outp);
    double *tree_depths_ptr  =  tree_depths.size()? REAL(tree_depths) : NULL;
    int    *tree_num_ptr     =  tree_num.size()?    INTEGER(tree_num) : NULL;

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    TreesIndexer*  indexer = NULL;
    if (!Rf_isNull(indexer_R_ptr) && R_ExternalPtrAddr(indexer_R_ptr) != NULL)
        indexer = static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (indexer != NULL && indexer->indices.empty())
        indexer = NULL;

    MissingAction missing_action = is_extended?
                                   ext_model_ptr->missing_action
                                     :
                                   model_ptr->missing_action;
    if (missing_action != Fail)
    {
        if (X_num.size()) numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, X_num.size(), Xcpp, nthreads);
        if (Xc.size())    Xc_ptr           = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, nthreads);
        if (Xr.size())    Xr_ptr           = set_R_nan_as_C_nan(Xr_ptr, Xr.size(), Xcpp, nthreads);
    }

    predict_iforest<double, int>(numeric_data_ptr, categ_data_ptr,
                                 true, (size_t)0, (size_t)0,
                                 Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                                 Xr_ptr, Xr_ind_ptr, Xr_indptr_ptr,
                                 nrows, nthreads, standardize,
                                 model_ptr, ext_model_ptr,
                                 depths_ptr, tree_num_ptr,
                                 tree_depths_ptr,
                                 indexer);
}

// [[Rcpp::export(rng = false)]]
void dist_iso(SEXP model_R_ptr, SEXP indexer_R_ptr,
              Rcpp::NumericVector tmat, Rcpp::NumericMatrix dmat,
              Rcpp::NumericMatrix rmat, bool is_extended,
              Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat,
              Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
              size_t nrows, int nthreads, bool assume_full_distr,
              bool standardize_dist, bool sq_dist, size_t n_from,
              bool use_reference_points, bool as_kernel)
{
    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    double*     Xc_ptr              =  NULL;
    int*        Xc_ind_ptr          =  NULL;
    int*        Xc_indptr_ptr       =  NULL;
    Rcpp::NumericVector Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr  =  REAL(X_num);
    }

    if (X_cat.size())
    {
        categ_data_ptr    =  INTEGER(X_cat);
    }

    if (Xc_indptr.size())
    {
        Xc_ptr         =  REAL(Xc);
        Xc_ind_ptr     =  INTEGER(Xc_ind);
        Xc_indptr_ptr  =  INTEGER(Xc_indptr);
    }

    double*  tmat_ptr    =  n_from? (double*)NULL : REAL(tmat);
    double*  dmat_ptr    =  (sq_dist & !n_from)? REAL(dmat) : NULL;
    double*  rmat_ptr    =  n_from? REAL(rmat) : NULL;

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    TreesIndexer*  indexer        =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    if (!Rf_isNull(indexer_R_ptr) && R_ExternalPtrAddr(indexer_R_ptr) != NULL)
        indexer = static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (indexer != NULL && (indexer->indices.empty() || (!as_kernel && indexer->indices.front().node_distances.empty())))
        indexer = NULL;

    if (use_reference_points && indexer != NULL && !indexer->indices.front().reference_points.empty()) {
        tmat_ptr = NULL;
        dmat_ptr = NULL;
        rmat_ptr = REAL(rmat);
    }
    else {
        use_reference_points = false;
    }


    MissingAction missing_action = is_extended?
                                   ext_model_ptr->missing_action
                                     :
                                   model_ptr->missing_action;
    if (missing_action != Fail)
    {
        if (X_num.size()) numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, X_num.size(), Xcpp, nthreads);
        if (Xc.size())    Xc_ptr           = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, nthreads);
    }


    calc_similarity(numeric_data_ptr, categ_data_ptr,
                    Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                    nrows, nthreads,
                    assume_full_distr, standardize_dist, as_kernel,
                    model_ptr, ext_model_ptr,
                    tmat_ptr, rmat_ptr, n_from, use_reference_points,
                    indexer, true, (size_t)0, (size_t)0);

    if (tmat.size() && dmat.ncol() > 0)
    {
        double diag_filler;
        if (as_kernel) {
            if (standardize_dist)
                diag_filler = 1.;
            else
                diag_filler = (model_ptr != NULL)? model_ptr->trees.size() : ext_model_ptr->hplanes.size();
        }
        else {
            if (standardize_dist)
                diag_filler = 0;
            else
                diag_filler = std::numeric_limits<double>::infinity();
        }
        tmat_to_dense(tmat_ptr, dmat_ptr, nrows, diag_filler);
    }
}

// [[Rcpp::export(rng = false)]]
Rcpp::List impute_iso(SEXP model_R_ptr, SEXP imputer_R_ptr, bool is_extended,
                      Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat,
                      Rcpp::NumericVector Xr, Rcpp::IntegerVector Xr_ind, Rcpp::IntegerVector Xr_indptr,
                      size_t nrows, int nthreads)
{
    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    double*     Xr_ptr              =  NULL;
    int*        Xr_ind_ptr          =  NULL;
    int*        Xr_indptr_ptr       =  NULL;

    if (X_num.size())
    {
        numeric_data_ptr  =  REAL(X_num);
    }

    if (X_cat.size())
    {
        categ_data_ptr    =  INTEGER(X_cat);
    }

    if (Xr_indptr.size())
    {
        Xr_ptr         =  REAL(Xr);
        Xr_ind_ptr     =  INTEGER(Xr_ind);
        Xr_indptr_ptr  =  INTEGER(Xr_indptr);
    }

    if (X_num.size()) numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, X_num.size(), nthreads);
    if (Xr.size())    Xr_ptr           = set_R_nan_as_C_nan(Xr_ptr, Xr.size(), nthreads);

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));

    Imputer* imputer_ptr = static_cast<Imputer*>(R_ExternalPtrAddr(imputer_R_ptr));


    impute_missing_values(numeric_data_ptr, categ_data_ptr, true,
                          Xr_ptr, Xr_ind_ptr, Xr_indptr_ptr,
                          nrows, nthreads,
                          model_ptr, ext_model_ptr,
                          *imputer_ptr);

    return Rcpp::List::create(
                Rcpp::_["X_num"] = (Xr.size())? (Xr) : (X_num),
                Rcpp::_["X_cat"] = X_cat
            );
}

// [[Rcpp::export(rng = false)]]
void drop_imputer(Rcpp::List lst_modify, Rcpp::List lst_modify2)
{
    Rcpp::RawVector empty_ser = Rcpp::RawVector();
    Rcpp::LogicalVector FalseObj = Rcpp::LogicalVector::create(false);
    Rcpp::XPtr<Imputer> imp_ptr = lst_modify["imp_ptr"];
    imp_ptr.release();

    lst_modify["imp_ser"] = empty_ser;
    lst_modify2["build_imputer"] = FalseObj;
}

// [[Rcpp::export(rng = false)]]
void drop_indexer(Rcpp::List lst_modify, Rcpp::List lst_modify2)
{
    Rcpp::XPtr<TreesIndexer> empty_ptr = Rcpp::XPtr<TreesIndexer>(nullptr, false);
    Rcpp::RawVector empty_ser = Rcpp::RawVector();
    Rcpp::CharacterVector empty_char = Rcpp::CharacterVector();
    Rcpp::XPtr<TreesIndexer> indexer = lst_modify["indexer"];
    indexer.release();

    lst_modify["ind_ser"] = empty_ser;
    lst_modify2["reference_names"] = empty_char;
}

// [[Rcpp::export(rng = false)]]
void drop_reference_points(Rcpp::List lst_modify, Rcpp::List lst_modify2)
{
    Rcpp::CharacterVector empty_char = Rcpp::CharacterVector();
    Rcpp::RawVector empty_ser = Rcpp::RawVector();
    Rcpp::XPtr<TreesIndexer> indexer_R_ptr = lst_modify["indexer"];
    TreesIndexer *indexer_ptr = indexer_R_ptr.get();
    if (indexer_ptr == NULL) {
        lst_modify["ind_ser"] = empty_ser;
        lst_modify2["reference_names"] = empty_char;
        return;
    }
    if (indexer_ptr->indices.empty()) {
        indexer_R_ptr.release();
        lst_modify["ind_ser"] = empty_ser;
        lst_modify2["reference_names"] = empty_char;
        return;
    }
    if (indexer_ptr->indices.front().reference_points.empty()) {
        lst_modify2["reference_names"] = empty_char;
        return;
    }

    std::unique_ptr<TreesIndexer> new_indexer(new TreesIndexer(*indexer_ptr));
    for (auto &tree : new_indexer->indices)
    {
        tree.reference_points.clear();
        tree.reference_indptr.clear();
        tree.reference_mapping.clear();
    }
    Rcpp::RawVector ind_ser = serialize_cpp_obj(new_indexer.get());
    *indexer_ptr = std::move(*new_indexer);
    new_indexer.release();
    lst_modify["ind_ser"] = ind_ser;
    lst_modify2["reference_names"] = empty_char;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List subset_trees
(
    SEXP model_R_ptr, SEXP imputer_R_ptr, SEXP indexer_R_ptr,
    bool is_extended, bool has_imputer,
    Rcpp::IntegerVector trees_take
)
{
    bool has_indexer = !Rf_isNull(indexer_R_ptr) && R_ExternalPtrAddr(indexer_R_ptr) != NULL;

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["ptr"] = R_NilValue,
        Rcpp::_["serialized"] = R_NilValue,
        Rcpp::_["imp_ptr"] = R_NilValue,
        Rcpp::_["imp_ser"] = R_NilValue,
        Rcpp::_["indexer"] = R_NilValue,
        Rcpp::_["ind_ser"] = R_NilValue
    );

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    Imputer*       imputer_ptr    =  NULL;
    TreesIndexer*  indexer_ptr    =  NULL;
    std::unique_ptr<IsoForest>     new_model_ptr(nullptr);
    std::unique_ptr<ExtIsoForest>  new_ext_model_ptr(nullptr);
    std::unique_ptr<Imputer>       new_imputer_ptr(nullptr);
    std::unique_ptr<TreesIndexer>  new_indexer_ptr(nullptr);

    if (is_extended) {
        ext_model_ptr      =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        new_ext_model_ptr  =  std::unique_ptr<ExtIsoForest>(new ExtIsoForest());
    }
    else {
        model_ptr          =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        new_model_ptr      =  std::unique_ptr<IsoForest>(new IsoForest());
    }

    
    if (has_imputer) {
        imputer_ptr        =  static_cast<Imputer*>(R_ExternalPtrAddr(imputer_R_ptr));
        new_imputer_ptr    =  std::unique_ptr<Imputer>(new Imputer());
    }

    if (has_indexer) {
        indexer_ptr        =  static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
        new_indexer_ptr    =  std::unique_ptr<TreesIndexer>(new TreesIndexer());
    }

    std::unique_ptr<size_t[]> trees_take_(new size_t[trees_take.size()]);
    for (decltype(trees_take.size()) ix = 0; ix < trees_take.size(); ix++)
        trees_take_[ix] = (size_t)(trees_take[ix] - 1);

    subset_model(model_ptr,      new_model_ptr.get(),
                 ext_model_ptr,  new_ext_model_ptr.get(),
                 imputer_ptr,    new_imputer_ptr.get(),
                 indexer_ptr,    new_indexer_ptr.get(),
                 trees_take_.get(), trees_take.size());
    trees_take_.reset();

    if (!is_extended)
        out["serialized"] = serialize_cpp_obj(new_model_ptr.get());
    else
        out["serialized"] = serialize_cpp_obj(new_ext_model_ptr.get());
    if (has_imputer)
        out["imp_ser"] = serialize_cpp_obj(new_imputer_ptr.get());
    if (has_indexer)
        out["ind_ser"] = serialize_cpp_obj(new_indexer_ptr.get());

    if (!is_extended) {
        out["ptr"] = Rcpp::unwindProtect(safe_XPtr<IsoForest>, new_model_ptr.get());
        new_model_ptr.release();
    }
    else {
        out["ptr"] = Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, new_ext_model_ptr.get());
        new_ext_model_ptr.release();
    }
    if (has_imputer) {
        out["imp_ptr"] = Rcpp::unwindProtect(safe_XPtr<Imputer>, new_imputer_ptr.get());
        new_imputer_ptr.release();
    }
    if (has_indexer) {
        out["indexer"] = Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, new_indexer_ptr.get());
        new_indexer_ptr.release();
    }
    return out;
}

// [[Rcpp::export(rng = false)]]
void inplace_set_to_zero(SEXP obj)
{
    auto obj_type = TYPEOF(obj);
    switch(obj_type)
    {
        case REALSXP:
        {
            REAL(obj)[0] = 0;
            break;
        }

        case INTSXP:
        {
            INTEGER(obj)[0] = 0;
            break;
        }

        case LGLSXP:
        {
            LOGICAL(obj)[0] = 0;
            break;
        }

        default:
        {
            Rcpp::stop("Model object has incorrect structure.\n");
        }
    }
}

// [[Rcpp::export(rng = false)]]
Rcpp::List get_n_nodes(SEXP model_R_ptr, bool is_extended, int nthreads)
{
    size_t ntrees;
    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    if (is_extended)
    {
        ext_model_ptr =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        ntrees        =  ext_model_ptr->hplanes.size();
    }
    else
    {
        model_ptr     =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        ntrees        =  model_ptr->trees.size();
    }

    Rcpp::IntegerVector n_nodes(ntrees);
    Rcpp::IntegerVector n_terminal(ntrees);
    if (is_extended)
        get_num_nodes(*ext_model_ptr, INTEGER(n_nodes), INTEGER(n_terminal), nthreads);
    else
        get_num_nodes(*model_ptr, INTEGER(n_nodes), INTEGER(n_terminal), nthreads);

    return Rcpp::List::create(
                Rcpp::_["total"]    = n_nodes,
                Rcpp::_["terminal"] = n_terminal
            );
}

// [[Rcpp::export(rng = false)]]
void append_trees_from_other(SEXP model_R_ptr, SEXP other_R_ptr,
                             SEXP imp_R_ptr, SEXP oimp_R_ptr,
                             SEXP ind_R_ptr, SEXP oind_R_ptr,
                             bool is_extended,
                             Rcpp::RawVector serialized_obj,
                             Rcpp::RawVector serialized_imputer,
                             Rcpp::RawVector serialized_indexer,
                             Rcpp::List &model_cpp_obj_update,
                             Rcpp::List &model_params_update)
{
    if ((!Rf_isNull(imp_R_ptr) && R_ExternalPtrAddr(imp_R_ptr) != NULL)
            &&
        !(!Rf_isNull(oimp_R_ptr) && R_ExternalPtrAddr(oimp_R_ptr) != NULL))
    {
        Rcpp::stop("Model to append trees to has imputer, but model to append from doesn't. Try dropping the imputer.\n");
    }
    if ((!Rf_isNull(ind_R_ptr) && R_ExternalPtrAddr(ind_R_ptr) != NULL)
            &&
        !(!Rf_isNull(oind_R_ptr) && R_ExternalPtrAddr(oind_R_ptr) != NULL))
    {
        Rcpp::stop("Model to append trees to has indexer, but model to append from doesn't. Try dropping the indexer.\n");
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["serialized"] = R_NilValue,
        Rcpp::_["imp_ser"] = R_NilValue,
        Rcpp::_["ind_ser"] = R_NilValue
    );

    Rcpp::IntegerVector ntrees_new = Rcpp::IntegerVector::create(Rf_asInteger(model_params_update["ntrees"]));

    IsoForest* model_ptr = NULL;
    IsoForest* other_ptr = NULL;
    ExtIsoForest* ext_model_ptr = NULL;
    ExtIsoForest* ext_other_ptr = NULL;
    Imputer* imputer_ptr  = NULL;
    Imputer* oimputer_ptr = NULL;
    TreesIndexer* indexer_ptr  = NULL;
    TreesIndexer* oindexer_ptr = NULL;
    size_t old_ntrees;

    if (is_extended) {
        ext_model_ptr = static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        ext_other_ptr = static_cast<ExtIsoForest*>(R_ExternalPtrAddr(other_R_ptr));
        old_ntrees = ext_model_ptr->hplanes.size();
    } else {
        model_ptr = static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        other_ptr = static_cast<IsoForest*>(R_ExternalPtrAddr(other_R_ptr));
        old_ntrees = model_ptr->trees.size();
    }

    if (!Rf_isNull(imp_R_ptr) && !Rf_isNull(oimp_R_ptr) &&
        R_ExternalPtrAddr(imp_R_ptr) != NULL &&
        R_ExternalPtrAddr(oimp_R_ptr) != NULL)
    {
        imputer_ptr  = static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));
        oimputer_ptr = static_cast<Imputer*>(R_ExternalPtrAddr(oimp_R_ptr));
    }

    if (!Rf_isNull(ind_R_ptr) && !Rf_isNull(oind_R_ptr) &&
        R_ExternalPtrAddr(ind_R_ptr) != NULL &&
        R_ExternalPtrAddr(oind_R_ptr) != NULL)
    {
        indexer_ptr  = static_cast<TreesIndexer*>(R_ExternalPtrAddr(ind_R_ptr));
        oindexer_ptr = static_cast<TreesIndexer*>(R_ExternalPtrAddr(oind_R_ptr));
    }

    merge_models(model_ptr, other_ptr,
                 ext_model_ptr, ext_other_ptr,
                 imputer_ptr, oimputer_ptr,
                 indexer_ptr, oindexer_ptr);

    Rcpp::RawVector new_serialized, new_imp_serialized, new_ind_serialized;
    size_t new_size;
    try
    {
        if (!is_extended)
        {
            if (serialized_obj.size() &&
                check_can_undergo_incremental_serialization(*model_ptr, (char*)RAW(serialized_obj)))
            {
                try {
                    new_size = serialized_obj.size()
                                + determine_serialized_size_additional_trees(*model_ptr, old_ntrees);
                    new_serialized = resize_vec(serialized_obj, new_size);
                    char *temp = (char*)RAW(new_serialized);
                    incremental_serialize_isotree(*model_ptr, temp);
                    out["serialized"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_singlevar;
                }
            }

            else {
                serialize_anew_singlevar:
                out["serialized"] = serialize_cpp_obj(model_ptr);
            }
        }

        else
        {
            if (serialized_obj.size() &&
                check_can_undergo_incremental_serialization(*ext_model_ptr, (char*)RAW(serialized_obj)))
            {
                try {
                    new_size = serialized_obj.size()
                                + determine_serialized_size_additional_trees(*ext_model_ptr, old_ntrees);
                    new_serialized = resize_vec(serialized_obj, new_size);
                    char *temp = (char*)RAW(new_serialized);
                    incremental_serialize_isotree(*ext_model_ptr, temp);
                    out["serialized"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ext;
                }
            }

            else {
                serialize_anew_ext:
                out["serialized"] = serialize_cpp_obj(ext_model_ptr);
            }
        }

        if (imputer_ptr != NULL)
        {
            if (serialized_imputer.size() &&
                check_can_undergo_incremental_serialization(*imputer_ptr, (char*)RAW(serialized_imputer)))
            {
                try {
                    new_size = serialized_obj.size()
                                + determine_serialized_size_additional_trees(*imputer_ptr, old_ntrees);
                    new_imp_serialized = resize_vec(serialized_imputer, new_size);
                    char *temp = (char*)RAW(new_imp_serialized);
                    incremental_serialize_isotree(*imputer_ptr, temp);
                    out["imp_ser"] = new_imp_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_imp;
                }
            }

            else {
                serialize_anew_imp:
                out["imp_ser"] = serialize_cpp_obj(imputer_ptr);
            }
        }

        if (indexer_ptr != NULL)
        {
            if (serialized_indexer.size() &&
                check_can_undergo_incremental_serialization(*indexer_ptr, (char*)RAW(serialized_indexer)))
            {
                try {
                    new_size = serialized_obj.size()
                                + determine_serialized_size_additional_trees(*indexer_ptr, old_ntrees);
                    new_ind_serialized = resize_vec(serialized_indexer, new_size);
                    char *temp = (char*)RAW(new_ind_serialized);
                    incremental_serialize_isotree(*indexer_ptr, temp);
                    out["ind_ser"] = new_ind_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ind;
                }
            }

            else {
                serialize_anew_ind:
                out["ind_ser"] = serialize_cpp_obj(indexer_ptr);
            }
        }
    }

    catch (...)
    {
        if (!is_extended)
            model_ptr->trees.resize(old_ntrees);
        else
            ext_model_ptr->hplanes.resize(old_ntrees);

        if (imputer_ptr != NULL)
            imputer_ptr->imputer_tree.resize(old_ntrees);
        if (indexer_ptr != NULL)
            indexer_ptr->indices.resize(old_ntrees);
        throw;
    }

    model_cpp_obj_update["serialized"] = out["serialized"];
    if (imputer_ptr)
        model_cpp_obj_update["imp_ser"] = out["imp_ser"];
    if (indexer_ptr)
        model_cpp_obj_update["ind_ser"] = out["ind_ser"];
    *(INTEGER(ntrees_new)) = is_extended? ext_model_ptr->hplanes.size() : model_ptr->trees.size();
    model_params_update["ntrees"] = ntrees_new;
}

SEXP alloc_List(void *data)
{
    return Rcpp::List(*(size_t*)data);
}

SEXP safe_CastString(void *data)
{
    return Rcpp::CharacterVector(*(std::string*)data);
}

// [[Rcpp::export(rng = false)]]
Rcpp::ListOf<Rcpp::CharacterVector> model_to_sql(SEXP model_R_ptr, bool is_extended,
                                                 Rcpp::CharacterVector numeric_colanmes,
                                                 Rcpp::CharacterVector categ_colnames,
                                                 Rcpp::ListOf<Rcpp::CharacterVector> categ_levels,
                                                 bool output_tree_num, bool single_tree, size_t tree_num,
                                                 int nthreads)
{
    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));

    std::vector<std::string> numeric_colanmes_cpp = Rcpp::as<std::vector<std::string>>(numeric_colanmes);
    std::vector<std::string> categ_colanmes_cpp = Rcpp::as<std::vector<std::string>>(categ_colnames);
    std::vector<std::vector<std::string>> categ_levels_cpp = Rcpp::as<std::vector<std::vector<std::string>>>(categ_levels);

    std::vector<std::string> res = generate_sql(model_ptr, ext_model_ptr,
                                                numeric_colanmes_cpp,
                                                categ_colanmes_cpp,
                                                categ_levels_cpp,
                                                output_tree_num, true, single_tree, tree_num,
                                                nthreads);
    /* TODO: this function could create objects through the ALTREP system instead.
       That way, it would avoid an extra copy of the data */
    size_t sz = res.size();
    Rcpp::List out = Rcpp::unwindProtect(alloc_List, (void*)&sz);
    for (size_t ix = 0; ix < res.size(); ix++)
        out[ix] = Rcpp::unwindProtect(safe_CastString, &(res[ix]));
    return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::CharacterVector model_to_sql_with_select_from(SEXP model_R_ptr, bool is_extended,
                                                    Rcpp::CharacterVector numeric_colanmes,
                                                    Rcpp::CharacterVector categ_colnames,
                                                    Rcpp::ListOf<Rcpp::CharacterVector> categ_levels,
                                                    Rcpp::CharacterVector table_from,
                                                    Rcpp::CharacterVector select_as,
                                                    int nthreads)
{
    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));

    std::vector<std::string> numeric_colanmes_cpp = Rcpp::as<std::vector<std::string>>(numeric_colanmes);
    std::vector<std::string> categ_colanmes_cpp = Rcpp::as<std::vector<std::string>>(categ_colnames);
    std::vector<std::vector<std::string>> categ_levels_cpp = Rcpp::as<std::vector<std::vector<std::string>>>(categ_levels);
    std::string table_from_cpp = Rcpp::as<std::string>(table_from);
    std::string select_as_cpp = Rcpp::as<std::string>(select_as);

    std::string out = generate_sql_with_select_from(model_ptr, ext_model_ptr,
                                                    table_from_cpp, select_as_cpp,
                                                    numeric_colanmes_cpp, categ_colanmes_cpp,
                                                    categ_levels_cpp,
                                                    true, nthreads);
    /* TODO: this function could create objects through the ALTREP system instead.
       That way, it would avoid an extra copy of the data */
    return Rcpp::unwindProtect(safe_CastString, &out);
}

// [[Rcpp::export(rng = false)]]
Rcpp::List copy_cpp_objects(SEXP model_R_ptr, bool is_extended, SEXP imp_R_ptr, bool has_imputer, SEXP ind_R_ptr)
{
    bool has_indexer = !Rf_isNull(ind_R_ptr) && R_ExternalPtrAddr(ind_R_ptr) != NULL;

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["ptr"]    =  R_NilValue,
        Rcpp::_["imp_ptr"]  =  R_NilValue,
        Rcpp::_["indexer"]  =  R_NilValue
    );

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    Imputer*       imputer_ptr    =  NULL;
    TreesIndexer*  indexer_ptr    =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    if (has_imputer)
        imputer_ptr    =  static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));
    if (has_indexer)
        indexer_ptr    =  static_cast<TreesIndexer*>(R_ExternalPtrAddr(ind_R_ptr));

    std::unique_ptr<IsoForest> copy_model(new IsoForest());
    std::unique_ptr<ExtIsoForest> copy_ext_model(new ExtIsoForest());
    std::unique_ptr<Imputer> copy_imputer(new Imputer());
    std::unique_ptr<TreesIndexer> copy_indexer(new TreesIndexer());

    if (model_ptr != NULL) 
        *copy_model = *model_ptr;
    if (ext_model_ptr != NULL)
        *copy_ext_model = *ext_model_ptr;
    if (imputer_ptr != NULL)
        *copy_imputer = *imputer_ptr;
    if (indexer_ptr != NULL)
        *copy_indexer = *indexer_ptr;

    if (is_extended) {
        out["ptr"]    =  Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, copy_ext_model.get());
        copy_ext_model.release();
    }
    else {
        out["ptr"]    =  Rcpp::unwindProtect(safe_XPtr<IsoForest>, copy_model.get());
        copy_model.release();
    }
    if (has_imputer) {
        out["imp_ptr"]  =  Rcpp::unwindProtect(safe_XPtr<Imputer>, copy_imputer.get());
        copy_imputer.release();
    }
    if (has_indexer) {
        out["indexer"]  =  Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, copy_indexer.get());
        copy_indexer.release();
    }
    return out;
}

// [[Rcpp::export(rng = false)]]
void build_tree_indices(Rcpp::List lst_modify, bool is_extended, bool with_distances, int nthreads)
{
    Rcpp::RawVector ind_ser = Rcpp::RawVector();
    Rcpp::List empty_lst = Rcpp::List::create(Rcpp::_["indexer"] = R_NilValue);
    std::unique_ptr<TreesIndexer> indexer(new TreesIndexer());

    if (!is_extended) {
        build_tree_indices(*indexer,
                           *static_cast<IsoForest*>(R_ExternalPtrAddr(lst_modify["ptr"])),
                           nthreads,
                           with_distances);
    }
    else {
        build_tree_indices(*indexer,
                           *static_cast<ExtIsoForest*>(R_ExternalPtrAddr(lst_modify["ptr"])),
                           nthreads,
                           with_distances);
    }

    ind_ser  =  serialize_cpp_obj(indexer.get());
    empty_lst["indexer"]     =  Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, indexer.get());
    if (!Rf_isNull(lst_modify["indexer"])) {
        Rcpp::XPtr<TreesIndexer> indexer_R_ptr = lst_modify["indexer"];
        indexer_R_ptr.release();
    }
    
    lst_modify["ind_ser"] = ind_ser;
    lst_modify["indexer"] = empty_lst["indexer"];
    indexer.release();
}

// [[Rcpp::export(rng = false)]]
bool check_node_indexer_has_distances(SEXP indexer_R_ptr)
{
    if (Rf_isNull(indexer_R_ptr) || R_ExternalPtrAddr(indexer_R_ptr) == NULL)
        return false;
    TreesIndexer *indexer = static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (indexer->indices.empty()) return false;
    return !indexer->indices.front().node_distances.empty();
}

// [[Rcpp::export(rng = false)]]
void set_reference_points(Rcpp::List lst_modify, Rcpp::List lst_modify2, SEXP rnames, bool is_extended,
                          Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat,
                          Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
                          size_t nrows, int nthreads, bool with_distances)
{
    Rcpp::RawVector ind_ser = Rcpp::RawVector();
    Rcpp::XPtr<TreesIndexer> indexer_R_ptr = lst_modify["indexer"];

    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    double*     Xc_ptr              =  NULL;
    int*        Xc_ind_ptr          =  NULL;
    int*        Xc_indptr_ptr       =  NULL;
    Rcpp::NumericVector Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr  =  REAL(X_num);
    }

    if (X_cat.size())
    {
        categ_data_ptr    =  INTEGER(X_cat);
    }

    if (Xc_indptr.size())
    {
        Xc_ptr         =  REAL(Xc);
        Xc_ind_ptr     =  INTEGER(Xc_ind);
        Xc_indptr_ptr  =  INTEGER(Xc_indptr);
    }

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    TreesIndexer*  indexer        =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(lst_modify["ptr"]));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(lst_modify["ptr"]));
    indexer            =  indexer_R_ptr.get();

    MissingAction missing_action = is_extended?
                                   ext_model_ptr->missing_action
                                     :
                                   model_ptr->missing_action;
    if (missing_action != Fail)
    {
        if (X_num.size()) numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, X_num.size(), Xcpp, nthreads);
        if (Xc.size())    Xc_ptr           = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, nthreads);
    }

    std::unique_ptr<TreesIndexer> new_indexer(new TreesIndexer(*indexer));

    set_reference_points(model_ptr, ext_model_ptr, new_indexer.get(),
                         with_distances,
                         numeric_data_ptr, categ_data_ptr,
                         true, (size_t)0, (size_t)0,
                         Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                         (double*)NULL, (int*)NULL, (int*)NULL,
                         nrows, nthreads);

    ind_ser = serialize_cpp_obj(new_indexer.get());
    *indexer = std::move(*new_indexer);
    new_indexer.release();
    lst_modify["ind_ser"] = ind_ser;
    lst_modify2["reference_names"] = rnames;
}

// [[Rcpp::export(rng = false)]]
bool check_node_indexer_has_references(SEXP indexer_R_ptr)
{
    if (Rf_isNull(indexer_R_ptr) || R_ExternalPtrAddr(indexer_R_ptr) == NULL)
        return false;
    TreesIndexer *indexer = static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (indexer->indices.empty())
        return false;
    if (indexer->indices.front().reference_points.empty())
        return false;
    else
        return true;
}

// [[Rcpp::export(rng = false)]]
int get_num_references(SEXP indexer_R_ptr)
{
    TreesIndexer *indexer = static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (indexer == NULL || indexer->indices.empty()) return 0;
    return indexer->indices.front().reference_points.size();
}

// [[Rcpp::export(rng = false)]]
SEXP get_null_R_pointer()
{
    return R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue);
}

/* This library will use different code paths for opening a file path
   in order to support non-ASCII characters, depending on compiler and
   platform support. */
#if (defined(_WIN32) || defined(_WIN64))
#   if defined(__GNUC__) && (__GNUC__ >= 5)
#       define USE_CODECVT
#       define TAKE_AS_UTF8 true
#   elif !defined(_FOR_CRAN)
#       define USE_RC_FOPEN
#       define TAKE_AS_UTF8 false
#   else
#       define USE_SIMPLE_FOPEN
#       define TAKE_AS_UTF8 false
#   endif
#else
#   define USE_SIMPLE_FOPEN
#   define TAKE_AS_UTF8 false
#endif

/* Now the actual implementations */
#ifdef USE_CODECVT
/* https://stackoverflow.com/questions/2573834/c-convert-string-or-char-to-wstring-or-wchar-t */
/*  */
#include <locale>
#include <codecvt>
#include <string>
FILE* R_fopen(Rcpp::CharacterVector fname, const char *mode)
{
    Rcpp::String s(fname[0], CE_UTF8);
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wide = converter.from_bytes(s.get_cstring());
    std::string mode__(mode);
    std::wstring mode_ = converter.from_bytes(mode__);
    return _wfopen(wide.c_str(), mode_.c_str());
}
#endif

#ifdef USE_RC_FOPEN
extern "C" {
    FILE *RC_fopen(const SEXP fn, const char *mode, const Rboolean expand);
}
FILE* R_fopen(Rcpp::CharacterVector fname, const char *mode)
{
    return RC_fopen(fname[0], mode, FALSE);
}
#endif

#ifdef USE_SIMPLE_FOPEN
FILE* R_fopen(Rcpp::CharacterVector fname, const char *mode)
{
    return fopen(fname[0], mode);
}
#endif

class FileOpener
{
public:
    FILE *handle = NULL;
    FileOpener(const SEXP fname, const char *mode)
    {
        if (this->handle != NULL)
            this->close_file();
        this->handle = R_fopen(fname, mode);
    }
    FILE *get_handle()
    {
        return this->handle;
    }
    void close_file()
    {
        if (this->handle != NULL) {
            fclose(this->handle);
            this->handle = NULL;
        }
    }
    ~FileOpener()
    {
        this->close_file();
    }
};

// [[Rcpp::export]]
void serialize_to_file
(
    Rcpp::RawVector serialized_obj,
    Rcpp::RawVector serialized_imputer,
    Rcpp::RawVector serialized_indexer,
    bool is_extended,
    Rcpp::RawVector metadata,
    Rcpp::CharacterVector fname
)
{
    FileOpener file_(fname[0], "wb");
    FILE *output_file = file_.get_handle();
    serialize_combined(
        is_extended? nullptr : (char*)RAW(serialized_obj),
        is_extended? (char*)RAW(serialized_obj) : nullptr,
        serialized_imputer.size()? (char*)RAW(serialized_imputer) : nullptr,
        serialized_indexer.size()? (char*)RAW(serialized_indexer) : nullptr,
        metadata.size()? (char*)RAW(metadata) : nullptr,
        metadata.size(),
        output_file
    );
}

// [[Rcpp::export]]
Rcpp::List deserialize_from_file(Rcpp::CharacterVector fname)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["ptr"] = R_NilValue,
        Rcpp::_["serialized"] = R_NilValue,
        Rcpp::_["imp_ptr"] = R_NilValue,
        Rcpp::_["imp_ser"] = R_NilValue,
        Rcpp::_["indexer"] = R_NilValue,
        Rcpp::_["ind_ser"] = R_NilValue,
        Rcpp::_["metadata"] = R_NilValue
    );

    FileOpener file_(fname[0], "rb");
    FILE *input_file = file_.get_handle();

    bool is_isotree_model;
    bool is_compatible;
    bool has_combined_objects;
    bool has_IsoForest;
    bool has_ExtIsoForest;
    bool has_Imputer;
    bool has_Indexer;
    bool has_metadata;
    size_t size_metadata;
    
    inspect_serialized_object(
        input_file,
        is_isotree_model,
        is_compatible,
        has_combined_objects,
        has_IsoForest,
        has_ExtIsoForest,
        has_Imputer,
        has_Indexer,
        has_metadata,
        size_metadata
    );

    if (!is_isotree_model || !has_combined_objects)
        Rcpp::stop("Input file is not a serialized isotree model.\n");
    if (!is_compatible)
        Rcpp::stop("Model file format is incompatible.\n");
    if (!size_metadata)
        Rcpp::stop("Input file does not contain metadata.\n");

    out["metadata"] = Rcpp::unwindProtect(alloc_RawVec, (void*)&size_metadata);

    std::unique_ptr<IsoForest> model(new IsoForest());
    std::unique_ptr<ExtIsoForest> model_ext(new ExtIsoForest());
    std::unique_ptr<Imputer> imputer(new Imputer());
    std::unique_ptr<TreesIndexer> indexer(new TreesIndexer());

    IsoForest *ptr_model = NULL;
    ExtIsoForest *ptr_model_ext = NULL;
    Imputer *ptr_imputer = NULL;
    TreesIndexer *ptr_indexer = NULL;
    char *ptr_metadata = (char*)RAW(out["metadata"]);

    if (has_IsoForest)
        ptr_model = model.get();
    if (has_ExtIsoForest)
        ptr_model_ext = model_ext.get();
    if (has_Imputer)
        ptr_imputer = imputer.get();
    if (has_Indexer)
        ptr_indexer = indexer.get();

    deserialize_combined(
        input_file,
        ptr_model,
        ptr_model_ext,
        ptr_imputer,
        ptr_indexer,
        ptr_metadata
    );

    if (has_IsoForest)
        out["serialized"] = serialize_cpp_obj(model.get());
    else
        out["serialized"] = serialize_cpp_obj(model_ext.get());
    if (has_Imputer)
        out["imp_ser"] = serialize_cpp_obj(imputer.get());
    if (has_Indexer)
        out["ind_ser"] = serialize_cpp_obj(indexer.get());

    if (has_IsoForest) {
        out["ptr"] = Rcpp::unwindProtect(safe_XPtr<IsoForest>, model.get());
        model.release();
    }
    else {
        out["ptr"] = Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, model_ext.get());
        model_ext.release();
    }
    if (has_Imputer) {
        out["imp_ptr"] = Rcpp::unwindProtect(safe_XPtr<Imputer>, imputer.get());
        imputer.release();
    }
    if (has_Indexer) {
        out["indexer"] = Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, indexer.get());
        indexer.release();
    }

    return out;
}

/* The functions below make for missing functionality in the
   'Matrix' and 'SparseM' packages for sub-setting the data */

// [[Rcpp::export(rng = false)]]
void call_sort_csc_indices(Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr)
{
    size_t ncols_numeric = Xc_indptr.size() - 1;
    sort_csc_indices(REAL(Xc), INTEGER(Xc_ind), INTEGER(Xc_indptr), ncols_numeric);
}

// [[Rcpp::export(rng = false)]]
void call_reconstruct_csr_sliced
(
    Rcpp::NumericVector orig_Xr, Rcpp::IntegerVector orig_Xr_indptr,
    Rcpp::NumericVector rec_Xr, Rcpp::IntegerVector rec_Xr_indptr,
    size_t nrows
)
{
    reconstruct_csr_sliced<double, int>(
        REAL(orig_Xr), INTEGER(orig_Xr_indptr),
        REAL(rec_Xr), INTEGER(rec_Xr_indptr),
        nrows
    );
}

// [[Rcpp::export(rng = false)]]
void call_reconstruct_csr_with_categ
(
    Rcpp::NumericVector orig_Xr, Rcpp::IntegerVector orig_Xr_ind, Rcpp::IntegerVector orig_Xr_indptr,
    Rcpp::NumericVector rec_Xr, Rcpp::IntegerVector rec_Xr_ind, Rcpp::IntegerVector rec_Xr_indptr,
    Rcpp::IntegerVector rec_X_cat,
    Rcpp::IntegerVector cols_numeric, Rcpp::IntegerVector cols_categ,
    size_t nrows, size_t ncols
)
{
    reconstruct_csr_with_categ<double, int, int>(
        REAL(orig_Xr), INTEGER(orig_Xr_ind), INTEGER(orig_Xr_indptr),
        REAL(rec_Xr), INTEGER(rec_Xr_ind), INTEGER(rec_Xr_indptr),
        INTEGER(rec_X_cat), true,
        INTEGER(cols_numeric), INTEGER(cols_categ),
        nrows, ncols, cols_numeric.size(), cols_categ.size()
    );
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector deepcopy_vector(Rcpp::NumericVector inp)
{
    return Rcpp::NumericVector(inp.begin(), inp.end());
}

Rcpp::IntegerMatrix csc_to_dense_int
(
    Rcpp::NumericVector Xc,
    Rcpp::IntegerVector Xc_ind,
    Rcpp::IntegerVector Xc_indptr,
    size_t nrows
)
{
    size_t ncols = Xc_indptr.size() - 1;
    Rcpp::IntegerMatrix out_(nrows, ncols);
    int *restrict out = INTEGER(out_);
    for (size_t col = 0; col < ncols; col++)
    {
        for (auto ix = Xc_indptr[col]; ix < Xc_indptr[col+1]; ix++)
            out[(size_t)Xc_ind[ix] + col*nrows]
                =
            (Xc[ix] >= 0 && !ISNAN(Xc[ix]))?
                (int)Xc[ix] : (int)(-1);
    }
    return out_;
}

template <class real_vec, class int_vec>
Rcpp::IntegerMatrix csr_to_dense_int
(
    real_vec Xr,
    int_vec Xr_ind,
    int_vec Xr_indptr,
    int ncols
)
{
    size_t nrows = Xr_indptr.size() - 1;
    size_t matrix_dims[] = {nrows, (size_t)ncols};
    Rcpp::IntegerMatrix out_ = Rcpp::unwindProtect(safe_int_matrix, (void*)matrix_dims);
    int *restrict out = INTEGER(out_);
    for (size_t row = 0; row < nrows; row++)
    {
        for (auto ix = Xr_indptr[row]; ix < Xr_indptr[row+1]; ix++)
            out[row + (size_t)Xr_ind[ix]*nrows]
                =
            (Xr[ix] >= 0 && !ISNAN(Xr[ix]))?
                (int)Xr[ix] : (int)(-1);
    }
    return out_;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List call_take_cols_by_slice_csr
(
    Rcpp::NumericVector Xr_,
    Rcpp::IntegerVector Xr_ind_,
    Rcpp::IntegerVector Xr_indptr,
    int ncols_take,
    bool as_dense
)
{
    /* Indices need to be sorted beforehand */
    double *restrict Xr = REAL(Xr_);
    int *restrict Xr_ind = INTEGER(Xr_ind_);
    size_t nrows = Xr_indptr.size() - 1;
    Rcpp::IntegerVector out_Xr_indptr(nrows+1);
    out_Xr_indptr[0] = 0;
    size_t total_size = 0;
    for (size_t row = 0; row < nrows; row++)
    {
        for (auto col = Xr_indptr[row]; col < Xr_indptr[row+1]; col++)
            total_size += Xr_ind[col] < ncols_take;
        out_Xr_indptr[row+1] = total_size;
    }

    Rcpp::NumericVector out_Xr_(total_size);
    Rcpp::IntegerVector out_Xr_ind_(total_size);
    double *restrict out_Xr = REAL(out_Xr_);
    int *restrict out_Xr_ind = INTEGER(out_Xr_ind_);

    size_t n_this;
    for (size_t row = 0; row < nrows; row++)
    {
        n_this = out_Xr_indptr[row+1] - out_Xr_indptr[row];
        if (n_this) {
            std::copy(Xr + Xr_indptr[row],
                      Xr + Xr_indptr[row] + n_this,
                      out_Xr + out_Xr_indptr[row]);
            std::copy(Xr_ind + Xr_indptr[row],
                      Xr_ind + Xr_indptr[row] + n_this,
                      out_Xr_ind + out_Xr_indptr[row]);
        }
    }

    if (!as_dense)
        return Rcpp::List::create(
            Rcpp::_["Xr"] = out_Xr_,
            Rcpp::_["Xr_ind"] = out_Xr_ind_,
            Rcpp::_["Xr_indptr"] = out_Xr_indptr
        );
    else
        return Rcpp::List::create(
            Rcpp::_["X_cat"] = csr_to_dense_int(out_Xr_,
                                                out_Xr_ind_,
                                                out_Xr_indptr,
                                                ncols_take)
        );
}

// [[Rcpp::export(rng = false)]]
Rcpp::List call_take_cols_by_index_csr
(
    Rcpp::NumericVector Xr,
    Rcpp::IntegerVector Xr_ind,
    Rcpp::IntegerVector Xr_indptr,
    Rcpp::IntegerVector cols_take,
    bool as_dense
)
{
    Rcpp::List out;
    if (!as_dense) {
        out = Rcpp::List::create(
            Rcpp::_["Xr"] = R_NilValue,
            Rcpp::_["Xr_ind"] = R_NilValue,
            Rcpp::_["Xr_indptr"] = R_NilValue
        );
    }
    else {
        out = Rcpp::List::create(
            Rcpp::_["X_cat"] = R_NilValue
        );
    }


    /* 'cols_take' should be sorted */
    int n_take = cols_take.size();
    int nrows = Xr_indptr.size() - 1;
    std::vector<double> out_Xr;
    std::vector<int> out_Xr_ind;
    std::vector<int> out_Xr_indptr(nrows + 1);

    int *curr_ptr;
    int *end_ptr;
    int *restrict ptr_Xr_ind = INTEGER(Xr_ind);
    int *restrict ptr_cols_take = INTEGER(cols_take);
    int *restrict ptr_cols_take_end = ptr_cols_take + n_take;
    int curr_col;
    int *search_res;

    for (int row = 0; row < nrows; row++)
    {
        curr_ptr = ptr_Xr_ind + Xr_indptr[row];
        end_ptr = ptr_Xr_ind + Xr_indptr[row+1];
        curr_col = 0;

        if (end_ptr == curr_ptr + 1)
        {
            search_res = std::lower_bound(ptr_cols_take, ptr_cols_take_end, *curr_ptr);
            curr_col = std::distance(ptr_cols_take, search_res);
            if (curr_col < n_take && *search_res == *curr_ptr)
            {
                out_Xr.push_back(Xr[std::distance(ptr_Xr_ind, curr_ptr)]);
                out_Xr_ind.push_back(curr_col);
            }
        }

        else
        if (end_ptr > curr_ptr)
        {
            while (true)
            {
                curr_ptr = std::lower_bound(curr_ptr, end_ptr, ptr_cols_take[curr_col]);
                
                if (curr_ptr >= end_ptr)
                {
                    break;
                }


                else if (*curr_ptr == ptr_cols_take[curr_col])
                {
                    out_Xr.push_back(Xr[std::distance(ptr_Xr_ind, curr_ptr)]);
                    out_Xr_ind.push_back(curr_col);
                    curr_ptr++;
                    curr_col++;

                    if (curr_ptr >= end_ptr || curr_col >= n_take)
                        break;
                }


                else
                {
                    curr_col = std::distance(
                        ptr_cols_take,
                        std::lower_bound(ptr_cols_take + curr_col, ptr_cols_take_end, *curr_ptr)
                    );

                    if (curr_col >= n_take)
                        break;

                    if (curr_col == *curr_ptr) {
                        out_Xr.push_back(Xr[std::distance(ptr_Xr_ind, curr_ptr)]);
                        out_Xr_ind.push_back(curr_col);
                        curr_ptr++;
                        curr_col++;
                    }

                    if (curr_ptr >= end_ptr || curr_col >= n_take)
                        break;
                }
            }
        }

        out_Xr_indptr[row+1] = out_Xr.size();
    }

    if (!as_dense)
    {
        out["Xr"] = Rcpp::unwindProtect(safe_copy_vec, (void*)&out_Xr);
        out["Xr_ind"] = Rcpp::unwindProtect(safe_copy_intvec, (void*)&out_Xr_ind);
        out["Xr_indptr"] = Rcpp::unwindProtect(safe_copy_intvec, (void*)&out_Xr_indptr);
    }

    else
    {
        out["X_cat"] = csr_to_dense_int(out_Xr,
                                        out_Xr_ind,
                                        out_Xr_indptr,
                                        n_take);
    }

    return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List call_take_cols_by_slice_csc
(
    Rcpp::NumericVector Xc,
    Rcpp::IntegerVector Xc_ind,
    Rcpp::IntegerVector Xc_indptr,
    size_t ncols_take,
    bool as_dense, size_t nrows
)
{
    Rcpp::IntegerVector out_Xc_indptr(ncols_take+1);
    size_t total_size = Xc_indptr[ncols_take+1];
    Rcpp::NumericVector out_Xc(REAL(Xc), REAL(Xc) + total_size);
    Rcpp::IntegerVector out_Xc_ind(INTEGER(Xc_ind), INTEGER(Xc_ind) + total_size);

    if (!as_dense)
        return Rcpp::List::create(
            Rcpp::_["Xc"] = out_Xc,
            Rcpp::_["Xc_ind"] = out_Xc_ind,
            Rcpp::_["Xc_indptr"] = out_Xc_indptr
        );
    else
        return Rcpp::List::create(
            Rcpp::_["X_cat"] = csc_to_dense_int(out_Xc,
                                                out_Xc_ind,
                                                out_Xc_indptr,
                                                nrows)
        );
}

// [[Rcpp::export(rng = false)]]
Rcpp::List call_take_cols_by_index_csc
(
    Rcpp::NumericVector Xc_,
    Rcpp::IntegerVector Xc_ind_,
    Rcpp::IntegerVector Xc_indptr,
    Rcpp::IntegerVector cols_take,
    bool as_dense, size_t nrows
)
{
    /* 'cols_take' should be sorted */
    double *restrict Xc = REAL(Xc_);
    int *restrict Xc_ind = INTEGER(Xc_ind_);
    size_t n_take = cols_take.size();
    Rcpp::IntegerVector out_Xc_indptr(n_take+1);
    size_t total_size = 0;

    for (size_t col = 0; col < n_take; col++)
        total_size += Xc_indptr[cols_take[col]+1] - Xc_indptr[cols_take[col]];

    Rcpp::NumericVector out_Xc_(total_size);
    Rcpp::IntegerVector out_Xc_ind_(total_size);
    double *restrict out_Xc = REAL(out_Xc_);
    int *restrict out_Xc_ind = INTEGER(out_Xc_ind_);

    total_size = 0;
    size_t n_this;
    out_Xc_indptr[0] = 0;
    for (size_t col = 0; col < n_take; col++)
    {
        n_this = Xc_indptr[cols_take[col]+1] - Xc_indptr[cols_take[col]];
        if (n_this) {
            std::copy(Xc + Xc_indptr[cols_take[col]],
                      Xc + Xc_indptr[cols_take[col]] + n_this,
                      out_Xc + total_size);
            std::copy(Xc_ind + Xc_indptr[cols_take[col]],
                      Xc_ind + Xc_indptr[cols_take[col]] + n_this,
                      out_Xc_ind + total_size);
        }
        total_size += n_this;
        out_Xc_indptr[col+1] = total_size;
    }

    if (!as_dense)
        return Rcpp::List::create(
            Rcpp::_["Xc"] = out_Xc_,
            Rcpp::_["Xc_ind"] = out_Xc_ind_,
            Rcpp::_["Xc_indptr"] = out_Xc_indptr
        );
    else
        return Rcpp::List::create(
            Rcpp::_["X_cat"] = csc_to_dense_int(out_Xc_,
                                                out_Xc_ind_,
                                                out_Xc_indptr,
                                                nrows)
        );
}

// [[Rcpp::export(rng = false)]]
void copy_csc_cols_by_slice
(
    Rcpp::NumericVector out_Xc_,
    Rcpp::IntegerVector out_Xc_indptr,
    Rcpp::NumericVector from_Xc_,
    Rcpp::IntegerVector from_Xc_indptr,
    size_t n_copy
)
{
    size_t total_size = from_Xc_indptr[n_copy+1];
    std::copy(REAL(from_Xc_), REAL(from_Xc_) + total_size, REAL(out_Xc_));
}

// [[Rcpp::export(rng = false)]]
void copy_csc_cols_by_index
(
    Rcpp::NumericVector out_Xc_,
    Rcpp::IntegerVector out_Xc_indptr,
    Rcpp::NumericVector from_Xc_,
    Rcpp::IntegerVector from_Xc_indptr,
    Rcpp::IntegerVector cols_copy
)
{
    size_t n_copy = cols_copy.size();
    double *restrict out_Xc = REAL(out_Xc_);
    double *restrict from_Xc = REAL(from_Xc_);

    for (size_t col = 0; col < n_copy; col++)
    {
        std::copy(from_Xc + from_Xc_indptr[col],
                  from_Xc + from_Xc_indptr[col+1],
                  out_Xc + out_Xc_indptr[cols_copy[col]]);
    }
}


// [[Rcpp::export(rng = false)]]
Rcpp::List assign_csc_cols
(
    Rcpp::NumericVector Xc_,
    Rcpp::IntegerVector Xc_ind_,
    Rcpp::IntegerVector Xc_indptr,
    Rcpp::IntegerVector X_cat_,
    Rcpp::IntegerVector cols_categ,
    Rcpp::IntegerVector cols_numeric,
    size_t nrows
)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["Xc"] = R_NilValue,
        Rcpp::_["Xc_ind"] = R_NilValue,
        Rcpp::_["Xc_indptr"] = R_NilValue
    );
    size_t ncols_tot = (size_t)cols_categ.size() + (size_t)cols_numeric.size();
    std::vector<double> out_Xc;
    std::vector<int> out_Xc_ind;
    std::vector<int> out_Xc_indptr(ncols_tot + 1);

    double *restrict Xc = REAL(Xc_);
    int *restrict Xc_ind = INTEGER(Xc_ind_);
    int *restrict X_cat = INTEGER(X_cat_);

    hashed_set<int> cols_categ_set(INTEGER(cols_categ), INTEGER(cols_categ) + cols_categ.size());
    hashed_set<int> cols_numeric_set(INTEGER(cols_numeric), INTEGER(cols_numeric) + cols_numeric.size());

    size_t curr_num = 0;
    size_t curr_cat = 0;
    bool has_zeros;
    size_t curr_size;

    for (size_t col = 0; col < ncols_tot; col++)
    {
        if (is_in_set((int)col, cols_numeric_set))
        {
            std::copy(Xc + Xc_indptr[curr_num],
                      Xc + Xc_indptr[curr_num+1],
                      std::back_inserter(out_Xc));
            std::copy(Xc_ind + Xc_indptr[curr_num],
                      Xc_ind + Xc_indptr[curr_num+1],
                      std::back_inserter(out_Xc_ind));
            curr_num++;
        }

        else if (is_in_set((int)col, cols_categ_set))
        {
            has_zeros = false;
            for (size_t row = 0; row < nrows; row++)
                if (X_cat[row + (size_t)curr_cat*nrows] == 0)
                    has_zeros = true;

            if (!has_zeros) {
                std::copy(X_cat + (size_t)curr_cat*nrows,
                          X_cat + ((size_t)curr_cat+1)*nrows,
                          std::back_inserter(out_Xc));
                curr_size = out_Xc_ind.size();
                out_Xc_ind.resize(curr_size + (size_t)nrows);
                std::iota(out_Xc_ind.begin() + curr_size, out_Xc_ind.end(), (int)0);
            }

            else {
                for (size_t row = 0; row < nrows; row++) {
                    if (X_cat[row + (size_t)curr_cat*nrows] > 0) {
                        out_Xc.push_back(X_cat[row + (size_t)curr_cat*nrows]);
                        out_Xc_ind.push_back((int)row);
                    }
                }
            }

            curr_cat++;
        }

        out_Xc_indptr[col+1] = out_Xc.size();
    }


    out["Xc"] = Rcpp::unwindProtect(safe_copy_vec, (void*)&out_Xc);
    out["Xc_ind"] = Rcpp::unwindProtect(safe_copy_intvec, (void*)&out_Xc_ind);
    out["Xc_indptr"] = Rcpp::unwindProtect(safe_copy_intvec, (void*)&out_Xc_indptr);
    return out;
}

/* These are helpers for dealing with large integers and R's copy-on-write semantics */

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector get_empty_tmat(int nrows_)
{
    size_t nrows = (size_t)nrows_;
    size_t tmat_size = (nrows * (nrows - (size_t)1)) / (size_t)2;
    return Rcpp::NumericVector((R_xlen_t)tmat_size);
}

// [[Rcpp::export(rng = false)]]
Rcpp::IntegerMatrix get_empty_int_mat(int nrows, int ncols)
{
    return Rcpp::IntegerMatrix(nrows, ncols);
}

// [[Rcpp::export(rng = false)]]
Rcpp::IntegerMatrix get_null_int_mat()
{
    return Rcpp::IntegerMatrix(0, 0);
}

// [[Rcpp::export(rng = false)]]
int get_ntrees(SEXP model_R_ptr, bool is_extended)
{
    if (is_extended) {
        ExtIsoForest* ext_model_ptr = static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        return ext_model_ptr->hplanes.size();
    }
    
    else {
        IsoForest* model_ptr = static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        return model_ptr->trees.size();
    }
}

// [[Rcpp::export(rng = false)]]
SEXP deepcopy_int(SEXP x)
{
    return Rf_ScalarInteger(Rf_asInteger(x));
}

// [[Rcpp::export(rng = false)]]
void modify_R_list_inplace(SEXP lst, int ix, SEXP el)
{
    SET_VECTOR_ELT(lst, ix, el);
}

// [[Rcpp::export(rng = false)]]
void addto_R_list_inplace(Rcpp::List &lst, Rcpp::String nm, SEXP el)
{
    lst[nm] = el;
}


// [[Rcpp::export(rng = false)]]
bool R_has_openmp()
{
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}

#endif /* _FOR_R */
