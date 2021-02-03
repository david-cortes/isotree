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
#ifdef _FOR_R

#include <Rcpp.h>
#include <Rcpp/unwindProtect.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(unwindProtect)]]

/* This is to serialize the model objects */
// [[Rcpp::depends(Rcereal)]]
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <string>

/* This is the package's header */
#include "isotree.hpp"

/* Library is templated, base R comes with only these 2 types though */
#include "model_joined.hpp"
#define real_t double
#define sparse_ix int
#include "instantiate_model.hpp"


/*  Note: the R version calls the 'sort_csc_indices' templated function,
    so it's not enough to just include 'isotree_exportable.hpp' and let
    the templates be instantiated elsewhere. */


SEXP alloc_RawVec(void *data)
{
    return Rcpp::RawVector(*(size_t*)data);
}

/* for model serialization and re-usage in R */
/* https://stackoverflow.com/questions/18474292/how-to-handle-c-internal-data-structure-in-r-in-order-to-allow-save-load */
/* this extra comment below the link is a workaround for Rcpp issue 675 in GitHub, do not remove it */
#include <Rinternals.h>
template <class T>
Rcpp::RawVector serialize_cpp_obj(T *model_outputs)
{
    std::stringstream ss;
    {
        cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
        oarchive(*model_outputs);
    }
    ss.seekg(0, ss.end);
    /* Checking for potential integer overflows */
    std::stringstream::pos_type vec_size = ss.tellg();
    if (vec_size <= 0) {
        Rcpp::Rcerr << "Error: model is too big to serialize, resulting object will not be usable.\n" << std::endl;
        return Rcpp::RawVector();
    }

    Rcpp::RawVector retval;
    size_t vec_size_ = (size_t)vec_size;
    retval = Rcpp::unwindProtect(alloc_RawVec, (void*)&vec_size_);
    if (!retval.size())
        return retval;
    ss.seekg(0, ss.beg);
    ss.read(reinterpret_cast<char*>(&retval[0]), retval.size());
    return retval;
}

template <class T>
SEXP deserialize_cpp_obj(Rcpp::RawVector src)
{
    std::stringstream ss;
    ss.write(reinterpret_cast<char*>(&src[0]), src.size());
    ss.seekg(0, ss.beg);
    std::unique_ptr<T> model_outputs = std::unique_ptr<T>(new T());
    {
        cereal::BinaryInputArchive iarchive(ss);
        iarchive(*model_outputs);
    }
    return Rcpp::XPtr<T>(model_outputs.release(), true);
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
Rcpp::LogicalVector check_null_ptr_model(SEXP ptr_model)
{
    return Rcpp::LogicalVector(R_ExternalPtrAddr(ptr_model) == NULL);
}

double* set_R_nan_as_C_nan(double *x, size_t n, std::vector<double> &v, int nthreads)
{
    v.assign(x, x + n);
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(x, v, n)
    for (size_t_for i = 0; i < n; i++)
        if (isnan(v[i]))
            v[i] = NAN;
    return v.data();
}

double* set_R_nan_as_C_nan(double *x, size_t n, int nthreads)
{
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(x, n)
    for (size_t_for i = 0; i < n; i++)
        if (isnan(x[i]))
            x[i] = NAN;
    return x;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List fit_model(Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat, Rcpp::IntegerVector ncat,
                     Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
                     Rcpp::NumericVector sample_weights, Rcpp::NumericVector col_weights,
                     size_t nrows, size_t ncols_numeric, size_t ncols_categ, size_t ndim, size_t ntry,
                     Rcpp::CharacterVector coef_type, bool coef_by_prop, bool with_replacement, bool weight_as_sample,
                     size_t sample_size, size_t ntrees,  size_t max_depth, size_t ncols_per_tree, bool limit_depth,
                     bool penalize_range, bool calc_dist, bool standardize_dist, bool sq_dist,
                     bool calc_depth, bool standardize_depth, bool weigh_by_kurt,
                     double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                     double prob_pick_by_gain_pl,  double prob_split_by_gain_pl, double min_gain,
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
    int      *  Xc_ind_ptr          =  NULL;
    int      *  Xc_indptr_ptr       =  NULL;
    double*     sample_weights_ptr  =  NULL;
    double*     col_weights_ptr     =  NULL;
    std::vector<double> Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr = &X_num[0];
        if (Rcpp::as<std::string>(missing_action) != std::string("fail"))
            numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, nrows * ncols_numeric, Xcpp, nthreads);
    }

    if (X_cat.size())
    {
        categ_data_ptr  =  &X_cat[0];
        ncat_ptr        =  &ncat[0];
    }

    if (Xc.size())
    {
        Xc_ptr          =  &Xc[0];
        Xc_ind_ptr      =  &Xc_ind[0];
        Xc_indptr_ptr   =  &Xc_indptr[0];
        if (Rcpp::as<std::string>(missing_action) != std::string("fail"))
            Xc_ptr = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, nthreads);
    }

    if (sample_weights.size())
    {
        sample_weights_ptr  =  &sample_weights[0];
    }

    if (col_weights.size())
    {
        col_weights_ptr     =  &col_weights[0];
    }

    CoefType        coef_type_C       =  Normal;
    CategSplit      cat_split_type_C  =  SubSet;
    NewCategAction  new_cat_action_C  =  Weighted;
    MissingAction   missing_action_C  =  Divide;
    UseDepthImp     depth_imp_C       =  Higher;
    WeighImpRows    weigh_imp_rows_C  =  Inverse;

    if (Rcpp::as<std::string>(coef_type) == std::string("uniform"))
    {
        coef_type_C       =  Uniform;
    }
    if (Rcpp::as<std::string>(cat_split_type) == std::string("single_categ"))
    {
        cat_split_type_C  =  SingleCateg;
    }
    if (Rcpp::as<std::string>(new_cat_action) == std::string("smallest"))
    {
        new_cat_action_C  =  Smallest;
    }
    else if (Rcpp::as<std::string>(new_cat_action) == std::string("random"))
    {
        new_cat_action_C  =  Random;
    }
    if (Rcpp::as<std::string>(missing_action) == std::string("impute"))
    {
        missing_action_C  =  Impute;
    }
    else if (Rcpp::as<std::string>(missing_action) == std::string("fail"))
    {
        missing_action_C  =  Fail;
    }
    if (Rcpp::as<std::string>(depth_imp) == std::string("lower"))
    {
        depth_imp_C       =  Lower;
    }
    else if (Rcpp::as<std::string>(depth_imp) == std::string("same"))
    {
        depth_imp_C       =  Same;
    }
    if (Rcpp::as<std::string>(weigh_imp_rows) == std::string("prop"))
    {
        weigh_imp_rows_C  =  Prop;
    }
    else if (Rcpp::as<std::string>(weigh_imp_rows) == std::string("flat"))
    {
        weigh_imp_rows_C  =  Flat;
    }

    Rcpp::NumericVector  tmat    =  Rcpp::NumericVector();
    Rcpp::NumericMatrix  dmat    =  Rcpp::NumericMatrix();
    Rcpp::NumericVector  depths  =  Rcpp::NumericVector();
    double*  tmat_ptr    =  NULL;
    double*  dmat_ptr    =  NULL;
    double*  depths_ptr  =  NULL;

    if (calc_dist)
    {
        tmat      =  Rcpp::NumericVector((nrows * (nrows - 1)) / 2);
        tmat_ptr  =  &tmat[0];
        if (sq_dist)
        {
            dmat      =  Rcpp::NumericMatrix(nrows);
            dmat_ptr  =  &dmat(0, 0);
        }
    }

    if (calc_depth)
    {
        depths      =  Rcpp::NumericVector(nrows);
        depths_ptr  =  &depths[0];
    }

    std::unique_ptr<IsoForest>     model_ptr      =  std::unique_ptr<IsoForest>();
    std::unique_ptr<ExtIsoForest>  ext_model_ptr  =  std::unique_ptr<ExtIsoForest>();
    std::unique_ptr<Imputer>       imputer_ptr    =  std::unique_ptr<Imputer>();

    if (ndim == 1)
        model_ptr      =  std::unique_ptr<IsoForest>(new IsoForest());
    else
        ext_model_ptr  =  std::unique_ptr<ExtIsoForest>(new ExtIsoForest());

    if (build_imputer)
        imputer_ptr    =  std::unique_ptr<Imputer>(new Imputer());

    int ret_val = 
    fit_iforest(model_ptr.get(), ext_model_ptr.get(),
                numeric_data_ptr,  ncols_numeric,
                categ_data_ptr,    ncols_categ,    ncat_ptr,
                Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                ndim, ntry, coef_type_C, coef_by_prop,
                sample_weights_ptr, with_replacement, weight_as_sample,
                nrows, sample_size, ntrees, max_depth, ncols_per_tree,
                limit_depth, penalize_range,
                standardize_dist, tmat_ptr,
                depths_ptr, standardize_depth,
                col_weights_ptr, weigh_by_kurt,
                prob_pick_by_gain_avg, prob_split_by_gain_avg,
                prob_pick_by_gain_pl,  prob_split_by_gain_pl,
                min_gain, missing_action_C,
                cat_split_type_C, new_cat_action_C,
                all_perm, imputer_ptr.get(), min_imp_obs,
                depth_imp_C, weigh_imp_rows_C, output_imputations,
                (uint64_t) random_seed, nthreads);
    Rcpp::checkUserInterrupt();

    if (ret_val == EXIT_FAILURE)
    {
        return Rcpp::List::create(Rcpp::_["err"] = Rcpp::LogicalVector::create(1));
    }

    if (calc_dist && sq_dist)
        tmat_to_dense(tmat_ptr, dmat_ptr, nrows, !standardize_dist);

    bool serialization_failed = false;
    Rcpp::RawVector serialized_obj;
    if (ndim == 1)
        serialized_obj  =  serialize_cpp_obj(model_ptr.get());
    else
        serialized_obj  =  serialize_cpp_obj(ext_model_ptr.get());
    if (!serialized_obj.size()) serialization_failed = true;
    if (serialization_failed) {
        if (ndim == 1)
            model_ptr.reset();
        else
            ext_model_ptr.reset();
    }

    Rcpp::List outp = Rcpp::List::create(
                Rcpp::_["serialized_obj"] = serialized_obj,
                Rcpp::_["depths"]    = depths,
                Rcpp::_["tmat"]      = tmat,
                Rcpp::_["dmat"]      = dmat
        );

    if (!serialization_failed)
    {
        if (ndim == 1)
            outp["model_ptr"]   =  Rcpp::XPtr<IsoForest>(model_ptr.release(), true);
        else
            outp["model_ptr"]   =  Rcpp::XPtr<ExtIsoForest>(ext_model_ptr.release(), true);
    } else
        outp["model_ptr"] = R_NilValue;

    if (build_imputer && !serialization_failed)
    {
        outp["imputer_ser"] =  serialize_cpp_obj(imputer_ptr.get());
        if (!Rf_xlength(outp["imputer_ser"]))
        {
            serialization_failed = true;
            imputer_ptr.reset();
            if (ndim == 1)
                model_ptr.reset();
            else
                ext_model_ptr.reset();
            outp["imputer_ptr"]  =  R_NilValue;
            outp["model_ptr"]    =  R_NilValue;
        } else
            outp["imputer_ptr"] =  Rcpp::XPtr<Imputer>(imputer_ptr.release(), true);
    }

    if (output_imputations && !serialization_failed)
    {
        outp["imputed_num"] = Rcpp::NumericVector(Xcpp.begin(), Xcpp.end());
        outp["imputed_cat"] = X_cat;
    }

    outp["err"] = Rcpp::LogicalVector::create(0);

    return outp;
}

// [[Rcpp::export(rng = false)]]
Rcpp::RawVector fit_tree(SEXP model_R_ptr, 
                         Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat, Rcpp::IntegerVector ncat,
                         Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
                         Rcpp::NumericVector sample_weights, Rcpp::NumericVector col_weights,
                         size_t nrows, size_t ncols_numeric, size_t ncols_categ,
                         size_t ndim, size_t ntry, Rcpp::CharacterVector coef_type, bool coef_by_prop,
                         size_t max_depth, size_t ncols_per_tree, bool limit_depth, bool penalize_range,
                         bool weigh_by_kurt,
                         double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                         double prob_pick_by_gain_pl,  double prob_split_by_gain_pl, double min_gain,
                         Rcpp::CharacterVector cat_split_type, Rcpp::CharacterVector new_cat_action,
                         Rcpp::CharacterVector missing_action, bool build_imputer, size_t min_imp_obs, SEXP imp_R_ptr,
                         Rcpp::CharacterVector depth_imp, Rcpp::CharacterVector weigh_imp_rows,
                         bool all_perm, uint64_t random_seed)
{
    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    int*        ncat_ptr            =  NULL;
    double*     Xc_ptr              =  NULL;
    int      *  Xc_ind_ptr          =  NULL;
    int      *  Xc_indptr_ptr       =  NULL;
    double*     sample_weights_ptr  =  NULL;
    double*     col_weights_ptr     =  NULL;
    std::vector<double> Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr = &X_num[0];
        if (Rcpp::as<std::string>(missing_action) != std::string("fail"))
            numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, nrows * ncols_numeric, Xcpp, 1);
    }

    if (X_cat.size())
    {
        categ_data_ptr  =  &X_cat[0];
        ncat_ptr        =  &ncat[0];
    }

    if (Xc.size())
    {
        Xc_ptr         =  &Xc[0];
        Xc_ind_ptr     =  &Xc_ind[0];
        Xc_indptr_ptr  =  &Xc_indptr[0];
        if (Rcpp::as<std::string>(missing_action) != std::string("fail"))
            Xc_ptr = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, 1);
    }

    if (sample_weights.size())
    {
        sample_weights_ptr  =  &sample_weights[0];
    }

    if (col_weights.size())
    {
        col_weights_ptr     =  &col_weights[0];
    }

    CoefType        coef_type_C       =  Normal;
    CategSplit      cat_split_type_C  =  SubSet;
    NewCategAction  new_cat_action_C  =  Weighted;
    MissingAction   missing_action_C  =  Divide;
    UseDepthImp     depth_imp_C       =  Higher;
    WeighImpRows    weigh_imp_rows_C  =  Inverse;

    if (Rcpp::as<std::string>(coef_type) == std::string("uniform"))
    {
        coef_type_C       =  Uniform;
    }
    if (Rcpp::as<std::string>(cat_split_type) == std::string("single_categ"))
    {
        cat_split_type_C  =  SingleCateg;
    }
    if (Rcpp::as<std::string>(new_cat_action) == std::string("smallest"))
    {
        new_cat_action_C  =  Smallest;
    }
    else if (Rcpp::as<std::string>(new_cat_action) == std::string("random"))
    {
        new_cat_action_C  =  Random;
    }
    if (Rcpp::as<std::string>(missing_action) == std::string("impute"))
    {
        missing_action_C  =  Impute;
    }
    else if (Rcpp::as<std::string>(missing_action) == std::string("fail"))
    {
        missing_action_C  =  Fail;
    }
    if (Rcpp::as<std::string>(depth_imp) == std::string("lower"))
    {
        depth_imp_C       =  Lower;
    }
    else if (Rcpp::as<std::string>(depth_imp) == std::string("same"))
    {
        depth_imp_C       =  Same;
    }
    if (Rcpp::as<std::string>(weigh_imp_rows) == std::string("prop"))
    {
        weigh_imp_rows_C  =  Prop;
    }
    else if (Rcpp::as<std::string>(weigh_imp_rows) == std::string("flat"))
    {
        weigh_imp_rows_C  =  Flat;
    }

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    Imputer*       imputer_ptr    = NULL;
    if (ndim == 1)
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));

    std::vector<ImputeNode>  *imp_ptr = NULL;
    if (build_imputer)
    {
        imputer_ptr = static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));
        imputer_ptr->imputer_tree.emplace_back();
        imp_ptr     = &imputer_ptr->imputer_tree.back();
    }

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
             depth_imp_C, weigh_imp_rows_C, all_perm,
             imp_ptr, min_imp_obs, (uint64_t)random_seed);

    if (ndim == 1)
        return serialize_cpp_obj(model_ptr);
    else
        return serialize_cpp_obj(ext_model_ptr);
}

// [[Rcpp::export(rng = false)]]
void predict_iso(SEXP model_R_ptr, Rcpp::NumericVector outp, Rcpp::IntegerVector tree_num, bool is_extended,
                 Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat,
                 Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
                 Rcpp::NumericVector Xr, Rcpp::IntegerVector Xr_ind, Rcpp::IntegerVector Xr_indptr,
                 size_t nrows, int nthreads, bool standardize)
{
    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    double*     Xc_ptr              =  NULL;
    int      *  Xc_ind_ptr          =  NULL;
    int      *  Xc_indptr_ptr       =  NULL;
    double*     Xr_ptr              =  NULL;
    int      *  Xr_ind_ptr          =  NULL;
    int      *  Xr_indptr_ptr       =  NULL;
    int      *  tree_num_ptr        =  NULL;
    std::vector<double> Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr  =  &X_num[0];
    }

    if (X_cat.size())
    {
        categ_data_ptr    =  &X_cat[0];
    }

    if (Xc_indptr.size())
    {
        if (Xc.size())
            Xc_ptr         =  &Xc[0];
        if (Xc_ind.size())
            Xc_ind_ptr     =  &Xc_ind[0];
        Xc_indptr_ptr      =  &Xc_indptr[0];
    }

    if (Xr_indptr.size())
    {
        if (Xr.size())
            Xr_ptr         =  &Xr[0];
        if (Xr_ind.size())
            Xr_ind_ptr     =  &Xr_ind[0];
        Xr_indptr_ptr      =  &Xr_indptr[0];
    }

    if (tree_num.size())
    {
        tree_num_ptr = &tree_num[0];
    }

    double* depths_ptr =  &outp[0];

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));

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
                                 depths_ptr, tree_num_ptr);
}

// [[Rcpp::export(rng = false)]]
void dist_iso(SEXP model_R_ptr, Rcpp::NumericVector tmat, Rcpp::NumericVector dmat,
              Rcpp::NumericVector rmat, bool is_extended,
              Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat,
              Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
              size_t nrows, int nthreads, bool assume_full_distr,
              bool standardize_dist, bool sq_dist, size_t n_from)
{
    double*     numeric_data_ptr    =  NULL;
    int*        categ_data_ptr      =  NULL;
    double*     Xc_ptr              =  NULL;
    int      *  Xc_ind_ptr          =  NULL;
    int      *  Xc_indptr_ptr       =  NULL;
    std::vector<double> Xcpp;

    if (X_num.size())
    {
        numeric_data_ptr  =  &X_num[0];
    }

    if (X_cat.size())
    {
        categ_data_ptr    =  &X_cat[0];
    }

    if (Xc_indptr.size())
    {
        if (Xc.size())
            Xc_ptr         =  &Xc[0];
        if (Xc_ind.size())
            Xc_ind_ptr     =  &Xc_ind[0];
        Xc_indptr_ptr  =  &Xc_indptr[0];
    }

    double*  tmat_ptr    =  n_from? (double*)NULL : &tmat[0];
    double*  dmat_ptr    =  (sq_dist & !n_from)? &dmat[0] : NULL;
    double*  rmat_ptr    =  n_from? &rmat[0] : NULL;

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));


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
                    nrows, nthreads, assume_full_distr, standardize_dist,
                    model_ptr, ext_model_ptr,
                    tmat_ptr, rmat_ptr, n_from);

    if (sq_dist & !n_from)
        tmat_to_dense(tmat_ptr, dmat_ptr, nrows, !standardize_dist);
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
    int      *  Xr_ind_ptr          =  NULL;
    int      *  Xr_indptr_ptr       =  NULL;

    if (X_num.size())
    {
        numeric_data_ptr  =  &X_num[0];
    }

    if (X_cat.size())
    {
        categ_data_ptr    =  &X_cat[0];
    }

    if (Xr_indptr.size())
    {
        if (Xr.size())
            Xr_ptr         =  &Xr[0];
        if (Xr_ind.size())
            Xr_ind_ptr     =  &Xr_ind[0];
        Xr_indptr_ptr  =  &Xr_indptr[0];
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
        get_num_nodes(*ext_model_ptr, &n_nodes[0], &n_terminal[0], nthreads);
    else
        get_num_nodes(*model_ptr, &n_nodes[0], &n_terminal[0], nthreads);

    return Rcpp::List::create(
                Rcpp::_["total"]    = n_nodes,
                Rcpp::_["terminal"] = n_terminal
            );
}

// [[Rcpp::export(rng = false)]]
Rcpp::List append_trees_from_other(SEXP model_R_ptr, SEXP other_R_ptr,
                                   SEXP imp_R_ptr, SEXP oimp_R_ptr,
                                   bool is_extended)
{
    Rcpp::List out;
    IsoForest* model_ptr = NULL;
    IsoForest* other_ptr = NULL;
    ExtIsoForest* ext_model_ptr = NULL;
    ExtIsoForest* ext_other_ptr = NULL;
    Imputer* imputer_ptr  = NULL;
    Imputer* oimputer_ptr = NULL;

    if (is_extended) {
        ext_model_ptr = static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        ext_other_ptr = static_cast<ExtIsoForest*>(R_ExternalPtrAddr(other_R_ptr));
    } else {
        model_ptr = static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        other_ptr = static_cast<IsoForest*>(R_ExternalPtrAddr(other_R_ptr));
    }

    if (!Rf_isNull(imp_R_ptr) && !Rf_isNull(oimp_R_ptr) &&
        R_ExternalPtrAddr(imp_R_ptr) != NULL &&
        R_ExternalPtrAddr(oimp_R_ptr) != NULL)
    {
        imputer_ptr  = static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));
        oimputer_ptr = static_cast<Imputer*>(R_ExternalPtrAddr(oimp_R_ptr));
    }

    merge_models(model_ptr, other_ptr,
                 ext_model_ptr, ext_other_ptr,
                 imputer_ptr, oimputer_ptr);


    if (is_extended)
        out["serialized"] = serialize_cpp_obj(ext_model_ptr);
    else
        out["serialized"] = serialize_cpp_obj(model_ptr);

    if (imputer_ptr != NULL && oimputer_ptr != NULL)
        out["imp_ser"] = serialize_cpp_obj(imputer_ptr);

    return out;
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
    return Rcpp::unwindProtect(safe_CastString, &out);
}

// [[Rcpp::export(rng = false)]]
Rcpp::List copy_cpp_objects(SEXP model_R_ptr, bool is_extended, SEXP imp_R_ptr, bool has_imputer)
{
    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    Imputer*       imputer_ptr    =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    if (has_imputer)
        imputer_ptr    =  static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));

    std::unique_ptr<IsoForest> copy_model(new IsoForest());
    std::unique_ptr<ExtIsoForest> copy_ext_model(new ExtIsoForest());
    std::unique_ptr<Imputer> copy_imputer(new Imputer());

    if (model_ptr != NULL) 
        *copy_model = *model_ptr;
    if (ext_model_ptr != NULL)
        *copy_ext_model = *ext_model_ptr;
    if (imputer_ptr != NULL)
        *copy_imputer = *imputer_ptr;

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["model_ptr"]    =  R_NilValue,
        Rcpp::_["imputer_ptr"]  =  R_NilValue
    );

    if (is_extended)
        out["model_ptr"]    =  Rcpp::XPtr<ExtIsoForest>(copy_ext_model.release(), true);
    else
        out["model_ptr"]    =  Rcpp::XPtr<IsoForest>(copy_model.release(), true);
    if (has_imputer)
        out["imputer_ptr"]  =  Rcpp::XPtr<Imputer>(copy_imputer.release(), true);
    return out;
}

// [[Rcpp::export(rng = false)]]
void call_sort_csc_indices(Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr)
{
    size_t ncols_numeric = Xc_indptr.size() - 1;
    sort_csc_indices(REAL(Xc), INTEGER(Xc_ind), INTEGER(Xc_indptr), ncols_numeric);
}

#endif /* _FOR_R */
