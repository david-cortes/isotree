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
*     Copyright (c) 2019-2023, David Cortes
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
#include <Rinternals.h>
#include <R_ext/Altrep.h>

#include <type_traits>

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

#define throw_mem_err() throw Rcpp::exception("Error: insufficient memory. Try smaller sample sizes and fewer trees.\n")

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
        throw Rcpp::exception("Unexpected error.");
    if (unlikely(serialized_size > (size_t)std::numeric_limits<R_xlen_t>::max()))
        throw Rcpp::exception("Resulting model is too large for R to handle.");
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
SEXP serialize_IsoForest_from_ptr(SEXP R_ptr)
{
    const IsoForest* model = (const IsoForest*)R_ExternalPtrAddr(R_ptr);
    return serialize_cpp_obj<IsoForest>(model);
}

// [[Rcpp::export(rng = false)]]
SEXP serialize_ExtIsoForest_from_ptr(SEXP R_ptr)
{
    const ExtIsoForest* model = (const ExtIsoForest*)R_ExternalPtrAddr(R_ptr);
    return serialize_cpp_obj<ExtIsoForest>(model);
}

// [[Rcpp::export(rng = false)]]
SEXP serialize_Imputer_from_ptr(SEXP R_ptr)
{
    const Imputer* model = (const Imputer*)R_ExternalPtrAddr(R_ptr);
    return serialize_cpp_obj<Imputer>(model);
}

// [[Rcpp::export(rng = false)]]
SEXP serialize_Indexer_from_ptr(SEXP R_ptr)
{
    const TreesIndexer* model = (const TreesIndexer*)R_ExternalPtrAddr(R_ptr);
    return serialize_cpp_obj<TreesIndexer>(model);
}

// [[Rcpp::export(rng = false)]]
Rcpp::LogicalVector check_null_ptr_model_internal(SEXP ptr_model)
{
    return Rcpp::LogicalVector(R_ExternalPtrAddr(ptr_model) == NULL);
}

static R_altrep_class_t altrepped_pointer_IsoForest;
static R_altrep_class_t altrepped_pointer_ExtIsoForest;
static R_altrep_class_t altrepped_pointer_Imputer;
static R_altrep_class_t altrepped_pointer_TreesIndexer;
static R_altrep_class_t altrepped_pointer_NullPointer;

template <class Model>
R_altrep_class_t get_altrep_obj_class()
{
    if (std::is_same<Model, IsoForest>::value) return altrepped_pointer_IsoForest;

    if (std::is_same<Model, ExtIsoForest>::value) return altrepped_pointer_ExtIsoForest;

    if (std::is_same<Model, Imputer>::value) return altrepped_pointer_Imputer;

    if (std::is_same<Model, TreesIndexer>::value) return altrepped_pointer_TreesIndexer;

    throw Rcpp::exception("Internal error. Please open a bug report.");
}

R_xlen_t altrepped_pointer_length(SEXP obj)
{
    return 1;
}

SEXP get_element_from_altrepped_obj(SEXP R_altrepped_obj, R_xlen_t idx)
{
    return R_altrep_data1(R_altrepped_obj);
}

template <class Model>
void delete_model_from_R_ptr(SEXP R_ptr)
{
    Model *cpp_ptr = (Model*)R_ExternalPtrAddr(R_ptr);
    delete cpp_ptr;
    R_SetExternalPtrAddr(R_ptr, nullptr);
    R_ClearExternalPtr(R_ptr);
}

template <class Model>
SEXP get_altrepped_pointer(void *void_ptr)
{
    SEXP R_ptr_name = PROTECT(Rf_mkString("ptr"));
    SEXP R_ptr_class = PROTECT(Rf_mkString("isotree_altrepped_handle"));
    SEXP R_ptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    SEXP out = PROTECT(R_new_altrep(get_altrep_obj_class<Model>(), R_NilValue, R_NilValue));

    std::unique_ptr<Model> *ptr = (std::unique_ptr<Model>*)void_ptr;
    R_SetExternalPtrAddr(R_ptr, ptr->get());
    R_RegisterCFinalizerEx(R_ptr, delete_model_from_R_ptr<Model>, TRUE);
    ptr->release();
    
    R_set_altrep_data1(out, R_ptr);
    Rf_setAttrib(out, R_NamesSymbol, R_ptr_name);
    Rf_setAttrib(out, R_ClassSymbol, R_ptr_class);

    UNPROTECT(4);
    return out;
}

template <class Model>
SEXP serialize_altrepped_pointer(SEXP altrepped_obj)
{
    try {
        Model *cpp_ptr = (Model*)R_ExternalPtrAddr(R_altrep_data1(altrepped_obj));
        R_xlen_t state_size = determine_serialized_size(*cpp_ptr);
        SEXP R_state = PROTECT(Rf_allocVector(RAWSXP, state_size));
        serialize_isotree(*cpp_ptr, (char*)RAW(R_state));
        UNPROTECT(1);
        return R_state;
    }
    catch (const std::exception &ex) {
        Rf_error("%s\n", ex.what());
    }

    return R_NilValue; /* <- won't be reached */
}

template <class Model>
SEXP deserialize_altrepped_pointer(SEXP cls, SEXP R_state)
{
    SEXP R_ptr_name = PROTECT(Rf_mkString("ptr"));
    SEXP R_ptr_class = PROTECT(Rf_mkString("isotree_altrepped_handle"));
    SEXP R_ptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    SEXP out = PROTECT(R_new_altrep(get_altrep_obj_class<Model>(), R_NilValue, R_NilValue));

    try {
        std::unique_ptr<Model> model(new Model());
        const char *inp = (const char*)RAW(R_state);
        deserialize_isotree(*model, inp);

        R_SetExternalPtrAddr(R_ptr, model.get());
        R_RegisterCFinalizerEx(R_ptr, delete_model_from_R_ptr<Model>, TRUE);
        model.release();
    }
    catch (const std::exception &ex) {
        Rf_error("%s\n", ex.what());
    }

    R_set_altrep_data1(out, R_ptr);
    Rf_setAttrib(out, R_NamesSymbol, R_ptr_name);
    Rf_setAttrib(out, R_ClassSymbol, R_ptr_class);

    UNPROTECT(4);
    return out;
}

template <class Model>
SEXP duplicate_altrepped_pointer(SEXP altrepped_obj, Rboolean deep)
{
    SEXP R_ptr_name = PROTECT(Rf_mkString("ptr"));
    SEXP R_ptr_class = PROTECT(Rf_mkString("isotree_altrepped_handle"));
    SEXP out = PROTECT(R_new_altrep(get_altrep_obj_class<Model>(), R_NilValue, R_NilValue));

    if (!deep) {
        R_set_altrep_data1(out, R_altrep_data1(altrepped_obj));
    }

    else {

        SEXP R_ptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));

        try {
            std::unique_ptr<Model> new_obj(new Model());
            Model *cpp_ptr = (Model*)R_ExternalPtrAddr(R_altrep_data1(altrepped_obj));
            *new_obj = *cpp_ptr;

            R_SetExternalPtrAddr(R_ptr, new_obj.get());
            R_RegisterCFinalizerEx(R_ptr, delete_model_from_R_ptr<Model>, TRUE);
            new_obj.release();
        }

        catch (const std::exception &ex) {
            Rf_error("%s\n", ex.what());
        }

        R_set_altrep_data1(out, R_ptr);
        UNPROTECT(1);
    }

    Rf_setAttrib(out, R_NamesSymbol, R_ptr_name);
    Rf_setAttrib(out, R_NamesSymbol, R_ptr_class);
    UNPROTECT(3);
    return out;
}

SEXP get_altrepped_null_pointer()
{
    SEXP R_ptr_name = PROTECT(Rf_mkString("ptr"));
    SEXP R_ptr_class = PROTECT(Rf_mkString("isotree_altrepped_handle"));
    SEXP R_ptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    SEXP out = PROTECT(R_new_altrep(altrepped_pointer_NullPointer, R_ptr, R_NilValue));
    Rf_setAttrib(out, R_NamesSymbol, R_ptr_name);
    Rf_setAttrib(out, R_ClassSymbol, R_ptr_class);
    UNPROTECT(4);
    return out;
}

SEXP safe_get_altrepped_null_pointer(void *unused)
{
    return get_altrepped_null_pointer();
}

SEXP serialize_altrepped_null(SEXP altrepped_obj)
{
    return Rf_allocVector(RAWSXP, 0);
}

SEXP deserialize_altrepped_null(SEXP cls, SEXP R_state)
{
    return get_altrepped_null_pointer();
}

SEXP duplicate_altrepped_pointer(SEXP altrepped_obj, Rboolean deep)
{
    return get_altrepped_null_pointer();
}

Rboolean inspect_altrepped_pointer(SEXP x, int pre, int deep, int pvec, void (*inspect_subtree)(SEXP, int, int, int))
{
    Rcpp::Rcout << "Altrepped pointer [address:" << R_ExternalPtrAddr(R_altrep_data1(x)) << "]\n";
    return TRUE;
}

template <class Model>
Model* get_pointer_from_altrep(SEXP altrepped_obj)
{
    return (Model*)R_ExternalPtrAddr(R_altrep_data1(altrepped_obj));
}

template <class Model>
Model* get_pointer_from_xptr(SEXP R_ptr)
{
    return (Model*)R_ExternalPtrAddr(R_ptr);
}

// [[Rcpp::init]]
void init_altrepped_vectors(DllInfo* dll)
{
    altrepped_pointer_IsoForest = R_make_altlist_class("altrepped_pointer_IsoForest", "isotree", dll);
    R_set_altrep_Length_method(altrepped_pointer_IsoForest, altrepped_pointer_length);
    R_set_altrep_Inspect_method(altrepped_pointer_IsoForest, inspect_altrepped_pointer);
    R_set_altrep_Serialized_state_method(altrepped_pointer_IsoForest, serialize_altrepped_pointer<IsoForest>);
    R_set_altrep_Unserialize_method(altrepped_pointer_IsoForest, deserialize_altrepped_pointer<IsoForest>);
    R_set_altrep_Duplicate_method(altrepped_pointer_IsoForest, duplicate_altrepped_pointer<IsoForest>);
    R_set_altlist_Elt_method(altrepped_pointer_IsoForest, get_element_from_altrepped_obj);

    altrepped_pointer_ExtIsoForest = R_make_altlist_class("altrepped_pointer_ExtIsoForest", "isotree", dll);
    R_set_altrep_Length_method(altrepped_pointer_ExtIsoForest, altrepped_pointer_length);
    R_set_altrep_Inspect_method(altrepped_pointer_ExtIsoForest, inspect_altrepped_pointer);
    R_set_altrep_Serialized_state_method(altrepped_pointer_ExtIsoForest, serialize_altrepped_pointer<ExtIsoForest>);
    R_set_altrep_Unserialize_method(altrepped_pointer_ExtIsoForest, deserialize_altrepped_pointer<ExtIsoForest>);
    R_set_altrep_Duplicate_method(altrepped_pointer_ExtIsoForest, duplicate_altrepped_pointer<ExtIsoForest>);
    R_set_altlist_Elt_method(altrepped_pointer_ExtIsoForest, get_element_from_altrepped_obj);

    altrepped_pointer_Imputer = R_make_altlist_class("altrepped_pointer_Imputer", "isotree", dll);
    R_set_altrep_Length_method(altrepped_pointer_Imputer, altrepped_pointer_length);
    R_set_altrep_Inspect_method(altrepped_pointer_Imputer, inspect_altrepped_pointer);
    R_set_altrep_Serialized_state_method(altrepped_pointer_Imputer, serialize_altrepped_pointer<Imputer>);
    R_set_altrep_Unserialize_method(altrepped_pointer_Imputer, deserialize_altrepped_pointer<Imputer>);
    R_set_altrep_Duplicate_method(altrepped_pointer_Imputer, duplicate_altrepped_pointer<Imputer>);
    R_set_altlist_Elt_method(altrepped_pointer_Imputer, get_element_from_altrepped_obj);

    altrepped_pointer_TreesIndexer = R_make_altlist_class("altrepped_pointer_TreesIndexer", "isotree", dll);
    R_set_altrep_Length_method(altrepped_pointer_TreesIndexer, altrepped_pointer_length);
    R_set_altrep_Inspect_method(altrepped_pointer_TreesIndexer, inspect_altrepped_pointer);
    R_set_altrep_Serialized_state_method(altrepped_pointer_TreesIndexer, serialize_altrepped_pointer<TreesIndexer>);
    R_set_altrep_Unserialize_method(altrepped_pointer_TreesIndexer, deserialize_altrepped_pointer<TreesIndexer>);
    R_set_altrep_Duplicate_method(altrepped_pointer_TreesIndexer, duplicate_altrepped_pointer<TreesIndexer>);
    R_set_altlist_Elt_method(altrepped_pointer_TreesIndexer, get_element_from_altrepped_obj);

    altrepped_pointer_NullPointer = R_make_altlist_class("altrepped_pointer_NullPointer", "isotree", dll);
    R_set_altrep_Length_method(altrepped_pointer_NullPointer, altrepped_pointer_length);
    R_set_altrep_Inspect_method(altrepped_pointer_NullPointer, inspect_altrepped_pointer);
    R_set_altrep_Serialized_state_method(altrepped_pointer_NullPointer, serialize_altrepped_null);
    R_set_altrep_Unserialize_method(altrepped_pointer_NullPointer, deserialize_altrepped_null);
    R_set_altrep_Duplicate_method(altrepped_pointer_NullPointer, duplicate_altrepped_pointer);
    R_set_altlist_Elt_method(altrepped_pointer_NullPointer, get_element_from_altrepped_obj);
}

double* set_R_nan_as_C_nan(double *x, size_t n, std::vector<double> &v, int nthreads)
{
    v.assign(x, x + n);
    for (size_t i = 0; i < n; i++)
        if (unlikely(std::isnan(v[i]))) v[i] = NAN;
    return v.data();
}

double* set_R_nan_as_C_nan(double *x, size_t n, Rcpp::NumericVector &v, int nthreads)
{
    v = Rcpp::NumericVector(x, x + n);
    for (size_t i = 0; i < n; i++)
        if (unlikely(std::isnan(v[i]))) v[i] = NAN;
    return REAL(v);
}

double* set_R_nan_as_C_nan(double *x, size_t n, int nthreads)
{
    for (size_t i = 0; i < n; i++)
        if (unlikely(std::isnan(x[i]))) x[i] = NAN;
    return x;
}

TreesIndexer* get_indexer_ptr_from_R_obj(SEXP indexer_R_ptr)
{
    if (Rf_isNull(indexer_R_ptr)) return nullptr;
    TreesIndexer *out = get_pointer_from_xptr<TreesIndexer>(indexer_R_ptr);
    if (out && out->indices.empty()) out = nullptr;
    return out;
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
                     int random_seed, bool use_long_double, int nthreads, bool lazy_serialization)
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
        Rcpp::_["depths"]    =  depths,
        Rcpp::_["tmat"]      =  tmat,
        Rcpp::_["dmat"]      =  dmat,
        Rcpp::_["model"]     =  R_NilValue,
        Rcpp::_["imputer"]   =  R_NilValue,
        Rcpp::_["indexer"]   =  R_NilValue,
        Rcpp::_["imputed_num"]      =  R_NilValue,
        Rcpp::_["imputed_cat"]      =  R_NilValue,
        Rcpp::_["err"]  =  Rcpp::LogicalVector::create(1)
    );

    Rcpp::List model_lst_nonlazy = Rcpp::List::create(
        Rcpp::_["ptr"]  =  R_NilValue,
        Rcpp::_["ser"]  =  R_NilValue
    );

    Rcpp::List imputer_lst_nonlazy = Rcpp::List::create(
        Rcpp::_["ptr"]  =  Rcpp::XPtr<void*>(nullptr, false),
        Rcpp::_["ser"]  =  R_NilValue
    );

    if (lazy_serialization) {
        outp["indexer"] = get_altrepped_null_pointer();
    }
    else {
        outp["indexer"] = Rcpp::List::create(
            Rcpp::_["ptr"]  =  Rcpp::XPtr<void*>(nullptr, false),
            Rcpp::_["ser"]  =  R_NilValue
        );
    }

    std::unique_ptr<IsoForest>     model_ptr(nullptr);
    std::unique_ptr<ExtIsoForest>  ext_model_ptr(nullptr);
    std::unique_ptr<Imputer>       imputer_ptr(nullptr);

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
                (uint64_t) random_seed, use_long_double, nthreads);

    Rcpp::checkUserInterrupt(); /* <- nothing is returned in this case */
    /* Note to self: the procedure has its own interrupt checker, so when an interrupt
       signal is triggered, first it will print a message about it, then re-issue the
       signal, then check for interrupt through Rcpp's, which will return nothing to
       the outside and will not raise any error. In this case, at least the user will
       see the error message. Note that Rcpp's interrupt non-return, unlike R's, triggers
       stack unwinding for C++ objects. */

    /* Note to self: since the function for fitting the model uses the C++ exception system,
       and the stop signals are translated into Rcpp stops, this section below should not
       be reachable anyhow. */
    if (ret_val == EXIT_FAILURE)
    {
        Rcpp::Rcerr << "Unexpected error" << std::endl;
        return Rcpp::unwindProtect(safe_errlist, nullptr);
    }

    if (calc_dist && sq_dist)
        tmat_to_dense(tmat_ptr, dmat_ptr, nrows, standardize_dist? 0. : std::numeric_limits<double>::infinity());

    bool serialization_failed = false;

    if (lazy_serialization)
    {
        if (ndim == 1) {
            outp["model"]    =  Rcpp::unwindProtect(get_altrepped_pointer<IsoForest>, (void*)&model_ptr);
        }
        else {
            outp["model"]    =  Rcpp::unwindProtect(get_altrepped_pointer<ExtIsoForest>, (void*)&ext_model_ptr);
        }

        if (build_imputer) {
            outp["imputer"]  =  Rcpp::unwindProtect(get_altrepped_pointer<Imputer>, (void*)&imputer_ptr);
        }
        else {
            outp["imputer"]  =  Rcpp::unwindProtect(safe_get_altrepped_null_pointer, nullptr);
        }
    }
    
    else
    {
        Rcpp::RawVector serialized_obj;
        /* Note to self: the serialization functions use unwind protection internally. */
        if (ndim == 1)
            serialized_obj  =  serialize_cpp_obj(model_ptr.get());
        else
            serialized_obj  =  serialize_cpp_obj(ext_model_ptr.get());

        if (unlikely(!serialized_obj.size())) serialization_failed = true;
        if (unlikely(serialization_failed)) {
            throw Rcpp::exception("Error: insufficient memory\n");
        }

        model_lst_nonlazy["ser"]       =  serialized_obj;
        if (ndim == 1) {
            model_lst_nonlazy["ptr"]   =  Rcpp::unwindProtect(safe_XPtr<IsoForest>, model_ptr.get());
            model_ptr.release();
        }
        else {
            model_lst_nonlazy["ptr"]   =  Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, ext_model_ptr.get());
            ext_model_ptr.release();
        }

        outp["model"] = model_lst_nonlazy;

        if (build_imputer)
        {
            imputer_lst_nonlazy["ser"] =  serialize_cpp_obj(imputer_ptr.get());
            if (!Rf_xlength(imputer_lst_nonlazy["ser"]))
            {
                throw Rcpp::exception("Error: insufficient memory\n");
            }

            imputer_lst_nonlazy["ptr"]  =  Rcpp::unwindProtect(safe_XPtr<Imputer>, imputer_ptr.get());
            imputer_ptr.release();
        }

        outp["imputer"] = imputer_lst_nonlazy;
    }

    if (output_imputations)
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
              uint64_t random_seed, bool use_long_double,
              Rcpp::List &model_cpp_obj_update, Rcpp::List &model_params_update,
              bool is_altrepped)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["model_ser"] = R_NilValue,
        Rcpp::_["imputer_ser"] = R_NilValue,
        Rcpp::_["indexer_ser"] = R_NilValue
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

    indexer_ptr = get_indexer_ptr_from_R_obj(indexer_R_ptr);

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
             (uint64_t)random_seed, use_long_double);
    
    Rcpp::RawVector new_serialized, new_imp_serialized, new_ind_serialized;
    size_t new_size;

    if (is_altrepped) goto dont_serialize;

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
                    out["model_ser"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_singlevar;
                }
            }

            else {
                serialize_anew_singlevar:
                out["model_ser"] = serialize_cpp_obj(model_ptr);
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
                    out["model_ser"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ext;
                }
            }

            else {
                serialize_anew_ext:
                out["model_ser"] = serialize_cpp_obj(ext_model_ptr);
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
                    out["imputer_ser"] = new_imp_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_imp;
                }
            }

            else {
                serialize_anew_imp:
                out["imputer_ser"] = serialize_cpp_obj(imputer_ptr);
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
                    out["indexer_ser"] = new_ind_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ind;
                }
            }

            else {
                serialize_anew_ind:
                out["indexer_ser"] = serialize_cpp_obj(indexer_ptr);
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

    {
        Rcpp::List model_lst = model_cpp_obj_update["model"];
        model_lst["ser"] = out["model_ser"];
        model_cpp_obj_update["model"] = model_lst;

        if (build_imputer)
        {
            Rcpp::List imputer_lst = model_cpp_obj_update["imputer"];
            imputer_lst["ser"] = out["imputer_ser"];
            model_cpp_obj_update["imputer"] = imputer_lst;
        }

        if (indexer_ptr)
        {
            Rcpp::List indexer_lst = model_cpp_obj_update["indexer"];
            indexer_lst["ser"] = out["indexer_ser"];
            model_cpp_obj_update["indexer"] = indexer_lst;
        }
    }
    
    dont_serialize:
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
        ext_model_ptr  =  get_pointer_from_xptr<ExtIsoForest>(model_R_ptr);
    else
        model_ptr      =  get_pointer_from_xptr<IsoForest>(model_R_ptr);
    TreesIndexer*  indexer = get_indexer_ptr_from_R_obj(indexer_R_ptr);

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
              size_t nrows, bool use_long_double, int nthreads, bool assume_full_distr,
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
    TreesIndexer*  indexer        =  get_indexer_ptr_from_R_obj(indexer_R_ptr);
    if (is_extended)
        ext_model_ptr  =  get_pointer_from_xptr<ExtIsoForest>(model_R_ptr);
    else
        model_ptr      =  get_pointer_from_xptr<IsoForest>(model_R_ptr);

    if (use_reference_points && indexer && !indexer->indices.front().reference_points.empty()) {
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
                    nrows, use_long_double, nthreads,
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
                      size_t nrows, bool use_long_double, int nthreads)
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
        ext_model_ptr  =  get_pointer_from_xptr<ExtIsoForest>(model_R_ptr);
    else
        model_ptr      =  get_pointer_from_xptr<IsoForest>(model_R_ptr);

    Imputer* imputer_ptr = get_pointer_from_xptr<Imputer>(imputer_R_ptr);

    if (!imputer_ptr) throw Rcpp::exception("Error: requested missing value imputation, but model was built without imputer.\n");


    impute_missing_values(numeric_data_ptr, categ_data_ptr, true,
                          Xr_ptr, Xr_ind_ptr, Xr_indptr_ptr,
                          nrows, use_long_double, nthreads,
                          model_ptr, ext_model_ptr,
                          *imputer_ptr);

    return Rcpp::List::create(
                Rcpp::_["X_num"] = (Xr.size())? (Xr) : (X_num),
                Rcpp::_["X_cat"] = X_cat
            );
}

// [[Rcpp::export(rng = false)]]
void drop_imputer(bool is_altrepped, bool free_cpp,
                  SEXP lst_imputer, Rcpp::List lst_cpp_objects, Rcpp::List lst_params)
{
    SEXP FalseObj = PROTECT(Rf_ScalarLogical(0));
    SEXP blank_ptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    SEXP altrepped_null = PROTECT(get_altrepped_null_pointer());
    
    if (is_altrepped) {

        if (free_cpp) {
            SEXP imp_R_ptr = R_altrep_data1(lst_imputer);
            Imputer* imputer_ptr = (Imputer*)R_ExternalPtrAddr(imp_R_ptr);
            delete imputer_ptr;
            R_SetExternalPtrAddr(imp_R_ptr, nullptr);
            R_ClearExternalPtr(imp_R_ptr);
        }
        
        lst_cpp_objects["imputer"] = altrepped_null;

    }

    else {

        if (free_cpp) {
            SEXP imp_R_ptr = VECTOR_ELT(lst_imputer, 0);
            Imputer* imputer_ptr = get_pointer_from_xptr<Imputer>(imp_R_ptr);
            delete imputer_ptr;
            R_SetExternalPtrAddr(imp_R_ptr, nullptr);
            R_ClearExternalPtr(imp_R_ptr);
            SET_VECTOR_ELT(lst_imputer, 0, imp_R_ptr);
        }

        SET_VECTOR_ELT(lst_imputer, 0, blank_ptr);
        SET_VECTOR_ELT(lst_imputer, 1, R_NilValue);
    }

    lst_params["build_imputer"] = FalseObj;
    UNPROTECT(3);
}

// [[Rcpp::export(rng = false)]]
void drop_indexer(bool is_altrepped, bool free_cpp,
                  SEXP lst_indexer, Rcpp::List lst_cpp_objects, Rcpp::List lst_metadata)
{
    SEXP empty_str = PROTECT(Rf_allocVector(STRSXP, 0));
    SEXP blank_ptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    SEXP altrepped_null = PROTECT(get_altrepped_null_pointer());
    
    if (is_altrepped) {

        if (free_cpp) {
            SEXP ind_R_ptr = R_altrep_data1(lst_indexer);
            TreesIndexer* indexer_ptr = (TreesIndexer*)R_ExternalPtrAddr(ind_R_ptr);
            delete indexer_ptr;
            R_SetExternalPtrAddr(ind_R_ptr, nullptr);
            R_ClearExternalPtr(ind_R_ptr);
        }
        
        lst_cpp_objects["indexer"] = altrepped_null;
    }

    else {

        if (free_cpp) {
            SEXP ind_R_ptr = VECTOR_ELT(lst_indexer, 0);
            TreesIndexer* indexer_ptr = get_pointer_from_xptr<TreesIndexer>(ind_R_ptr);
            delete indexer_ptr;
            R_SetExternalPtrAddr(ind_R_ptr, nullptr);
            R_ClearExternalPtr(ind_R_ptr);
            SET_VECTOR_ELT(lst_indexer, 0, ind_R_ptr);
        }

        SET_VECTOR_ELT(lst_indexer, 0, blank_ptr);
        SET_VECTOR_ELT(lst_indexer, 1, R_NilValue);
    }

    lst_metadata["reference_names"] = empty_str;
    UNPROTECT(3);
}

// [[Rcpp::export(rng = false)]]
void drop_reference_points(bool is_altrepped, SEXP lst_indexer, Rcpp::List lst_cpp_objects, Rcpp::List lst_metadata)
{
    SEXP empty_str = PROTECT(Rf_allocVector(STRSXP, 0));

    if (is_altrepped)
    {
        SEXP ind_R_ptr = R_altrep_data1(lst_indexer);
        TreesIndexer* indexer_ptr = (TreesIndexer*)R_ExternalPtrAddr(ind_R_ptr);
        if (!indexer_ptr) return;

        for (auto &tree : indexer_ptr->indices)
        {
            tree.reference_points.clear();
            tree.reference_indptr.clear();
            tree.reference_mapping.clear();
        }
    }

    else
    {
        SEXP ind_R_ptr = VECTOR_ELT(lst_indexer, 0);
        TreesIndexer* indexer_ptr = get_pointer_from_xptr<TreesIndexer>(ind_R_ptr);
        if (!indexer_ptr) return;
        
        std::unique_ptr<TreesIndexer> new_indexer(new TreesIndexer(*indexer_ptr));
        for (auto &tree : new_indexer->indices)
        {
            tree.reference_points.clear();
            tree.reference_indptr.clear();
            tree.reference_mapping.clear();
        }

        SET_VECTOR_ELT(lst_indexer, 1, serialize_cpp_obj(new_indexer.get()));
        *indexer_ptr = std::move(*new_indexer);
        new_indexer.release();
    }

    lst_metadata["reference_names"] = empty_str;
    UNPROTECT(1);
}

// [[Rcpp::export(rng = false)]]
Rcpp::List subset_trees
(
    SEXP model_R_ptr, SEXP imputer_R_ptr, SEXP indexer_R_ptr,
    bool is_extended, bool is_altrepped,
    Rcpp::IntegerVector trees_take
)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["model"] = R_NilValue,
        Rcpp::_["imputer"] = R_NilValue,
        Rcpp::_["indexer"] = R_NilValue
    );
    Rcpp::List lst_model = Rcpp::List::create(
        Rcpp::_["ptr"] = R_NilValue,
        Rcpp::_["ser"] = R_NilValue
    );
    Rcpp::List lst_imputer = Rcpp::List::create(
        Rcpp::_["ptr"] = Rcpp::XPtr<void*>(nullptr, false),
        Rcpp::_["ser"] = R_NilValue
    );
    Rcpp::List lst_indexer = Rcpp::List::create(
        Rcpp::_["ptr"] = Rcpp::XPtr<void*>(nullptr, false),
        Rcpp::_["ser"] = R_NilValue
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
    
    imputer_ptr          =  static_cast<Imputer*>(R_ExternalPtrAddr(imputer_R_ptr));
    if (imputer_ptr) {
        new_imputer_ptr  =  std::unique_ptr<Imputer>(new Imputer());
    }

    indexer_ptr          =  static_cast<TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (indexer_ptr) {
        new_indexer_ptr  =  std::unique_ptr<TreesIndexer>(new TreesIndexer());
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

    if (is_altrepped)
    {
        out["model"] = is_extended?
                        Rcpp::unwindProtect(get_altrepped_pointer<ExtIsoForest>, (void*)&new_ext_model_ptr)
                        :
                        Rcpp::unwindProtect(get_altrepped_pointer<IsoForest>, (void*)&new_model_ptr);
        out["imputer"] = imputer_ptr?
                          Rcpp::unwindProtect(get_altrepped_pointer<Imputer>, (void*)&new_imputer_ptr)
                          :
                          Rcpp::unwindProtect(safe_get_altrepped_null_pointer, nullptr);
        out["indexer"] = indexer_ptr?
                          Rcpp::unwindProtect(get_altrepped_pointer<TreesIndexer>, (void*)&new_indexer_ptr)
                          :
                          Rcpp::unwindProtect(safe_get_altrepped_null_pointer, nullptr);
    }

    else
    {
        lst_model["ser"] = is_extended? serialize_cpp_obj(new_ext_model_ptr.get()) : serialize_cpp_obj(new_model_ptr.get());
        if (imputer_ptr) lst_imputer["ser"] = serialize_cpp_obj(new_imputer_ptr.get());
        if (indexer_ptr) lst_indexer["ser"] = serialize_cpp_obj(new_indexer_ptr.get());

        lst_model["ptr"] = is_extended?
                            Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, new_ext_model_ptr.get())
                            :
                            Rcpp::unwindProtect(safe_XPtr<IsoForest>, new_model_ptr.get());
        new_model_ptr.release();

        if (imputer_ptr) {
            lst_imputer["ptr"] = Rcpp::unwindProtect(safe_XPtr<Imputer>, new_imputer_ptr.get());
            new_imputer_ptr.release();
        }

        if (indexer_ptr) {
            lst_indexer["ptr"] = Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, new_indexer_ptr.get());
            new_indexer_ptr.release();
        }

        out["model"] = lst_model;
        out["imputer"] = lst_imputer;
        out["indexer"] = lst_indexer;
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
        ext_model_ptr =  get_pointer_from_xptr<ExtIsoForest>(model_R_ptr);
        ntrees        =  ext_model_ptr->hplanes.size();
    }
    else
    {
        model_ptr     =  get_pointer_from_xptr<IsoForest>(model_R_ptr);
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
                             Rcpp::List &model_params_update,
                             bool is_altrepped)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["model_ser"] = R_NilValue,
        Rcpp::_["imputer_ser"] = R_NilValue,
        Rcpp::_["indexer_ser"] = R_NilValue
    );

    Rcpp::IntegerVector ntrees_new = Rcpp::IntegerVector::create(Rf_asInteger(model_params_update["ntrees"]));

    IsoForest* model_ptr = nullptr;
    IsoForest* other_ptr = nullptr;
    ExtIsoForest* ext_model_ptr = nullptr;
    ExtIsoForest* ext_other_ptr = nullptr;
    Imputer* imputer_ptr  = static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));
    Imputer* oimputer_ptr = static_cast<Imputer*>(R_ExternalPtrAddr(oimp_R_ptr));
    TreesIndexer* indexer_ptr  = get_indexer_ptr_from_R_obj(ind_R_ptr);
    TreesIndexer* oindexer_ptr = get_indexer_ptr_from_R_obj(oind_R_ptr);
    size_t old_ntrees;

    if (is_extended) {
        ext_model_ptr = static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        ext_other_ptr = static_cast<ExtIsoForest*>(R_ExternalPtrAddr(other_R_ptr));
        old_ntrees = ext_model_ptr->hplanes.size();
        if (ext_model_ptr == ext_other_ptr) {
            throw Rcpp::exception("Error: attempting to append trees from one model to itself.");
        }
    } else {
        model_ptr = static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        other_ptr = static_cast<IsoForest*>(R_ExternalPtrAddr(other_R_ptr));
        old_ntrees = model_ptr->trees.size();
        if (model_ptr == other_ptr) {
            throw Rcpp::exception("Error: attempting to append trees from one model to itself.");
        }
    }

    if (imputer_ptr && !oimputer_ptr) {
        throw Rcpp::exception("Model to append trees to has imputer, but model to append from doesn't. Try dropping the imputer.\n");
    }

    if (indexer_ptr && !oindexer_ptr) {
        throw Rcpp::exception("Model to append trees to has indexer, but model to append from doesn't. Try dropping the indexer.\n");
    }


    merge_models(model_ptr, other_ptr,
                 ext_model_ptr, ext_other_ptr,
                 imputer_ptr, oimputer_ptr,
                 indexer_ptr, oindexer_ptr);

    Rcpp::RawVector new_serialized, new_imp_serialized, new_ind_serialized;
    size_t new_size;

    if (is_altrepped) goto dont_serialize;

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
                    out["model_ser"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_singlevar;
                }
            }

            else {
                serialize_anew_singlevar:
                out["model_ser"] = serialize_cpp_obj(model_ptr);
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
                    out["model_ser"] = new_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ext;
                }
            }

            else {
                serialize_anew_ext:
                out["model_ser"] = serialize_cpp_obj(ext_model_ptr);
            }
        }

        if (imputer_ptr)
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
                    out["imputer_ser"] = new_imp_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_imp;
                }
            }

            else {
                serialize_anew_imp:
                out["imputer_ser"] = serialize_cpp_obj(imputer_ptr);
            }
        }

        if (indexer_ptr)
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
                    out["indexer_ser"] = new_ind_serialized;
                }

                catch (std::runtime_error &e) {
                    goto serialize_anew_ind;
                }
            }

            else {
                serialize_anew_ind:
                out["indexer_ser"] = serialize_cpp_obj(indexer_ptr);
            }
        }
    }

    catch (...)
    {
        if (!is_extended)
            model_ptr->trees.resize(old_ntrees);
        else
            ext_model_ptr->hplanes.resize(old_ntrees);

        if (imputer_ptr)
            imputer_ptr->imputer_tree.resize(old_ntrees);
        if (indexer_ptr)
            indexer_ptr->indices.resize(old_ntrees);
        throw;
    }

    {
        Rcpp::List model_lst = model_cpp_obj_update["model"];
        model_lst["ser"] = out["model_ser"];
        model_cpp_obj_update["model"] = model_lst;

        if (imputer_ptr)
        {
            Rcpp::List imputer_lst = model_cpp_obj_update["imputer"];
            imputer_lst["ser"] = out["imputer_ser"];
            model_cpp_obj_update["imputer"] = imputer_lst;
        }

        if (indexer_ptr)
        {
            Rcpp::List indexer_lst = model_cpp_obj_update["indexer"];
            indexer_lst["ser"] = out["indexer_ser"];
            model_cpp_obj_update["indexer"] = indexer_lst;
        }
    }

    dont_serialize:
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
    const IsoForest*     model_ptr      =  nullptr;
    const ExtIsoForest*  ext_model_ptr  =  nullptr;
    if (is_extended)
        ext_model_ptr  =  static_cast<const ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<const IsoForest*>(R_ExternalPtrAddr(model_R_ptr));

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
Rcpp::ListOf<Rcpp::CharacterVector> model_to_graphviz(SEXP model_R_ptr, bool is_extended,
                                                      SEXP indexer_R_ptr,
                                                      Rcpp::CharacterVector numeric_colanmes,
                                                      Rcpp::CharacterVector categ_colnames,
                                                      Rcpp::ListOf<Rcpp::CharacterVector> categ_levels,
                                                      bool output_tree_num, bool single_tree, size_t tree_num,
                                                      int nthreads)
{
    const IsoForest*     model_ptr      =  nullptr;
    const ExtIsoForest*  ext_model_ptr  =  nullptr;
    const TreesIndexer*  indexer        =  nullptr;
    if (is_extended)
        ext_model_ptr  =  static_cast<const ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<const IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    indexer = get_indexer_ptr_from_R_obj(indexer_R_ptr);

    std::vector<std::string> numeric_colanmes_cpp = Rcpp::as<std::vector<std::string>>(numeric_colanmes);
    std::vector<std::string> categ_colanmes_cpp = Rcpp::as<std::vector<std::string>>(categ_colnames);
    std::vector<std::vector<std::string>> categ_levels_cpp = Rcpp::as<std::vector<std::vector<std::string>>>(categ_levels);

    std::vector<std::string> res = generate_dot(model_ptr, ext_model_ptr, indexer,
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
Rcpp::ListOf<Rcpp::CharacterVector> model_to_json(SEXP model_R_ptr, bool is_extended,
                                                  SEXP indexer_R_ptr,
                                                  Rcpp::CharacterVector numeric_colanmes,
                                                  Rcpp::CharacterVector categ_colnames,
                                                  Rcpp::ListOf<Rcpp::CharacterVector> categ_levels,
                                                  bool output_tree_num, bool single_tree, size_t tree_num,
                                                  int nthreads)
{
    const IsoForest*     model_ptr      =  nullptr;
    const ExtIsoForest*  ext_model_ptr  =  nullptr;
    const TreesIndexer*  indexer        =  nullptr;
    if (is_extended)
        ext_model_ptr  =  static_cast<const ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<const IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    indexer = get_indexer_ptr_from_R_obj(indexer_R_ptr);

    std::vector<std::string> numeric_colanmes_cpp = Rcpp::as<std::vector<std::string>>(numeric_colanmes);
    std::vector<std::string> categ_colanmes_cpp = Rcpp::as<std::vector<std::string>>(categ_colnames);
    std::vector<std::vector<std::string>> categ_levels_cpp = Rcpp::as<std::vector<std::vector<std::string>>>(categ_levels);

    std::vector<std::string> res = generate_json(model_ptr, ext_model_ptr, indexer,
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
Rcpp::List copy_cpp_objects(SEXP model_R_ptr, bool is_extended, SEXP imp_R_ptr, SEXP ind_R_ptr, bool lazy_serialization)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["model"]    =  Rcpp::XPtr<void*>(nullptr, false),
        Rcpp::_["imputer"]  =  Rcpp::XPtr<void*>(nullptr, false),
        Rcpp::_["indexer"]  =  Rcpp::XPtr<void*>(nullptr, false)
    );

    IsoForest*     model_ptr      =  NULL;
    ExtIsoForest*  ext_model_ptr  =  NULL;
    Imputer*       imputer_ptr    =  NULL;
    TreesIndexer*  indexer_ptr    =  NULL;
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
    if (R_ExternalPtrAddr(imp_R_ptr))
        imputer_ptr    =  static_cast<Imputer*>(R_ExternalPtrAddr(imp_R_ptr));
    if (R_ExternalPtrAddr(ind_R_ptr))
        indexer_ptr    =  static_cast<TreesIndexer*>(R_ExternalPtrAddr(ind_R_ptr));

    std::unique_ptr<IsoForest> copy_model(new IsoForest());
    std::unique_ptr<ExtIsoForest> copy_ext_model(new ExtIsoForest());
    std::unique_ptr<Imputer> copy_imputer(new Imputer());
    std::unique_ptr<TreesIndexer> copy_indexer(new TreesIndexer());

    if (model_ptr) 
        *copy_model = *model_ptr;
    if (ext_model_ptr)
        *copy_ext_model = *ext_model_ptr;
    if (imputer_ptr)
        *copy_imputer = *imputer_ptr;
    if (indexer_ptr)
        *copy_indexer = *indexer_ptr;

    if (lazy_serialization)
    {
        if (is_extended) {
            out["model"]    =  Rcpp::unwindProtect(get_altrepped_pointer<ExtIsoForest>, (void*)&copy_ext_model);
        }
        else {
            out["model"]    =  Rcpp::unwindProtect(get_altrepped_pointer<IsoForest>, (void*)&copy_model);
        }

        if (imputer_ptr) {
            out["imputer"]  =  Rcpp::unwindProtect(get_altrepped_pointer<Imputer>, (void*)&copy_imputer);
        }
        else {
            out["imputer"]  =  Rcpp::unwindProtect(safe_get_altrepped_null_pointer, nullptr);
        }
        
        if (indexer_ptr) {
            out["indexer"]  =  Rcpp::unwindProtect(get_altrepped_pointer<TreesIndexer>, (void*)&copy_indexer);
        }
        else {
            out["indexer"]  =  Rcpp::unwindProtect(safe_get_altrepped_null_pointer, nullptr);
        }
    }

    else
    {
        if (is_extended) {
            out["model"]    =  Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, copy_ext_model.get());
            copy_ext_model.release();
        }
        else {
            out["model"]    =  Rcpp::unwindProtect(safe_XPtr<IsoForest>, copy_model.get());
            copy_model.release();
        }
        if (imputer_ptr) {
            out["imputer"]  =  Rcpp::unwindProtect(safe_XPtr<Imputer>, copy_imputer.get());
            copy_imputer.release();
        }
        if (indexer_ptr) {
            out["indexer"]  =  Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, copy_indexer.get());
            copy_indexer.release();
        }
    }
    
    return out;
}

// [[Rcpp::export(rng = false)]]
void build_tree_indices(Rcpp::List lst_cpp_objects, SEXP ptr_model, bool is_altrepped, bool is_extended, bool with_distances, int nthreads)
{
    Rcpp::List lst_out = Rcpp::List::create(
        Rcpp::_["ptr"] = R_NilValue,
        Rcpp::_["ser"] = R_NilValue
    );
    std::unique_ptr<TreesIndexer> indexer(new TreesIndexer());

    if (!is_extended) {
        build_tree_indices(*indexer,
                           *static_cast<IsoForest*>(R_ExternalPtrAddr(ptr_model)),
                           nthreads,
                           with_distances);
    }
    else {
        build_tree_indices(*indexer,
                           *static_cast<ExtIsoForest*>(R_ExternalPtrAddr(ptr_model)),
                           nthreads,
                           with_distances);
    }

    if (is_altrepped) {
        lst_cpp_objects["indexer"] = Rcpp::unwindProtect(get_altrepped_pointer<TreesIndexer>, (void*)&indexer);
    }

    else {
        lst_out["ser"] = serialize_cpp_obj(indexer.get());
        lst_out["ptr"] =  Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, indexer.get());
        indexer.release();
        lst_cpp_objects["indexer"] = lst_out;
    }
}

// [[Rcpp::export(rng = false)]]
bool check_node_indexer_has_distances(SEXP indexer_R_ptr)
{
    const TreesIndexer *indexer = (const TreesIndexer*)R_ExternalPtrAddr(indexer_R_ptr);
    if (!indexer) return false;
    return !indexer->indices.front().node_distances.empty();
}

// [[Rcpp::export(rng = false)]]
void set_reference_points(Rcpp::List lst_cpp_objects, SEXP ptr_model, SEXP ind_R_ptr, bool is_altrepped,
                          Rcpp::List lst_metadata, SEXP rnames, bool is_extended,
                          Rcpp::NumericVector X_num, Rcpp::IntegerVector X_cat,
                          Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind, Rcpp::IntegerVector Xc_indptr,
                          size_t nrows, int nthreads, bool with_distances)
{
    Rcpp::List lst_out = Rcpp::List::create(
        Rcpp::_["ptr"] = R_NilValue,
        Rcpp::_["ser"] = R_NilValue
    );

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

    IsoForest*     model_ptr      =  nullptr;
    ExtIsoForest*  ext_model_ptr  =  nullptr;
    TreesIndexer*  indexer        =  static_cast<TreesIndexer*>(R_ExternalPtrAddr(ind_R_ptr));
    if (is_extended)
        ext_model_ptr  =  static_cast<ExtIsoForest*>(R_ExternalPtrAddr(ptr_model));
    else
        model_ptr      =  static_cast<IsoForest*>(R_ExternalPtrAddr(ptr_model));

    MissingAction missing_action = is_extended?
                                   ext_model_ptr->missing_action
                                     :
                                   model_ptr->missing_action;
    if (missing_action != Fail)
    {
        if (X_num.size()) numeric_data_ptr = set_R_nan_as_C_nan(numeric_data_ptr, X_num.size(), Xcpp, nthreads);
        if (Xc.size())    Xc_ptr           = set_R_nan_as_C_nan(Xc_ptr, Xc.size(), Xcpp, nthreads);
    }

    std::unique_ptr<TreesIndexer> new_indexer(is_altrepped? nullptr : (new TreesIndexer(*indexer)));
    TreesIndexer *indexer_use = is_altrepped? indexer : new_indexer.get();

    /* Note: if using an altrepped pointer, the indexer is modified in-place. If that fails,
    it will end up overwitten, with the previous references taken away. OTOH, if using
    a pointer + serialized, and it fails, it should not overwrite anything, and thus
    should not re-assign here immediately. */
    if (is_altrepped) {
        lst_metadata["reference_names"] = rnames;
    }

    set_reference_points(model_ptr, ext_model_ptr, indexer_use,
                         with_distances,
                         numeric_data_ptr, categ_data_ptr,
                         true, (size_t)0, (size_t)0,
                         Xc_ptr, Xc_ind_ptr, Xc_indptr_ptr,
                         (double*)NULL, (int*)NULL, (int*)NULL,
                         nrows, nthreads);

    if (!is_altrepped) {
        lst_out["ser"] = serialize_cpp_obj(new_indexer.get());
        *indexer = std::move(*new_indexer);
        lst_metadata["reference_names"] = rnames;
    }
}

// [[Rcpp::export(rng = false)]]
bool check_node_indexer_has_references(SEXP indexer_R_ptr)
{
    const TreesIndexer *indexer = (const TreesIndexer*)R_ExternalPtrAddr(indexer_R_ptr);
    if (!indexer) return false;
    return !(indexer->indices.front().reference_points.empty());
}

// [[Rcpp::export(rng = false)]]
int get_num_references(SEXP indexer_R_ptr)
{
    const TreesIndexer *indexer = static_cast<const TreesIndexer*>(R_ExternalPtrAddr(indexer_R_ptr));
    if (!indexer || indexer->indices.empty()) return 0;
    return indexer->indices.front().reference_points.size();
}

// [[Rcpp::export(rng = false)]]
SEXP get_null_R_pointer_internal(bool altrepped)
{
    if (!altrepped) {
        return R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue);
    }
    else {
        SEXP R_ptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
        SEXP out = PROTECT(R_new_altrep(altrepped_pointer_NullPointer, R_ptr, R_NilValue));
        UNPROTECT(2);
        return out;
    }
}

// [[Rcpp::export(rng = false)]]
bool compare_pointers(SEXP obj1, SEXP obj2)
{
    return R_ExternalPtrAddr(obj1) == R_ExternalPtrAddr(obj2);
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
Rcpp::List deserialize_from_file(Rcpp::CharacterVector fname, bool lazy_serialization)
{
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["model"] = R_NilValue,
        Rcpp::_["imputer"] = R_NilValue,
        Rcpp::_["indexer"] = R_NilValue,
        Rcpp::_["metadata"] = R_NilValue
    );

    if (!lazy_serialization) {
        out["model"] = Rcpp::List::create(
            Rcpp::_["ptr"] = Rcpp::XPtr<void*>(nullptr, false),
            Rcpp::_["ser"] = R_NilValue
        );
        out["imputer"] = Rcpp::List::create(
            Rcpp::_["ptr"] = Rcpp::XPtr<void*>(nullptr, false),
            Rcpp::_["ser"] = R_NilValue
        );
        out["indexer"] = Rcpp::List::create(
            Rcpp::_["ptr"] = Rcpp::XPtr<void*>(nullptr, false),
            Rcpp::_["ser"] = R_NilValue
        );
    }

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

    if (lazy_serialization)
    {
        if (has_IsoForest)
            out["model"] = Rcpp::unwindProtect(get_altrepped_pointer<IsoForest>, &model);
        else
            out["model"] = Rcpp::unwindProtect(get_altrepped_pointer<ExtIsoForest>, &model_ext);
        
        if (has_Imputer)
            out["imputer"] = Rcpp::unwindProtect(get_altrepped_pointer<Imputer>, &imputer);
        else
            out["imputer"] = Rcpp::unwindProtect(safe_get_altrepped_null_pointer, nullptr);

        if (has_Imputer)
            out["indexer"] = Rcpp::unwindProtect(get_altrepped_pointer<TreesIndexer>, &indexer);
        else
            out["indexer"] = Rcpp::unwindProtect(safe_get_altrepped_null_pointer, nullptr);
    }

    else
    {
        Rcpp::List tmp_model = out["model"];
        Rcpp::List tmp_imputer = out["imputer"];
        Rcpp::List tmp_indexer = out["indexer"];
        
        if (has_IsoForest)
            tmp_model["ser"] = serialize_cpp_obj(model.get());
        else
            tmp_model["ser"] = serialize_cpp_obj(model_ext.get());
        
        if (has_Imputer)
            tmp_imputer["ser"] = serialize_cpp_obj(imputer.get());
        
        if (has_Indexer)
            tmp_indexer["ser"] = serialize_cpp_obj(indexer.get());

        if (has_IsoForest) {
            tmp_model["ptr"] = Rcpp::unwindProtect(safe_XPtr<IsoForest>, model.get());
            model.release();
        }
        else {
            tmp_model["ptr"] = Rcpp::unwindProtect(safe_XPtr<ExtIsoForest>, model_ext.get());
            model_ext.release();
        }
        if (has_Imputer) {
            tmp_imputer["ptr"] = Rcpp::unwindProtect(safe_XPtr<Imputer>, imputer.get());
            imputer.release();
        }
        if (has_Indexer) {
            tmp_indexer["ptr"] = Rcpp::unwindProtect(safe_XPtr<TreesIndexer>, indexer.get());
            indexer.release();
        }

        out["model"] = tmp_model;
        out["imputer"] = tmp_imputer;
        out["indexer"] = tmp_indexer;
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
        const ExtIsoForest* ext_model_ptr = static_cast<const ExtIsoForest*>(R_ExternalPtrAddr(model_R_ptr));
        return ext_model_ptr->hplanes.size();
    }
    
    else {
        const IsoForest* model_ptr = static_cast<const IsoForest*>(R_ExternalPtrAddr(model_R_ptr));
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
