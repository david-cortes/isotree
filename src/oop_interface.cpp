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
#if !defined(_FOR_R) && !defined(_FOR_PYTHON)
#include "isotree.hpp"
#include "isotree_exportable.hpp"
#include "oop_interface.hpp"
using namespace isotree;

IsolationForest::IsolationForest
(
    size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
    bool with_replacement, bool weight_as_sample,
    size_t sample_size, size_t ntrees,
    size_t max_depth, size_t ncols_per_tree, bool   limit_depth,
    bool penalize_range, bool standardize_data,
    ScoringMetric scoring_metric, bool fast_bratio, bool weigh_by_kurt,
    double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
    double prob_pick_col_by_range, double prob_pick_col_by_var,
    double prob_pick_col_by_kurt,
    double min_gain, MissingAction missing_action,
    CategSplit cat_split_type, NewCategAction new_cat_action,
    bool   all_perm, bool build_imputer, size_t min_imp_obs,
    UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
    uint64_t random_seed, int nthreads
)
        :
        ndim(ndim),
        ntry(ntry),
        coef_type(coef_type),
        coef_by_prop(coef_by_prop),
        with_replacement(with_replacement),
        weight_as_sample(weight_as_sample),
        sample_size(sample_size),
        ntrees(ntrees),
        max_depth(max_depth),
        ncols_per_tree(ncols_per_tree),
        limit_depth(limit_depth),
        penalize_range(penalize_range),
        standardize_data(standardize_data),
        scoring_metric(scoring_metric),
        fast_bratio(fast_bratio),
        weigh_by_kurt(weigh_by_kurt),
        prob_pick_by_gain_pl(prob_pick_by_gain_pl),
        prob_pick_by_gain_avg(prob_pick_by_gain_avg),
        prob_pick_col_by_range(prob_pick_col_by_range),
        prob_pick_col_by_var(prob_pick_col_by_var),
        prob_pick_col_by_kurt(prob_pick_col_by_kurt),
        min_gain(min_gain),
        missing_action(missing_action),
        cat_split_type(cat_split_type),
        new_cat_action(new_cat_action),
        all_perm(all_perm),
        build_imputer(build_imputer),
        min_imp_obs(min_imp_obs),
        depth_imp(depth_imp),
        weigh_imp_rows(weigh_imp_rows),
        random_seed(random_seed)
    {}


void IsolationForest::fit(double X[], size_t nrows, size_t ncols)
{
    this->check_params();
    this->override_previous_fit();

    auto retcode = fit_iforest(
        (this->ndim == 1)? &this->model : nullptr,
        (this->ndim != 1)? &this->model_ext : nullptr,
        X,  ncols,
        (int*)nullptr, (size_t)0, (int*)nullptr,
        (double*)nullptr, (int*)nullptr, (int*)nullptr,
        this->ndim, this->ntry, this->coef_type, this->coef_by_prop,
        (double*)nullptr, this->with_replacement, this->weight_as_sample,
        nrows, this->sample_size, this->ntrees,
        this->max_depth, this->ncols_per_tree,
        this->limit_depth, this->penalize_range, this->standardize_data,
        this->scoring_metric, this->fast_bratio,
        false, (double*)nullptr,
        (double*)nullptr, true,
        (double*)nullptr, this->weigh_by_kurt,
        this->prob_pick_by_gain_pl,
        this->prob_pick_by_gain_avg,
        this->prob_pick_col_by_range,
        this->prob_pick_col_by_var,
        this->prob_pick_col_by_kurt,
        this->min_gain, this->missing_action,
        this->cat_split_type, this->new_cat_action,
        this->all_perm, &this->imputer, this->min_imp_obs,
        this->depth_imp, this->weigh_imp_rows, false,
        this->random_seed, this->nthreads
    );
    if (retcode != EXIT_SUCCESS) unexpected_error();
    this->is_fitted = true;
}

void IsolationForest::fit(double numeric_data[],   size_t ncols_numeric,  size_t nrows,
                          int    categ_data[],     size_t ncols_categ,    int ncat[],
                          double sample_weights[], double col_weights[])
{
    this->check_params();
    this->override_previous_fit();

    auto retcode = fit_iforest(
        (this->ndim == 1)? &this->model : nullptr,
        (this->ndim != 1)? &this->model_ext : nullptr,
        numeric_data,  ncols_numeric,
        categ_data, ncols_categ, ncat,
        (double*)nullptr, (int*)nullptr, (int*)nullptr,
        this->ndim, this->ntry, this->coef_type, this->coef_by_prop,
        sample_weights, this->with_replacement, this->weight_as_sample,
        nrows, this->sample_size, this->ntrees,
        this->max_depth, this->ncols_per_tree,
        this->limit_depth, this->penalize_range, this->standardize_data,
        this->scoring_metric, this->fast_bratio,
        false, (double*)nullptr,
        (double*)nullptr, true,
        col_weights, this->weigh_by_kurt,
        this->prob_pick_by_gain_pl,
        this->prob_pick_by_gain_avg,
        this->prob_pick_col_by_range,
        this->prob_pick_col_by_var,
        this->prob_pick_col_by_kurt,
        this->min_gain, this->missing_action,
        this->cat_split_type, this->new_cat_action,
        this->all_perm, &this->imputer, this->min_imp_obs,
        this->depth_imp, this->weigh_imp_rows, false,
        this->random_seed, this->nthreads
    );
    if (retcode != EXIT_SUCCESS) unexpected_error();
    this->is_fitted = true;
}

void IsolationForest::fit(double Xc[], int Xc_ind[], int Xc_indptr[],
                          size_t ncols_numeric,      size_t nrows,
                          int    categ_data[],       size_t ncols_categ,   int ncat[],
                          double sample_weights[],   double col_weights[])
{
    this->check_params();
    this->override_previous_fit();

    auto retcode = fit_iforest(
        (this->ndim == 1)? &this->model : nullptr,
        (this->ndim != 1)? &this->model_ext : nullptr,
        (double*)nullptr,  ncols_numeric,
        categ_data, ncols_categ, ncat,
        Xc, Xc_ind, Xc_indptr,
        this->ndim, this->ntry, this->coef_type, this->coef_by_prop,
        sample_weights, this->with_replacement, this->weight_as_sample,
        nrows, this->sample_size, this->ntrees,
        this->max_depth, this->ncols_per_tree,
        this->limit_depth, this->penalize_range, this->standardize_data,
        this->scoring_metric, this->fast_bratio,
        false, (double*)nullptr,
        (double*)nullptr, true,
        col_weights, this->weigh_by_kurt,
        this->prob_pick_by_gain_pl,
        this->prob_pick_by_gain_avg,
        this->prob_pick_col_by_range,
        this->prob_pick_col_by_var,
        this->prob_pick_col_by_kurt,
        this->min_gain, this->missing_action,
        this->cat_split_type, this->new_cat_action,
        this->all_perm, &this->imputer, this->min_imp_obs,
        this->depth_imp, this->weigh_imp_rows, false,
        this->random_seed, this->nthreads
    );
    if (retcode != EXIT_SUCCESS) unexpected_error();
    this->is_fitted = true;
}

std::vector<double> IsolationForest::predict(double X[], size_t nrows, bool standardize)
{
    this->check_is_fitted();
    this->check_nthreads();
    std::vector<double> out(nrows);
    predict_iforest(
        X, (int*)nullptr,
        true, (size_t)0, (size_t)0,
        (double*)nullptr, (int*)nullptr, (int*)nullptr,
        (double*)nullptr, (int*)nullptr, (int*)nullptr,
        nrows, this->nthreads, standardize,
        (!this->model.trees.empty())? &this->model : nullptr,
        (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
        out.data(), (int*)nullptr, (double*)nullptr,
        (TreesIndexer*)nullptr);
    return out;
}

void IsolationForest::predict(double numeric_data[], int categ_data[], bool is_col_major,
                              size_t nrows, size_t ld_numeric, size_t ld_categ, bool standardize,
                              double output_depths[], int tree_num[], double per_tree_depths[])
{
    this->check_is_fitted();
    this->check_nthreads();
    if ((tree_num || per_tree_depths) && !this->check_can_predict_per_tree())
        throw std::runtime_error("Cannot predict tree numbers/depths with this model.\n");
    predict_iforest(
        numeric_data, categ_data,
        is_col_major, ld_numeric, ld_categ,
        (double*)nullptr, (int*)nullptr, (int*)nullptr,
        (double*)nullptr, (int*)nullptr, (int*)nullptr,
        nrows, this->nthreads, standardize,
        (!this->model.trees.empty())? &this->model : nullptr,
        (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
        output_depths, tree_num, per_tree_depths,
        (!this>indexer.indices.empty())? &this->indexer : nullptr);
}

void IsolationForest::predict(double X_sparse[], int X_ind[], int X_indptr[], bool is_csc,
                              int categ_data[], bool is_col_major, size_t ld_categ, size_t nrows, bool standardize,
                              double output_depths[], int tree_num[], double per_tree_depths[])
{
    this->check_is_fitted();
    this->check_nthreads();
    if ((tree_num || per_tree_depths) && !this->check_can_predict_per_tree())
        throw std::runtime_error("Cannot predict tree numbers/depths with this model.\n");
    std::vector<double> out(nrows);
    predict_iforest(
        (double*)nullptr, categ_data,
        is_col_major, (size_t)0, ld_categ,
        is_csc? X_sparse : (double*)nullptr, is_csc? X_ind : (int*)nullptr, is_csc? X_indptr : (int*)nullptr,
        is_csc? (double*)nullptr : X_sparse, is_csc? (int*)nullptr : X_ind, is_csc? (int*)nullptr : X_indptr,
        nrows, this->nthreads, standardize,
        (!this->model.trees.empty())? &this->model : nullptr,
        (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
        output_depths, tree_num, per_tree_depths,
        (!this>indexer.indices.empty())? &this->indexer : nullptr);
}

std::vector<double> IsolationForest::predict_distance(double X[], size_t nrows,
                                                      bool assume_full_distr, bool standardize_dist,
                                                      bool triangular)
{
    this->check_is_fitted();
    this->check_nthreads();
    std::vector<double> tmat(calc_ncomb(nrows));
    std::vector<double> dmat(triangular? square(nrows) : 0);

    calc_similarity(X, (int*)nullptr,
                    (double*)nullptr, (int*)nullptr, (int*)nullptr,
                    nrows, this->nthreads, assume_full_distr, standardize_dist,
                    (!this->model.trees.empty())? &this->model : nullptr,
                    (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
                    tmat.data(), (double*)nullptr, (size_t)0,
                    (!this->indexer.indices.empty())? &this->indexer : nullptr,
                    true, (size_t)0, (size_t)0);
    if (!triangular)
        tmat_to_dense(tmat.data(), dmat.data(), nrows, false, !standardize_dist);
    return (triangular? tmat : dmat);
}

void IsolationForest::predict_distance(double numeric_data[], int categ_data[],
                                       size_t nrows,
                                       bool assume_full_distr, bool standardize_dist,
                                       bool triangular,
                                       double dist_matrix[])
{
    this->check_is_fitted();
    this->check_nthreads();
    std::vector<double> tmat(triangular? 0 : calc_ncomb(nrows));

    calc_similarity(numeric_data, categ_data,
                    (double*)nullptr, (int*)nullptr, (int*)nullptr,
                    nrows, this->nthreads, assume_full_distr, standardize_dist,
                    (!this->model.trees.empty())? &this->model : nullptr,
                    (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
                    triangular? dist_matrix : tmat.data(),
                    (double*)nullptr,
                    (size_t)0,
                    (!this->indexer.indices.empty())? &this->indexer : nullptr,
                    true, (size_t)0, (size_t)0);
    if (!triangular)
        tmat_to_dense(tmat.data(), dist_matrix, nrows, false, !standardize_dist);
}

void IsolationForest::predict_distance(double Xc[], int Xc_ind[], int Xc_indptr[], int categ_data[],
                                       size_t nrows, bool assume_full_distr, bool standardize_dist,
                                       bool triangular,
                                       double dist_matrix[])
{
    this->check_is_fitted();
    this->check_nthreads();
    std::vector<double> tmat(triangular? 0 : calc_ncomb(nrows));

    calc_similarity((double*)nullptr, (int*)nullptr,
                    Xc, Xc_ind, Xc_indptr,
                    nrows, this->nthreads, assume_full_distr, standardize_dist,
                    (!this->model.trees.empty())? &this->model : nullptr,
                    (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
                    triangular? dist_matrix : tmat.data(),
                    (double*)nullptr,
                    (size_t)0,
                    (!this->indexer.indices.empty())? &this->indexer : nullptr,
                    true, (size_t)0, (size_t)0);
    if (!triangular)
        tmat_to_dense(tmat.data(), dist_matrix, nrows, false, !standardize_dist);
}

void IsolationForest::impute(double X[], size_t nrows)
{
    this->check_is_fitted();
    this->check_nthreads();
    if (this->imputer.imputer_tree.empty())
        throw std::runtime_error("Model was built without imputation capabilities.\n");
    impute_missing_values(X, (int*)nullptr, true,
                          (double*)nullptr, (int*)nullptr, (int*)nullptr,
                          nrows, this->nthreads,
                          (!this->model.trees.empty())? &this->model : nullptr,
                          (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
                          this->imputer);
}

void IsolationForest::impute(double numeric_data[], int categ_data[], bool is_col_major, size_t nrows)
{
    this->check_is_fitted();
    if (this->imputer.imputer_tree.empty())
        throw std::runtime_error("Model was built without imputation capabilities.\n");
    this->check_nthreads();
    impute_missing_values(numeric_data, categ_data, is_col_major,
                          (double*)nullptr, (int*)nullptr, (int*)nullptr,
                          nrows, this->nthreads,
                          (!this->model.trees.empty())? &this->model : nullptr,
                          (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
                          this->imputer);
}

void IsolationForest::impute(double Xr[], int Xr_ind[], int Xr_indptr[],
                             int categ_data[], bool is_col_major, size_t nrows)
{
    this->check_is_fitted();
    if (this->imputer.imputer_tree.empty())
        throw std::runtime_error("Model was built without imputation capabilities.\n");
    this->check_nthreads();
    impute_missing_values((double*)nullptr, categ_data, is_col_major,
                          Xr, Xr_ind, Xr_indptr,
                          nrows, this->nthreads,
                          (!this->model.trees.empty())? &this->model : nullptr,
                          (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
                          this->imputer);
}

void IsolationForest::build_indexer(const bool with_distances)
{
    this->check_is_fitted();
    if (!this->indexer.indices.empty())
        return;
    if (this->missing_action == Divide)
        throw std::runtime_error("Cannot build tree indexer when using 'missing_action=Divide'.\n");
    if (!this->model.trees.empty() && this->new_cat_action == Weighted)
        throw std::runtime_error("Cannot build tree indexer when using 'new_cat_action=Weighted' with single-variable model.\n");
    
    if (!this->model.trees.empty())
        build_tree_indices(this->indexer, this->model, this->nthreads, with_distances);
    else if (!this->model_ext.hplanes.empty())
        build_tree_indices(this->indexer, this->model_ext, this->nthreads, with_distances);
    else
        unexpected_error();
}

void IsolationForest::serialize(FILE *out) const
{
    this->serialize_template(out);
}

void IsolationForest::serialize(std::ostream &out) const
{
    this->serialize_template(out);
}

IsolationForest IsolationForest::deserialize(FILE *inp, int nthreads)
{
    return deserialize_template(inp, nthreads);
}

IsolationForest IsolationForest::deserialize(std::istream &inp, int nthreads)
{
    return deserialize_template(inp, nthreads);
}

std::ostream& operator<<(std::ostream &ost, const IsolationForest &model)
{
    model.serialize(ost);
    return ost;
}


std::ostream& isotree::operator<<(std::ostream &ost, const IsolationForest &model)
{
    model.serialize(ost);
    return ost;
}

std::istream& operator>>(std::istream &ist, IsolationForest &model)
{
    model = IsolationForest::deserialize(ist, -1);
    return ist;
}

std::istream& isotree::operator>>(std::istream &ist, IsolationForest &model)
{
    model = IsolationForest::deserialize(ist, -1);
    return ist;
}

IsoForest& IsolationForest::get_model()
{
    if (this->ndim != 1)
        throw std::runtime_error("Error: class contains an 'ExtIsoForest' model only.\n");
    return this->model;
}

ExtIsoForest& IsolationForest::get_model_ext()
{
    if (this->ndim == 1)
        throw std::runtime_error("Error: class contains an 'IsoForest' model only.\n");
    return this->model_ext;
}

Imputer& IsolationForest::get_imputer()
{
    if (!this->build_imputer)
        throw std::runtime_error("Error: model does not contain imputer.\n");
    return this->imputer;
}

TreesIndexer& IsolationForest::get_indexer()
{
    if (this->indexer.indices.empty() && (!this->model.trees.empty() || !this->model_ext.hplanes.empty()))
        throw std::runtime_error("Error: model does not contain indexer.\n");
    return this->indexer;
}

void IsolationForest::check_nthreads()
{
    if (this->nthreads < 0) {
        #ifdef _OPENMP
        this->nthreads = omp_get_max_threads() + this->nthreads + 1;
        #else
        this->nthreads = 1;
        #endif
    }
    if (nthreads <= 0) {
        fprintf(stderr, "'isotree' got invalid 'nthreads', will set to 1.\n");
        this->nthreads = 1;
    }
    #ifndef _OPENMP
    else if (nthreads > 1) {
        fprintf(stderr,
                "Passed nthreads:%d to 'isotree', but library was compiled without multithreading.\n",
                this->nthreads);
        this->nthreads = 1;
    }
    #endif
}

size_t IsolationForest::get_ntrees() const
{
    if (!this->model.trees.empty())
        return this->model.trees.size();
    else if (!this->model_ext.hplanes.empty())
        return this->model_ext.hplanes.size();
    else
        throw std::runtime_error("Model is not fitted or is corrupted.\n");
}

bool IsolationForest::check_can_predict_per_tree() const
{
    if (!this->model.trees.empty())
    {
        if (this->model.missing_action == Divide)
            return false;
        if (this->model.new_cat_action == Weighted && this->cat_split_type != SingleCateg)
        {
            for (const std::vector<IsoTree> &tree : this->model.trees)
                for (const IsoTree &node : tree)
                    if (node.col_type == Categorical)
                        return false;
        }
    }

    return true;
}

void IsolationForest::override_previous_fit()
{
    if (this->is_fitted) {
        this->model = IsoForest();
        this->model_ext = ExtIsoForest();
        this->imputer = Imputer();
        this->indexer = TreesIndexer();
    }
}

void IsolationForest::check_params()
{
    this->check_nthreads();

    if (this->prob_pick_by_gain_avg < 0) throw std::runtime_error("'prob_pick_by_gain_avg' must be >= 0.\n");
    if (this->prob_pick_by_gain_pl < 0) throw std::runtime_error("'prob_pick_by_gain_pl' must be >= 0.\n");
    if (this->prob_pick_col_by_range < 0) throw std::runtime_error("'prob_pick_col_by_range' must be >= 0.\n");
    if (this->prob_pick_col_by_var < 0) throw std::runtime_error("'prob_pick_col_by_var' must be >= 0.\n");
    if (this->prob_pick_col_by_kurt < 0) throw std::runtime_error("'prob_pick_col_by_kurt' must be >= 0.\n");

    if (prob_pick_by_gain_avg + prob_pick_by_gain_pl
        > 1. + 2. * std::numeric_limits<double>::epsilon())
        throw std::runtime_error("Probabilities for gain-based splits sum to more than 1.\n");

    if (prob_pick_col_by_var + prob_pick_col_by_var + prob_pick_col_by_kurt
        > 1. + 2. * std::numeric_limits<double>::epsilon())
        throw std::runtime_error("Probabilities for column choices sum to more than 1.\n");

    if (min_gain < 0)
        throw std::runtime_error("'min_gain' cannot be negative.\n");

    if (this->ndim != 1) {
        if (this->missing_action == Divide)
            throw std::runtime_error("'missing_action' = 'Divide' not supported in extended model.\n");
    }

    if (this->coef_type != Uniform && this->coef_type != Normal)
        throw std::runtime_error("Invalid 'coef_type'.\n");
    if (this->missing_action != Divide && this->missing_action != Impute && this->missing_action != Fail)
        throw std::runtime_error("Invalid 'missing_action'.\n");
    if (this->cat_split_type != SubSet && this->cat_split_type != SingleCateg)
        throw std::runtime_error("Invalid 'cat_split_type'.\n");
    if (this->new_cat_action != Weighted && this->new_cat_action != Smallest && this->new_cat_action != Random)
        throw std::runtime_error("Invalid 'new_cat_action'.\n");
    if (this->depth_imp != Lower && this->depth_imp != Higher && this->depth_imp != Same)
        throw std::runtime_error("Invalid 'depth_imp'.\n");
    if (this->weigh_imp_rows != Inverse && this->weigh_imp_rows != Prop && this->weigh_imp_rows != Flat)
        throw std::runtime_error("Invalid 'weigh_imp_rows'.\n");

    if (this->sample_size > 0 && this->sample_size <= 2)
        throw std::runtime_error("'sample_size' must be greater than 2.\n");

    if (this->penalize_range && (this->scoring_metric == Density || this->scoring_metric == AdjDensity))
        throw std::runtime_error("'penalize_range' is incompatible with density scoring.\n");
}

void IsolationForest::check_is_fitted() const
{
    if (!this->is_fitted)
        throw std::runtime_error("Model has not been fitted.\n");
}

template <class otype>
void IsolationForest::serialize_template(otype &out) const
{
    this->check_is_fitted();

    serialize_combined(
        (!this->model.trees.empty())? &this->model : nullptr,
        (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
        (!this->imputer.imputer_tree.empty())? &this->imputer : nullptr,
        (!this->indexer.indices.empty())? &this->indexer : nullptr,
        (char*)nullptr,
        (size_t)0,
        out
    );
}

IsolationForest::IsolationForest(int nthreads, size_t ndim, size_t ntrees, bool build_imputer)
    :
    nthreads(nthreads),
    ndim(ndim),
    ntrees(ntrees),
    build_imputer(build_imputer) {this->is_fitted = true;};

template <class itype>
IsolationForest IsolationForest::deserialize_template(itype &inp, int nthreads)
{
    bool is_isotree_model = false;
    bool is_compatible = false;
    bool has_combined_objects = false;
    bool has_IsoForest = false;
    bool has_ExtIsoForest = false;
    bool has_Imputer = false;
    bool has_Indexer = false;
    bool has_metadata = false;
    size_t size_metadata = 0;
    inspect_serialized_object(
        inp,
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
    if (is_isotree_model && is_compatible && !has_combined_objects)
        throw std::runtime_error("Serialized model is not compatible.\n");

    IsoForest model = IsoForest();
    ExtIsoForest model_ext = ExtIsoForest();
    Imputer imputer = Imputer();
    TreesIndexer indexer = TreesIndexer();

    deserialize_combined(
        inp,
        &model,
        &model_ext,
        &imputer,
        &indexer,
        (char*)nullptr
    );

    if (model.trees.empty() && model_ext.hplanes.empty())
        throw std::runtime_error("Error: model contains no trees.\n");

    size_t ntrees;
    size_t ndim = 3;
    bool build_imputer = false;

    if (!model.trees.empty()) {
        ntrees = model.trees.size();
        ndim = 1;
    }
    else {
        ntrees = model_ext.hplanes.size();
    }
    if (!imputer.imputer_tree.empty()) {
        if (imputer.imputer_tree.size() != ntrees)
            throw std::runtime_error("Error: imputer has incorrect number of trees.\n");
        build_imputer = true;
    }
    if (!indexer.indices.empty()) {
        if (indexer.indices.size() != ntrees)
            throw std::runtime_error("Error: indexer has incorrect number of trees.\n");
    }

    IsolationForest out = IsolationForest(nthreads, ndim, ntrees, build_imputer);

    if (!model.trees.empty()) {
        out.get_model() = std::move(model);
        out.penalize_range = out.get_model().has_range_penalty;
    }
    else {
        out.get_model_ext() = std::move(model_ext);
        out.penalize_range = out.get_model_ext().has_range_penalty;
    }
    if (!imputer.imputer_tree.empty())
        out.get_imputer() = std::move(imputer);
    if (!indexer.indices.empty())
        out.indexer = std::move(indexer);

    return out;
}

#endif
