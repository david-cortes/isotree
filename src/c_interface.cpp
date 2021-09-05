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
#if !defined(_FOR_R) && !defined(_FOR_PYTHON)

#include "oop_interface.hpp"

using std::cerr;
using isotree::IsolationForest;

struct IsoTree_Params {
    int nthreads = -1; /* <- May be manually changed at any time */

    uint64_t random_seed = 1;

    /*  General tree construction parameters  */
    size_t ndim = 3;
    size_t ntry = 3;
    CoefType coef_type = Normal; /* only for ndim>1 */
    bool   with_replacement = false;
    bool   weight_as_sample = true;
    size_t sample_size = 0;
    size_t ntrees = 500;
    size_t max_depth = 0;
    size_t ncols_per_tree = 0;
    bool   limit_depth = true;
    bool   penalize_range = false;
    bool   weigh_by_kurt = false;
    double prob_pick_by_gain_avg = 0.;
    double prob_split_by_gain_avg = 0.; /* only for ndim==1 */
    double prob_pick_by_gain_pl = 0.;
    double prob_split_by_gain_pl = 0.;  /* only for ndim==1 */
    double min_gain = 0.;
    MissingAction missing_action = Impute;

    /*  For categorical variables  */
    CategSplit cat_split_type = SubSet;
    NewCategAction new_cat_action = Weighted;
    bool   coef_by_prop = false;
    bool   all_perm = false;

    /*  For imputation methods (when using 'build_imputer=true' and calling 'impute')  */
    bool   build_imputer = false;
    size_t min_imp_obs = 3;
    UseDepthImp depth_imp = Higher;
    WeighImpRows weigh_imp_rows = Inverse;
};

extern "C" {

ISOTREE_EXPORTED
void* allocate_default_isotree_parameters()
{
    return new IsoTree_Params();
}

ISOTREE_EXPORTED
void delete_isotree_parameters(void *isotree_parameters)
{
    IsoTree_Params *params = (IsoTree_Params*)isotree_parameters;
    delete params;
}

ISOTREE_EXPORTED
void set_isotree_parameters
(
    void *isotree_parameters,
    int*       nthreads,
    uint64_t*  random_seed,
    size_t*    ndim,
    size_t*    ntry,
    uint8_t*   coef_type,
    uint8_t*   with_replacement,
    uint8_t*   weight_as_sample,
    size_t*    sample_size,
    size_t*    ntrees,
    size_t*    max_depth,
    size_t*    ncols_per_tree,
    uint8_t*   limit_depth,
    uint8_t*   penalize_range,
    uint8_t*   weigh_by_kurt,
    double*    prob_pick_by_gain_avg,
    double*    prob_split_by_gain_avg,
    double*    prob_pick_by_gain_pl,
    double*    prob_split_by_gain_pl,
    double*    min_gain,
    uint8_t*   missing_action,
    uint8_t*   cat_split_type,
    uint8_t*   new_cat_action,
    uint8_t*   coef_by_prop,
    uint8_t*   all_perm,
    uint8_t*   build_imputer,
    size_t*    min_imp_obs,
    uint8_t*   depth_imp,
    uint8_t*   weigh_imp_rows
)
{
    IsoTree_Params *params = (IsoTree_Params*)isotree_parameters;
    if (nthreads) params->nthreads = *nthreads;
    if (random_seed) params->random_seed = *random_seed;
    if (ndim) params->ndim = *ndim;
    if (ntry) params->ntry = *ntry;
    if (coef_type) params->coef_type = (CoefType)*coef_type;
    if (with_replacement) params->with_replacement = *with_replacement;
    if (weight_as_sample) params->weight_as_sample = *weight_as_sample;
    if (sample_size) params->sample_size = *sample_size;
    if (ntrees) params->ntrees = *ntrees;
    if (max_depth) params->max_depth = *max_depth;
    if (ncols_per_tree) params->ncols_per_tree = *ncols_per_tree;
    if (limit_depth) params->limit_depth = *limit_depth;
    if (penalize_range) params->penalize_range = *penalize_range;
    if (weigh_by_kurt) params->weigh_by_kurt = *weigh_by_kurt;
    if (prob_pick_by_gain_avg) params->prob_pick_by_gain_avg = *prob_pick_by_gain_avg;
    if (prob_split_by_gain_avg) params->prob_split_by_gain_avg = *prob_split_by_gain_avg;
    if (prob_pick_by_gain_pl) params->prob_pick_by_gain_pl = *prob_pick_by_gain_pl;
    if (prob_split_by_gain_pl) params->prob_split_by_gain_pl = *prob_split_by_gain_pl;
    if (min_gain) params->min_gain = *min_gain;
    if (missing_action) params->missing_action = (MissingAction)*missing_action;
    if (cat_split_type) params->cat_split_type = (CategSplit)*cat_split_type;
    if (new_cat_action) params->new_cat_action = (NewCategAction)*new_cat_action;
    if (coef_by_prop) params->coef_by_prop = *coef_by_prop;
    if (all_perm) params->all_perm = *all_perm;
    if (build_imputer) params->build_imputer = *build_imputer;
    if (min_imp_obs) params->min_imp_obs = *min_imp_obs;
    if (depth_imp) params->depth_imp = (UseDepthImp)*depth_imp;
    if (weigh_imp_rows) params->weigh_imp_rows = (WeighImpRows)*weigh_imp_rows;
}

ISOTREE_EXPORTED
void get_isotree_parameters
(
    const void *isotree_parameters,
    int*       nthreads,
    uint64_t*  random_seed,
    size_t*    ndim,
    size_t*    ntry,
    uint8_t*   coef_type,
    uint8_t*   with_replacement,
    uint8_t*   weight_as_sample,
    size_t*    sample_size,
    size_t*    ntrees,
    size_t*    max_depth,
    size_t*    ncols_per_tree,
    uint8_t*   limit_depth,
    uint8_t*   penalize_range,
    uint8_t*   weigh_by_kurt,
    double*    prob_pick_by_gain_avg,
    double*    prob_split_by_gain_avg,
    double*    prob_pick_by_gain_pl,
    double*    prob_split_by_gain_pl,
    double*    min_gain,
    uint8_t*   missing_action,
    uint8_t*   cat_split_type,
    uint8_t*   new_cat_action,
    uint8_t*   coef_by_prop,
    uint8_t*   all_perm,
    uint8_t*   build_imputer,
    size_t*    min_imp_obs,
    uint8_t*   depth_imp,
    uint8_t*   weigh_imp_rows
)
{
    const IsoTree_Params *params = (IsoTree_Params*)isotree_parameters;
    *nthreads = params->nthreads;
    *random_seed = params->random_seed;
    *ndim = params->ndim;
    *ntry = params->ntry;
    *coef_type = params->coef_type;
    *with_replacement = params->with_replacement;
    *weight_as_sample = params->weight_as_sample;
    *sample_size = params->sample_size;
    *ntrees = params->ntrees;
    *max_depth = params->max_depth;
    *ncols_per_tree = params->ncols_per_tree;
    *limit_depth = params->limit_depth;
    *penalize_range = params->penalize_range;
    *weigh_by_kurt = params->weigh_by_kurt;
    *prob_pick_by_gain_avg = params->prob_pick_by_gain_avg;
    *prob_split_by_gain_avg = params->prob_split_by_gain_avg;
    *prob_pick_by_gain_pl = params->prob_pick_by_gain_pl;
    *prob_split_by_gain_pl = params->prob_split_by_gain_pl;
    *min_gain = params->min_gain;
    *missing_action = params->missing_action;
    *cat_split_type = params->cat_split_type;
    *new_cat_action = params->new_cat_action;
    *coef_by_prop = params->coef_by_prop;
    *all_perm = params->all_perm;
    *build_imputer = params->build_imputer;
    *min_imp_obs = params->min_imp_obs;
    *depth_imp = params->depth_imp;
    *weigh_imp_rows = params->weigh_imp_rows;
}



void* isotree_fit
(
    const void *isotree_parameters,
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
)
{
    if (!ncols_numeric && !ncols_categ) {
        cerr << "Data has no columns" << std::endl;
        return nullptr;
    }
    if (categ_data && !ncateg) {
        cerr << "Must pass 'ncateg' if there is categorical data" << std::endl;
        return nullptr;
    }
    if (!isotree_parameters) {
        cerr << "Passed NULL 'isotree_parameters' to 'isotree_fit'." << std::endl;
        return nullptr;
    }

    const IsoTree_Params *params = (const IsoTree_Params*)isotree_parameters;
    try
    {
        std::unique_ptr<IsolationForest> iso(
            new IsolationForest(
                params->ndim, params->ntry, params->coef_type, params->coef_by_prop,
                params->with_replacement, params->weight_as_sample,
                params->sample_size, params->ntrees,
                params->max_depth, params->ncols_per_tree,
                params->limit_depth, params->penalize_range, params->weigh_by_kurt,
                params->prob_pick_by_gain_avg, params->prob_split_by_gain_avg,
                params->prob_pick_by_gain_pl,  params->prob_split_by_gain_pl,
                params->min_gain, params->missing_action,
                params->cat_split_type, params->new_cat_action,
                params->all_perm, params->build_imputer, params->min_imp_obs,
                params->depth_imp, params->weigh_imp_rows,
                params->random_seed, params->nthreads
            )
        );

        if (numeric_data && !categ_data && !csc_indptr) {
            iso->fit(numeric_data, nrows, ncols_numeric);
        }

        else if (categ_data && !csc_indptr) {
            iso->fit(numeric_data, ncols_numeric, nrows,
                     categ_data, ncols_categ, ncateg,
                     row_weights, column_weights);
        }

        else if (csc_indptr) {
            iso->fit(csc_values, csc_indices, csc_indptr,
                     ncols_numeric, nrows,
                     categ_data, ncols_categ, ncateg,
                     row_weights, column_weights);
        }

        else {
            throw std::runtime_error("Invalid input data.\n");
        }
        
        return iso.release();
    }

    catch (std::exception &e)
    {
        cerr << e.what();
        cerr.flush();
        return nullptr;
    }
}

void delete_isotree_model(void *isotree_model)
{
    IsolationForest *ptr = (IsolationForest*)isotree_model;
    delete ptr;
}

void isotree_predict
(
    void *isotree_model,
    double *output_scores,
    int *output_tree_num,
    uint8_t standardize_scores,
    size_t nrows,
    uint8_t is_col_major,
    double *numeric_data,
    size_t ld_numeric,
    int *categ_data,
    size_t ld_categ,
    uint8_t is_csc,
    double *sparse_values,
    int *sparse_indices,
    int *sparse_indptr
)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_predict'." << std::endl;
        return;
    }
    if (!output_scores) {
        cerr << "Passed NULL 'output_scores' to 'isotree_predict'." << std::endl;
        return;
    }
    IsolationForest *model = (IsolationForest*)isotree_model;

    if (!sparse_indptr) {
        model->predict(numeric_data, categ_data, (bool)is_col_major,
                       nrows, ld_numeric, ld_categ, (bool)standardize_scores,
                       output_scores, output_tree_num);
    }

    else {
        model->predict(sparse_values, sparse_indices, sparse_indptr, (bool)is_csc,
                       categ_data, (bool)is_col_major, ld_categ, nrows, (bool)standardize_scores,
                       output_scores, output_tree_num);
    }
}

void isotree_predict_distance
(
    void *isotree_model,
    uint8_t output_triangular,
    uint8_t standardize_distance,
    uint8_t assume_full_distr,
    double *output_dist,
    size_t nrows,
    double *numeric_data,
    int *categ_data,
    double *csc_values,
    int *csc_indices,
    int *csc_indptr
)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_predict_distance'." << std::endl;
        return;
    }
    if (!output_dist) {
        cerr << "Passed NULL 'output_dist' to 'isotree_predict_distance'." << std::endl;
        return;
    }
    IsolationForest *model = (IsolationForest*)isotree_model;

    if (!csc_indptr) {
        model->predict_distance(numeric_data, categ_data,
                                nrows,
                                (bool) assume_full_distr, (bool) standardize_distance,
                                (bool) output_triangular,
                                output_dist);
    }

    else {
        model->predict_distance(csc_values, csc_indices, csc_indptr, categ_data,
                                nrows, (bool) assume_full_distr, (bool) standardize_distance,
                                (bool) output_triangular,
                                output_dist);
    }
}

void isotree_impute
(
    void *isotree_model,
    size_t nrows,
    uint8_t is_col_major,
    double *numeric_data,
    int *categ_data,
    double *csr_values,
    int *csr_indices,
    int *csr_indptr
)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_impute'." << std::endl;
        return;
    }

    IsolationForest *model = (IsolationForest*)isotree_model;

    if (!csr_indptr) {
        model->impute(numeric_data, categ_data, (bool) is_col_major, nrows);
    }

    else {
        model->impute(csr_values, csr_indices, csr_indptr,
                      categ_data, (bool) is_col_major, nrows);
    }
}

void isotree_serialize_to_file(const void *isotree_model, FILE *output)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_serialize_to_file'." << std::endl;
        return;
    }
    if (!output) {
        cerr << "Passed invalid file handle to 'isotree_serialize_to_file'." << std::endl;
        return;
    }

    const IsolationForest *model = (const IsolationForest*)isotree_model;
    
    try
    {
        model->serialize(output);
    }

    catch (std::exception &e)
    {
        cerr << e.what();
        cerr.flush();
    }
}

void* isotree_deserialize_from_file(FILE *serialized_model, int nthreads)
{
    if (!serialized_model) {
        cerr << "Passed invalid file handle to 'isotree_deserialize_from_file'." << std::endl;
        return nullptr;
    }

    try
    {
        #if __cplusplus >= 201402L
        auto out = std::make_unique<IsolationForest>(IsolationForest::deserialize(serialized_model, nthreads));
        #else
        std::unique_ptr<IsolationForest> out(new IsolationForest(std::forward<IsolationForest>(
            IsolationForest::deserialize(serialized_model, nthreads)
        )));
        #endif
        return (void*)out.release();
    }

    catch (std::exception &e)
    {
        cerr << e.what();
        cerr.flush();
        return nullptr;
    }
}

size_t isotree_serialize_get_raw_size(const void *isotree_model)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_serialize_get_raw_size'." << std::endl;
        return 0;
    }

    try
    {
        const IsolationForest *model = (const IsolationForest*)isotree_model;
        return determine_serialized_size_combined(
            model->model.trees.size()? &model->model : nullptr,
            model->model_ext.hplanes.size()? &model->model_ext : nullptr,
            model->imputer.imputer_tree.size()? &model->imputer : nullptr,
            (size_t)0
        );
    }

    catch (std::exception &e)
    {
        cerr << e.what();
        cerr.flush();
        return 0;
    }
}

void isotree_serialize_to_raw(const void *isotree_model, char *output)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_serialize_to_raw'." << std::endl;
        return;
    }

    const IsolationForest *model = (const IsolationForest*)isotree_model;

    try
    {
        serialize_combined(
            model->model.trees.size()? &model->model : nullptr,
            model->model_ext.hplanes.size()? &model->model_ext : nullptr,
            model->imputer.imputer_tree.size()? &model->imputer : nullptr,
            (char*)nullptr,
            (size_t)0,
            output
        );
    }

    catch (std::exception &e)
    {
        cerr << e.what();
        cerr.flush();
    }
}


void* isotree_deserialize_from_raw(const char *serialized_model, int nthreads)
{
    if (!serialized_model) {
        cerr << "Passed NULL 'serialized_model' to 'isotree_deserialize_from_raw'." << std::endl;
        return nullptr;
    }

    try
    {
        bool is_isotree_model = false;
        bool is_compatible = false;
        bool has_combined_objects = false;
        bool has_IsoForest = false;
        bool has_ExtIsoForest = false;
        bool has_Imputer = false;
        bool has_metadata = false;
        size_t size_metadata = 0;
        inspect_serialized_object(
            serialized_model,
            is_isotree_model,
            is_compatible,
            has_combined_objects,
            has_IsoForest,
            has_ExtIsoForest,
            has_Imputer,
            has_metadata,
            size_metadata
        );
        if (is_isotree_model && is_compatible && !has_combined_objects)
            throw std::runtime_error("Serialized model is not compatible.\n");

        IsoForest model = IsoForest();
        ExtIsoForest model_ext = ExtIsoForest();
        Imputer imputer = Imputer();

        deserialize_combined(
            serialized_model,
            &model,
            &model_ext,
            &imputer,
            (char*)nullptr
        );

        if (!model.trees.size() && !model_ext.hplanes.size())
            throw std::runtime_error("Error: model contains no trees.\n");

        size_t ntrees;
        size_t ndim = 3;
        bool build_imputer = false;

        if (model.trees.size()) {
            ntrees = model.trees.size();
            ndim = 1;
        }
        else {
            ntrees = model_ext.hplanes.size();
        }
        if (imputer.imputer_tree.size()) {
            if (imputer.imputer_tree.size() != ntrees)
                throw std::runtime_error("Error: imputer has incorrect number of trees.\n");
            build_imputer = true;
        }

        std::unique_ptr<IsolationForest> out(new IsolationForest());
        out->nthreads = nthreads;
        out->ndim = ndim;
        out->ntrees = ntrees;
        out->build_imputer = build_imputer;

        if (model.trees.size())
            out->get_model() = std::move(model);
        else
            out->get_model_ext() = std::move(model_ext);
        if (imputer.imputer_tree.size())
            out->get_imputer() = std::move(imputer);

        return out.release();
    }

    catch (std::exception &e)
    {
        cerr << e.what();
        cerr.flush();
        return nullptr;
    }
}

ISOTREE_EXPORTED
void isotree_set_num_threads(void *isotree_model, int nthreads)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_set_num_threads'." << std::endl;
        return;
    }
    if (nthreads < 0) {
        #ifndef _OPENMP
        nthreads = 1;
        #else
        nthreads = omp_get_max_threads() + nthreads + 1;
        #endif
    }
    IsolationForest *model = (IsolationForest*)isotree_model;
    model->nthreads = nthreads;
}

ISOTREE_EXPORTED
int isotree_get_num_threads(const void *isotree_model)
{
    if (!isotree_model) {
        cerr << "Passed NULL 'isotree_model' to 'isotree_get_num_threads'." << std::endl;
        return -INT_MAX;
    }
    IsolationForest *model = (IsolationForest*)isotree_model;
    return model->nthreads;
}


} /* extern "C" */

#endif
