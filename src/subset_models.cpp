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
#include "isotree.hpp" 

/* Create a model containing a sub-set of the trees from another model
* 
* Parameters
* ==========
* - model (in)
*       Pointer to isolation forest model wich has already been fit through 'fit_iforest',
*       from which the desired trees will be copied into a new model object.
*       Pass NULL if using the extended model.
* - ext_model (in)
*       Pointer to extended isolation forest model which has already been fit through 'fit_iforest',
*       from which the desired trees will be copied into a new model object.
*       Pass NULL if using the single-variable model.
* - imputer (in)
*       Pointer to imputation object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest'.
*       Pass NULL if the model was built without an imputer.
* - model_new (out)
*       Pointer to already-allocated isolation forest model, which will be reset and to
*       which the selected trees from 'model' will be copied.
*       Pass NULL if using the extended model.
* - ext_model_new (out)
*       Pointer to already-allocated extended isolation forest model, which will be reset and to
*       which the selected hyperplanes from 'ext_model' will be copied.
*       Pass NULL if using the single-variable model.
* - imputer_new (out)
*       Pointer to already-allocated imputation object, which will be reset and to
*       which the selected nodes from 'imputer' (matching to those of either 'model'
*       or 'ext_model') will be copied.
*       Pass NULL if the model was built without an imputer.
*/
void subset_model(IsoForest*     model,      IsoForest*     model_new,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                  Imputer*       imputer,    Imputer*       imputer_new,
                  size_t *trees_take, size_t ntrees_take)
{
    if (model != NULL)
    {
        if (model_new == NULL)
            throw std::runtime_error("Must pass an already-allocated 'model_new'.\n");
        if (imputer != NULL && model->trees.size() != imputer->imputer_tree.size())
            throw std::runtime_error("Number of trees in imputer does not match with model.\n");
        if (ext_model != NULL)
            throw std::runtime_error("Should pass only one of 'model' or 'ext_model'.\n");
        model_new->new_cat_action = model->new_cat_action;
        model_new->cat_split_type = model->cat_split_type;
        model_new->missing_action = model->missing_action;
        model_new->exp_avg_depth = model->exp_avg_depth;
        model_new->exp_avg_sep = model->exp_avg_sep;
        model_new->orig_sample_size = model->orig_sample_size;

        model_new->trees.resize(ntrees_take);
        for (size_t ix = 0; ix < ntrees_take; ix++)
            model_new->trees[ix] = model->trees[trees_take[ix]];
    }

    else if (ext_model != NULL)
    {
        if (ext_model_new == NULL)
            throw std::runtime_error("Must pass an already-allocated 'ext_model_new'.");
        if (imputer != NULL && ext_model->hplanes.size() != imputer->imputer_tree.size())
            throw std::runtime_error("Number of trees in imputer does not match with model.\n");
        if (model != NULL)
            throw std::runtime_error("Should pass only one of 'model' or 'ext_model'.\n");
        ext_model_new->new_cat_action = ext_model->new_cat_action;
        ext_model_new->cat_split_type = ext_model->cat_split_type;
        ext_model_new->missing_action = ext_model->missing_action;
        ext_model_new->exp_avg_depth = ext_model->exp_avg_depth;
        ext_model_new->exp_avg_sep = ext_model->exp_avg_sep;
        ext_model_new->orig_sample_size = ext_model->orig_sample_size;

        ext_model_new->hplanes.resize(ntrees_take);
        for (size_t ix = 0; ix < ntrees_take; ix++)
            ext_model_new->hplanes[ix] = ext_model->hplanes[trees_take[ix]];
    }

    if (imputer != NULL)
    {
        if (imputer_new == NULL)
            throw std::runtime_error("Must pass an already-allocated 'imputer_new'.");
        imputer_new->ncols_numeric = imputer->ncols_numeric;
        imputer_new->ncols_categ = imputer->ncols_categ;
        imputer_new->ncat = imputer->ncat;
        imputer_new->col_means = imputer->col_means;
        imputer_new->col_modes = imputer->col_modes;

        imputer_new->imputer_tree.resize(ntrees_take);
        for (size_t ix = 0; ix < ntrees_take; ix++)
            imputer_new->imputer_tree[ix] = imputer->imputer_tree[trees_take[ix]];
    }
}
