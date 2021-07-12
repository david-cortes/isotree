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

/* Append trees from one model into another
* 
* Parameters
* ==========
* - model (in, out)
*       Pointer to isolation forest model wich has already been fit through 'fit_iforest'.
*       The trees from 'other' will be merged into this (will be at the end of vector member 'trees').
*       Both 'model' and 'other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - other
*       Pointer to isolation forest model which has already been fit through 'fit_iforest'.
*       The trees from this object will be added into 'model' (this object will not be modified).
*       Both 'model' and 'other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - ext_model (in, out)
*       Pointer to extended isolation forest model which has already been fit through 'fit_iforest'.
*       The trees/hyperplanes from 'ext_other' will be merged into this (will be at the end of vector member 'hplanes').
*       Both 'ext_model' and 'ext_other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - ext_other
*       Pointer to extended isolation forest model which has already been fit through 'fit_iforest'.
*       The trees/hyperplanes from this object will be added into 'ext_model' (this object will not be modified).
*       Both 'ext_model' and 'ext_other' must have been fit with the same hyperparameters
*       in order for this merge to work correctly - at the very least, should have
*       the same 'missing_action', 'cat_split_type', 'new_cat_action'.
*       Should only pass one of 'model'+'other' or 'ext_model'+'ext_other'.
*       Pass NULL if this is not to be used.
* - imputer (in, out)
*       Pointer to imputation object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest'.
*       The imputation nodes from 'iother' will be merged into this (will be at the end of vector member 'imputer_tree').
*       Hyperparameters related to imputation might differ between 'imputer' and 'iother' ('imputer' will preserve its
*       hyperparameters after the merge).
*       Pass NULL if this is not to be used.
* - iother
*       Pointer to imputation object which has already been fit through 'fit_iforest' along with
*       either 'model' or 'ext_model' in the same call to 'fit_iforest'.
*       The imputation nodes from this object will be added into 'imputer' (this object will not be modified).
*       Hyperparameters related to imputation might differ between 'imputer' and 'iother' ('imputer' will preserve its
*       hyperparameters after the merge).
*       Pass NULL if this is not to be used.
*/
void merge_models(IsoForest*     model,      IsoForest*     other,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                  Imputer*       imputer,    Imputer*       iother)
{
    size_t curr_size_model = (model != NULL)? (model->trees.size()) : 0;
    size_t curr_size_model_ext = (ext_model != NULL)? (ext_model->hplanes.size()) : 0;
    size_t curr_size_imputer = (imputer != NULL)? (imputer->imputer_tree.size()) : 0;

    try
    {
        if (model != NULL && other != NULL)
        {
            if (model == other)
            {
                auto other_copy = *other;
                merge_models(model, &other_copy, ext_model, ext_other, imputer, iother);
                return;
            }
            model->trees.insert(model->trees.end(),
                                other->trees.begin(),
                                other->trees.end());

        }

        if (ext_model != NULL && ext_other != NULL)
        {
            if (ext_model == ext_other)
            {
                auto other_copy = *ext_other;
                merge_models(model, other, ext_model, &other_copy, imputer, iother);
                return;
            }
            ext_model->hplanes.insert(ext_model->hplanes.end(),
                                      ext_other->hplanes.begin(),
                                      ext_other->hplanes.end());
        }

        if (imputer != NULL && iother != NULL)
        {
            if (imputer == iother)
            {
                auto other_copy = *iother;
                merge_models(model, other, ext_model, ext_other, imputer, &other_copy);
                return;
            }
            imputer->imputer_tree.insert(imputer->imputer_tree.end(),
                                         iother->imputer_tree.begin(),
                                         iother->imputer_tree.end());
        }
    }

    catch(...)
    {
        if (model != NULL) model->trees.resize(curr_size_model);
        if (ext_model != NULL) ext_model->hplanes.resize(curr_size_model_ext);
        if (imputer != NULL) imputer->imputer_tree.resize(curr_size_imputer);
        throw;
    }
}
