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
#include "isotree.hpp"

template <class Model, class real_t, class sparse_ix>
void set_reference_points(TreesIndexer &indexer, Model &model, const bool with_distances,
                          real_t *restrict numeric_data, int *restrict categ_data,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                          real_t *restrict Xr, sparse_ix *restrict Xr_ind, sparse_ix *restrict Xr_indptr,
                          size_t nrows, int nthreads)
{
    if (indexer.indices.empty() || (with_distances && indexer.indices.front().node_distances.empty()))
    {
        build_tree_indices(indexer, model, nthreads, with_distances);
    }

    if (!indexer.indices.front().reference_points.empty())
    {
        for (auto &tree : indexer.indices)
        {
            tree.reference_points.clear();
            tree.reference_indptr.clear();
            tree.reference_mapping.clear();
        }
    }


    size_t ntrees = get_ntrees(model);
    std::unique_ptr<double[]> ignored(new double[nrows]);
    std::unique_ptr<sparse_ix[]> node_indices_predict(new sparse_ix[nrows * ntrees]);
    void *model_ptr = (std::is_same<Model, IsoForest>::value)? &model : NULL;
    void *model_ext_ptr = (std::is_same<Model, ExtIsoForest>::value)? &model : NULL;
    predict_iforest(numeric_data, categ_data,
                    is_col_major, ld_numeric, ld_categ,
                    Xc, Xc_ind, Xc_indptr,
                    Xr, Xr_ind, Xr_indptr,
                    nrows, nthreads, false,
                    (IsoForest*)model_ptr, (ExtIsoForest*)model_ext_ptr,
                    ignored.get(), node_indices_predict.get(),
                    (double*)NULL,
                    &indexer);
    ignored.reset();

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(indexer, node_indices_predict, ntrees)
    for (size_t_for tree = 0; tree < (decltype(tree))ntrees; tree++)
    {
        indexer.indices[tree].reference_points.assign(node_indices_predict.get() + tree*nrows,
                                                      node_indices_predict.get() + (tree+1)*nrows);
        indexer.indices[tree].reference_points.shrink_to_fit();
        build_ref_node(indexer.indices[tree]);
    }
}

template <class real_t, class sparse_ix>
void set_reference_points(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
                          const bool with_distances,
                          real_t *restrict numeric_data, int *restrict categ_data,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                          real_t *restrict Xr, sparse_ix *restrict Xr_ind, sparse_ix *restrict Xr_indptr,
                          size_t nrows, int nthreads)
{
    try
    {
        if (model_outputs != NULL)
            set_reference_points(*indexer, *model_outputs, with_distances,
                                 numeric_data, categ_data,
                                 is_col_major, ld_numeric, ld_categ,
                                 Xc, Xc_ind, Xc_indptr,
                                 Xr, Xr_ind, Xr_indptr,
                                 nrows, nthreads);
        else
            set_reference_points(*indexer, *model_outputs_ext, with_distances,
                                 numeric_data, categ_data,
                                 is_col_major, ld_numeric, ld_categ,
                                 Xc, Xc_ind, Xc_indptr,
                                 Xr, Xr_ind, Xr_indptr,
                                 nrows, nthreads);
    }

    catch (...)
    {
        for (auto &tree : indexer->indices)
        {
            tree.reference_points.clear();
            tree.reference_indptr.clear();
            tree.reference_mapping.clear();
        }

        throw;
    }
}
