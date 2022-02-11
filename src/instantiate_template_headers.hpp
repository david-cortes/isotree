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

ISOTREE_EXPORTED int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees,
                size_t max_depth,   size_t ncols_per_tree,
                bool   limit_depth, bool penalize_range, bool standardize_data,
                ScoringMetric scoring_metric, bool fast_bratio,
                bool   standardize_dist, double tmat[],
                double output_depths[], bool standardize_depth,
                real_t col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                double prob_pick_by_full_gain, double prob_pick_by_dens,
                double prob_pick_col_by_range, double prob_pick_col_by_var,
                double prob_pick_col_by_kurt,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, bool use_long_double, int nthreads)
{
    return fit_iforest<real_t, sparse_ix>
               (model_outputs, model_outputs_ext,
                numeric_data,  ncols_numeric,
                categ_data,    ncols_categ,    ncat,
                Xc, Xc_ind, Xc_indptr,
                ndim, ntry, coef_type, coef_by_prop,
                sample_weights, with_replacement, weight_as_sample,
                nrows, sample_size, ntrees,
                max_depth,   ncols_per_tree,
                limit_depth, penalize_range, standardize_data,
                scoring_metric, fast_bratio,
                standardize_dist, tmat,
                output_depths, standardize_depth,
                col_weights, weigh_by_kurt,
                prob_pick_by_gain_pl, prob_pick_by_gain_avg,
                prob_pick_by_full_gain, prob_pick_by_dens,
                prob_pick_col_by_range, prob_pick_col_by_var,
                prob_pick_col_by_kurt,
                min_gain, missing_action,
                cat_split_type, new_cat_action,
                all_perm, imputer, min_imp_obs,
                depth_imp, weigh_imp_rows, impute_at_fit,
                random_seed, use_long_double, nthreads);
}
ISOTREE_EXPORTED int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
             real_t numeric_data[],  size_t ncols_numeric,
             int    categ_data[],    size_t ncols_categ,    int ncat[],
             real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
             real_t sample_weights[], size_t nrows,
             size_t max_depth,     size_t ncols_per_tree,
             bool   limit_depth,   bool penalize_range, bool standardize_data,
             bool   fast_bratio,
             real_t col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
             double prob_pick_by_full_gain, double prob_pick_by_dens,
             double prob_pick_col_by_range, double prob_pick_col_by_var,
             double prob_pick_col_by_kurt,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, Imputer *imputer, size_t min_imp_obs,
             TreesIndexer *indexer,
             real_t ref_numeric_data[], int ref_categ_data[],
             bool ref_is_col_major, size_t ref_ld_numeric, size_t ref_ld_categ,
             real_t ref_Xc[], sparse_ix ref_Xc_ind[], sparse_ix ref_Xc_indptr[],
             uint64_t random_seed, bool use_long_double)
{
    return add_tree<real_t, sparse_ix>
            (model_outputs, model_outputs_ext,
             numeric_data,  ncols_numeric,
             categ_data,    ncols_categ,    ncat,
             Xc, Xc_ind, Xc_indptr,
             ndim, ntry, coef_type, coef_by_prop,
             sample_weights, nrows,
             max_depth,     ncols_per_tree,
             limit_depth,   penalize_range, standardize_data,
             fast_bratio,
             col_weights, weigh_by_kurt,
             prob_pick_by_gain_pl, prob_pick_by_gain_avg,
             prob_pick_by_full_gain, prob_pick_by_dens,
             prob_pick_col_by_range, prob_pick_col_by_var,
             prob_pick_col_by_kurt,
             min_gain, missing_action,
             cat_split_type, new_cat_action,
             depth_imp, weigh_imp_rows,
             all_perm, imputer, min_imp_obs,
             indexer,
             ref_numeric_data, ref_categ_data,
             ref_is_col_major, ref_ld_numeric, ref_ld_categ,
             ref_Xc, ref_Xc_ind, ref_Xc_indptr,
             random_seed, use_long_double);
}
ISOTREE_EXPORTED void predict_iforest(real_t numeric_data[], int categ_data[],
                     bool is_col_major, size_t ncols_numeric, size_t ncols_categ,
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[],
                     double per_tree_depths[],
                     TreesIndexer *indexer)
{
    predict_iforest<real_t, sparse_ix>
                    (numeric_data, categ_data,
                     is_col_major, ncols_numeric, ncols_categ,
                     Xc, Xc_ind, Xc_indptr,
                     Xr, Xr_ind, Xr_indptr,
                     nrows, nthreads, standardize,
                     model_outputs, model_outputs_ext,
                     output_depths,   tree_num,
                     per_tree_depths,
                     indexer);
}
ISOTREE_EXPORTED void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, bool use_long_double, int nthreads,
                     bool assume_full_distr, bool standardize_dist, bool as_kernel,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from, bool use_indexed_references,
                     TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ)
{
    calc_similarity<real_t, sparse_ix>
                    (numeric_data, categ_data,
                     Xc, Xc_ind, Xc_indptr,
                     nrows, use_long_double, nthreads,
                     assume_full_distr, standardize_dist, as_kernel,
                     model_outputs, model_outputs_ext,
                     tmat, rmat, n_from, use_indexed_references,
                     indexer, is_col_major, ld_numeric, ld_categ);
}
ISOTREE_EXPORTED void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, bool use_long_double, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer)
{
    impute_missing_values<real_t, sparse_ix>
                          (numeric_data, categ_data, is_col_major,
                           Xr, Xr_ind, Xr_indptr,
                           nrows, use_long_double, nthreads,
                           model_outputs, model_outputs_ext,
                           imputer);
}

ISOTREE_EXPORTED void set_reference_points(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
                          const bool with_distances,
                          real_t *numeric_data, int *categ_data,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          real_t *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
                          real_t *Xr, sparse_ix *Xr_ind, sparse_ix *Xr_indptr,
                          size_t nrows, int nthreads)
{
    set_reference_points<real_t, sparse_ix>
                        (model_outputs, model_outputs_ext, indexer,
                         with_distances,
                         numeric_data, categ_data,
                         is_col_major, ld_numeric, ld_categ,
                         Xc, Xc_ind, Xc_indptr,
                         Xr, Xr_ind, Xr_indptr,
                         nrows, nthreads);
}

#ifndef _NO_REAL_T
ISOTREE_EXPORTED void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept
{
    get_num_nodes<sparse_ix>(model_outputs, n_nodes, n_terminal, nthreads);
}
ISOTREE_EXPORTED void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept
{
    get_num_nodes<sparse_ix>(model_outputs, n_nodes, n_terminal, nthreads);
}
#endif

