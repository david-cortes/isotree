int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees, size_t max_depth,
                bool   limit_depth, bool penalize_range,
                bool   standardize_dist, double tmat[],
                double output_depths[], bool standardize_depth,
                real_t col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, int nthreads)
{
    return fit_iforest<real_t, sparse_ix>
               (model_outputs, model_outputs_ext,
                numeric_data,  ncols_numeric,
                categ_data,    ncols_categ,    ncat,
                Xc, Xc_ind, Xc_indptr,
                ndim, ntry, coef_type, coef_by_prop,
                sample_weights, with_replacement, weight_as_sample,
                nrows, sample_size, ntrees, max_depth,
                limit_depth, penalize_range,
                standardize_dist, tmat,
                output_depths, standardize_depth,
                col_weights, weigh_by_kurt,
                prob_pick_by_gain_avg, prob_split_by_gain_avg,
                prob_pick_by_gain_pl,  prob_split_by_gain_pl,
                min_gain, missing_action,
                cat_split_type, new_cat_action,
                all_perm, imputer, min_imp_obs,
                depth_imp, weigh_imp_rows, impute_at_fit,
                random_seed, nthreads);
}
int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
             real_t numeric_data[],  size_t ncols_numeric,
             int    categ_data[],    size_t ncols_categ,    int ncat[],
             real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
             real_t sample_weights[], size_t nrows, size_t max_depth,
             bool   limit_depth,   bool penalize_range,
             real_t col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
             double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, std::vector<ImputeNode> *impute_nodes, size_t min_imp_obs,
             uint64_t random_seed)
{
    return add_tree<real_t, sparse_ix>
            (model_outputs, model_outputs_ext,
             numeric_data,  ncols_numeric,
             categ_data,    ncols_categ,    ncat,
             Xc, Xc_ind, Xc_indptr,
             ndim, ntry, coef_type, coef_by_prop,
             sample_weights, nrows, max_depth,
             limit_depth,   penalize_range,
             col_weights, weigh_by_kurt,
             prob_pick_by_gain_avg, prob_split_by_gain_avg,
             prob_pick_by_gain_pl,  prob_split_by_gain_pl,
             min_gain, missing_action,
             cat_split_type, new_cat_action,
             depth_imp, weigh_imp_rows,
             all_perm, impute_nodes, min_imp_obs,
             random_seed);
}
void predict_iforest(real_t numeric_data[], int categ_data[],
                     bool is_col_major, size_t ncols_numeric, size_t ncols_categ,
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[])
{
    predict_iforest<real_t, sparse_ix>
                    (numeric_data, categ_data,
                     is_col_major, ncols_numeric, ncols_categ,
                     Xc, Xc_ind, Xc_indptr,
                     Xr, Xr_ind, Xr_indptr,
                     nrows, nthreads, standardize,
                     model_outputs, model_outputs_ext,
                     output_depths,   tree_num);
}
void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, int nthreads, bool assume_full_distr, bool standardize_dist,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from)
{
    calc_similarity<real_t, sparse_ix>
                    (numeric_data, categ_data,
                     Xc, Xc_ind, Xc_indptr,
                     nrows, nthreads, assume_full_distr, standardize_dist,
                     model_outputs, model_outputs_ext,
                     tmat, rmat, n_from);
}
void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer)
{
    impute_missing_values<real_t, sparse_ix>
                          (numeric_data, categ_data, is_col_major,
                           Xr, Xr_ind, Xr_indptr,
                           nrows, nthreads,
                           model_outputs, model_outputs_ext,
                           imputer);
}

#ifndef _NO_REAL_T
void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads)
{
    get_num_nodes<sparse_ix>(model_outputs, n_nodes, n_terminal, nthreads);
}
void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads)
{
    get_num_nodes<sparse_ix>(model_outputs, n_nodes, n_terminal, nthreads);
}
#endif

