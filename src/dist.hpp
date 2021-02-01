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


/* Calculate distance or similarity between data points
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data for which to make calculations. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.),
*       and the column order must be the same as in the data that was used to fit the model.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data for which to make calculations. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.),
*       and the column order must be the same as in the data that was used to fit the model.
*       Pass NULL if there are no categorical columns.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data' or 'Xc' + 'Xc_ind' + 'Xc_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSC format.
* - Xc_indptr[ncols_categ + 1]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end.
*       Pass NULL if there are no sparse numeric columns in CSC format.
*       If making calculations between two sets of observations/rows (see documentation for 'rmat'),
*       the first group is assumed to be the earlier rows here.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'Xr, 'categ_data'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* - assume_full_distr
*       Whether to assume that the fitted model represents a full population distribution (will use a
*       standardizing criterion assuming infinite sample, and the results of the similarity between two points
*       at prediction time will not depend on the prescence of any third point that is similar to them, but will
*       differ more compared to the pairwise distances between points from which the model was fit). If passing
*       'false', will calculate pairwise distances as if the new observations at prediction time were added to
*       the sample to which each tree was fit, which will make the distances between two points potentially vary
*       according to other newly introduced points.
* - standardize_dist
*       Whether to standardize the resulting average separation depths between rows according
*       to the expected average separation depth in a similar way as when predicting outlierness,
*       in order to obtain a standardized distance. If passing 'false', will output the average
*       separation depth instead.
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the calculations are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the calculations are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - tmat[nrows * (nrows - 1) / 2] (out)
*       Pointer to array where the resulting pairwise distances or average separation depths will
*       be written into. As the output is a symmetric matrix, this function will only fill in the
*       upper-triangular part, in which entry 0 <= i < j < n will be located at position
*           p(i,j) = (i * (n - (i+1)/2) + j - i - 1).
*       Can be converted to a dense square matrix through function 'tmat_to_dense'.
*       The array must already be initialized to zeros.
*       If calculating distance/separation from a group of points to another group of points,
*       pass NULL here and use 'rmat' instead.
* - rmat[nrows1 * nrows2] (out)
*       Pointer to array where to write the distances or separation depths between each row in
*       one set of observations and each row in a different set of observations. If doing these
*       calculations for all pairs of observations/rows, pass 'rmat' instead.
*       Will take the first group of observations as the rows in this matrix, and the second
*       group as the columns. The groups are assumed to be in the same data arrays, with the
*       first group corresponding to the earlier rows there.
*       This matrix will be used in row-major order (i.e. entries 1..n_from contain the first row).
*       Must be already initialized to zeros.
*       Ignored when 'tmat' is passed.
* - n_from
*       When calculating distances between two groups of points, this indicates the number of
*       observations/rows belonging to the first group (the rows in 'rmat'), which will be
*       assumed to be the first 'n_from' rows.
*       Ignored when 'tmat' is passed.
*/
template <class real_t, class sparse_ix>
void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, int nthreads, bool assume_full_distr, bool standardize_dist,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from)
{
    PredictionData<real_t, sparse_ix>
                   prediction_data = {numeric_data, categ_data, nrows,
                                      false, 0, 0,
                                      Xc, Xc_ind, Xc_indptr,
                                      NULL, NULL, NULL};

    size_t ntrees = (model_outputs != NULL)? model_outputs->trees.size() : model_outputs_ext->hplanes.size();

    if (tmat != NULL) n_from = 0;

    if ((size_t)nthreads > ntrees)
        nthreads = (int)ntrees;
    #ifdef _OPENMP
    std::vector<WorkerForSimilarity> worker_memory(nthreads);
    #else
    std::vector<WorkerForSimilarity> worker_memory(1);
    #endif

    /* Global variable that determines if the procedure receives a stop signal */
    SignalSwitcher ss = SignalSwitcher();
    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return;
    #endif

    if (model_outputs != NULL)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(ntrees, worker_memory, prediction_data, model_outputs)
        for (size_t_for tree = 0; tree < ntrees; tree++)
        {
            initialize_worker_for_sim(worker_memory[omp_get_thread_num()], prediction_data,
                                      model_outputs, NULL, n_from, assume_full_distr);
            traverse_tree_sim(worker_memory[omp_get_thread_num()],
                              prediction_data,
                              *model_outputs,
                              model_outputs->trees[tree],
                              (size_t)0);
        }
    }

    else
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(ntrees, worker_memory, prediction_data, model_outputs_ext)
        for (size_t_for hplane = 0; hplane < ntrees; hplane++)
        {
            initialize_worker_for_sim(worker_memory[omp_get_thread_num()], prediction_data,
                                      NULL, model_outputs_ext, n_from, assume_full_distr);
            traverse_hplane_sim(worker_memory[omp_get_thread_num()],
                                prediction_data,
                                *model_outputs_ext,
                                model_outputs_ext->hplanes[hplane],
                                (size_t)0);
        }    
    }

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return;
    #endif
    
    /* gather and transform the results */
    gather_sim_result< PredictionData<real_t, sparse_ix>, InputData<real_t, sparse_ix>, WorkerMemory<ImputedData<sparse_ix>> >
                     (&worker_memory, NULL,
                      &prediction_data, NULL,
                      model_outputs, model_outputs_ext,
                      tmat, rmat, n_from,
                      ntrees, assume_full_distr,
                      standardize_dist, nthreads);

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return;
    #endif
}

template <class PredictionData>
void traverse_tree_sim(WorkerForSimilarity   &workspace,
                       PredictionData        &prediction_data,
                       IsoForest             &model_outputs,
                       std::vector<IsoTree>  &trees,
                       size_t                curr_tree)
{
    if (interrupt_switch)
        return;

    if (workspace.st == workspace.end)
        return;

    if (!workspace.tmat_sep.size())
    {
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);
        if (workspace.ix_arr[workspace.st] >= workspace.n_from)
            return;
        if (workspace.ix_arr[workspace.end] < workspace.n_from)
            return;
    }

    /* Note: the first separation step will not be added here, as it simply consists of adding +1
       to every combination regardless. It has to be added at the end in 'gather_sim_result' to
       obtain the average separation depth. */
    if (trees[curr_tree].score >= 0.)
    {
        long double rem = (long double) trees[curr_tree].remainder;
        if (!workspace.weights_arr.size())
        {
            rem += (long double)(workspace.end - workspace.st + 1);
            if (workspace.tmat_sep.size())
                increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                                      prediction_data.nrows, workspace.tmat_sep.data(),
                                      workspace.assume_full_distr? 3. : expected_separation_depth(rem));
            else
                increase_comb_counter_in_groups(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                workspace.n_from, prediction_data.nrows, workspace.rmat.data(),
                                                workspace.assume_full_distr? 3. : expected_separation_depth(rem));
        }

        else
        {
            if (!workspace.assume_full_distr)
            {
                rem += std::accumulate(workspace.ix_arr.begin() + workspace.st,
                                       workspace.ix_arr.begin() + workspace.end,
                                       (long double) 0.,
                                       [&workspace](long double curr, size_t ix)
                                                      {return curr + (long double)workspace.weights_arr[ix];}
                                      );
            }

            if (workspace.tmat_sep.size())
                increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                                      prediction_data.nrows, workspace.tmat_sep.data(),
                                      workspace.weights_arr.data(),
                                      workspace.assume_full_distr? 3. : expected_separation_depth(rem));
            else
                increase_comb_counter_in_groups(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                workspace.n_from, prediction_data.nrows,
                                                workspace.rmat.data(), workspace.weights_arr.data(),
                                                workspace.assume_full_distr? 3. : expected_separation_depth(rem));
        }
        return;
    }

    else if (curr_tree > 0)
    {
        if (workspace.tmat_sep.size())
            if (!workspace.weights_arr.size())
                increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                                      prediction_data.nrows, workspace.tmat_sep.data(), -1.);
            else
                increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                                      prediction_data.nrows, workspace.tmat_sep.data(),
                                      workspace.weights_arr.data(), -1.);
        else
            if (!workspace.weights_arr.size())
                increase_comb_counter_in_groups(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                workspace.n_from, prediction_data.nrows, workspace.rmat.data(), -1.);
            else
                increase_comb_counter_in_groups(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                workspace.n_from, prediction_data.nrows,
                                                workspace.rmat.data(), workspace.weights_arr.data(), -1.);
    }


    /* divide according to tree */
    if (prediction_data.Xc_indptr != NULL && workspace.tmat_sep.size())
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);
    size_t st_NA, end_NA, split_ix;
    switch(trees[curr_tree].col_type)
    {
        case Numeric:
        {
            if (prediction_data.Xc_indptr == NULL)
                divide_subset_split(workspace.ix_arr.data(),
                                    prediction_data.numeric_data + prediction_data.nrows * trees[curr_tree].col_num,
                                    workspace.st, workspace.end, trees[curr_tree].num_split,
                                    model_outputs.missing_action, st_NA, end_NA, split_ix);
            else
                divide_subset_split(workspace.ix_arr.data(), workspace.st, workspace.end, trees[curr_tree].col_num,
                                    prediction_data.Xc, prediction_data.Xc_ind, prediction_data.Xc_indptr,
                                    trees[curr_tree].num_split, model_outputs.missing_action,
                                    st_NA, end_NA, split_ix);

            break;
        }

        case Categorical:
        {
            switch(model_outputs.cat_split_type)
            {
                case SingleCateg:
                {
                    divide_subset_split(workspace.ix_arr.data(),
                                        prediction_data.categ_data + prediction_data.nrows * trees[curr_tree].col_num,
                                        workspace.st, workspace.end, trees[curr_tree].chosen_cat,
                                         model_outputs.missing_action, st_NA, end_NA, split_ix);
                    break;
                }

                case SubSet:
                {
                    if (!trees[curr_tree].cat_split.size())
                        divide_subset_split(workspace.ix_arr.data(),
                                            prediction_data.categ_data + prediction_data.nrows * trees[curr_tree].col_num,
                                            workspace.st, workspace.end,
                                            model_outputs.missing_action, model_outputs.new_cat_action,
                                            trees[curr_tree].pct_tree_left < .5, st_NA, end_NA, split_ix);
                    else
                        divide_subset_split(workspace.ix_arr.data(),
                                            prediction_data.categ_data + prediction_data.nrows * trees[curr_tree].col_num,
                                            workspace.st, workspace.end, trees[curr_tree].cat_split.data(),
                                            (int) trees[curr_tree].cat_split.size(),
                                            model_outputs.missing_action, model_outputs.new_cat_action,
                                            (bool)(trees[curr_tree].pct_tree_left < .5), st_NA, end_NA, split_ix);
                    break;
                }
            }
            break;
        }
    }


    /* continue splitting recursively */
    size_t orig_end = workspace.end;
    switch(model_outputs.missing_action)
    {
        case Impute:
        {
            split_ix = (trees.back().pct_tree_left >= .5)? end_NA : st_NA;
        }

        case Fail:
        {
            if (split_ix > workspace.st)
            {
                workspace.end = split_ix - 1;
                traverse_tree_sim(workspace,
                                  prediction_data,
                                  model_outputs,
                                  trees,
                                  trees[curr_tree].tree_left);
            }


            if (split_ix < orig_end)
            {
                workspace.st  = split_ix;
                workspace.end = orig_end;
                traverse_tree_sim(workspace,
                                  prediction_data,
                                  model_outputs,
                                  trees,
                                  trees[curr_tree].tree_right);
            }
            break;
        }

        case Divide: /* new_cat_action = 'Weighted' will also fall here */
        {
            std::vector<double> weights_arr;
            std::vector<size_t> ix_arr;
            if (end_NA > workspace.st)
            {
                weights_arr.assign(workspace.weights_arr.begin(),
                                   workspace.weights_arr.begin() + end_NA);
                ix_arr.assign(workspace.ix_arr.begin(),
                              workspace.ix_arr.begin() + end_NA);
            }

            if (end_NA > workspace.st)
            {
                workspace.end = end_NA - 1;
                for (size_t row = st_NA; row < end_NA; row++)
                    workspace.weights_arr[workspace.ix_arr[row]] *= trees[curr_tree].pct_tree_left;
                traverse_tree_sim(workspace,
                                  prediction_data,
                                  model_outputs,
                                  trees,
                                  trees[curr_tree].tree_left);
            }

            if (st_NA < orig_end)
            {
                workspace.st = st_NA;
                workspace.end = orig_end;
                if (weights_arr.size())
                {
                    std::copy(weights_arr.begin(),
                              weights_arr.end(),
                              workspace.weights_arr.begin());
                    std::copy(ix_arr.begin(),
                              ix_arr.end(),
                              workspace.ix_arr.begin());
                    weights_arr.clear();
                    weights_arr.shrink_to_fit();
                    ix_arr.clear();
                    ix_arr.shrink_to_fit();
                }

                for (size_t row = st_NA; row < end_NA; row++)
                    workspace.weights_arr[workspace.ix_arr[row]] *= (1 - trees[curr_tree].pct_tree_left);
                traverse_tree_sim(workspace,
                                  prediction_data,
                                  model_outputs,
                                  trees,
                                  trees[curr_tree].tree_right);
            }
            break;
        }
    }
}

template <class PredictionData>
void traverse_hplane_sim(WorkerForSimilarity     &workspace,
                         PredictionData          &prediction_data,
                         ExtIsoForest            &model_outputs,
                         std::vector<IsoHPlane>  &hplanes,
                         size_t                  curr_tree)
{
    if (interrupt_switch)
        return;
    
    if (workspace.st == workspace.end)
        return;

    if (!workspace.tmat_sep.size())
    {
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);
        if (workspace.ix_arr[workspace.st] >= workspace.n_from)
            return;
        if (workspace.ix_arr[workspace.end] < workspace.n_from)
            return;
    }

    /* Note: the first separation step will not be added here, as it simply consists of adding +1
       to every combination regardless. It has to be added at the end in 'gather_sim_result' to
       obtain the average separation depth. */
    if (hplanes[curr_tree].score >= 0)
    {
        if (workspace.tmat_sep.size())
            increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                                  prediction_data.nrows, workspace.tmat_sep.data(),
                                  workspace.assume_full_distr? 3. : 
                                  expected_separation_depth((long double) hplanes[curr_tree].remainder
                                                              + (long double)(workspace.end - workspace.st + 1))
                                  );
        else
            increase_comb_counter_in_groups(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.n_from,
                                            prediction_data.nrows, workspace.rmat.data(),
                                            workspace.assume_full_distr? 3. : 
                                            expected_separation_depth((long double) hplanes[curr_tree].remainder
                                                                        + (long double)(workspace.end - workspace.st + 1))
                                            );
        return;
    }

    else if (curr_tree > 0)
    {
        if (workspace.tmat_sep.size())
            increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                                  prediction_data.nrows, workspace.tmat_sep.data(), -1.);
        else
            increase_comb_counter_in_groups(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.n_from,
                                            prediction_data.nrows, workspace.rmat.data(), -1.);
    }

    if (prediction_data.Xc_indptr != NULL && workspace.tmat_sep.size())
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);

    /* reconstruct linear combination */
    size_t ncols_numeric = 0;
    size_t ncols_categ   = 0;
    std::fill(workspace.comb_val.begin(), workspace.comb_val.begin() + (workspace.end - workspace.st + 1), 0);
    if (prediction_data.categ_data != NULL || prediction_data.Xc_indptr != NULL)
    {
        for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
        {
            switch(hplanes[curr_tree].col_type[col])
            {
                case Numeric:
                {
                    if (prediction_data.Xc_indptr == NULL)
                        add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                        prediction_data.numeric_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
                                        hplanes[curr_tree].coef[ncols_numeric], (double)0, hplanes[curr_tree].mean[ncols_numeric],
                                        (model_outputs.missing_action == Fail)?  workspace.comb_val[0] : hplanes[curr_tree].fill_val[col],
                                        model_outputs.missing_action, NULL, NULL, false);
                    else
                        add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end,
                                        hplanes[curr_tree].col_num[col], workspace.comb_val.data(),
                                        prediction_data.Xc, prediction_data.Xc_ind, prediction_data.Xc_indptr,
                                        hplanes[curr_tree].coef[ncols_numeric], (double)0, hplanes[curr_tree].mean[ncols_numeric],
                                        (model_outputs.missing_action == Fail)?  workspace.comb_val[0] : hplanes[curr_tree].fill_val[col],
                                        model_outputs.missing_action, NULL, NULL, false);
                    ncols_numeric++;
                    break;
                }

                case Categorical:
                {
                    switch(model_outputs.cat_split_type)
                    {
                        case SingleCateg:
                        {
                            add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                            prediction_data.categ_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
                                            (int)0, NULL, hplanes[curr_tree].fill_new[ncols_categ],
                                            hplanes[curr_tree].chosen_cat[ncols_categ],
                                            (model_outputs.missing_action == Fail)?  workspace.comb_val[0] : hplanes[curr_tree].fill_val[col],
                                            workspace.comb_val[0], NULL, NULL, model_outputs.new_cat_action,
                                            model_outputs.missing_action, SingleCateg, false);
                            break;
                        }

                        case SubSet:
                        {
                            add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                            prediction_data.categ_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
                                            (int) hplanes[curr_tree].cat_coef[ncols_categ].size(),
                                            hplanes[curr_tree].cat_coef[ncols_categ].data(), (double) 0, (int) 0,
                                            (model_outputs.missing_action == Fail)? workspace.comb_val[0] : hplanes[curr_tree].fill_val[col],
                                            hplanes[curr_tree].fill_new[ncols_categ], NULL, NULL,
                                            model_outputs.new_cat_action, model_outputs.missing_action, SubSet, false);
                            break;
                        }
                    }
                    ncols_categ++;
                    break;
                }
            } 
        }
    }

    
    else /* faster version for numerical-only */
    {
        for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
            add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                            prediction_data.numeric_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
                            hplanes[curr_tree].coef[col], (double)0, hplanes[curr_tree].mean[col],
                            (model_outputs.missing_action == Fail)?  workspace.comb_val[0] : hplanes[curr_tree].fill_val[col],
                            model_outputs.missing_action, NULL, NULL, false);
    }

    /* divide data */
    size_t split_ix = divide_subset_split(workspace.ix_arr.data(), workspace.comb_val.data(),
                                          workspace.st, workspace.end, hplanes[curr_tree].split_point);

    /* continue splitting recursively */
    size_t orig_end = workspace.end;
    if (split_ix > workspace.st)
    {
        workspace.end = split_ix - 1;
        traverse_hplane_sim(workspace,
                            prediction_data,
                            model_outputs,
                            hplanes,
                            hplanes[curr_tree].hplane_left);
    }

    if (split_ix < orig_end)
    {
        workspace.st  = split_ix;
        workspace.end = orig_end;
        traverse_hplane_sim(workspace,
                            prediction_data,
                            model_outputs,
                            hplanes,
                            hplanes[curr_tree].hplane_right);
    }

}

template <class PredictionData, class InputData, class WorkerMemory>
void gather_sim_result(std::vector<WorkerForSimilarity> *worker_memory,
                       std::vector<WorkerMemory> *worker_memory_m,
                       PredictionData *prediction_data, InputData *input_data,
                       IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                       double *restrict tmat, double *restrict rmat, size_t n_from,
                       size_t ntrees, bool assume_full_distr,
                       bool standardize_dist, int nthreads)
{
    if (interrupt_switch)
        return;
    
    size_t ncomb = (prediction_data != NULL)?
                    (prediction_data->nrows * (prediction_data->nrows - 1)) / 2
                        :
                    (input_data->nrows * (input_data->nrows - 1)) / 2;
    size_t n_to  = (prediction_data != NULL)? (prediction_data->nrows - n_from) : 0;

    #ifdef _OPENMP
    if (nthreads > 1)
    {
        if (worker_memory != NULL)
        {
            for (WorkerForSimilarity &w : *worker_memory)
            {
                if (w.tmat_sep.size())
                {
                    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(ncomb, tmat, w, worker_memory)
                    for (size_t_for ix = 0; ix < ncomb; ix++)
                        tmat[ix] += w.tmat_sep[ix];
                }
                else if (w.rmat.size())
                {
                    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(rmat, w, worker_memory)
                    for (size_t_for ix = 0; ix < w.rmat.size(); ix++)
                        rmat[ix] += w.rmat[ix];
                }
            }
        }

        else
        {
            for (WorkerMemory &w : *worker_memory_m)
            {
                if (w.tmat_sep.size())
                {
                    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(ncomb, tmat, w, worker_memory_m)
                    for (size_t_for ix = 0; ix < ncomb; ix++)
                        tmat[ix] += w.tmat_sep[ix];
                }
            }
        }
    }
    
    else
    #endif
    {
        if (worker_memory != NULL)
        {
            if ((*worker_memory)[0].tmat_sep.size())
                std::copy((*worker_memory)[0].tmat_sep.begin(), (*worker_memory)[0].tmat_sep.end(), tmat);
            else
                std::copy((*worker_memory)[0].rmat.begin(), (*worker_memory)[0].rmat.end(), rmat);
        }
        
        else
        {
            std::copy((*worker_memory_m)[0].tmat_sep.begin(), (*worker_memory_m)[0].tmat_sep.end(), tmat);
        }
    }

    double ntrees_dbl = (double) ntrees;
    if (standardize_dist)
    {
        /* Note: the separation distances up this point are missing the first hop, which is always
           a +1 to every combination. Thus, it needs to be added back for the average separation depth.
           For the standardized metric, it takes the expected divisor as 2(=3-1) instead of 3, given
           that every combination will always get a +1 at the beginning. Since what's obtained here
           is a sum across all trees, adding this +1 means adding the number of trees. */
        double div_trees = ntrees_dbl;
        if (assume_full_distr)
        {
            div_trees *= 2;
        }

        else if (input_data != NULL)
        {
            div_trees *= (expected_separation_depth(input_data->nrows) - 1);
        }

        else
        {
            div_trees *= ((
                               (model_outputs != NULL)?
                                expected_separation_depth_hotstart(model_outputs->exp_avg_sep,
                                                                    model_outputs->orig_sample_size,
                                                                    model_outputs->orig_sample_size + prediction_data->nrows)
                                    :
                                expected_separation_depth_hotstart(model_outputs_ext->exp_avg_sep,
                                                                    model_outputs_ext->orig_sample_size,
                                                                    model_outputs_ext->orig_sample_size + prediction_data->nrows)
                          ) - 1);
        }

        
        if (tmat != NULL)
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(ncomb, tmat, ntrees_dbl, div_trees)
            for (size_t_for ix = 0; ix < ncomb; ix++)
                tmat[ix] = exp2( - tmat[ix] / div_trees);
        else
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(ncomb, rmat, ntrees_dbl, div_trees)
            for (size_t_for ix = 0; ix < n_from * n_to; ix++)
                rmat[ix] = exp2( - rmat[ix] / div_trees);
    }
    
    else
    {
        if (tmat != NULL)
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(ncomb, tmat, ntrees_dbl)
            for (size_t_for ix = 0; ix < ncomb; ix++)
                tmat[ix] = (tmat[ix] + ntrees) / ntrees_dbl;
        else
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(n_from, rmat, ntrees_dbl)
            for (size_t_for ix = 0; ix < n_from * n_to; ix++)
                rmat[ix] = (rmat[ix] + ntrees) / ntrees_dbl;
    }
}

template <class PredictionData>
void initialize_worker_for_sim(WorkerForSimilarity  &workspace,
                               PredictionData       &prediction_data,
                               IsoForest            *model_outputs,
                               ExtIsoForest         *model_outputs_ext,
                               size_t                n_from,
                               bool                  assume_full_distr)
{
    workspace.st  = 0;
    workspace.end = prediction_data.nrows - 1;
    workspace.n_from = n_from;
    workspace.assume_full_distr = assume_full_distr; /* doesn't need to have one copy per worker */

    if (!workspace.ix_arr.size())
    {
        workspace.ix_arr.resize(prediction_data.nrows);
        std::iota(workspace.ix_arr.begin(), workspace.ix_arr.end(), (size_t)0);
        if (!n_from)
          workspace.tmat_sep.resize((prediction_data.nrows * (prediction_data.nrows - 1)) / 2, 0);
        else
          workspace.rmat.resize(prediction_data.nrows * n_from, 0);
    }

    if (model_outputs != NULL && (model_outputs->missing_action == Divide || model_outputs->new_cat_action == Weighted))
    {
        if (!workspace.weights_arr.size())
            workspace.weights_arr.resize(prediction_data.nrows, 1);
        else
            std::fill(workspace.weights_arr.begin(), workspace.weights_arr.end(), 1);
    }

    if (model_outputs_ext != NULL)
    {
        if (!workspace.comb_val.size())
            workspace.comb_val.resize(prediction_data.nrows, 0);
        else
            std::fill(workspace.comb_val.begin(), workspace.comb_val.end(), 0);
    }
}
