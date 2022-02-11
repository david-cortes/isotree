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


/* for use in regular model */
template <class InputData, class WorkerMemory>
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params, IsoTree &tree)
{
    if (tree.col_num < input_data.ncols_numeric)
    {
        tree.col_type = Numeric;

        if (input_data.Xc_indptr == NULL)
            get_range(workspace.ix_arr.data(), input_data.numeric_data + input_data.nrows * tree.col_num,
                      workspace.st, workspace.end, model_params.missing_action,
                      workspace.xmin, workspace.xmax, workspace.unsplittable);
        else
            get_range(workspace.ix_arr.data(), workspace.st, workspace.end, tree.col_num,
                      input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                      model_params.missing_action, workspace.xmin, workspace.xmax, workspace.unsplittable);
    }

    else
    {
        tree.col_num -= input_data.ncols_numeric;
        tree.col_type = Categorical;

        get_categs(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * tree.col_num,
                   workspace.st, workspace.end, input_data.ncat[tree.col_num],
                   model_params.missing_action, workspace.categs.data(), workspace.npresent, workspace.unsplittable);
    }
}

/* for use in extended model */
template <class InputData, class WorkerMemory>
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params)
{
    if (workspace.col_chosen < input_data.ncols_numeric)
    {
        workspace.col_type = Numeric;

        if (input_data.Xc_indptr == NULL)
            get_range(workspace.ix_arr.data(), input_data.numeric_data + input_data.nrows * workspace.col_chosen,
                      workspace.st, workspace.end, model_params.missing_action,
                      workspace.xmin, workspace.xmax, workspace.unsplittable);
        else
            get_range(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.col_chosen,
                      input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                      model_params.missing_action, workspace.xmin, workspace.xmax, workspace.unsplittable);
    }

    else
    {
        workspace.col_type = Categorical;
        workspace.col_chosen -= input_data.ncols_numeric;

        get_categs(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * workspace.col_chosen,
                   workspace.st, workspace.end, input_data.ncat[workspace.col_chosen],
                   model_params.missing_action, workspace.categs.data(), workspace.npresent, workspace.unsplittable);
    }
}

/* for use in regular model with ntry>1 */
template <class InputData, class WorkerMemory>
void get_split_range_v2(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params)
{
    get_split_range(workspace, input_data, model_params);
    if (workspace.col_type == Categorical)
        workspace.col_chosen += input_data.ncols_numeric;
}

template <class InputData, class WorkerMemory>
int choose_cat_from_present(WorkerMemory &workspace, InputData &input_data, size_t col_num)
{
    int chosen_cat = std::uniform_int_distribution<int>
                                    (0, workspace.npresent - 1)
                                    (workspace.rnd_generator);
    workspace.ncat_tried = 0;
    for (int cat = 0; cat < input_data.ncat[col_num]; cat++)
    {
        if (workspace.categs[cat] > 0)
        {
            if (workspace.ncat_tried == chosen_cat)
                return cat;
            else
                workspace.ncat_tried++;
        }
    }

    return -1; /* this will never be reached, but CRAN complains otherwise */
}

bool is_col_taken(std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s,
                  size_t col_num)
{
    if (!col_is_taken.empty())
        return col_is_taken[col_num];
    else
        return col_is_taken_s.find(col_num) != col_is_taken_s.end();
}

template <class InputData>
void set_col_as_taken(std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s,
                      InputData &input_data, size_t col_num, ColType col_type)
{
    col_num += ((col_type == Numeric)? 0 : input_data.ncols_numeric);
    if (!col_is_taken.empty())
        col_is_taken[col_num] = true;
    else
        col_is_taken_s.insert(col_num);
}

template <class InputData>
void set_col_as_taken(std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s,
                      InputData &input_data, size_t col_num)
{
    if (!col_is_taken.empty())
        col_is_taken[col_num] = true;
    else
        col_is_taken_s.insert(col_num);
}

template <class InputData, class WorkerMemory>
void add_separation_step(WorkerMemory &workspace, InputData &input_data, double remainder)
{
    if (!workspace.changed_weights)
        increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                              input_data.nrows, workspace.tmat_sep.data(), remainder);
    else if (!workspace.weights_arr.empty())
        increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                              input_data.nrows, workspace.tmat_sep.data(), workspace.weights_arr.data(), remainder);
    else
        increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                              input_data.nrows, workspace.tmat_sep.data(), workspace.weights_map, remainder);
}

template <class InputData, class WorkerMemory, class ldouble_safe>
void add_remainder_separation_steps(WorkerMemory &workspace, InputData &input_data, ldouble_safe sum_weight)
{
    if ((workspace.end - workspace.st) > 0 && (!workspace.changed_weights || sum_weight > 0))
    {
        double expected_dsep;
        if (!workspace.changed_weights)
            expected_dsep = expected_separation_depth(workspace.end - workspace.st + 1);
        else
            expected_dsep = expected_separation_depth(sum_weight);

        add_separation_step(workspace, input_data, expected_dsep + 1);
    }
}

template <class PredictionData, class sparse_ix>
void remap_terminal_trees(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                          PredictionData &prediction_data, sparse_ix *restrict tree_num, int nthreads)
{
    size_t ntrees = (model_outputs != NULL)? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
    size_t max_tree, curr_term;
    std::vector<sparse_ix> tree_mapping;
    if (model_outputs != NULL)
    {
        max_tree = std::accumulate(model_outputs->trees.begin(),
                                   model_outputs->trees.end(),
                                   (size_t)0,
                                   [](const size_t curr_max, const std::vector<IsoTree> &tr)
                                   {return std::max(curr_max, tr.size());});
        tree_mapping.resize(max_tree);
        for (size_t tree = 0; tree < ntrees; tree++)
        {
            std::fill(tree_mapping.begin(), tree_mapping.end(), (size_t)0);
            curr_term = 0;
            for (size_t node = 0; node < model_outputs->trees[tree].size(); node++)
                if (model_outputs->trees[tree][node].tree_left == 0)
                    tree_mapping[node] = curr_term++;

            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(tree_num, tree_mapping, tree, prediction_data)
            for (size_t_for row = 0; row < (decltype(row))prediction_data.nrows; row++)
                tree_num[row + tree * prediction_data.nrows] = tree_mapping[tree_num[row + tree * prediction_data.nrows]];
        }
    }

    else
    {
        max_tree = std::accumulate(model_outputs_ext->hplanes.begin(),
                                   model_outputs_ext->hplanes.end(),
                                   (size_t)0,
                                   [](const size_t curr_max, const std::vector<IsoHPlane> &tr)
                                   {return std::max(curr_max, tr.size());});
        tree_mapping.resize(max_tree);
        for (size_t tree = 0; tree < ntrees; tree++)
        {
            std::fill(tree_mapping.begin(), tree_mapping.end(), (size_t)0);
            curr_term = 0;
            for (size_t node = 0; node < model_outputs_ext->hplanes[tree].size(); node++)
                if (model_outputs_ext->hplanes[tree][node].hplane_left == 0)
                    tree_mapping[node] = curr_term++;
            
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(tree_num, tree_mapping, tree, prediction_data)
            for (size_t_for row = 0; row < (decltype(row))prediction_data.nrows; row++)
                tree_num[row + tree * prediction_data.nrows] = tree_mapping[tree_num[row + tree * prediction_data.nrows]];
        }
    }
}


template <class WorkerMemory>
RecursionState::RecursionState(WorkerMemory &workspace, bool full_state)
{
    this->full_state = full_state;

    this->split_ix         =  workspace.split_ix;
    this->end              =  workspace.end;
    if (!workspace.col_sampler.has_weights())
        this->sampler_pos  =  workspace.col_sampler.curr_pos;
    else {
        this->col_sampler_weights = workspace.col_sampler.tree_weights;
        this->n_dropped           = workspace.col_sampler.n_dropped;
    }

    if (this->full_state)
    {
        this->st           = workspace.st;
        this->st_NA        = workspace.st_NA;
        this->end_NA       = workspace.end_NA;

        this->changed_weights = workspace.changed_weights;

        /* for the extended model, it's not necessary to copy everything */
        if (workspace.comb_val.empty() && workspace.st_NA < workspace.end_NA)
        {
            this->ix_arr = std::vector<size_t>(workspace.ix_arr.begin() + workspace.st_NA,
                                               workspace.ix_arr.begin() + workspace.end_NA);
            if (this->changed_weights)
            {
                size_t tot = workspace.end_NA - workspace.st_NA;
                this->weights_arr = std::unique_ptr<double[]>(new double[tot]);
                if (!workspace.weights_arr.empty())
                    for (size_t ix = 0; ix < tot; ix++)
                        this->weights_arr[ix] = workspace.weights_arr[workspace.ix_arr[ix + workspace.st_NA]];
                else
                    for (size_t ix = 0; ix < tot; ix++)
                        this->weights_arr[ix] = workspace.weights_map[workspace.ix_arr[ix + workspace.st_NA]];
            }
        }
    }
}


template <class WorkerMemory>
void RecursionState::restore_state(WorkerMemory &workspace)
{
    workspace.split_ix         =  this->split_ix;
    workspace.end              =  this->end;
    if (!workspace.col_sampler.has_weights())
        workspace.col_sampler.curr_pos = this->sampler_pos;
    else  {
        workspace.col_sampler.tree_weights  =  std::move(this->col_sampler_weights);
        workspace.col_sampler.n_dropped     =  this->n_dropped;
    }

    if (this->full_state)
    {
        workspace.st         =  this->st;
        workspace.st_NA      =  this->st_NA;
        workspace.end_NA     =  this->end_NA;

        workspace.changed_weights = this->changed_weights;

        if (workspace.comb_val.empty() && !this->ix_arr.empty())
        {
            std::copy(this->ix_arr.begin(),
                      this->ix_arr.end(),
                      workspace.ix_arr.begin() + this->st_NA);
            if (this->changed_weights)
            {
                size_t tot = workspace.end_NA - workspace.st_NA;
                if (!workspace.weights_arr.empty())
                    for (size_t ix = 0; ix < tot; ix++)
                        workspace.weights_arr[workspace.ix_arr[ix + workspace.st_NA]] = this->weights_arr[ix];
                else
                    for (size_t ix = 0; ix < tot; ix++)
                        workspace.weights_map[workspace.ix_arr[ix + workspace.st_NA]] = this->weights_arr[ix];
            }
        }
    }
}

template <class InputData, class ldouble_safe>
std::vector<double> calc_kurtosis_all_data(InputData &input_data, ModelParams &model_params, RNG_engine &rnd_generator)
{
    std::unique_ptr<double[]> buffer_double;
    std::unique_ptr<size_t[]> buffer_size_t;
    if (input_data.ncols_categ)
    {
        buffer_double = std::unique_ptr<double[]>(new double[input_data.max_categ]);
        if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
            buffer_size_t = std::unique_ptr<size_t[]>(new size_t[input_data.max_categ + 1]);
    }


    std::vector<double> kurt_weights(input_data.ncols_numeric + input_data.ncols_categ);
    for (size_t col = 0; col < input_data.ncols_tot; col++)
    {
        if (col < input_data.ncols_numeric)
        {
            if (input_data.Xc_indptr == NULL)
            {
                if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
                {
                    kurt_weights[col]
                        = calc_kurtosis<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                                        input_data.numeric_data + col * input_data.nrows,
                                        input_data.nrows, model_params.missing_action);
                }

                else
                {
                    kurt_weights[col]
                        = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                 ldouble_safe>(
                                                 input_data.numeric_data + col * input_data.nrows, input_data.nrows,
                                                 model_params.missing_action, input_data.sample_weights);
                }
            }

            else
            {
                if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
                {
                    kurt_weights[col]
                        = calc_kurtosis<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                        typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                        ldouble_safe>(
                                        col, input_data.nrows,
                                        input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                        model_params.missing_action);
                }

                else
                {
                    kurt_weights[col]
                        = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                 typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                 ldouble_safe>(
                                                 col, input_data.nrows,
                                                 input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                 model_params.missing_action, input_data.sample_weights);
                }
            }
        }

        else
        {
            if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
            {
                kurt_weights[col]
                        = calc_kurtosis<ldouble_safe>(input_data.nrows,
                                        input_data.categ_data + (col- input_data.ncols_numeric) * input_data.nrows,
                                        input_data.ncat[col - input_data.ncols_numeric],
                                        buffer_size_t.get(), buffer_double.get(),
                                        model_params.missing_action, model_params.cat_split_type, rnd_generator);
            }

            else
            {
                kurt_weights[col]
                        = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.sample_weights)>::type,
                                                 ldouble_safe>(
                                                 input_data.nrows,
                                                 input_data.categ_data + (col- input_data.ncols_numeric) * input_data.nrows,
                                                 input_data.ncat[col - input_data.ncols_numeric],
                                                 buffer_double.get(),
                                                 model_params.missing_action, model_params.cat_split_type,
                                                 rnd_generator, input_data.sample_weights);
            }
        }
    }

    for (auto &w : kurt_weights) w = (w == -HUGE_VAL)? 0. : std::fmax(1e-8, -1. + w);
    if (input_data.col_weights != NULL)
    {
        for (size_t col = 0; col < input_data.ncols_tot; col++)
        {
            if (kurt_weights[col] <= 0) continue;
            kurt_weights[col] *= input_data.col_weights[col];
            kurt_weights[col] = std::fmax(kurt_weights[col], 1e-100);
        }
    }

    return kurt_weights;
}

template <class InputData, class WorkerMemory>
void calc_ranges_all_cols(InputData &input_data, WorkerMemory &workspace, ModelParams &model_params,
                          double *restrict ranges, double *restrict saved_xmin, double *restrict saved_xmax)
{
    workspace.col_sampler.prepare_full_pass();
    while (workspace.col_sampler.sample_col(workspace.col_chosen))
    {
        get_split_range(workspace, input_data, model_params);

        if (workspace.unsplittable) {
            workspace.col_sampler.drop_col(workspace.col_chosen);
            ranges[workspace.col_chosen] = 0;
            if (saved_xmin != NULL) {
                saved_xmin[workspace.col_chosen] = 0;
                saved_xmax[workspace.col_chosen] = 0;
            }
        }
        else {
            ranges[workspace.col_chosen] = workspace.xmax - workspace.xmin;
            if (workspace.tree_kurtoses != NULL) {
                ranges[workspace.col_chosen] *= workspace.tree_kurtoses[workspace.col_chosen];
                ranges[workspace.col_chosen] = std::fmax(ranges[workspace.col_chosen], 1e-100);
            }
            else if (input_data.col_weights != NULL) {
                ranges[workspace.col_chosen] *= input_data.col_weights[workspace.col_chosen];
                ranges[workspace.col_chosen] = std::fmax(ranges[workspace.col_chosen], 1e-100);
            }
            if (saved_xmin != NULL) {
                saved_xmin[workspace.col_chosen] = workspace.xmin;
                saved_xmax[workspace.col_chosen] = workspace.xmax;
            }
        }
    }
}

template <class InputData, class WorkerMemory, class ldouble_safe>
void calc_var_all_cols(InputData &input_data, WorkerMemory &workspace, ModelParams &model_params,
                       double *restrict variances, double *restrict saved_xmin, double *restrict saved_xmax,
                       double *restrict saved_means, double *restrict saved_sds)
{
    double xmean, xsd;
    if (saved_means != NULL)
        workspace.has_saved_stats = true;

    workspace.col_sampler.prepare_full_pass();
    while (workspace.col_sampler.sample_col(workspace.col_chosen))
    {
        if (workspace.col_chosen < input_data.ncols_numeric)
        {
            get_split_range(workspace, input_data, model_params);
            if (workspace.unsplittable)
            {
                workspace.col_sampler.drop_col(workspace.col_chosen);
                variances[workspace.col_chosen] = 0;
                if (saved_xmin != NULL)
                {
                    saved_xmin[workspace.col_chosen] = 0;
                    saved_xmax[workspace.col_chosen] = 0;
                }
                continue;
            }

            if (saved_xmin != NULL)
            {
                saved_xmin[workspace.col_chosen] = workspace.xmin;
                saved_xmax[workspace.col_chosen] = workspace.xmax;
            }


            if (input_data.Xc_indptr == NULL)
            {
                if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                {
                    calc_mean_and_sd<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                                     workspace.ix_arr.data(), workspace.st, workspace.end,
                                     input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                     model_params.missing_action, xsd, xmean);
                }

                else if (!workspace.weights_arr.empty())
                {
                    calc_mean_and_sd_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                              decltype(workspace.weights_arr), ldouble_safe>(
                                              workspace.ix_arr.data(), workspace.st, workspace.end,
                                              input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                              workspace.weights_arr,
                                              model_params.missing_action, xsd, xmean);
                }

                else
                {
                    calc_mean_and_sd_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                              decltype(workspace.weights_map), ldouble_safe>(
                                              workspace.ix_arr.data(), workspace.st, workspace.end,
                                              input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                              workspace.weights_map,
                                              model_params.missing_action, xsd, xmean);
                }
            }

            else
            {
                if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                {
                    calc_mean_and_sd<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                     typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                     ldouble_safe>(
                                     workspace.ix_arr.data(), workspace.st, workspace.end,
                                     workspace.col_chosen,
                                     input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                     xsd, xmean);
                }

                else if (!workspace.weights_arr.empty())
                {
                    calc_mean_and_sd_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                              typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                              decltype(workspace.weights_arr), ldouble_safe>(
                                              workspace.ix_arr.data(), workspace.st, workspace.end,
                                              workspace.col_chosen,
                                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                              xsd, xmean, workspace.weights_arr);
                }

                else
                {
                    calc_mean_and_sd_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                              typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                              decltype(workspace.weights_map), ldouble_safe>(
                                              workspace.ix_arr.data(), workspace.st, workspace.end,
                                              workspace.col_chosen,
                                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                              xsd, xmean, workspace.weights_map);
                }
            }

            if (saved_means != NULL) saved_means[workspace.col_chosen] = xmean;
            if (saved_sds != NULL) saved_sds[workspace.col_chosen] = xsd;
        }

        else
        {
            size_t col = workspace.col_chosen - input_data.ncols_numeric;
            if (workspace.weights_arr.empty() && workspace.weights_map.empty())
            {
                if (workspace.buffer_szt.size() < (size_t)2 * (size_t)input_data.ncat[col] + 1)
                    workspace.buffer_szt.resize((size_t)2 * (size_t)input_data.ncat[col] + 1);
                xsd = expected_sd_cat<size_t, ldouble_safe>(
                                      workspace.ix_arr.data(), workspace.st, workspace.end,
                                      input_data.categ_data + col * input_data.nrows,
                                      input_data.ncat[col],
                                      model_params.missing_action,
                                      workspace.buffer_szt.data(),
                                      workspace.buffer_szt.data() + input_data.ncat[col] + 1,
                                      workspace.buffer_dbl.data());
            }

            else if (!workspace.weights_arr.empty())
            {
                if (workspace.buffer_dbl.size() < (size_t)2 * (size_t)input_data.ncat[col] + 1)
                    workspace.buffer_dbl.resize((size_t)2 * (size_t)input_data.ncat[col] + 1);
                xsd = expected_sd_cat_weighted<decltype(workspace.weights_arr), size_t, ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows,
                                               input_data.ncat[col],
                                               model_params.missing_action, workspace.weights_arr,
                                               workspace.buffer_dbl.data(),
                                               workspace.buffer_szt.data(),
                                               workspace.buffer_dbl.data() + input_data.ncat[col] + 1);
            }

            else
            {
                if (workspace.buffer_dbl.size() < (size_t)2 * (size_t)input_data.ncat[col] + 1)
                    workspace.buffer_dbl.resize((size_t)2 * (size_t)input_data.ncat[col] + 1);
                xsd = expected_sd_cat_weighted<decltype(workspace.weights_map), size_t, ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows,
                                               input_data.ncat[col],
                                               model_params.missing_action, workspace.weights_map,
                                               workspace.buffer_dbl.data(),
                                               workspace.buffer_szt.data(),
                                               workspace.buffer_dbl.data() + input_data.ncat[col] + 1);
            }
        }

        if (xsd)
        {
            variances[workspace.col_chosen] = square(xsd);
            if (workspace.tree_kurtoses != NULL)
                variances[workspace.col_chosen] *= workspace.tree_kurtoses[workspace.col_chosen];
            else if (input_data.col_weights != NULL)
                variances[workspace.col_chosen] *= input_data.col_weights[workspace.col_chosen];
            variances[workspace.col_chosen] = std::fmax(variances[workspace.col_chosen], 1e-100);
        }

        else
        {
            workspace.col_sampler.drop_col(workspace.col_chosen);
            variances[workspace.col_chosen] = 0;
        }
    }
}

template <class InputData, class WorkerMemory, class ldouble_safe>
void calc_kurt_all_cols(InputData &input_data, WorkerMemory &workspace, ModelParams &model_params,
                        double *restrict kurtosis, double *restrict saved_xmin, double *restrict saved_xmax)
{
    workspace.col_sampler.prepare_full_pass();
    while (workspace.col_sampler.sample_col(workspace.col_chosen))
    {
        if (saved_xmin != NULL)
        {
            get_split_range(workspace, input_data, model_params);
            if (workspace.unsplittable)
            {
                workspace.col_sampler.drop_col(workspace.col_chosen);
                continue;
            }

            if (saved_xmin != NULL)
            {
                saved_xmin[workspace.col_chosen] = workspace.xmin;
                saved_xmax[workspace.col_chosen] = workspace.xmax;
            }
        }

        if (workspace.col_chosen < input_data.ncols_numeric)
        {
            if (input_data.Xc_indptr == NULL)
            {
                if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                {
                    kurtosis[workspace.col_chosen] =
                        calc_kurtosis<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                      ldouble_safe>(
                                      workspace.ix_arr.data(), workspace.st, workspace.end,
                                      input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                      model_params.missing_action);
                }

                else if (!workspace.weights_arr.empty())
                {
                    kurtosis[workspace.col_chosen] =
                        calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                               decltype(workspace.weights_arr), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                               model_params.missing_action, workspace.weights_arr);
                }

                else
                {
                    kurtosis[workspace.col_chosen] =
                        calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                               decltype(workspace.weights_map), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                               model_params.missing_action, workspace.weights_map);
                }
            }

            else
            {
                if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                {
                    kurtosis[workspace.col_chosen] =
                        calc_kurtosis<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                      typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                      ldouble_safe>(
                                      workspace.ix_arr.data(), workspace.st, workspace.end,
                                      workspace.col_chosen,
                                      input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                      model_params.missing_action);
                }

                else if (!workspace.weights_arr.empty())
                {
                    kurtosis[workspace.col_chosen] =
                        calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                               typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                               decltype(workspace.weights_arr), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               workspace.col_chosen,
                                               input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                               model_params.missing_action, workspace.weights_arr);
                }

                else
                {
                    kurtosis[workspace.col_chosen] =
                        calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                               typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                               decltype(workspace.weights_map), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               workspace.col_chosen,
                                               input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                               model_params.missing_action, workspace.weights_map);
                }
            }
        }

        else
        {
            size_t col = workspace.col_chosen - input_data.ncols_numeric;
            if (workspace.weights_arr.empty() && workspace.weights_map.empty())
            {
                kurtosis[workspace.col_chosen] =
                    calc_kurtosis<ldouble_safe>(
                                  workspace.ix_arr.data(), workspace.st, workspace.end,
                                  input_data.categ_data + col * input_data.nrows,
                                  input_data.ncat[col],
                                  workspace.buffer_szt.data(), workspace.buffer_dbl.data(),
                                  model_params.missing_action, model_params.cat_split_type,
                                  workspace.rnd_generator);
            }

            else if (!workspace.weights_arr.empty())
            {
                kurtosis[workspace.col_chosen] =
                    calc_kurtosis_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                           workspace.ix_arr.data(), workspace.st, workspace.end,
                                           input_data.categ_data + col * input_data.nrows,
                                           input_data.ncat[col],
                                           workspace.buffer_dbl.data(),
                                           model_params.missing_action, model_params.cat_split_type,
                                           workspace.rnd_generator, workspace.weights_arr);
            }

            else
            {
                kurtosis[workspace.col_chosen] =
                    calc_kurtosis_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                           workspace.ix_arr.data(), workspace.st, workspace.end,
                                           input_data.categ_data + col * input_data.nrows,
                                           input_data.ncat[col],
                                           workspace.buffer_dbl.data(),
                                           model_params.missing_action, model_params.cat_split_type,
                                           workspace.rnd_generator, workspace.weights_map);
            }
        }

        if (kurtosis[workspace.col_chosen] == -HUGE_VAL)
            workspace.col_sampler.drop_col(workspace.col_chosen);

        kurtosis[workspace.col_chosen] = (kurtosis[workspace.col_chosen] == -HUGE_VAL)?
                                            0. : std::fmax(1e-8, -1. + kurtosis[workspace.col_chosen]);
        if (input_data.col_weights != NULL && kurtosis[workspace.col_chosen] > 0)
        {
            kurtosis[workspace.col_chosen] *= input_data.col_weights[workspace.col_chosen];
            kurtosis[workspace.col_chosen] = std::fmax(kurtosis[workspace.col_chosen], 1e-100);
        }
    }
}

bool is_boxed_metric(const ScoringMetric scoring_metric)
{
    return scoring_metric == BoxedDensity ||
           scoring_metric == BoxedDensity2 ||
           scoring_metric == BoxedRatio;
}
