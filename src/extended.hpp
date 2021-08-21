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

template <class InputData, class WorkerMemory>
void split_hplane_recursive(std::vector<IsoHPlane>   &hplanes,
                            WorkerMemory             &workspace,
                            InputData                &input_data,
                            ModelParams              &model_params,
                            std::vector<ImputeNode> *impute_nodes,
                            size_t                   curr_depth)
{
    if (interrupt_switch) return;
    long double sum_weight = -HUGE_VAL;
    size_t hplane_from = hplanes.size() - 1;
    std::unique_ptr<RecursionState> recursion_state;
    std::vector<bool> col_is_taken;
    hashed_set<size_t> col_is_taken_s;

    /* calculate imputation statistics if desired */
    if (impute_nodes != NULL)
    {
        if (input_data.Xc_indptr != NULL)
            std::sort(workspace.ix_arr.begin() + workspace.st,
                      workspace.ix_arr.begin() + workspace.end + 1);
        build_impute_node(impute_nodes->back(), workspace,
                          input_data, model_params,
                          *impute_nodes, curr_depth,
                          model_params.min_imp_obs);
    }

    /* check for potential isolated leafs or unique splits */
    if (workspace.end == workspace.st || (workspace.end - workspace.st) == 1 || curr_depth >= model_params.max_depth)
        goto terminal_statistics;

    /* when using weights, the split should stop when the sum of weights is <= 1 */
    sum_weight = calculate_sum_weights(workspace.ix_arr, workspace.st, workspace.end, curr_depth,
                                       workspace.weights_arr, workspace.weights_map);

    if (curr_depth > 0 && (workspace.weights_arr.size() || workspace.weights_map.size()) && sum_weight <= 1)
        goto terminal_statistics;

    /* for sparse matrices, need to sort the indices */
    if (input_data.Xc_indptr != NULL && impute_nodes == NULL)
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);

    /* pick column to split according to criteria */
    workspace.prob_split_type = workspace.rbin(workspace.rnd_generator);

    if (
            workspace.prob_split_type
                < (
                    model_params.prob_pick_by_gain_avg +
                    model_params.prob_pick_by_gain_pl
                  )
        )
    {
        workspace.ntry = model_params.ntry;
        hplanes.back().score = -HUGE_VAL; /* this keeps track of the gain */
        if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg)
            workspace.criterion = Averaged;
        else
            workspace.criterion = Pooled;
    }

    else
    {
        workspace.criterion = NoCrit;
        workspace.ntry = 1;
    }

    if (workspace.criterion != NoCrit && (workspace.weights_arr.size() || workspace.weights_map.size()))
    {
        if (workspace.weights_arr.size())
        {
            for (size_t row = workspace.st; row <= workspace.end; row++)
                workspace.sample_weights[row-workspace.st] = workspace.weights_arr[workspace.ix_arr[row]];
        }

        else
        {
            for (size_t row = workspace.st; row <= workspace.end; row++)
                workspace.sample_weights[row-workspace.st] = workspace.weights_map[workspace.ix_arr[row]];
        }
    }

    workspace.ntaken_best = 0;

    for (size_t attempt = 0; attempt < workspace.ntry; attempt++)
    {
        if (input_data.ncols_tot < 1e5 ||
            ((long double)model_params.ndim / (long double)workspace.col_sampler.get_remaining_cols()) > .25
            )
        {
            if (!col_is_taken.size())
                col_is_taken.resize(input_data.ncols_tot, false);
            else
                col_is_taken.assign(input_data.ncols_tot, false);
        }
        else {
            col_is_taken_s.clear();
            col_is_taken_s.reserve(model_params.ndim);
        }
        workspace.ntaken = 0;
        workspace.ntried = 0;
        std::fill(workspace.comb_val.begin(),
                  workspace.comb_val.begin() + (workspace.end - workspace.st + 1),
                  (double)0);

        if (model_params.ndim >= input_data.ncols_tot)
            workspace.col_sampler.prepare_full_pass();
        else if (workspace.try_all)
            workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);
        size_t threshold_shuffle = (workspace.col_sampler.get_remaining_cols() + 1) / 2;

        while (
                workspace.try_all?
                workspace.col_sampler.sample_col(workspace.col_chosen)
                    :
                workspace.col_sampler.sample_col(workspace.col_chosen, workspace.rnd_generator)
            )
        {
            if (interrupt_switch) return;
            
            workspace.ntried++;
            if (!workspace.try_all && workspace.ntried >= threshold_shuffle)
            {
                workspace.try_all = true;
                workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);
            }

            if (is_col_taken(col_is_taken, col_is_taken_s, workspace.col_chosen))
                continue;

            get_split_range(workspace, input_data, model_params);
            if (workspace.unsplittable)
            {
                workspace.col_sampler.drop_col(workspace.col_chosen
                                                 +
                                               ((workspace.col_type == Numeric)?
                                                 (size_t)0 : input_data.ncols_numeric));
            }

            else
            {
                add_chosen_column(workspace, input_data, model_params, col_is_taken, col_is_taken_s);
                if (++workspace.ntaken >= model_params.ndim)
                    break;
            }
        }

        if (!workspace.ntaken && !workspace.ntaken_best)
            goto terminal_statistics;
        else if (!workspace.ntaken)
            break;


        /* evaluate gain if necessary */
        if (workspace.criterion != NoCrit)
        {
            if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                workspace.this_gain = eval_guided_crit(workspace.comb_val.data(), workspace.end - workspace.st + 1,
                                                       workspace.criterion, model_params.min_gain, workspace.ntry == 1,
                                                       workspace.buffer_dbl.data(), workspace.this_split_point,
                                                       workspace.xmin, workspace.xmax);
            else if (workspace.weights_arr.size())
                workspace.this_gain = eval_guided_crit_weighted(workspace.comb_val.data(), workspace.end - workspace.st + 1,
                                                                workspace.criterion, model_params.min_gain, workspace.ntry == 1,
                                                                workspace.buffer_dbl.data(), workspace.this_split_point,
                                                                workspace.xmin, workspace.xmax,
                                                                workspace.sample_weights.data(), workspace.buffer_szt.data());
            else
                workspace.this_gain = eval_guided_crit_weighted(workspace.comb_val.data(), workspace.end - workspace.st + 1,
                                                                workspace.criterion, model_params.min_gain, workspace.ntry == 1,
                                                                workspace.buffer_dbl.data(), workspace.this_split_point,
                                                                workspace.xmin, workspace.xmax,
                                                                workspace.sample_weights.data(), workspace.buffer_szt.data());
        }
        
        /* pass to the output object */
        if (workspace.ntry == 1 || workspace.this_gain > hplanes.back().score)
        {
            /* these should be shrunk later according to what ends up used */
            hplanes.back().score = workspace.this_gain;
            workspace.ntaken_best = workspace.ntaken;
            if (workspace.criterion != NoCrit)
            {
                hplanes.back().split_point = workspace.this_split_point;
                if (model_params.penalize_range)
                {
                    hplanes.back().range_low  = workspace.xmin - workspace.xmax + hplanes.back().split_point;
                    hplanes.back().range_high = workspace.xmax - workspace.xmin + hplanes.back().split_point;
                }
            }
            hplanes.back().col_num.assign(workspace.col_take.begin(), workspace.col_take.begin() + workspace.ntaken);
            hplanes.back().col_type.assign(workspace.col_take_type.begin(), workspace.col_take_type.begin() + workspace.ntaken);
            if (input_data.ncols_numeric)
            {
                hplanes.back().coef.assign(workspace.ext_coef.begin(), workspace.ext_coef.begin() + workspace.ntaken);
                hplanes.back().mean.assign(workspace.ext_mean.begin(), workspace.ext_mean.begin() + workspace.ntaken);
            }

            if (model_params.missing_action != Fail)
                hplanes.back().fill_val.assign(workspace.ext_fill_val.begin(), workspace.ext_fill_val.begin() + workspace.ntaken);

            if (input_data.ncols_categ)
            {
                hplanes.back().fill_new.assign(workspace.ext_fill_new.begin(), workspace.ext_fill_new.begin() + workspace.ntaken);
                switch(model_params.cat_split_type)
                {
                    case SingleCateg:
                    {
                        hplanes.back().chosen_cat.assign(workspace.chosen_cat.begin(),
                                                         workspace.chosen_cat.begin() + workspace.ntaken);
                        break;
                    }

                    case SubSet:
                    {
                        if (hplanes.back().cat_coef.size() < workspace.ntaken)
                             hplanes.back().cat_coef.assign(workspace.ext_cat_coef.begin(),
                                                            workspace.ext_cat_coef.begin() + workspace.ntaken);
                        else
                            for (size_t col = 0; col < workspace.ntaken_best; col++)
                                std::copy(workspace.ext_cat_coef[col].begin(),
                                          workspace.ext_cat_coef[col].end(),
                                          hplanes.back().cat_coef[col].begin());
                        break;
                    }
                }
            }
        }

    }

    col_is_taken.clear();
    col_is_taken.shrink_to_fit();
    col_is_taken_s.clear();

    /* if the best split is not good enough, don't split any further */
    if (workspace.criterion != NoCrit && hplanes.back().score <= 0)
        goto terminal_statistics;
    
    /* now need to reproduce the same split from before */
    if (workspace.criterion != NoCrit)
    {
        std::fill(workspace.comb_val.begin(),
                  workspace.comb_val.begin() + (workspace.end - workspace.st + 1),
                  (double)0);
        for (size_t col = 0; col < workspace.ntaken_best; col++)
        {
            switch(hplanes.back().col_type[col])
            {
                case Numeric:
                {
                    if (input_data.Xc_indptr == NULL)
                    {
                        add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                        input_data.numeric_data + hplanes.back().col_num[col] * input_data.nrows,
                                        hplanes.back().coef[col], (double)0, hplanes.back().mean[col],
                                        hplanes.back().fill_val.size()? hplanes.back().fill_val[col] : workspace.this_split_point, /* second case is not used */
                                        model_params.missing_action, NULL, NULL, false);
                    }

                    else
                    {
                        add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end,
                                        hplanes.back().col_num[col], workspace.comb_val.data(),
                                        input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                        hplanes.back().coef[col], (double)0, hplanes.back().mean[col],
                                        hplanes.back().fill_val.size()? hplanes.back().fill_val[col] : workspace.this_split_point, /* second case is not used */
                                        model_params.missing_action, NULL, NULL, false);
                    }

                    break;
                }

                case Categorical:
                {
                    add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                    input_data.categ_data + hplanes.back().col_num[col] * input_data.nrows,
                                    input_data.ncat[hplanes.back().col_num[col]],
                                    (model_params.cat_split_type == SubSet)? hplanes.back().cat_coef[col].data() : NULL,
                                    (model_params.cat_split_type == SingleCateg)? hplanes.back().fill_new[col] : (double)0,
                                    (model_params.cat_split_type == SingleCateg)? hplanes.back().chosen_cat[col] : 0,
                                    (hplanes.back().fill_val.size())? hplanes.back().fill_val[col] : workspace.this_split_point, /* second case is not used */
                                    (model_params.cat_split_type == SubSet)? hplanes.back().fill_new[col] : workspace.this_split_point, /* second case is not used */
                                    NULL, NULL, model_params.new_cat_action, model_params.missing_action,
                                    model_params.cat_split_type, false);
                    break;
                }

                default:
                {
                    unexpected_error();
                    break;
                }
            }
        }
    }

    /* get the range */
    if (workspace.criterion == NoCrit)
    {
        workspace.xmin =  HUGE_VAL;
        workspace.xmax = -HUGE_VAL;
        for (size_t row = 0; row < (workspace.end - workspace.st + 1); row++)
        {
            workspace.xmin = (workspace.xmin > workspace.comb_val[row])? workspace.comb_val[row] : workspace.xmin;
            workspace.xmax = (workspace.xmax < workspace.comb_val[row])? workspace.comb_val[row] : workspace.xmax;
        }
        if (workspace.xmin == workspace.xmax)
            goto terminal_statistics; /* in theory, could try again too, this could just be an unlucky case */
        
        hplanes.back().split_point =
            std::uniform_real_distribution<double>(workspace.xmin, workspace.xmax)
                                                  (workspace.rnd_generator);

        /* determine acceptable range */
        if (model_params.penalize_range)
        {
            hplanes.back().range_low  = workspace.xmin - workspace.xmax + hplanes.back().split_point;
            hplanes.back().range_high = workspace.xmax - workspace.xmin + hplanes.back().split_point;
        }
    }

    if (model_params.missing_action == Fail && is_na_or_inf(hplanes.back().split_point))
        throw std::runtime_error("Data has missing values. Try using a different value for 'missing_action'.\n");

    /* divide */
    workspace.split_ix = divide_subset_split(workspace.ix_arr.data(), workspace.comb_val.data(),
                                             workspace.st, workspace.end, hplanes.back().split_point);

    /* set as non-terminal */
    hplanes.back().score = -1;

    /* add another round of separation depth for distance */
    if (model_params.calc_dist && curr_depth > 0)
        add_separation_step(workspace, input_data, (double)(-1));

    /* simplify vectors according to what ends up used */
    if (input_data.ncols_categ || workspace.ntaken_best < model_params.ndim)
        simplify_hplane(hplanes.back(), workspace, input_data, model_params);

    shrink_to_fit_hplane(hplanes.back(), false);

    /* now split */

    /* back-up where it was */
    recursion_state = std::unique_ptr<RecursionState>(new RecursionState(workspace, true));

    /* follow left branch */
    hplanes[hplane_from].hplane_left = hplanes.size();
    hplanes.emplace_back();
    if (impute_nodes != NULL) impute_nodes->emplace_back(hplane_from);
    workspace.end = workspace.split_ix - 1;
    split_hplane_recursive(hplanes,
                           workspace,
                           input_data,
                           model_params,
                           impute_nodes,
                           curr_depth + 1);


    /* follow right branch */
    hplanes[hplane_from].hplane_right = hplanes.size();
    recursion_state->restore_state(workspace);
    hplanes.emplace_back();
    if (impute_nodes != NULL) impute_nodes->emplace_back(hplane_from);
    workspace.st = workspace.split_ix;
    split_hplane_recursive(hplanes,
                           workspace,
                           input_data,
                           model_params,
                           impute_nodes,
                           curr_depth + 1);

    return;

    terminal_statistics:
    {
        if (!workspace.weights_arr.size() && !workspace.weights_map.size())
        {
            hplanes.back().score = (double)(curr_depth + expected_avg_depth(workspace.end - workspace.st + 1));
        }

        else
        {
            if (sum_weight == -HUGE_VAL)
                sum_weight = calculate_sum_weights(workspace.ix_arr, workspace.st, workspace.end, curr_depth,
                                                   workspace.weights_arr, workspace.weights_map);
            hplanes.back().score = (double)(curr_depth + expected_avg_depth(sum_weight));
        }

        /* don't leave any vector initialized */
        shrink_to_fit_hplane(hplanes.back(), true);

        hplanes.back().remainder = workspace.weights_arr.size()?
                                   sum_weight : (workspace.weights_map.size()?
                                                 sum_weight : ((double)(workspace.end - workspace.st + 1))
                                                 );

        /* for distance, assume also the elements keep being split */
        if (model_params.calc_dist)
            add_remainder_separation_steps(workspace, input_data, sum_weight);

        /* add this depth right away if requested */
        if (workspace.row_depths.size())
            for (size_t row = workspace.st; row <= workspace.end; row++)
                workspace.row_depths[workspace.ix_arr[row]] += hplanes.back().score;

        /* add imputations from node if requested */
        if (model_params.impute_at_fit)
            add_from_impute_node(impute_nodes->back(), workspace, input_data);
    }
}


template <class InputData, class WorkerMemory>
void add_chosen_column(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params,
                       std::vector<bool> &col_is_taken, hashed_set<size_t> &col_is_taken_s)
{
    set_col_as_taken(col_is_taken, col_is_taken_s, input_data, workspace.col_chosen, workspace.col_type);
    workspace.col_take[workspace.ntaken]      = workspace.col_chosen;
    workspace.col_take_type[workspace.ntaken] = workspace.col_type;

    switch(workspace.col_type)
    {
        case Numeric:
        {
            switch(model_params.coef_type)
            {
                case Uniform:
                {
                    workspace.ext_coef[workspace.ntaken] = workspace.coef_unif(workspace.rnd_generator);
                    break;
                }

                case Normal:
                {
                    workspace.ext_coef[workspace.ntaken] = workspace.coef_norm(workspace.rnd_generator);
                    break;
                }
            }

            if (input_data.Xc_indptr == NULL)
            {
                if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                {
                    calc_mean_and_sd(workspace.ix_arr.data(), workspace.st, workspace.end,
                                     input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                     model_params.missing_action, workspace.ext_sd, workspace.ext_mean[workspace.ntaken]);
                    add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                    input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                    workspace.ext_coef[workspace.ntaken], workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                    workspace.ext_fill_val[workspace.ntaken], model_params.missing_action,
                                    workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true);
                }
                else if (workspace.weights_arr.size())
                {
                    calc_mean_and_sd_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                              input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                              workspace.weights_arr,
                                              model_params.missing_action, workspace.ext_sd,
                                              workspace.ext_mean[workspace.ntaken]);
                    add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                             input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                             workspace.ext_coef[workspace.ntaken], workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                             workspace.ext_fill_val[workspace.ntaken], model_params.missing_action,
                                             workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true,
                                             workspace.weights_arr);
                }

                else
                {
                    calc_mean_and_sd_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                              input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                              workspace.weights_map,
                                              model_params.missing_action, workspace.ext_sd,
                                              workspace.ext_mean[workspace.ntaken]);
                    add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                             input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                             workspace.ext_coef[workspace.ntaken], workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                             workspace.ext_fill_val[workspace.ntaken], model_params.missing_action,
                                             workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true,
                                             workspace.weights_map);
                }
            }

            else
            {
                if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                {
                    calc_mean_and_sd(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.col_chosen,
                                     input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                     workspace.ext_sd, workspace.ext_mean[workspace.ntaken]);
                    add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end,
                                    workspace.col_chosen, workspace.comb_val.data(),
                                    input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                    workspace.ext_coef[workspace.ntaken], workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                    workspace.ext_fill_val[workspace.ntaken], model_params.missing_action,
                                    workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true);
                }

                else if (workspace.weights_arr.size())
                {
                    calc_mean_and_sd_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.col_chosen,
                                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                              workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                              workspace.weights_arr);
                    add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                             workspace.col_chosen, workspace.comb_val.data(),
                                             input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                             workspace.ext_coef[workspace.ntaken], workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                             workspace.ext_fill_val[workspace.ntaken], model_params.missing_action,
                                             workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true,
                                             workspace.weights_arr);
                }

                else
                {
                    calc_mean_and_sd_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.col_chosen,
                                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                              workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                              workspace.weights_map);
                    add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end,
                                             workspace.col_chosen, workspace.comb_val.data(),
                                             input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                             workspace.ext_coef[workspace.ntaken], workspace.ext_sd, workspace.ext_mean[workspace.ntaken],
                                             workspace.ext_fill_val[workspace.ntaken], model_params.missing_action,
                                             workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true,
                                             workspace.weights_map);
                }

                
            }
            break;
        }

        case Categorical:
        {
            switch(model_params.cat_split_type)
            {
                case SingleCateg:
                {
                    workspace.chosen_cat[workspace.ntaken] = choose_cat_from_present(workspace, input_data, workspace.col_chosen);
                    workspace.ext_fill_new[workspace.ntaken] = workspace.coef_norm(workspace.rnd_generator);
                    if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                    {
                        add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                        input_data.categ_data + workspace.col_chosen * input_data.nrows,
                                        input_data.ncat[workspace.col_chosen],
                                        NULL, workspace.ext_fill_new[workspace.ntaken],
                                        workspace.chosen_cat[workspace.ntaken],
                                        workspace.ext_fill_val[workspace.ntaken], workspace.ext_fill_new[workspace.ntaken],
                                        NULL, NULL, model_params.new_cat_action, model_params.missing_action, SingleCateg, true);
                    }

                    else if (workspace.weights_arr.size())
                    {
                        add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                                 input_data.categ_data + workspace.col_chosen * input_data.nrows,
                                                 input_data.ncat[workspace.col_chosen],
                                                 NULL, workspace.ext_fill_new[workspace.ntaken],
                                                 workspace.chosen_cat[workspace.ntaken],
                                                 workspace.ext_fill_val[workspace.ntaken], workspace.ext_fill_new[workspace.ntaken],
                                                 NULL, model_params.new_cat_action, model_params.missing_action, SingleCateg, true,
                                                 workspace.weights_arr);
                    }

                    else
                    {
                        add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                                 input_data.categ_data + workspace.col_chosen * input_data.nrows,
                                                 input_data.ncat[workspace.col_chosen],
                                                 NULL, workspace.ext_fill_new[workspace.ntaken],
                                                 workspace.chosen_cat[workspace.ntaken],
                                                 workspace.ext_fill_val[workspace.ntaken], workspace.ext_fill_new[workspace.ntaken],
                                                 NULL, model_params.new_cat_action, model_params.missing_action, SingleCateg, true,
                                                 workspace.weights_map);
                    }

                    break;
                }

                case SubSet:
                {
                    for (int cat = 0; cat < input_data.ncat[workspace.col_chosen]; cat++)
                        workspace.ext_cat_coef[workspace.ntaken][cat] = workspace.coef_norm(workspace.rnd_generator);

                    if (model_params.coef_by_prop)
                    {
                        int ncat = input_data.ncat[workspace.col_chosen];
                        size_t *restrict counts = workspace.buffer_szt.data();
                        size_t *restrict sorted_ix = workspace.buffer_szt.data() + ncat;
                        /* calculate counts and sort by them */
                        std::fill(counts, counts + ncat, (size_t)0);
                        for (size_t ix = workspace.st; ix <= workspace.end; ix++)
                            if (input_data.categ_data[workspace.col_chosen * input_data.nrows + ix] >= 0)
                                counts[input_data.categ_data[workspace.col_chosen * input_data.nrows + ix]]++;
                        std::iota(sorted_ix, sorted_ix + ncat, (size_t)0);
                        std::sort(sorted_ix, sorted_ix + ncat,
                                  [&counts](const size_t a, const size_t b){return counts[a] < counts[b];});
                        /* now re-order the coefficients accordingly */
                        std::sort(workspace.ext_cat_coef[workspace.ntaken].begin(),
                                  workspace.ext_cat_coef[workspace.ntaken].begin() + ncat);
                        std::copy(workspace.ext_cat_coef[workspace.ntaken].begin(),
                                  workspace.ext_cat_coef[workspace.ntaken].begin() + ncat,
                                  workspace.buffer_dbl.begin());
                        for (int ix = 0; ix < ncat; ix++)
                            workspace.ext_cat_coef[workspace.ntaken][ix] = workspace.buffer_dbl[sorted_ix[ix]];
                    }

                    if (!workspace.weights_arr.size() && !workspace.weights_map.size())
                    {
                        add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                        input_data.categ_data + workspace.col_chosen * input_data.nrows,
                                        input_data.ncat[workspace.col_chosen],
                                        workspace.ext_cat_coef[workspace.ntaken].data(), (double)0, (int)0,
                                        workspace.ext_fill_val[workspace.ntaken], workspace.ext_fill_new[workspace.ntaken],
                                        workspace.buffer_szt.data(), workspace.buffer_szt.data() + input_data.max_categ + 1,
                                        model_params.new_cat_action, model_params.missing_action, SubSet, true);
                    }

                    else if (workspace.weights_arr.size())
                    {
                        add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                                 input_data.categ_data + workspace.col_chosen * input_data.nrows,
                                                 input_data.ncat[workspace.col_chosen],
                                                 workspace.ext_cat_coef[workspace.ntaken].data(), (double)0, (int)0,
                                                 workspace.ext_fill_val[workspace.ntaken], workspace.ext_fill_new[workspace.ntaken],
                                                 workspace.buffer_szt.data(),
                                                 model_params.new_cat_action, model_params.missing_action, SubSet, true,
                                                 workspace.weights_arr);
                    }

                    else
                    {
                        add_linear_comb_weighted(workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                                 input_data.categ_data + workspace.col_chosen * input_data.nrows,
                                                 input_data.ncat[workspace.col_chosen],
                                                 workspace.ext_cat_coef[workspace.ntaken].data(), (double)0, (int)0,
                                                 workspace.ext_fill_val[workspace.ntaken], workspace.ext_fill_new[workspace.ntaken],
                                                 workspace.buffer_szt.data(),
                                                 model_params.new_cat_action, model_params.missing_action, SubSet, true,
                                                 workspace.weights_map);
                    }

                    break;
                }
            }
            break;
        }

        default:
        {
            unexpected_error();
            break;
        }
    }
}

void shrink_to_fit_hplane(IsoHPlane &hplane, bool clear_vectors)
{
    if (clear_vectors)
    {
        hplane.col_num.clear();
        hplane.col_type.clear();
        hplane.coef.clear();
        hplane.mean.clear();
        hplane.cat_coef.clear();
        hplane.chosen_cat.clear();
        hplane.fill_val.clear();
        hplane.fill_new.clear();
    }

    hplane.col_num.shrink_to_fit();
    hplane.col_type.shrink_to_fit();
    hplane.coef.shrink_to_fit();
    hplane.mean.shrink_to_fit();
    hplane.cat_coef.shrink_to_fit();
    hplane.chosen_cat.shrink_to_fit();
    hplane.fill_val.shrink_to_fit();
    hplane.fill_new.shrink_to_fit();
}

template <class InputData, class WorkerMemory>
void simplify_hplane(IsoHPlane &hplane, WorkerMemory &workspace, InputData &input_data, ModelParams &model_params)
{
    if (workspace.ntaken_best < model_params.ndim)
    {
        hplane.col_num.resize(workspace.ntaken_best);
        hplane.col_type.resize(workspace.ntaken_best);
        if (model_params.missing_action != Fail)
            hplane.fill_val.resize(workspace.ntaken_best);
    }

    size_t ncols_numeric = 0;
    size_t ncols_categ   = 0;

    if (input_data.ncols_categ)
    {
        for (size_t col = 0; col < workspace.ntaken_best; col++)
        {
            switch(hplane.col_type[col])
            {
                case Numeric:
                {
                    workspace.ext_coef[ncols_numeric] = hplane.coef[col];
                    workspace.ext_mean[ncols_numeric] = hplane.mean[col];
                    ncols_numeric++;
                    break;
                }

                case Categorical:
                {
                    workspace.ext_fill_new[ncols_categ] = hplane.fill_new[col];
                    switch(model_params.cat_split_type)
                    {
                        case SingleCateg:
                        {
                            workspace.chosen_cat[ncols_categ] = hplane.chosen_cat[col];
                            break;
                        }

                        case SubSet:
                        {
                            std::copy(hplane.cat_coef[col].begin(),
                                      hplane.cat_coef[col].begin() + input_data.ncat[hplane.col_num[col]],
                                      workspace.ext_cat_coef[ncols_categ].begin());
                            break;
                        }
                    }
                    ncols_categ++;
                    break;
                }

                default:
                {
                    unexpected_error();
                    break;
                }
            }
        }
    }

    else
    {
        ncols_numeric = workspace.ntaken_best;
    }


    hplane.coef.resize(ncols_numeric);
    hplane.mean.resize(ncols_numeric);
    if (input_data.ncols_numeric)
    {
        std::copy(workspace.ext_coef.begin(), workspace.ext_coef.begin() + ncols_numeric, hplane.coef.begin());
        std::copy(workspace.ext_mean.begin(), workspace.ext_mean.begin() + ncols_numeric, hplane.mean.begin());
    }

    /* If there are no categorical columns, all of them will be numerical and there is no need to reorder */
    if (ncols_categ)
    {
        hplane.fill_new.resize(ncols_categ);
        std::copy(workspace.ext_fill_new.begin(),
                  workspace.ext_fill_new.begin() + ncols_categ,
                  hplane.fill_new.begin());

        hplane.cat_coef.resize(ncols_categ);
        switch(model_params.cat_split_type)
        {
            case SingleCateg:
            {
                hplane.chosen_cat.resize(ncols_categ);
                std::copy(workspace.chosen_cat.begin(),
                          workspace.chosen_cat.begin() + ncols_categ,
                          hplane.chosen_cat.begin());
                hplane.cat_coef.clear();
                break;
            }

            case SubSet:
            {
                hplane.chosen_cat.clear();
                ncols_categ = 0;
                for (size_t col = 0; col < workspace.ntaken_best; col++)
                {
                    if (hplane.col_type[col] == Categorical)
                    {
                        hplane.cat_coef[ncols_categ].resize(input_data.ncat[hplane.col_num[col]]);
                        std::copy(workspace.ext_cat_coef[ncols_categ].begin(),
                                  workspace.ext_cat_coef[ncols_categ].begin()
                                   + input_data.ncat[hplane.col_num[col]],
                                  hplane.cat_coef[ncols_categ].begin());
                        hplane.cat_coef[ncols_categ].shrink_to_fit();
                        ncols_categ++;
                    }
                }
                break;
            }
        }
    }

    else
    {
        hplane.cat_coef.clear();
        hplane.chosen_cat.clear();
        hplane.fill_new.clear();
    }
}
