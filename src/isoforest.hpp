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

template <class InputData, class WorkerMemory, class ldouble_safe>
void split_itree_recursive(std::vector<IsoTree>     &trees,
                           WorkerMemory             &workspace,
                           InputData                &input_data,
                           ModelParams              &model_params,
                           std::vector<ImputeNode> *impute_nodes,
                           size_t                   curr_depth)
{
    if (interrupt_switch) return;
    ldouble_safe sum_weight = -HUGE_VAL;

    /* calculate imputation statistics if desired */
    if (impute_nodes != NULL)
    {
        if (input_data.Xc_indptr != NULL)
            std::sort(workspace.ix_arr.begin() + workspace.st,
                      workspace.ix_arr.begin() + workspace.end + 1);
        build_impute_node<decltype(input_data), decltype(workspace), ldouble_safe>(
                          impute_nodes->back(), workspace,
                          input_data, model_params,
                          *impute_nodes, curr_depth,
                          model_params.min_imp_obs);
    }

    /* check for potential isolated leafs or unique splits */
    if (workspace.end == workspace.st || (workspace.end - workspace.st) == 1 || curr_depth >= model_params.max_depth)
        goto terminal_statistics;

    /* when using weights, the split should stop when the sum of weights is <= 1 */
    if (workspace.changed_weights)
    {
        sum_weight = calculate_sum_weights<ldouble_safe>(
                                           workspace.ix_arr, workspace.st, workspace.end, curr_depth,
                                           workspace.weights_arr, workspace.weights_map);
        if (curr_depth > 0 && sum_weight <= 1)
            goto terminal_statistics;
    }

    /* if there's no columns left to split, can end here */
    if (!workspace.col_sampler.get_remaining_cols())
        goto terminal_statistics;

    /* for sparse matrices, need to sort the indices */
    if (input_data.Xc_indptr != NULL && impute_nodes == NULL)
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);

    /* pick column to split according to criteria */
    workspace.prob_split_type = workspace.rbin(workspace.rnd_generator);


    /* case1: guided, pick column and/or point with best gain */
    if (
            workspace.prob_split_type
                < (
                    model_params.prob_pick_by_gain_avg +
                    model_params.prob_pick_by_gain_pl +
                    model_params.prob_pick_by_full_gain +
                    model_params.prob_pick_by_dens
                  )
        )
    {
        /* case 1.1: column and/or threshold is/are decided by averaged gain */
        if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg)
            workspace.criterion = Averaged;

        /* case 1.2: column and/or threshold is/are decided by pooled gain */
        else if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg +
                                             model_params.prob_pick_by_gain_pl)
            workspace.criterion = Pooled;
        /* case 1.3: column and/or threshold is/are decided by full gain (pooled gain in all columns) */
        else if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg +
                                             model_params.prob_pick_by_gain_pl +
                                             model_params.prob_pick_by_full_gain)
            workspace.criterion = FullGain;
        /* case 1.4: column and/or threshold is/are decided by density pooled gain */
        else
            workspace.criterion = DensityCrit;

        workspace.determine_split = model_params.ntry <= 1 || workspace.col_sampler.get_remaining_cols() == 1;

        if (workspace.criterion == FullGain)
        {
            workspace.col_sampler.get_array_remaining_cols(workspace.col_indices);
        }
    }

    /* case2: column and split point is decided at random */
    else
    {
        workspace.criterion = NoCrit;
        workspace.determine_split = true;
    }

    /* pick column selection method also according to criteria */
    if (
        (workspace.criterion != NoCrit &&
         std::max(workspace.ntry, (size_t)1) >= workspace.col_sampler.get_remaining_cols())
            ||
        (workspace.col_sampler.get_remaining_cols() == 1)
    ) {
        workspace.prob_split_type = 0;
    }
    else {
        workspace.prob_split_type = workspace.rbin(workspace.rnd_generator);
    }

    if (
            workspace.prob_split_type
                < model_params.prob_pick_col_by_range
        )
    {
        workspace.col_criterion = ByRange;
        if (curr_depth == 0 && is_boxed_metric(model_params.scoring_metric))
        {
            workspace.has_saved_stats = false;
            for (size_t col = 0; col < input_data.ncols_numeric; col++)
                workspace.node_col_weights[col] = workspace.density_calculator.box_high[col]
                                                   - workspace.density_calculator.box_low[col];

            add_col_weights_to_ranges:
            if (workspace.tree_kurtoses != NULL)
            {
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (workspace.node_col_weights[col] <= 0) continue;
                    workspace.node_col_weights[col] *= workspace.tree_kurtoses[col];
                    workspace.node_col_weights[col]  = std::fmax(workspace.node_col_weights[col], 1e-100);
                }
            }
            else if (input_data.col_weights != NULL)
            {
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (workspace.node_col_weights[col] <= 0) continue;
                    workspace.node_col_weights[col] *= input_data.col_weights[col];
                    workspace.node_col_weights[col]  = std::fmax(workspace.node_col_weights[col], 1e-100);
                }
            }
        }

        else if (curr_depth == 0 &&
                 model_params.sample_size == input_data.nrows &&
                 !model_params.with_replacement &&
                 input_data.range_low != NULL &&
                 model_params.ncols_per_tree == input_data.ncols_tot)
        {
            workspace.has_saved_stats = false;
            for (size_t col = 0; col < input_data.ncols_numeric; col++)
                workspace.node_col_weights[col] = input_data.range_high[col]
                                                   - input_data.range_low[col];
            goto add_col_weights_to_ranges;
        }

        else
        {
            workspace.has_saved_stats = workspace.criterion == NoCrit;
            calc_ranges_all_cols(input_data, workspace, model_params, workspace.node_col_weights.data(),
                                 workspace.has_saved_stats? workspace.saved_stat1.data() : NULL,
                                 workspace.has_saved_stats? workspace.saved_stat2.data() : NULL);
        }
    }

    else if (
            workspace.prob_split_type
                < (model_params.prob_pick_col_by_range +
                   model_params.prob_pick_col_by_var)
        )
    {
        workspace.col_criterion = ByVar;
        workspace.has_saved_stats = workspace.criterion == NoCrit;
        calc_var_all_cols<InputData, WorkerMemory, ldouble_safe>(
                          input_data, workspace, model_params,
                          workspace.node_col_weights.data(),
                          workspace.has_saved_stats? workspace.saved_stat1.data() : NULL,
                          workspace.has_saved_stats? workspace.saved_stat2.data() : NULL,
                          NULL, NULL);
    }

    else if (
            workspace.prob_split_type
                < (model_params.prob_pick_col_by_range +
                   model_params.prob_pick_col_by_var +
                   model_params.prob_pick_col_by_kurt)
        )
    {
        workspace.col_criterion = ByKurt;
        workspace.has_saved_stats = workspace.criterion == NoCrit;
        calc_kurt_all_cols<decltype(input_data), decltype(workspace), ldouble_safe>(
                           input_data, workspace, model_params, workspace.node_col_weights.data(),
                           workspace.has_saved_stats? workspace.saved_stat1.data() : NULL,
                           workspace.has_saved_stats? workspace.saved_stat2.data() : NULL);
    }

    else
    {
        workspace.col_criterion = Uniformly;
    }

    if (workspace.col_criterion != Uniformly)
    {
        if (!workspace.node_col_sampler.initialize(workspace.node_col_weights.data(),
                                                   &workspace.col_sampler.col_indices,
                                                   workspace.col_sampler.curr_pos,
                                                   (workspace.criterion == NoCrit)? 1 : model_params.ntry,
                                                   false))
        {
            goto terminal_statistics;
        }
    }

    /* when column is chosen at random */
    if (workspace.determine_split)
    {
        if (workspace.col_criterion != Uniformly)
        {
            if (!workspace.node_col_sampler.sample_col(trees.back().col_num, workspace.rnd_generator))
            {
                goto terminal_statistics;
            }

            if (trees.back().col_num < input_data.ncols_numeric)
            {
                trees.back().col_type = Numeric;
                if (workspace.has_saved_stats)
                {
                    workspace.xmin = workspace.saved_stat1[trees.back().col_num];
                    workspace.xmax = workspace.saved_stat2[trees.back().col_num];
                }

                else
                {
                    get_split_range(workspace, input_data, model_params, trees.back());
                    if (workspace.unsplittable)
                        unexpected_error();
                }
            }

            else
            {
                get_split_range(workspace, input_data, model_params, trees.back());
                if (workspace.unsplittable)
                    unexpected_error();
            }

            goto produce_split;
        }

        if (!workspace.col_sampler.has_weights())
        {
            while (workspace.col_sampler.sample_col(trees.back().col_num, workspace.rnd_generator))
            {
                if (interrupt_switch) return;
                
                get_split_range(workspace, input_data, model_params, trees.back());
                if (workspace.unsplittable)
                    workspace.col_sampler.drop_col(trees.back().col_num + ((trees.back().col_type == Numeric)? (size_t)0 : input_data.ncols_numeric));
                else
                    goto produce_split;
            }
            goto terminal_statistics;
        }

        else
        {
            if (workspace.try_all)
                workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);
            workspace.ntried = 0;
            size_t threshold_shuffle = (workspace.col_sampler.get_remaining_cols() + 1) / 2;

            while (
                    workspace.try_all?
                    workspace.col_sampler.sample_col(trees.back().col_num)
                        :
                    workspace.col_sampler.sample_col(trees.back().col_num, workspace.rnd_generator)
                   )
            {
                if (interrupt_switch) return;

                get_split_range(workspace, input_data, model_params, trees.back());
                if (workspace.unsplittable)
                {
                    workspace.col_sampler.drop_col(trees.back().col_num + ((trees.back().col_type == Numeric)? (size_t)0 : input_data.ncols_numeric));
                    workspace.ntried++;
                    if (!workspace.try_all && workspace.ntried >= threshold_shuffle)
                    {
                        workspace.try_all = true;
                        workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);
                    }
                }

                else
                {
                    goto produce_split;
                }
            }
            goto terminal_statistics;
        }
    }


    /* when choosing both column and threshold */
    else
    {
        if (model_params.ntry >= workspace.col_sampler.get_remaining_cols())
            workspace.col_sampler.prepare_full_pass();
        else if (workspace.try_all && workspace.col_criterion == Uniformly)
            workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);

        std::vector<bool> col_is_taken;
        hashed_set<size_t> col_is_taken_s;
        if (model_params.ntry < workspace.col_sampler.get_remaining_cols() && workspace.col_criterion == Uniformly)
        {
            if (input_data.ncols_tot < 1e5 ||
                ((ldouble_safe)model_params.ntry / (ldouble_safe)workspace.col_sampler.get_remaining_cols()) > .25
                )
            {
                col_is_taken.resize(input_data.ncols_tot, false);
            }
            else {
                col_is_taken_s.reserve(model_params.ntry);
            }
        }

        size_t threshold_shuffle = (workspace.col_sampler.get_remaining_cols() + 1) / 2;
        workspace.ntried = 0; /* <- used to determine when to shuffle the remainder */
        workspace.ntaken = 0; /* <- used to count how many columns have been evaluated */
        trees.back().score = -HUGE_VAL; /* this is used to track the best gain */

        while (
                (workspace.col_criterion != Uniformly)?
                workspace.node_col_sampler.sample_col(workspace.col_chosen, workspace.rnd_generator)
                    :
                (workspace.try_all?
                 workspace.col_sampler.sample_col(workspace.col_chosen)
                     :
                 workspace.col_sampler.sample_col(workspace.col_chosen, workspace.rnd_generator))
            )
        {
            if (interrupt_switch) return;

            if (workspace.col_criterion != Uniformly)
            {
                workspace.ntaken++;
                goto probe_this_col;
            }
            
            workspace.ntried++;
            if (!workspace.try_all && workspace.ntried >= threshold_shuffle)
            {
                workspace.try_all = true;
                workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);
            }

            if ((col_is_taken.size() || col_is_taken_s.size()) && !workspace.try_all)
            {
                if (is_col_taken(col_is_taken, col_is_taken_s, workspace.col_chosen))
                    continue;
                set_col_as_taken(col_is_taken, col_is_taken_s, input_data, workspace.col_chosen);
            }

            get_split_range_v2(workspace, input_data, model_params);
            if (workspace.unsplittable)
            {
                workspace.col_sampler.drop_col(workspace.col_chosen);
                continue;
            }

            else
            {
                probe_this_col:
                if (workspace.col_chosen < input_data.ncols_numeric)
                {
                    if (input_data.Xc_indptr == NULL)
                    {
                        if (!workspace.changed_weights)
                            workspace.this_gain = eval_guided_crit<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                   ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                                                   workspace.buffer_dbl.data(), false,
                                                                   workspace.imputed_x_buffer.data(),
                                                                   &workspace.saved_xmedian,
                                                                   workspace.split_ix, workspace.this_split_point,
                                                                   workspace.xmin, workspace.xmax,
                                                                   workspace.criterion, model_params.min_gain,
                                                                   model_params.missing_action,
                                                                   workspace.col_indices.data(),
                                                                   workspace.col_sampler.get_remaining_cols(),
                                                                   model_params.ncols_per_tree < input_data.ncols_tot,
                                                                   input_data.X_row_major.data(),
                                                                   input_data.ncols_numeric,
                                                                   input_data.Xr.data(),
                                                                   input_data.Xr_ind.data(),
                                                                   input_data.Xr_indptr.data());
                        else if (!workspace.weights_arr.empty())
                            workspace.this_gain = eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                            decltype(workspace.weights_arr), ldouble_safe>(
                                                                            workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                            input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                                                            workspace.buffer_dbl.data(), false,
                                                                            workspace.imputed_x_buffer.data(),
                                                                            &workspace.saved_xmedian,
                                                                            workspace.split_ix, workspace.this_split_point,
                                                                            workspace.xmin, workspace.xmax,
                                                                            workspace.criterion, model_params.min_gain,
                                                                            model_params.missing_action,
                                                                            workspace.col_indices.data(),
                                                                            workspace.col_sampler.get_remaining_cols(),
                                                                            model_params.ncols_per_tree < input_data.ncols_tot,
                                                                            input_data.X_row_major.data(),
                                                                            input_data.ncols_numeric,
                                                                            input_data.Xr.data(),
                                                                            input_data.Xr_ind.data(),
                                                                            input_data.Xr_indptr.data(),
                                                                            workspace.weights_arr);
                        else
                            workspace.this_gain = eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                            decltype(workspace.weights_map), ldouble_safe>(
                                                                            workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                            input_data.numeric_data + workspace.col_chosen * input_data.nrows,
                                                                            workspace.buffer_dbl.data(), false,
                                                                            workspace.imputed_x_buffer.data(),
                                                                            &workspace.saved_xmedian,
                                                                            workspace.split_ix, workspace.this_split_point,
                                                                            workspace.xmin, workspace.xmax,
                                                                            workspace.criterion, model_params.min_gain,
                                                                            model_params.missing_action,
                                                                            workspace.col_indices.data(),
                                                                            workspace.col_sampler.get_remaining_cols(),
                                                                            model_params.ncols_per_tree < input_data.ncols_tot,
                                                                            input_data.X_row_major.data(),
                                                                            input_data.ncols_numeric,
                                                                            input_data.Xr.data(),
                                                                            input_data.Xr_ind.data(),
                                                                            input_data.Xr_indptr.data(),
                                                                            workspace.weights_map);
                    }

                    else
                    {
                        if (!workspace.changed_weights)
                            workspace.this_gain = eval_guided_crit<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                   typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                   ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   workspace.col_chosen, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                   workspace.buffer_dbl.data(), workspace.buffer_szt.data(), false,
                                                                   &workspace.saved_xmedian,
                                                                   workspace.this_split_point, workspace.xmin, workspace.xmax,
                                                                   workspace.criterion, model_params.min_gain, model_params.missing_action,
                                                                   workspace.col_indices.data(),
                                                                   workspace.col_sampler.get_remaining_cols(),
                                                                   model_params.ncols_per_tree < input_data.ncols_tot,
                                                                   input_data.X_row_major.data(),
                                                                   input_data.ncols_numeric,
                                                                   input_data.Xr.data(),
                                                                   input_data.Xr_ind.data(),
                                                                   input_data.Xr_indptr.data());
                        else if (!workspace.weights_arr.empty())
                            workspace.this_gain = eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                            typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                            decltype(workspace.weights_arr), ldouble_safe>(
                                                                            workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                            workspace.col_chosen, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                            workspace.buffer_dbl.data(), workspace.buffer_szt.data(), false,
                                                                            &workspace.saved_xmedian,
                                                                            workspace.this_split_point, workspace.xmin, workspace.xmax,
                                                                            workspace.criterion, model_params.min_gain, model_params.missing_action,
                                                                            workspace.col_indices.data(),
                                                                            workspace.col_sampler.get_remaining_cols(),
                                                                            model_params.ncols_per_tree < input_data.ncols_tot,
                                                                            input_data.X_row_major.data(),
                                                                            input_data.ncols_numeric,
                                                                            input_data.Xr.data(),
                                                                            input_data.Xr_ind.data(),
                                                                            input_data.Xr_indptr.data(),
                                                                            workspace.weights_arr);
                        else
                            workspace.this_gain = eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                            typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                            decltype(workspace.weights_map),
                                                                            ldouble_safe>(
                                                                            workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                            workspace.col_chosen, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                            workspace.buffer_dbl.data(), workspace.buffer_szt.data(), false,
                                                                            &workspace.saved_xmedian,
                                                                            workspace.this_split_point, workspace.xmin, workspace.xmax,
                                                                            workspace.criterion, model_params.min_gain, model_params.missing_action,
                                                                            workspace.col_indices.data(),
                                                                            workspace.col_sampler.get_remaining_cols(),
                                                                            model_params.ncols_per_tree < input_data.ncols_tot,
                                                                            input_data.X_row_major.data(),
                                                                            input_data.ncols_numeric,
                                                                            input_data.Xr.data(),
                                                                            input_data.Xr_ind.data(),
                                                                            input_data.Xr_indptr.data(),
                                                                            workspace.weights_map);
                    }
                }

                else
                {
                    if (!workspace.changed_weights)
                        workspace.this_gain = eval_guided_crit<ldouble_safe>(
                                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                                               input_data.categ_data + (workspace.col_chosen - input_data.ncols_numeric) * input_data.nrows,
                                                               input_data.ncat[workspace.col_chosen - input_data.ncols_numeric],
                                                               &workspace.saved_cat_mode,
                                                               workspace.buffer_szt.data(), workspace.buffer_szt.data() + input_data.max_categ,
                                                               workspace.buffer_dbl.data(), workspace.this_categ, workspace.this_split_categ.data(),
                                                               workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                               model_params.all_perm, model_params.missing_action, model_params.cat_split_type);
                    else if (!workspace.weights_arr.empty())
                        workspace.this_gain = eval_guided_crit_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                                                        workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                        input_data.categ_data + (workspace.col_chosen - input_data.ncols_numeric) * input_data.nrows,
                                                                        input_data.ncat[workspace.col_chosen - input_data.ncols_numeric],
                                                                        &workspace.saved_cat_mode,
                                                                        workspace.buffer_szt.data(),
                                                                        workspace.buffer_dbl.data(), workspace.this_categ, workspace.this_split_categ.data(),
                                                                        workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                                        model_params.all_perm, model_params.missing_action, model_params.cat_split_type,
                                                                        workspace.weights_arr);
                    else
                        workspace.this_gain = eval_guided_crit_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                                                        workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                        input_data.categ_data + (workspace.col_chosen - input_data.ncols_numeric) * input_data.nrows,
                                                                        input_data.ncat[workspace.col_chosen - input_data.ncols_numeric],
                                                                        &workspace.saved_cat_mode,
                                                                        workspace.buffer_szt.data(),
                                                                        workspace.buffer_dbl.data(), workspace.this_categ, workspace.this_split_categ.data(),
                                                                        workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                                        model_params.all_perm, model_params.missing_action, model_params.cat_split_type,
                                                                        workspace.weights_map);
                }

                if (std::isnan(workspace.this_gain) || workspace.this_gain <= -HUGE_VAL)
                    continue;


                if (workspace.this_gain > trees.back().score)
                {
                    if (workspace.col_chosen < input_data.ncols_numeric)
                    {
                        trees.back().score     = workspace.this_gain;
                        trees.back().col_num   = workspace.col_chosen;
                        trees.back().col_type  = Numeric;
                        trees.back().num_split = workspace.this_split_point;
                        if (model_params.penalize_range)
                        {
                            trees.back().range_low  = workspace.xmin - workspace.xmax + trees.back().num_split;
                            trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
                        }

                        if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
                        {
                            workspace.density_calculator.save_range(workspace.xmin, workspace.xmax);
                        }

                        workspace.best_xmedian = workspace.saved_xmedian;
                    }

                    else
                    {
                        trees.back().score    = workspace.this_gain;
                        trees.back().col_num  = workspace.col_chosen - input_data.ncols_numeric;
                        trees.back().col_type = Categorical;
                        switch (model_params.cat_split_type)
                        {
                            case SingleCateg:
                            {
                                trees.back().chosen_cat = workspace.this_categ;
                                break;
                            }

                            case SubSet:
                            {
                                trees.back().cat_split.assign(workspace.this_split_categ.begin(),
                                                              workspace.this_split_categ.begin()
                                                                + input_data.ncat[trees.back().col_num]);
                                break;
                            }
                        }

                        workspace.best_cat_mode = workspace.saved_cat_mode;

                        if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
                        {
                            if (model_params.scoring_metric == Density)
                            {
                                switch (model_params.cat_split_type)
                                {
                                    case SingleCateg:
                                    {
                                        workspace.density_calculator.save_n_present(workspace.buffer_szt.data(),
                                                                                    input_data.ncat[trees.back().col_num]);
                                        break;
                                    }

                                    case SubSet:
                                    {
                                        workspace.density_calculator.save_n_present_and_left(
                                            workspace.this_split_categ.data(),
                                            input_data.ncat[trees.back().col_num]
                                        );
                                        break;
                                    }
                                }
                            }

                            else
                            {
                                workspace.density_calculator.save_counts(workspace.buffer_szt.data(),
                                                                         input_data.ncat[trees.back().col_num]);
                            }
                        }
                    }
                }

                if (++workspace.ntaken >= model_params.ntry)
                    break;
            }
        }

        if (!workspace.ntaken)
            goto terminal_statistics;

        if (trees.back().score <= 0.)
            goto terminal_statistics;
        else
            trees.back().score = 0.;
    }


    /* for numeric, choose a random point, or pick the best point as determined earlier */
    produce_split:
    if (trees.back().col_type == Numeric)
    {
        if (workspace.determine_split)
        {
            switch(workspace.criterion)
            {
                case NoCrit:
                {
                    trees.back().num_split = sample_random_uniform(workspace.xmin, workspace.xmax, workspace.rnd_generator);
                    break;
                }

                default:
                {
                    if (input_data.Xc_indptr == NULL)
                    {
                        if (!workspace.changed_weights)
                            workspace.this_gain =
                                eval_guided_crit<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                                                 workspace.ix_arr.data(), workspace.st, workspace.end,
                                                 input_data.numeric_data + trees.back().col_num * input_data.nrows,
                                                 workspace.buffer_dbl.data(), true,
                                                 workspace.imputed_x_buffer.data(),
                                                 &workspace.best_xmedian,
                                                 workspace.split_ix, trees.back().num_split,
                                                 workspace.xmin, workspace.xmax,
                                                 workspace.criterion, model_params.min_gain,
                                                 model_params.missing_action,
                                                 workspace.col_indices.data(),
                                                 workspace.col_sampler.get_remaining_cols(),
                                                 model_params.ncols_per_tree < input_data.ncols_tot,
                                                 input_data.X_row_major.data(),
                                                 input_data.ncols_numeric,
                                                 input_data.Xr.data(),
                                                 input_data.Xr_ind.data(),
                                                 input_data.Xr_indptr.data());
                        else if (!workspace.weights_arr.empty())
                            workspace.this_gain =
                                eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, decltype(workspace.weights_arr), ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                                          input_data.numeric_data + trees.back().col_num * input_data.nrows,
                                                          workspace.buffer_dbl.data(), true,
                                                          workspace.imputed_x_buffer.data(),
                                                          &workspace.best_xmedian,
                                                          workspace.split_ix, trees.back().num_split,
                                                          workspace.xmin, workspace.xmax,
                                                          workspace.criterion, model_params.min_gain,
                                                          model_params.missing_action,
                                                          workspace.col_indices.data(),
                                                          workspace.col_sampler.get_remaining_cols(),
                                                          model_params.ncols_per_tree < input_data.ncols_tot,
                                                          input_data.X_row_major.data(),
                                                          input_data.ncols_numeric,
                                                          input_data.Xr.data(),
                                                          input_data.Xr_ind.data(),
                                                          input_data.Xr_indptr.data(),
                                                          workspace.weights_arr);
                        else
                            workspace.this_gain =
                                eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, decltype(workspace.weights_map), ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                                          input_data.numeric_data + trees.back().col_num * input_data.nrows,
                                                          workspace.buffer_dbl.data(), true,
                                                          workspace.imputed_x_buffer.data(),
                                                          &workspace.best_xmedian,
                                                          workspace.split_ix, trees.back().num_split,
                                                          workspace.xmin, workspace.xmax,
                                                          workspace.criterion, model_params.min_gain,
                                                          model_params.missing_action,
                                                          workspace.col_indices.data(),
                                                          workspace.col_sampler.get_remaining_cols(),
                                                          model_params.ncols_per_tree < input_data.ncols_tot,
                                                          input_data.X_row_major.data(),
                                                          input_data.ncols_numeric,
                                                          input_data.Xr.data(),
                                                          input_data.Xr_ind.data(),
                                                          input_data.Xr_indptr.data(),
                                                          workspace.weights_map);

                        if (std::isnan(workspace.this_gain) || workspace.this_gain <= -HUGE_VAL)
                            goto terminal_statistics;

                        if (
                            model_params.missing_action == Fail
                             ||
                            (model_params.missing_action == Impute && input_data.Xc_indptr == NULL)
                        ) /* data is already split in this case */
                        {
                            if (model_params.missing_action == Impute)
                            {
                                workspace.st_NA = workspace.split_ix + 1;
                                workspace.end_NA = workspace.st_NA;
                            }
                            
                            workspace.split_ix++;
                            if (model_params.penalize_range)
                            {
                                trees.back().range_low  = workspace.xmin - workspace.xmax + trees.back().num_split;
                                trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
                            }
                            goto follow_branches;
                        }
                    }

                    else
                    {
                        if (!workspace.changed_weights)
                            workspace.this_gain =
                                eval_guided_crit<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                 typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                 ldouble_safe>(
                                                 workspace.ix_arr.data(), workspace.st, workspace.end,
                                                 trees.back().col_num, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                 workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true,
                                                 &workspace.best_xmedian,
                                                 trees.back().num_split, workspace.xmin, workspace.xmax,
                                                 workspace.criterion, model_params.min_gain,
                                                 model_params.missing_action,
                                                 workspace.col_indices.data(),
                                                 workspace.col_sampler.get_remaining_cols(),
                                                 model_params.ncols_per_tree < input_data.ncols_tot,
                                                 input_data.X_row_major.data(),
                                                 input_data.ncols_numeric,
                                                 input_data.Xr.data(),
                                                 input_data.Xr_ind.data(),
                                                 input_data.Xr_indptr.data());
                        else if (!workspace.weights_arr.empty())
                            workspace.this_gain =
                                eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                          typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                          decltype(workspace.weights_arr),
                                                          ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                                          trees.back().col_num, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                          workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true,
                                                          &workspace.best_xmedian,
                                                          trees.back().num_split, workspace.xmin, workspace.xmax,
                                                          workspace.criterion, model_params.min_gain,
                                                          model_params.missing_action,
                                                          workspace.col_indices.data(),
                                                          workspace.col_sampler.get_remaining_cols(),
                                                          model_params.ncols_per_tree < input_data.ncols_tot,
                                                          input_data.X_row_major.data(),
                                                          input_data.ncols_numeric,
                                                          input_data.Xr.data(),
                                                          input_data.Xr_ind.data(),
                                                          input_data.Xr_indptr.data(),
                                                          workspace.weights_arr);
                        else
                            workspace.this_gain =
                                eval_guided_crit_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                          typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                          decltype(workspace.weights_map),
                                                          ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                                          trees.back().col_num, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                          workspace.buffer_dbl.data(), workspace.buffer_szt.data(), true,
                                                          &workspace.best_xmedian,
                                                          trees.back().num_split, workspace.xmin, workspace.xmax,
                                                          workspace.criterion, model_params.min_gain,
                                                          model_params.missing_action,
                                                          workspace.col_indices.data(),
                                                          workspace.col_sampler.get_remaining_cols(),
                                                          model_params.ncols_per_tree < input_data.ncols_tot,
                                                          input_data.X_row_major.data(),
                                                          input_data.ncols_numeric,
                                                          input_data.Xr.data(),
                                                          input_data.Xr_ind.data(),
                                                          input_data.Xr_indptr.data(),
                                                          workspace.weights_map);
                    }

                    if (std::isnan(workspace.this_gain) || workspace.this_gain <= -HUGE_VAL)
                        goto terminal_statistics;

                    break;
                }
            }

            if (model_params.penalize_range)
            {
                trees.back().range_low  = workspace.xmin - workspace.xmax + trees.back().num_split;
                trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
            }
        }

        if (model_params.missing_action == Fail && std::isnan(trees.back().num_split))
            throw std::runtime_error("Data has missing values. Try using a different value for 'missing_action'.\n");

        /* TODO: make this work, can end up messing with the start and end indices somehow */
        /* It should also consider that 'split_ix' might not match when using missing_action == Impute */
        // if (input_data.Xc_indptr == NULL && model_params.missing_action == Fail && workspace.ntaken == 1)
        //     goto follow_branches;
        
        if (input_data.Xc_indptr == NULL)
            divide_subset_split(workspace.ix_arr.data(), input_data.numeric_data + input_data.nrows * trees.back().col_num,
                                workspace.st, workspace.end, trees.back().num_split, model_params.missing_action,
                                workspace.st_NA, workspace.end_NA, workspace.split_ix);
        else
            divide_subset_split(workspace.ix_arr.data(), workspace.st, workspace.end, trees.back().col_num,
                                input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr, trees.back().num_split,
                                model_params.missing_action, workspace.st_NA, workspace.end_NA, workspace.split_ix);
    } 

    /* for categorical, there are different ways of splitting */
    else
    {
        /* if the columns is binary, there's only one possible split */
        if (input_data.ncat[trees.back().col_num] <= 2)
        {
            trees.back().chosen_cat = 0;
            divide_subset_split(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * trees.back().col_num,
                                workspace.st, workspace.end, (int)0, model_params.missing_action,
                                workspace.st_NA, workspace.end_NA, workspace.split_ix);
            trees.back().cat_split.clear();
            trees.back().cat_split.shrink_to_fit();
        }

        /* otherwise, split according to desired type (single/subset) */
        /* TODO: refactor this */
        else 
        {

            switch (model_params.cat_split_type)
            {

                case SingleCateg:
                {

                    if (workspace.determine_split)
                    {
                        switch (workspace.criterion)
                        {
                            case NoCrit:
                            {
                                trees.back().chosen_cat = choose_cat_from_present(workspace, input_data, trees.back().col_num);
                                break;
                            }

                            default:
                            {
                                if (!workspace.changed_weights)
                                    workspace.this_gain =
                                        eval_guided_crit<ldouble_safe>(
                                                         workspace.ix_arr.data(), workspace.st, workspace.end,
                                                         input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                         &workspace.best_cat_mode,
                                                         workspace.buffer_szt.data(), workspace.buffer_szt.data() + input_data.max_categ,
                                                         workspace.buffer_dbl.data(), trees.back().chosen_cat, workspace.this_split_categ.data(),
                                                         workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                         model_params.all_perm, model_params.missing_action, model_params.cat_split_type);
                                else if (!workspace.weights_arr.empty())
                                    workspace.this_gain =
                                        eval_guided_crit_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                                                  workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                  input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                                  &workspace.best_cat_mode,
                                                                  workspace.buffer_szt.data(),
                                                                  workspace.buffer_dbl.data(), trees.back().chosen_cat, workspace.this_split_categ.data(),
                                                                  workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                                  model_params.all_perm, model_params.missing_action, model_params.cat_split_type,
                                                                  workspace.weights_arr);
                                else
                                    workspace.this_gain =
                                        eval_guided_crit_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                                                  workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                  input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                                  &workspace.best_cat_mode,
                                                                  workspace.buffer_szt.data(),
                                                                  workspace.buffer_dbl.data(), trees.back().chosen_cat, workspace.this_split_categ.data(),
                                                                  workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                                  model_params.all_perm, model_params.missing_action, model_params.cat_split_type,
                                                                  workspace.weights_map);
                                
                                if (std::isnan(workspace.this_gain) || workspace.this_gain <= -HUGE_VAL)
                                    goto terminal_statistics;

                                break;
                            }
                        }
                    }


                    divide_subset_split(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * trees.back().col_num,
                                        workspace.st, workspace.end, trees.back().chosen_cat, model_params.missing_action,
                                        workspace.st_NA, workspace.end_NA, workspace.split_ix);
                    break;
                }


                case SubSet:
                {

                    if (workspace.determine_split)
                    {
                        switch(workspace.criterion)
                        {
                            case NoCrit:
                            {
                                workspace.unsplittable = true;
                                while(workspace.unsplittable)
                                {
                                    workspace.npresent = 0;
                                    workspace.ncols_tried = 0;
                                    for (int cat = 0; cat < input_data.ncat[trees.back().col_num]; cat++)
                                    {
                                        if (workspace.categs[cat] >= 0)
                                        {
                                            workspace.categs[cat]  =  workspace.rbin(workspace.rnd_generator) < 0.5;
                                            workspace.npresent    +=  workspace.categs[cat];
                                            workspace.ncols_tried += !workspace.categs[cat];
                                        }
                                        workspace.unsplittable = !(workspace.npresent && workspace.ncols_tried);
                                    }
                                }

                                trees.back().cat_split.assign(workspace.categs.begin(), workspace.categs.begin() + input_data.ncat[trees.back().col_num]);
                                break; /* NoCrit */
                            }

                            default:
                            {
                                trees.back().cat_split.resize(input_data.ncat[trees.back().col_num]);
                                if (!workspace.changed_weights)
                                    workspace.this_gain =
                                        eval_guided_crit<ldouble_safe>(
                                                         workspace.ix_arr.data(), workspace.st, workspace.end,
                                                         input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                         &workspace.best_cat_mode,
                                                         workspace.buffer_szt.data(), workspace.buffer_szt.data() + input_data.max_categ,
                                                         workspace.buffer_dbl.data(), trees.back().chosen_cat, trees.back().cat_split.data(),
                                                         workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                         model_params.all_perm, model_params.missing_action, model_params.cat_split_type);
                                else if (!workspace.weights_arr.empty())
                                    workspace.this_gain =
                                        eval_guided_crit_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                                                  workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                  input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                                  &workspace.best_cat_mode,
                                                                  workspace.buffer_szt.data(),
                                                                  workspace.buffer_dbl.data(), trees.back().chosen_cat, trees.back().cat_split.data(),
                                                                  workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                                  model_params.all_perm, model_params.missing_action, model_params.cat_split_type,
                                                                  workspace.weights_arr);
                                else
                                    workspace.this_gain =
                                        eval_guided_crit_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                                                  workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                  input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                                  &workspace.best_cat_mode,
                                                                  workspace.buffer_szt.data(),
                                                                  workspace.buffer_dbl.data(), trees.back().chosen_cat, trees.back().cat_split.data(),
                                                                  workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                                  model_params.all_perm, model_params.missing_action, model_params.cat_split_type,
                                                                  workspace.weights_map);

                                if (std::isnan(workspace.this_gain) || workspace.this_gain <= -HUGE_VAL)
                                    goto terminal_statistics;
                                break;
                            }
                        }
                    }

                    if (model_params.new_cat_action == Random)
                    {
                        if (model_params.scoring_metric == Density)
                        {
                            workspace.density_calculator.save_n_present_and_left(trees.back().cat_split.data(), input_data.ncat[trees.back().col_num]);
                        }

                        for (int cat = 0; cat < input_data.ncat[trees.back().col_num]; cat++)
                            if (trees.back().cat_split[cat] < 0)
                                trees.back().cat_split[cat] = workspace.rbin(workspace.rnd_generator) < 0.5;
                    }

                    divide_subset_split(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * trees.back().col_num,
                                        workspace.st, workspace.end, trees.back().cat_split.data(), model_params.missing_action,
                                        workspace.st_NA, workspace.end_NA, workspace.split_ix);
                }

            }

        }

    }


    /* if it hasn't reached the limit, continue splitting from here */
    follow_branches:
    {
        /* add another round of separation depth for distance */
        if (model_params.calc_dist && curr_depth > 0)
            add_separation_step(workspace, input_data, (double)(-1));

        /* if it split by a categorical variable with only 2 values,
           the column will no longer be splittable in either branch */
        if (trees.back().col_type == Categorical &&
            ((model_params.cat_split_type == SubSet && trees.back().cat_split.empty()) ||
             (model_params.cat_split_type == SingleCateg && input_data.ncat[trees.back().col_num] == 2)))
        {
            workspace.col_sampler.drop_col(trees.back().col_num + input_data.ncols_numeric,
                                           workspace.end - workspace.st + 1);
        }
        
        size_t tree_from = trees.size() - 1;
        std::unique_ptr<RecursionState>
        recursion_state(new RecursionState(workspace, model_params.missing_action != Fail));
        trees.back().score = -1;

        /* compute statistics for NAs and remember recursion indices/weights */
        if (model_params.missing_action != Fail)
        {
            if (
                model_params.missing_action == Impute &&
                workspace.criterion != NoCrit &&
                workspace.st_NA < workspace.end_NA
            ) {
                bool move_NAs_left;
                if (trees.back().col_type == Numeric)
                {
                    move_NAs_left = workspace.best_xmedian <= trees.back().num_split;
                }

                else
                {
                    if (trees.back().cat_split.empty())
                        move_NAs_left = workspace.best_cat_mode == trees.back().chosen_cat;
                    else
                        move_NAs_left = trees.back().cat_split[workspace.best_cat_mode] == 1;
                }

                if (move_NAs_left)
                    workspace.st_NA = workspace.end_NA;
                else
                    workspace.end_NA = workspace.st_NA;
            }

            if (!workspace.changed_weights)
            {
                trees.back().pct_tree_left = (ldouble_safe)(workspace.st_NA - workspace.st)
                                                /
                                             (ldouble_safe)(workspace.end - workspace.st + 1 - (workspace.end_NA - workspace.st_NA));

                if (model_params.missing_action == Divide && workspace.st_NA < workspace.end_NA)
                {
                    workspace.changed_weights = true;

                    if (input_data.Xc_indptr != NULL && model_params.sample_size < input_data.nrows / 20) {
                        workspace.weights_arr.clear();
                        workspace.weights_map.reserve(workspace.end - workspace.st + 1);
                        for (size_t row = workspace.st; row < workspace.end_NA; row++)
                            workspace.weights_map[workspace.ix_arr[row]] = 1;
                    }

                    else {
                        workspace.weights_arr.resize(input_data.nrows);
                        for (size_t row = workspace.st; row < workspace.end_NA; row++)
                            workspace.weights_arr[workspace.ix_arr[row]] = 1;
                    }
                }
            }

            else
            {
                ldouble_safe sum_weight_left = 0;
                ldouble_safe sum_weight_right = 0;

                if (!workspace.weights_arr.empty()) {
                    for (size_t row = workspace.st; row < workspace.st_NA; row++)
                        sum_weight_left += workspace.weights_arr[workspace.ix_arr[row]];
                    for (size_t row = workspace.end_NA; row <= workspace.end; row++)
                        sum_weight_right += workspace.weights_arr[workspace.ix_arr[row]];
                }

                else {
                    for (size_t row = workspace.st; row < workspace.st_NA; row++)
                        sum_weight_left += workspace.weights_map[workspace.ix_arr[row]];
                    for (size_t row = workspace.end_NA; row <= workspace.end; row++)
                        sum_weight_right += workspace.weights_map[workspace.ix_arr[row]];
                }

                trees.back().pct_tree_left = sum_weight_left / (sum_weight_left + sum_weight_right);
            }

            switch(model_params.missing_action)
            {
                case Impute:
                {
                    if (trees.back().pct_tree_left >= .5)
                        workspace.end = workspace.end_NA - 1;
                    else
                        workspace.end = workspace.st_NA - 1;
                    break;
                }


                case Divide:
                {
                    if (!workspace.weights_arr.empty())
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_arr[workspace.ix_arr[row]] *= trees.back().pct_tree_left;
                    else
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_map[workspace.ix_arr[row]] *= trees.back().pct_tree_left;
                    workspace.end = workspace.end_NA - 1;
                    break;
                }

                default:
                {
                    unexpected_error();
                    break;
                }
            }
        }

        else
        {
            trees.back().pct_tree_left = (ldouble_safe) (workspace.split_ix - workspace.st)
                                            /
                                         (ldouble_safe) (workspace.end - workspace.st + 1);
            workspace.end = workspace.split_ix - 1;
        }

        /* Depending on the scoring metric, might need to calculate fractions of data and volume */
        if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
        {
            switch (trees.back().col_type)
            {
                case Numeric:
                {
                    if (!workspace.determine_split)
                        workspace.density_calculator.restore_range(workspace.xmin, workspace.xmax);

                    if (model_params.scoring_metric == Density)
                        workspace.density_calculator.push_density(workspace.xmin, workspace.xmax, trees.back().num_split);
                    else
                        workspace.density_calculator.push_adj(workspace.xmin, workspace.xmax,
                                                              trees.back().num_split, trees.back().pct_tree_left,
                                                              model_params.scoring_metric);
                    break;
                }

                case Categorical:
                {
                    switch (model_params.cat_split_type)
                    {
                        case SingleCateg:
                        {
                            if (model_params.scoring_metric == Density)
                            {
                                if (workspace.determine_split)
                                {
                                    if (workspace.criterion == NoCrit)
                                        workspace.density_calculator.push_density(workspace.npresent);
                                    else
                                        workspace.density_calculator.push_density(workspace.buffer_szt.data(),
                                                                                  input_data.ncat[trees.back().col_num]);
                                }

                                else
                                {
                                    workspace.density_calculator.push_density(workspace.density_calculator.counts.data(),
                                                                              input_data.ncat[trees.back().col_num]);
                                }
                            }

                            else
                            {
                                if (workspace.determine_split)
                                {
                                    if (workspace.criterion == NoCrit)
                                    {
                                        count_categs(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                     input_data.categ_data + trees.back().col_num * input_data.nrows,
                                                     input_data.ncat[trees.back().col_num],
                                                     workspace.density_calculator.counts.data());
                                        workspace.density_calculator.push_adj(workspace.density_calculator.counts.data(),
                                                                              input_data.ncat[trees.back().col_num],
                                                                              trees.back().chosen_cat,
                                                                              model_params.scoring_metric);
                                    }

                                    else
                                    {
                                        workspace.density_calculator.push_adj(workspace.buffer_szt.data(),
                                                                              input_data.ncat[trees.back().col_num],
                                                                              trees.back().chosen_cat,
                                                                              model_params.scoring_metric);
                                    }
                                }

                                else
                                {

                                    workspace.density_calculator.push_adj(workspace.density_calculator.counts.data(),
                                                                          input_data.ncat[trees.back().col_num],
                                                                          trees.back().chosen_cat,
                                                                          model_params.scoring_metric);
                                }
                            }
                            break;
                        }

                        case SubSet:
                        {
                            if (model_params.scoring_metric == Density)
                            {
                                if (!trees.back().cat_split.size())
                                {
                                    workspace.density_calculator.push_density();
                                }
                                
                                else
                                {
                                    workspace.density_calculator.push_density(workspace.density_calculator.n_left,
                                                                              workspace.density_calculator.n_present);
                                }

                            }

                            else
                            {
                                if (!trees.back().cat_split.size())
                                {
                                    workspace.density_calculator.push_adj(trees.back().pct_tree_left,
                                                                          model_params.scoring_metric);
                                }

                                else
                                {
                                    if (workspace.determine_split)
                                    {
                                        if (workspace.criterion == NoCrit)
                                        {
                                            count_categs(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                         input_data.categ_data + trees.back().col_num * input_data.nrows,
                                                         input_data.ncat[trees.back().col_num],
                                                         workspace.density_calculator.counts.data());
                                            workspace.density_calculator.push_adj(trees.back().cat_split.data(),
                                                                                  workspace.density_calculator.counts.data(),
                                                                                  input_data.ncat[trees.back().col_num],
                                                                                  model_params.scoring_metric);
                                        }

                                        else
                                        {
                                            workspace.density_calculator.push_adj(trees.back().cat_split.data(),
                                                                                  workspace.buffer_szt.data(),
                                                                                  input_data.ncat[trees.back().col_num],
                                                                                  model_params.scoring_metric);
                                        }
                                    }

                                    else
                                    {
                                        workspace.density_calculator.push_adj(trees.back().cat_split.data(),
                                                                              workspace.density_calculator.counts.data(),
                                                                              input_data.ncat[trees.back().col_num],
                                                                              model_params.scoring_metric);
                                    }
                                }
                            }
                            break;
                        }
                    }
                    break;
                }

                default:
                {
                    assert(0);
                }
            }
        }

        else if (is_boxed_metric(model_params.scoring_metric))
        {
            switch (trees.back().col_type)
            {
                case Numeric:
                {
                    workspace.density_calculator.push_bdens(trees.back().num_split, trees.back().col_num);
                    break;
                }

                case Categorical:
                {
                    switch (model_params.cat_split_type)
                    {
                        case SingleCateg:
                        {
                            workspace.density_calculator.push_bdens((int)1, trees.back().col_num);
                            break;
                        }

                        case SubSet:
                        {
                            if (trees.back().cat_split.empty())
                            {
                                workspace.density_calculator.push_bdens((int)1, trees.back().col_num);
                            }

                            else
                            {
                                workspace.density_calculator.push_bdens(trees.back().cat_split, trees.back().col_num);
                            }
                            break;
                        }
                    }
                    break;
                }

                default:
                {
                    assert(0);
                }
            }
        }

        /* Branch where to assign new categories can be pre-determined in this case */
        if (
            trees.back().col_type       == Categorical &&
            model_params.cat_split_type == SubSet      &&
            input_data.ncat[trees.back().col_num] > 2  &&
            model_params.new_cat_action == Smallest
            )
        {
            bool new_to_left = trees.back().pct_tree_left < 0.5;
            for (int cat = 0; cat < input_data.ncat[trees.back().col_num]; cat++)
                if (trees.back().cat_split[cat] < 0)
                    trees.back().cat_split[cat] = new_to_left;
        }

        /* If doing single-category splits, the branch that got only one category will not
           be splittable anymore, so it can be dropped for the remainder of that branch */
        if (trees.back().col_type == Categorical &&
            model_params.cat_split_type == SingleCateg &&
            input_data.ncat[trees.back().col_num] > 2 /* <- in this case, would have been dropped earlier */
            )
        {
            workspace.col_sampler.drop_col(trees.back().col_num + input_data.ncols_numeric,
                                           workspace.end - workspace.st + 1);
        }

        /* left branch */
        trees.back().tree_left = trees.size();
        trees.emplace_back();
        if (impute_nodes != NULL) impute_nodes->emplace_back(tree_from);
        split_itree_recursive<InputData, WorkerMemory, ldouble_safe>(
                              trees,
                              workspace,
                              input_data,
                              model_params,
                              impute_nodes,
                              curr_depth + 1);


        /* right branch */
        recursion_state->restore_state(workspace);
        if (is_boxed_metric(model_params.scoring_metric))
        {
            if (trees[tree_from].col_type == Numeric)
                workspace.density_calculator.pop_bdens(trees[tree_from].col_num);
            else
                workspace.density_calculator.pop_bdens_cat(trees[tree_from].col_num);
        }
        else if (model_params.scoring_metric != Depth)
        {
            workspace.density_calculator.pop();
        }
        if (model_params.missing_action != Fail)
        {
            switch(model_params.missing_action)
            {
                case Impute:
                {
                    if (trees[tree_from].pct_tree_left >= .5)
                        workspace.st = workspace.end_NA;
                    else
                        workspace.st = workspace.st_NA;
                    break;
                }

                case Divide:
                {
                    if (!workspace.changed_weights && workspace.st_NA < workspace.end_NA)
                    {
                        workspace.changed_weights = true;

                        if (!workspace.weights_arr.empty()) {
                            for (size_t row = workspace.st_NA; row <= workspace.end; row++)
                                workspace.weights_arr[workspace.ix_arr[row]] = 1;
                        }

                        else {
                            for (size_t row = workspace.st_NA; row <= workspace.end; row++)
                                workspace.weights_map[workspace.ix_arr[row]] = 1;
                        }
                    }

                    if (!workspace.weights_arr.empty())
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_arr[workspace.ix_arr[row]] *= (1. - trees[tree_from].pct_tree_left);
                    else
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_map[workspace.ix_arr[row]] *= (1. - trees[tree_from].pct_tree_left);
                    workspace.st = workspace.st_NA;
                    break;
                }

                default:
                {
                    unexpected_error();
                    break;
                }
            }
        }

        else
        {
            workspace.st = workspace.split_ix;
        }

        trees[tree_from].tree_right = trees.size();
        trees.emplace_back();
        if (impute_nodes != NULL) impute_nodes->emplace_back(tree_from);
        split_itree_recursive<InputData, WorkerMemory, ldouble_safe>(
                              trees,
                              workspace,
                              input_data,
                              model_params,
                              impute_nodes,
                              curr_depth + 1);
        if (is_boxed_metric(model_params.scoring_metric))
        {
            if (trees[tree_from].col_type == Numeric)
                workspace.density_calculator.pop_bdens_right(trees[tree_from].col_num);
            else
                workspace.density_calculator.pop_bdens_cat_right(trees[tree_from].col_num);
        }
        else if (model_params.scoring_metric != Depth)
        {
            workspace.density_calculator.pop_right();
        }
    }
    return;

    /* if it reached the limit, calculate terminal statistics */
    terminal_statistics:
    {
        trees.back().tree_left = 0;

        if (workspace.changed_weights)
        {
            if (sum_weight <= -HUGE_VAL)
                sum_weight = calculate_sum_weights<ldouble_safe>(
                                                   workspace.ix_arr, workspace.st, workspace.end, curr_depth,
                                                   workspace.weights_arr, workspace.weights_map);
        }

        switch (model_params.scoring_metric)
        {
            case Depth:
            {
                if (!workspace.changed_weights)
                    trees.back().score = curr_depth + expected_avg_depth<ldouble_safe>(workspace.end - workspace.st + 1);
                else
                    trees.back().score = curr_depth + expected_avg_depth<ldouble_safe>(sum_weight);
                break;
            }

            case AdjDepth:
            {
                if (!workspace.changed_weights)
                    trees.back().score = workspace.density_calculator.calc_adj_depth() + expected_avg_depth<ldouble_safe>(workspace.end - workspace.st + 1);
                else
                    trees.back().score = workspace.density_calculator.calc_adj_depth() + expected_avg_depth<ldouble_safe>(sum_weight);
                break;
            }

            case Density:
            {
                if (!workspace.changed_weights)
                    trees.back().score = workspace.density_calculator.calc_density(workspace.end - workspace.st + 1, model_params.sample_size);
                else
                    trees.back().score = workspace.density_calculator.calc_density(sum_weight, model_params.sample_size);
                break;
            }

            case AdjDensity:
            {
                trees.back().score = workspace.density_calculator.calc_adj_density();
                break;
            }

            case BoxedRatio:
            {
                trees.back().score = workspace.density_calculator.calc_bratio();
                break;
            }

            case BoxedDensity:
            {
                if (!workspace.changed_weights)
                    trees.back().score = workspace.density_calculator.calc_bdens(workspace.end - workspace.st + 1, model_params.sample_size);
                else
                    trees.back().score = workspace.density_calculator.calc_bdens(sum_weight, model_params.sample_size);
                break;
            }

            case BoxedDensity2:
            {
                if (!workspace.changed_weights)
                    trees.back().score = workspace.density_calculator.calc_bdens2(workspace.end - workspace.st + 1, model_params.sample_size);
                else
                    trees.back().score = workspace.density_calculator.calc_bdens2(sum_weight, model_params.sample_size);
                break;
            }
        }

        trees.back().cat_split.clear();
        trees.back().cat_split.shrink_to_fit();

        trees.back().remainder = workspace.changed_weights?
                                    (double)sum_weight : (double)(workspace.end - workspace.st + 1);

        /* for distance, assume also the elements keep being split */
        if (model_params.calc_dist)
            add_remainder_separation_steps<InputData, WorkerMemory, ldouble_safe>(workspace, input_data, sum_weight);

        /* add this depth right away if requested */
        if (workspace.row_depths.size())
        {
            if (!workspace.changed_weights)
            {
                for (size_t row = workspace.st; row <= workspace.end; row++)
                    workspace.row_depths[workspace.ix_arr[row]] += trees.back().score;
            }

            else if (!workspace.weights_arr.empty())
            {
                for (size_t row = workspace.st; row <= workspace.end; row++)
                    workspace.row_depths[workspace.ix_arr[row]] += workspace.weights_arr[workspace.ix_arr[row]] * trees.back().score;
            }

            else
            {
                for (size_t row = workspace.st; row <= workspace.end; row++)
                    workspace.row_depths[workspace.ix_arr[row]] += workspace.weights_map[workspace.ix_arr[row]] * trees.back().score;
            }
        }

        /* add imputations from node if requested */
        if (model_params.impute_at_fit)
            add_from_impute_node(impute_nodes->back(), workspace, input_data);
    }

}
