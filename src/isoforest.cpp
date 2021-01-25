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
*     Copyright (c) 2020, David Cortes
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

void split_itree_recursive(std::vector<IsoTree>     &trees,
                           WorkerMemory             &workspace,
                           InputData                &input_data,
                           ModelParams              &model_params,
                           std::vector<ImputeNode> *impute_nodes,
                           size_t                   curr_depth)
{
    if (interrupt_switch) return;
    long double sum_weight = -HUGE_VAL;

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

    /* check for potential isolated leafs */
    if (workspace.end == workspace.st || curr_depth >= model_params.max_depth)
        goto terminal_statistics;

    /* with 2 observations and no weights, there's only 1 potential or assumed split */
    if ((workspace.end - workspace.st) == 1 && !workspace.weights_arr.size() && !workspace.weights_map.size())
        goto terminal_statistics;

    /* when using weights, the split should stop when the sum of weights is <= 2 */
    sum_weight = calculate_sum_weights(workspace.ix_arr, workspace.st, workspace.end, curr_depth,
                                       workspace.weights_arr, workspace.weights_map);

    if (curr_depth > 0 && (workspace.weights_arr.size() || workspace.weights_map.size()) && sum_weight < 2.5)
        goto terminal_statistics;

    /* for sparse matrices, need to sort the indices */
    if (input_data.Xc_indptr != NULL && impute_nodes == NULL)
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);

    /* pick column to split according to criteria */
    workspace.prob_split_type = workspace.rbin(workspace.rnd_generator);

    /* case1: guided, pick column and point with best gain */
    if (
            workspace.prob_split_type
                < (
                    model_params.prob_pick_by_gain_avg +
                    model_params.prob_pick_by_gain_pl
                  )
        )
    {
        workspace.determine_split = false;

        /* case 1.1: column is decided by averaged gain */
        if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg)
            workspace.criterion = Averaged;

        /* case 1.2: column is decided by pooled gain */
        else
            workspace.criterion = Pooled;

        /* evaluate gain for all columns */
        trees.back().score = -HUGE_VAL; /* this is used to track the best gain */
        if (input_data.Xc_indptr == NULL)
        {
            for (size_t col = 0; col < input_data.ncols_numeric; col++)
            {
                workspace.this_gain = eval_guided_crit(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                       input_data.numeric_data + col * input_data.nrows,
                                                       workspace.split_ix, workspace.this_split_point,
                                                       workspace.xmin, workspace.xmax,
                                                       workspace.criterion, model_params.min_gain,
                                                       model_params.missing_action);
                if (workspace.this_gain <= -HUGE_VAL)
                {
                    workspace.cols_possible[col] = false;
                }

                else if (workspace.this_gain > trees.back().score)
                {
                    trees.back().score     = workspace.this_gain;
                    trees.back().col_num   = col;
                    trees.back().num_split = workspace.this_split_point;
                    if (model_params.penalize_range)
                    {
                        trees.back().range_low  = workspace.xmin - workspace.xmax + trees.back().num_split;
                        trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
                    }
                }
            }

        }

        else
        {
            for (size_t col = 0; col < input_data.ncols_numeric; col++)
            {
                workspace.this_gain = eval_guided_crit(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                       col, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                       workspace.buffer_dbl.data(), workspace.buffer_szt.data(),
                                                       workspace.this_split_point, workspace.xmin, workspace.xmax,
                                                       workspace.criterion, model_params.min_gain, model_params.missing_action);
                if (workspace.this_gain <= -HUGE_VAL)
                {
                    workspace.cols_possible[col] = false;
                }

                else if (workspace.this_gain > trees.back().score)
                {
                    trees.back().score     = workspace.this_gain;
                    trees.back().col_num   = col;
                    trees.back().num_split = workspace.this_split_point;
                    if (model_params.penalize_range)
                    {
                        trees.back().range_low  = workspace.xmin - workspace.xmax + trees.back().num_split;
                        trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
                    }
                }
            }
        }

        for (size_t col = 0; col < input_data.ncols_categ; col++)
        {
            workspace.this_gain = eval_guided_crit(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                   input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                                   workspace.buffer_szt.data(), workspace.buffer_szt.data() + input_data.max_categ,
                                                   workspace.buffer_dbl.data(), workspace.this_categ, workspace.this_split_categ.data(),
                                                   workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                   model_params.all_perm, model_params.missing_action, model_params.cat_split_type);
            if (workspace.this_gain <= -HUGE_VAL)
            {
                workspace.cols_possible[col + input_data.ncols_numeric] = false;
            }

            else if (workspace.this_gain > trees.back().score)
            {
                trees.back().score    = workspace.this_gain;
                trees.back().col_num  = col + input_data.ncols_numeric;
                switch(model_params.cat_split_type)
                {
                    case SingleCateg:
                    {
                        trees.back().chosen_cat = workspace.this_categ;
                        break;
                    }

                    case SubSet:
                    {
                        trees.back().cat_split.assign(workspace.this_split_categ.begin(),
                                                      workspace.this_split_categ.begin() + input_data.ncat[col]);
                        break;
                    }
                }
            }
        }


        if (trees.back().score <= 0.)
            goto terminal_statistics;
        else
            trees.back().score = 0.;

        if (trees.back().col_num < input_data.ncols_numeric)
        {
            trees.back().col_type = Numeric;
        }

        else
        {
            trees.back().col_type  = Categorical;
            trees.back().col_num  -= input_data.ncols_numeric;
        }
    }

    /* case2: column is chosen at random */
    else
    {
        workspace.determine_split = true;

        /* case 2.1: split point is chosen according to gain (averaged) */
        if (
            workspace.prob_split_type
                < (
                    model_params.prob_pick_by_gain_avg +
                    model_params.prob_pick_by_gain_pl  +
                    model_params.prob_split_by_gain_avg
                  )
            )
            workspace.criterion = Averaged;

        /* case 2.2: split point is chosen according to gain (pooled) */
        else if (
                    workspace.prob_split_type
                        < (
                            model_params.prob_pick_by_gain_avg +
                            model_params.prob_pick_by_gain_pl  +
                            model_params.prob_split_by_gain_avg  +
                            model_params.prob_split_by_gain_pl
                          )
            )
            workspace.criterion = Pooled;

        /* case 2.3: split point is chosen randomly (like in the original paper) */
        else
            workspace.criterion = NoCrit;

        if (workspace.go_to_shuffle)
            goto probe_all;


        /* pick column at random */
        decide_column(input_data.ncols_numeric, input_data.ncols_categ,
                      trees.back().col_num, trees.back().col_type,
                      workspace.rnd_generator, workspace.runif,
                      workspace.col_sampler);

        /* get the range of possible splits */
        get_split_range(workspace, input_data, model_params, trees.back());

        /* if it's not possible to split, will have to try more */
        if (workspace.unsplittable)
        {
            /* keep track of which columns are tried */
            add_unsplittable_col(workspace, trees.back(), input_data);

            /* try more random columns for {(1/2) * ncols} times */
            workspace.ncols_tried = 1;
            do
            {
                decide_column(input_data.ncols_numeric, input_data.ncols_categ,
                              trees.back().col_num, trees.back().col_type,
                              workspace.rnd_generator, workspace.runif,
                              workspace.col_sampler);
                if (check_is_splittable_col(workspace, trees.back(), input_data))
                {
                    get_split_range(workspace, input_data, model_params, trees.back());
                    if (!workspace.unsplittable)
                        break;
                    else
                        add_unsplittable_col(workspace, trees.back(), input_data);
                }
                workspace.ncols_tried++;
            }
            while (workspace.ncols_tried < input_data.ncols_tot / 2);

            /* if that didn't work, try to check all columns in random order */
            if (workspace.unsplittable)
            {
                probe_all:
                workspace.go_to_shuffle = true;
                if (!workspace.col_sampler.is_initialized())
                {
                    if (workspace.cols_shuffled.size() < input_data.ncols_tot)
                        workspace.cols_shuffled.resize(input_data.ncols_tot);
                    std::iota(workspace.cols_shuffled.begin(), workspace.cols_shuffled.end(), (size_t)0);
                    std::shuffle(workspace.cols_shuffled.begin(), workspace.cols_shuffled.end(), workspace.rnd_generator);
                }

                else
                {
                    workspace.col_sampler.shuffle_cols(workspace.cols_shuffled, workspace.rnd_generator);
                }
                
                for (size_t col : workspace.cols_shuffled)
                {
                    if (!workspace.cols_possible[col]) continue;
                    
                    if (col < input_data.ncols_numeric)
                    {
                        if (input_data.Xc_indptr == NULL)
                            get_range(workspace.ix_arr.data(), input_data.numeric_data + input_data.nrows * col,
                                      workspace.st, workspace.end, model_params.missing_action,
                                      workspace.xmin, workspace.xmax, workspace.unsplittable);
                        else
                            get_range(workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                      input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                      model_params.missing_action, workspace.xmin, workspace.xmax, workspace.unsplittable);
                    }

                    else
                    {
                        get_categs(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * (col - input_data.ncols_numeric),
                                   workspace.st, workspace.end, input_data.ncat[col - input_data.ncols_numeric],
                                   model_params.missing_action, workspace.categs.data(), workspace.npresent, workspace.unsplittable);
                    }

                    if (workspace.unsplittable)
                    {
                        workspace.cols_possible[col] = !workspace.unsplittable;
                        workspace.col_sampler.drop_col(col);
                    }

                    else
                    {
                        if (col < input_data.ncols_numeric)
                        {
                            trees.back().col_num  = col;
                            trees.back().col_type = Numeric;
                        }

                        else
                        {
                            trees.back().col_num  = col - input_data.ncols_numeric;
                            trees.back().col_type = Categorical;
                        }

                        goto picked_col;
                    }
                }
                goto terminal_statistics;
            }

            /* finally, check the range if needed, and later decide on the split point */
            picked_col:
            if (workspace.criterion == NoCrit)
                get_split_range(workspace, input_data, model_params, trees.back());
        }

    }


    /* for numeric, choose a random point, or pick the best point as determined earlier */
    if (trees.back().col_type == Numeric)
    {
        if (workspace.determine_split)
        {
            switch(workspace.criterion)
            {
                case NoCrit:
                {
                    trees.back().num_split = std::uniform_real_distribution<double>
                                                (workspace.xmin, workspace.xmax)
                                                (workspace.rnd_generator);
                    break;
                }

                default:
                {
                    if (input_data.Xc_indptr == NULL)
                    {
                        eval_guided_crit(workspace.ix_arr.data(), workspace.st, workspace.end,
                                         input_data.numeric_data + trees.back().col_num * input_data.nrows,
                                         workspace.split_ix, trees.back().num_split,
                                         workspace.xmin, workspace.xmax,
                                         workspace.criterion, model_params.min_gain,
                                         model_params.missing_action);
                        if (model_params.missing_action == Fail) /* data is already split */
                        {
                            workspace.split_ix++;
                            goto follow_branches;
                        }
                    }

                    else
                    {
                        eval_guided_crit(workspace.ix_arr.data(), workspace.st, workspace.end,
                                         trees.back().col_num, input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                         workspace.buffer_dbl.data(), workspace.buffer_szt.data(),
                                         trees.back().num_split, workspace.xmin, workspace.xmax,
                                         workspace.criterion, model_params.min_gain,
                                         model_params.missing_action);
                    }
                    break;
                }
            }

            if (model_params.penalize_range)
            {
                trees.back().range_low  = workspace.xmin - workspace.xmax + trees.back().num_split;
                trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
            }
        }
        
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

            switch(model_params.cat_split_type)
            {

                case SingleCateg:
                {

                    if (workspace.determine_split)
                    {
                        switch(workspace.criterion)
                        {
                            case NoCrit:
                            {
                                trees.back().chosen_cat = choose_cat_from_present(workspace, input_data, trees.back().col_num);
                                break;
                            }

                            default:
                            {
                                eval_guided_crit(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                 input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                 workspace.buffer_szt.data(), workspace.buffer_szt.data() + input_data.max_categ,
                                                 workspace.buffer_dbl.data(), trees.back().chosen_cat, workspace.this_split_categ.data(),
                                                 workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                 model_params.all_perm, model_params.missing_action, model_params.cat_split_type);
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
                                eval_guided_crit(workspace.ix_arr.data(), workspace.st, workspace.end,
                                                 input_data.categ_data + trees.back().col_num * input_data.nrows, input_data.ncat[trees.back().col_num],
                                                 workspace.buffer_szt.data(), workspace.buffer_szt.data() + input_data.max_categ,
                                                 workspace.buffer_dbl.data(), trees.back().chosen_cat, trees.back().cat_split.data(),
                                                 workspace.buffer_chr.data(), workspace.criterion, model_params.min_gain,
                                                 model_params.all_perm, model_params.missing_action, model_params.cat_split_type);
                                break;
                            }
                        }
                    }

                    if (model_params.new_cat_action == Random)
                        for (int cat = 0; cat < input_data.ncat[trees.back().col_num]; cat++)
                            if (trees.back().cat_split[cat] < 0)
                                trees.back().cat_split[cat] = workspace.rbin(workspace.rnd_generator) < 0.5;

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
        
        size_t tree_from = trees.size() - 1;
        std::unique_ptr<RecursionState>
        recursion_state(new RecursionState(workspace, model_params.missing_action != Fail));
        trees.back().score = -1;

        /* compute statistics for NAs and remember recursion indices/weights */
        if (model_params.missing_action != Fail)
        {
            trees.back().pct_tree_left = (long double)(workspace.st_NA - workspace.st)
                                            /
                                         (long double)(workspace.end - workspace.st + 1 - (workspace.end_NA - workspace.st_NA));

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
                    if (workspace.weights_map.size())
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_map[workspace.ix_arr[row]] *= trees.back().pct_tree_left;
                    else
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_arr[workspace.ix_arr[row]] *= trees.back().pct_tree_left;
                    workspace.end = workspace.end_NA - 1;
                    break;
                }
            }
        }

        else
        {
            trees.back().pct_tree_left = (long double) (workspace.split_ix - workspace.st)
                                            /
                                         (long double) (workspace.end - workspace.st + 1);
            workspace.end = workspace.split_ix - 1;
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

        /* left branch */
        trees.back().tree_left = trees.size();
        trees.emplace_back();
        if (impute_nodes != NULL) impute_nodes->emplace_back(tree_from);
        split_itree_recursive(trees,
                              workspace,
                              input_data,
                              model_params,
                              impute_nodes,
                              curr_depth + 1);


        /* right branch */
        recursion_state->restore_state(workspace);
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
                    if (workspace.weights_map.size())
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_map[workspace.ix_arr[row]] *= (1 - trees[tree_from].pct_tree_left);
                    else
                        for (size_t row = workspace.st_NA; row < workspace.end_NA; row++)
                            workspace.weights_arr[workspace.ix_arr[row]] *= (1 - trees[tree_from].pct_tree_left);
                    workspace.st = workspace.st_NA;
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
        split_itree_recursive(trees,
                              workspace,
                              input_data,
                              model_params,
                              impute_nodes,
                              curr_depth + 1);
    }
    return;

    /* if it reached the limit, calculate terminal statistics */
    terminal_statistics:
    {
        if (!workspace.weights_arr.size() && !workspace.weights_map.size())
        {
            trees.back().score = (double)(curr_depth + expected_avg_depth(workspace.end - workspace.st + 1));
        }

        else
        {
            if (sum_weight == -HUGE_VAL)
                sum_weight = calculate_sum_weights(workspace.ix_arr, workspace.st, workspace.end, curr_depth,
                                                   workspace.weights_arr, workspace.weights_map);
            trees.back().score = (double)(curr_depth + expected_avg_depth(sum_weight));
        }

        trees.back().cat_split.clear();
        trees.back().cat_split.shrink_to_fit();

        trees.back().remainder = workspace.weights_arr.size()?
                                 sum_weight : (workspace.weights_map.size()?
                                               sum_weight : ((double)(workspace.end - workspace.st + 1))
                                               );

        /* for distance, assume also the elements keep being split */
        if (model_params.calc_dist)
            add_remainder_separation_steps(workspace, input_data, sum_weight);

        /* add this depth right away if requested */
        if (workspace.row_depths.size())
        {
            if (!workspace.weights_arr.size() && !workspace.weights_map.size())
            {
                for (size_t row = workspace.st; row <= workspace.end; row++)
                    workspace.row_depths[workspace.ix_arr[row]] += trees.back().score;
            }

            else if (workspace.weights_arr.size())
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
