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
*     Copyright (c) 2019, David Cortes
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

void decide_column(size_t ncols_numeric, size_t ncols_categ, size_t &col_chosen, ColType &col_type,
                   RNG_engine &rnd_generator, std::uniform_int_distribution<size_t> &runif,
                   std::discrete_distribution<size_t> &col_sampler)
{
    if (!col_sampler.max())
        col_chosen = runif(rnd_generator);
    else
        col_chosen = col_sampler(rnd_generator);

    if (col_chosen >= ncols_numeric)
    {
        col_chosen -= ncols_numeric;
        col_type    = Categorical;
    }

    else { col_type = Numeric; }
}

void add_unsplittable_col(WorkerMemory &workspace, IsoTree &tree, InputData &input_data)
{
    if (tree.col_type == Numeric)
        workspace.cols_possible[tree.col_num] = false;
    else
        workspace.cols_possible[tree.col_num + input_data.ncols_numeric] = false;
}

void add_unsplittable_col(WorkerMemory &workspace, InputData &input_data)
{
    if (workspace.col_type == Numeric)
        workspace.cols_possible[workspace.col_chosen] = false;
    else
        workspace.cols_possible[workspace.col_chosen + input_data.ncols_numeric] = false;
}

bool check_is_not_unsplittable_col(WorkerMemory &workspace, IsoTree &tree, InputData &input_data)
{
    if (tree.col_type == Numeric)
        return workspace.cols_possible[tree.col_num];
    else
        return workspace.cols_possible[tree.col_num + input_data.ncols_numeric];
}

/* for use in regular model */
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params, IsoTree &tree)
{
    if (tree.col_type == Numeric)
    {
        if (input_data.Xc == NULL)
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
        get_categs(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * tree.col_num,
                   workspace.st, workspace.end, input_data.ncat[tree.col_num],
                   model_params.missing_action, workspace.categs.data(), workspace.npresent, workspace.unsplittable);
    }
}

/* for use in extended model */
void get_split_range(WorkerMemory &workspace, InputData &input_data, ModelParams &model_params)
{
    if (workspace.col_type == Numeric)
    {
        if (input_data.Xc == NULL)
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
        get_categs(workspace.ix_arr.data(), input_data.categ_data + input_data.nrows * workspace.col_chosen,
                   workspace.st, workspace.end, input_data.ncat[workspace.col_chosen],
                   model_params.missing_action, workspace.categs.data(), workspace.npresent, workspace.unsplittable);
    }
}

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

void update_col_sampler(WorkerMemory &workspace, InputData &input_data)
{
    if (!workspace.col_sampler.max())
        return;

    std::vector<double> col_weights = workspace.col_sampler.probabilities();
    for (size_t col = 0; col < input_data.ncols_numeric; col++)
        if (!workspace.cols_possible[col])
            col_weights[col] = 0;
    for (size_t col = 0; col < input_data.ncols_categ; col++)
        if (!workspace.cols_possible[col + input_data.ncols_numeric])
            col_weights[col + input_data.ncols_numeric] = 0;
    workspace.col_sampler = std::discrete_distribution<size_t>(col_weights.begin(), col_weights.end());
}

bool is_col_taken(std::vector<bool> &col_is_taken, std::unordered_set<size_t> &col_is_taken_s,
                  InputData &input_data, size_t col_num, ColType col_type)
{
    col_num += ((col_type == Categorical)? 0 : input_data.ncols_categ);
    if (col_is_taken.size())
        return col_is_taken[col_num];
    else
        return col_is_taken_s.find(col_num) != col_is_taken_s.end();
}

void set_col_as_taken(std::vector<bool> &col_is_taken, std::unordered_set<size_t> &col_is_taken_s,
                      InputData &input_data, size_t col_num, ColType col_type)
{
    col_num += ((col_type == Categorical)? 0 : input_data.ncols_categ);
    if (col_is_taken.size())
        col_is_taken[col_num] = true;
    else
        col_is_taken_s.insert(col_num);
}

void add_separation_step(WorkerMemory &workspace, InputData &input_data, double remainder)
{
    if (workspace.weights_arr.size())
        increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                              input_data.nrows, workspace.tmat_sep.data(), workspace.weights_arr.data(), remainder);
    else if (workspace.weights_map.size())
        increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                              input_data.nrows, workspace.tmat_sep.data(), workspace.weights_map, remainder);
    else
        increase_comb_counter(workspace.ix_arr.data(), workspace.st, workspace.end,
                              input_data.nrows, workspace.tmat_sep.data(), remainder);
}

void add_remainder_separation_steps(WorkerMemory &workspace, InputData &input_data, long double sum_weight)
{
    if (
            ((workspace.end - workspace.st) > 0 && !workspace.weights_arr.size() && !workspace.weights_map.size()) ||
            (sum_weight > 1 && (workspace.weights_arr.size() || workspace.weights_map.size()))
        )
    {
        double expected_dsep;
        if (!workspace.weights_arr.size() && !workspace.weights_map.size())
            expected_dsep = expected_separation_depth(workspace.end - workspace.st + 1);
        else
            expected_dsep = expected_separation_depth(sum_weight);

        add_separation_step(workspace, input_data, expected_dsep + 1);
    }
}

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
                if (model_outputs->trees[tree][node].score >= 0)
                    tree_mapping[node] = curr_term++;

            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(tree_num, tree_mapping, tree, prediction_data)
            for (size_t_for row = 0; row < prediction_data.nrows; row++)
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
                if (model_outputs_ext->hplanes[tree][node].score >= 0)
                    tree_mapping[node] = curr_term++;
            
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(tree_num, tree_mapping, tree, prediction_data)
            for (size_t_for row = 0; row < prediction_data.nrows; row++)
                tree_num[row + tree * prediction_data.nrows] = tree_mapping[tree_num[row + tree * prediction_data.nrows]];
        }
    }
}

void backup_recursion_state(WorkerMemory &workspace, RecursionState &recursion_state)
{
    recursion_state.st            = workspace.st;
    recursion_state.st_NA         = workspace.st_NA;
    recursion_state.end_NA        = workspace.end_NA;
    recursion_state.split_ix      = workspace.split_ix;
    recursion_state.end           = workspace.end;
    recursion_state.cols_possible = workspace.cols_possible;
    recursion_state.col_sampler   = workspace.col_sampler;

    /* for the extended model, it's not necessary to copy everything */
    if (!workspace.comb_val.size())
    {
        /* TODO: here only need to copy the left half, as the right one is untouched */
        recursion_state.ix_arr        = workspace.ix_arr;
        recursion_state.weights_map   = workspace.weights_map;
        recursion_state.weights_arr   = workspace.weights_arr;
    }
}


void restore_recursion_state(WorkerMemory &workspace, RecursionState &recursion_state)
{
    workspace.st            = recursion_state.st;
    workspace.st_NA         = recursion_state.st_NA;
    workspace.end_NA        = recursion_state.end_NA;
    workspace.split_ix      = recursion_state.split_ix;
    workspace.end           = recursion_state.end;
    workspace.cols_possible = std::move(recursion_state.cols_possible);
    workspace.col_sampler   = std::move(recursion_state.col_sampler);

    if (!workspace.comb_val.size())
    {
        /* TODO: here only need to copy the left half, as the right one is untouched */
        workspace.ix_arr        = std::move(recursion_state.ix_arr);
        workspace.weights_map   = std::move(recursion_state.weights_map);
        workspace.weights_arr   = std::move(recursion_state.weights_arr);
    }
}
