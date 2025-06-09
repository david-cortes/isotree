/*    Isolation forests and variations thereof, with adjustments for incorporation
*     of categorical variables and missing values.
*     Written for C++11 standard and aimed at being used in R and Python.
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


/* TODO: this file is a complete mess, needs a refactor from scratch along with the data structs */

/* Impute missing values in new data
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric] (in, out)
*       Pointer to numeric data in which missing values will be imputed. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
*       Imputations will overwrite values in this same array.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data in which missing values will be imputed. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no categorical columns.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
*       Imputations will overwrite values in this same array.
* - is_col_major
*       Whether 'numeric_data' and 'categ_data' come in column-major order, like the data to which the
*       model was fit. If passing 'false', will assume they are in row-major order. Note that most of
*       the functions in this library work only with column-major order, but here both are suitable
*       and row-major is preferred. Both arrays must have the same orientation (row/column major).
* - ncols_categ
*       Number of categorical columns in the data.
* - ncat[ncols_categ]
*       Number of categories in each categorical column. E.g. if the highest code for a column is '4',
*       the number of categories for that column is '5' (zero is one category).
*       Must be the same as was passed to 'fit_iforest'.
* - Xr[nnz] (in, out)
*       Pointer to numeric data in sparse numeric matrix in CSR format (row-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
*       Imputations will overwrite values in this same array.
* - Xr_ind[nnz]
*       Pointer to column indices to which each non-zero entry in 'Xr' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSR format.
* - Xr_indptr[nrows + 1]
*       Pointer to row index pointers that tell at entry [row] where does row 'row'
*       start and at entry [row + 1] where does row 'row' end.
*       Pass NULL if there are no sparse numeric columns in CSR format.
* - nrows
*       Number of rows in 'numeric_data', 'Xc', 'Xr, 'categ_data'.
* - use_long_double
*       Whether to use 'long double' (extended precision) type for the calculations. This makes them
*       more accurate (provided that the compiler used has wider long doubles than doubles), but
*       slower - especially in platforms in which 'long double' is a software-emulated type (e.g.
*       Power8 platforms).
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - impute_nodes
*       Pointer to fitted imputation node objects for the same trees as in 'model_outputs' or 'model_outputs_ext',
*       as produced from function 'fit_iforest',
*/
template <class real_t, class sparse_ix>
void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, bool use_long_double, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer)
{
    if (use_long_double && !has_long_double()) {
        use_long_double = false;
        print_errmsg("Passed 'use_long_double=true', but library was compiled without long double support.\n");
    }
    #ifndef NO_LONG_DOUBLE
    if (likely(!use_long_double))
    #endif
        impute_missing_values_internal<real_t, sparse_ix, double>(
            numeric_data, categ_data, is_col_major,
            Xr, Xr_ind, Xr_indptr,
            nrows, nthreads,
            model_outputs, model_outputs_ext,
            imputer
        );
    #ifndef NO_LONG_DOUBLE
    else
        impute_missing_values_internal<real_t, sparse_ix, long double>(
            numeric_data, categ_data, is_col_major,
            Xr, Xr_ind, Xr_indptr,
            nrows, nthreads,
            model_outputs, model_outputs_ext,
            imputer
        );
    #endif
}

template <class real_t, class sparse_ix, class ldouble_safe>
void impute_missing_values_internal(
                           real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer)
{
    PredictionData<real_t, sparse_ix>
                   prediction_data = {numeric_data, categ_data, nrows,
                                      is_col_major, imputer.ncols_numeric, imputer.ncols_categ,
                                      NULL, NULL, NULL,
                                      Xr, Xr_ind, Xr_indptr};

    std::vector<size_t> ix_arr(nrows);
    std::iota(ix_arr.begin(), ix_arr.end(), (size_t) 0);

    size_t end = check_for_missing(prediction_data, imputer, ix_arr.data(), nthreads);

    if (end == 0)
        return;

    if ((size_t)nthreads > end)
        nthreads = (int)end;
    #ifdef _OPENMP
        std::vector<ImputedData<sparse_ix, ldouble_safe>> imp_memory(nthreads);
    #else
        std::vector<ImputedData<sparse_ix, ldouble_safe>> imp_memory(1);
    #endif

    bool threw_exception = false;
    std::exception_ptr ex = NULL;

    if (model_outputs != NULL)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(end, imp_memory, prediction_data, model_outputs, ix_arr, imputer, ex, threw_exception)
        for (size_t_for row = 0; row < (decltype(row))end; row++)
        {
            if (threw_exception) continue;
            try
            {
                initialize_impute_calc(imp_memory[omp_get_thread_num()], prediction_data, imputer, ix_arr[row]);

                for (std::vector<IsoTree> &tree : model_outputs->trees)
                {
                    traverse_itree(tree,
                                   *model_outputs,
                                   prediction_data,
                                   &imputer.imputer_tree[&tree - &(model_outputs->trees[0])],
                                   &imp_memory[omp_get_thread_num()],
                                   (double) 1,
                                   ix_arr[row],
                                   (sparse_ix*)NULL,
                                   (double*)NULL,
                                   (size_t) 0);
                }

                apply_imputation_results(prediction_data, imp_memory[omp_get_thread_num()], imputer, (size_t) ix_arr[row]);
            }

            catch(...)
            {
                #pragma omp critical
                {
                    if (!threw_exception)
                    {
                        threw_exception = true;
                        ex = std::current_exception();
                    }
                }
            }

        }
    }

    else
    {
        double temp;
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(end, imp_memory, prediction_data, model_outputs_ext, ix_arr, imputer, ex, threw_exception) \
                private(temp)
        for (size_t_for row = 0; row < (decltype(row))end; row++)
        {
            if (threw_exception) continue;
            try
            {
                initialize_impute_calc(imp_memory[omp_get_thread_num()], prediction_data, imputer, ix_arr[row]);

                for (std::vector<IsoHPlane> &hplane : model_outputs_ext->hplanes)
                {
                    traverse_hplane(hplane,
                                    *model_outputs_ext,
                                    prediction_data,
                                    temp,
                                    &imputer.imputer_tree[&hplane - &(model_outputs_ext->hplanes[0])],
                                    &imp_memory[omp_get_thread_num()],
                                    (sparse_ix*)NULL,
                                    (double*)NULL,
                                    ix_arr[row]);
                }

                apply_imputation_results(prediction_data, imp_memory[omp_get_thread_num()], imputer, (size_t) ix_arr[row]);
            }

            catch (...)
            {
                #pragma omp critical
                {
                    if (!threw_exception)
                    {
                        threw_exception = true;
                        ex = std::current_exception();
                    }
                }
            }

        }
    }

    if (threw_exception)
        std::rethrow_exception(ex);
}

template <class InputData, class ldouble_safe>
void initialize_imputer(Imputer &imputer, InputData &input_data, size_t ntrees, int nthreads)
{
    imputer.ncols_numeric  =  input_data.ncols_numeric;
    imputer.ncols_categ    =  input_data.ncols_categ;
    imputer.ncat.assign(input_data.ncat, input_data.ncat + input_data.ncols_categ);
    if (imputer.col_means.size())
    {
        imputer.col_means.resize(input_data.ncols_numeric);
        std::fill(imputer.col_means.begin(), imputer.col_means.end(), 0);
    }

    else
    {
        imputer.col_means.resize(input_data.ncols_numeric, 0);
    }

    imputer.col_modes.resize(input_data.ncols_categ);
    imputer.imputer_tree = std::vector<std::vector<ImputeNode>>(ntrees);

    /* TODO: here should use sample weights if specified as density */
    size_t offset, cnt;
    if (input_data.numeric_data != NULL)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) private(cnt, offset) shared(input_data, imputer)
        for (size_t_for col = 0; col < (decltype(col))input_data.ncols_numeric; col++)
        {
            cnt    = input_data.nrows;
            offset = col * input_data.nrows;
            for (size_t row = 0; row < input_data.nrows; row++)
            {
                imputer.col_means[col] += (!is_na_or_inf(input_data.numeric_data[row + offset]))?
                                           input_data.numeric_data[row + offset] : 0;
                cnt -= is_na_or_inf(input_data.numeric_data[row + offset]);
            }
            imputer.col_means[col] /= (ldouble_safe) cnt;
            if (!cnt) imputer.col_means[col] = NAN;
        }
    }

    else if (input_data.Xc_indptr != NULL)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) private(cnt) shared(input_data, imputer)
        for (size_t_for col = 0; col < (decltype(col))input_data.ncols_numeric; col++)
        {
            cnt = input_data.nrows;
            for (auto ix = input_data.Xc_indptr[col]; ix < input_data.Xc_indptr[col + 1]; ix++)
            {
                imputer.col_means[col] += (!is_na_or_inf(input_data.Xc[ix]))?
                                           input_data.Xc[ix] : 0;
                cnt -= is_na_or_inf(input_data.Xc[ix]);
            }
            imputer.col_means[col] /= (ldouble_safe) cnt;
            if (!cnt) imputer.col_means[col] = NAN;
        }
    }

    if (input_data.categ_data != NULL)
    {
        std::vector<size_t> cat_counts(input_data.max_categ);
        #pragma omp parallel for schedule(static) num_threads(nthreads) firstprivate(cat_counts) private(offset) shared(input_data, imputer)
        for (size_t_for col = 0; col < (decltype(col))input_data.ncols_categ; col++)
        {
            std::fill(cat_counts.begin(), cat_counts.end(), 0);
            offset = col * input_data.nrows;
            for (size_t row = 0; row < input_data.nrows; row++)
            {
                if (input_data.categ_data[row + offset] >= 0)
                    cat_counts[input_data.categ_data[row + offset]]++;
            }
            imputer.col_modes[col] = (int) std::distance(cat_counts.begin(),
                                                         std::max_element(cat_counts.begin(),
                                                                           cat_counts.begin() + input_data.ncat[col]));
        }
    }
}


/* https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
template <class InputData, class WorkerMemory, class ldouble_safe>
void build_impute_node(ImputeNode &imputer,    WorkerMemory &workspace,
                       InputData  &input_data, ModelParams  &model_params,
                       std::vector<ImputeNode> &imputer_tree,
                       size_t curr_depth, size_t min_imp_obs)
{
    double wsum;
    bool has_weights = workspace.weights_arr.size() || workspace.weights_map.size();
    if (!has_weights)
        wsum = (double)(workspace.end - workspace.st + 1);
    else
        wsum = calculate_sum_weights<ldouble_safe>(
                                     workspace.ix_arr, workspace.st, workspace.end, curr_depth,
                                     workspace.weights_arr, workspace.weights_map);

    imputer.num_sum.resize(input_data.ncols_numeric, 0);
    imputer.num_weight.resize(input_data.ncols_numeric, 0);
    imputer.cat_sum.resize(input_data.ncols_categ);
    imputer.cat_weight.resize(input_data.ncols_categ, 0);
    imputer.num_sum.shrink_to_fit();
    imputer.num_weight.shrink_to_fit();
    imputer.cat_sum.shrink_to_fit();
    imputer.cat_weight.shrink_to_fit();

    /* Note: in theory, 'num_weight' could be initialized to 'wsum',
       and the entries could get subtracted the weight afterwards, but due to rounding
       error, this could produce some cases of no-present observations having positive
       weight, or cases of negative weight, so it's better to add it for each row after
       checking for possible NAs, even though it's less computationally efficient.
       For sparse matrices it's done the other way as otherwise it would be too slow. */

    for (size_t col = 0; col < input_data.ncols_categ; col++)
    {
        imputer.cat_sum[col].resize(input_data.ncat[col]);
        imputer.cat_sum[col].shrink_to_fit();
    }

    double  xnum;
    int     xcat;
    double  weight;
    size_t  ix;

    if ((input_data.Xc_indptr == NULL && input_data.ncols_numeric) || input_data.ncols_categ)
    {
        if (!has_weights)
        {
            size_t cnt;
            if (input_data.numeric_data != NULL)
            {
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    cnt = 0;
                    for (size_t row = workspace.st; row <= workspace.end; row++)
                    {
                        xnum = input_data.numeric_data[workspace.ix_arr[row] + col * input_data.nrows];
                        if (!is_na_or_inf(xnum))
                        {
                            cnt++;
                            imputer.num_sum[col] += (xnum - imputer.num_sum[col]) / (ldouble_safe)cnt;
                        }
                    }
                    imputer.num_weight[col] = (double) cnt;
                }
            }

            if (input_data.categ_data != NULL)
            {
                for (size_t col = 0; col < input_data.ncols_categ; col++)
                {
                    cnt = 0;
                    for (size_t row = workspace.st; row <= workspace.end; row++)
                    {
                        xcat = input_data.categ_data[workspace.ix_arr[row] + col * input_data.nrows];
                        if (xcat >= 0)
                        {
                            cnt++;
                            imputer.cat_sum[col][xcat]++; /* later gets divided */
                        }
                    }
                    imputer.cat_weight[col] = (double) cnt;
                }
            }

        }

        else
        {
            ldouble_safe prod_sum, corr, val, diff;
            if (input_data.numeric_data != NULL)
            {
                 for (size_t col = 0; col < input_data.ncols_numeric; col++)
                 {
                    prod_sum = 0; corr = 0;
                    for (size_t row = workspace.st; row <= workspace.end; row++)
                    {
                        xnum = input_data.numeric_data[workspace.ix_arr[row] + col * input_data.nrows];
                        if (!is_na_or_inf(xnum))
                        {
                            if (workspace.weights_arr.size())
                                weight = workspace.weights_arr[workspace.ix_arr[row]];
                            else
                                weight = workspace.weights_map[workspace.ix_arr[row]];

                            imputer.num_weight[col] += weight; /* these are always <= 1 */
                            val      =  (xnum * weight) - corr;
                            diff     =  prod_sum + val;
                            corr     =  (diff - prod_sum) - val;
                            prod_sum =  diff;
                        }
                    }
                    imputer.num_sum[col] = prod_sum / imputer.num_weight[col];
                 }
            }


            if (input_data.ncols_categ)
            {
                for (size_t row = workspace.st; row <= workspace.end; row++)
                {
                    ix = workspace.ix_arr[row];
                    if (workspace.weights_arr.size())
                        weight = workspace.weights_arr[ix];
                    else
                        weight = workspace.weights_map[ix];

                    for (size_t col = 0; col < input_data.ncols_categ; col++)
                    {
                        xcat = input_data.categ_data[ix + col * input_data.nrows];
                        if (xcat >= 0)
                        {
                            imputer.cat_sum[col][xcat] += weight; /* later gets divided */
                            imputer.cat_weight[col]    += weight;
                        }
                    }
                }
            }
        }
    }

    if (input_data.Xc_indptr != NULL) /* sparse numeric */
    {
        size_t *ix_arr = workspace.ix_arr.data();
        size_t st_col, end_col, ind_end_col, curr_pos;
        std::fill(imputer.num_weight.begin(), imputer.num_weight.end(), wsum);

        for (size_t col = 0; col < input_data.ncols_numeric; col++)
        {
            st_col      =  input_data.Xc_indptr[col];
            end_col     =  input_data.Xc_indptr[col + 1] - 1;
            ind_end_col =  input_data.Xc_ind[end_col];
            curr_pos    =  st_col;
            for (size_t *row = std::lower_bound(ix_arr + workspace.st, ix_arr + workspace.end + 1, input_data.Xc_ind[st_col]);
                 row != ix_arr + workspace.end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (input_data.Xc_ind[curr_pos] == static_cast<typename std::remove_pointer<decltype(input_data.Xc_ind)>::type>(*row))
                {
                    xnum = input_data.Xc[curr_pos];
                    if (workspace.weights_arr.size())
                        weight = workspace.weights_arr[*row];
                    else if (workspace.weights_map.size())
                        weight = workspace.weights_map[*row];
                    else
                        weight = 1;

                    if (!is_na_or_inf(xnum))
                    {
                        imputer.num_sum[col]    += weight * xnum;
                    }

                    else
                    {
                        imputer.num_weight[col] -= weight;
                    }

                    if (row == ix_arr + workspace.end || curr_pos == end_col) break;
                    curr_pos = std::lower_bound(input_data.Xc_ind + curr_pos, input_data.Xc_ind + end_col + 1, *(++row)) - input_data.Xc_ind;
                }

                else
                {
                    if (input_data.Xc_ind[curr_pos] > static_cast<typename std::remove_pointer<decltype(input_data.Xc_ind)>::type>(*row))
                        row = std::lower_bound(row + 1, ix_arr + workspace.end + 1, input_data.Xc_ind[curr_pos]);
                    else
                        curr_pos = std::lower_bound(input_data.Xc_ind + curr_pos + 1, input_data.Xc_ind + end_col + 1, *row) - input_data.Xc_ind;
                }
            }

            imputer.num_sum[col] /= imputer.num_weight[col];
        }
    }

    /* if any value is not possible to impute, look it up from the parent tree, but assign a lesser weight
       Note: in theory, the parent node should always have some imputation value for every variable, but due to
       numeric rounding errors, it might have a weight of zero, so in those cases it's looked up higher up the
       tree instead. */
    size_t look_aboves, curr_tree;
    double min_imp_obs_dbl = (double) min_imp_obs;
    if (imputer.num_sum.size())
    {
        for (size_t col = 0; col < input_data.ncols_numeric; col++)
        {
            if (imputer.num_weight[col] < min_imp_obs_dbl)
            {
                look_aboves = 1;
                curr_tree   = imputer.parent;
                while (true)
                {
                    if (!is_na_or_inf(imputer_tree[curr_tree].num_sum[col]))
                    {
                        imputer.num_sum[col]     =  imputer_tree[curr_tree].num_sum[col] / imputer_tree[curr_tree].num_weight[col];
                        imputer.num_weight[col]  =  wsum / (double)(2 * look_aboves);
                        break;
                    }

                    else if (curr_tree > 0)
                    {
                        curr_tree = imputer_tree[curr_tree].parent;
                        look_aboves++;
                    }

                    else /* will only happen if every single value is missing */
                    {
                        imputer.num_sum[col]    = NAN;
                        imputer.num_weight[col] = 0;
                        break;
                    }
                }
            }
        }
    }

    if (imputer.cat_sum.size())
    {
        for (size_t col = 0; col < input_data.ncols_categ; col++)
        {
            if (imputer.cat_weight[col] >= min_imp_obs_dbl)
            {
                for (double &cat : imputer.cat_sum[col])
                    cat /= imputer.cat_weight[col];
            }

            else
            {
                look_aboves = 1;
                curr_tree   = imputer.parent;
                while (true)
                {
                    if (imputer_tree[curr_tree].cat_weight[col] > 0)
                    {
                        for (int cat = 0; cat < input_data.ncat[col]; cat++)
                        {
                            imputer.cat_sum[col][cat]
                                +=
                            (imputer_tree[curr_tree].cat_sum[col][cat] > 0)?
                                (imputer_tree[curr_tree].cat_sum[col][cat] / imputer_tree[curr_tree].cat_weight[col]) : 0.;
                            imputer.cat_weight[col]    =  wsum / (double)(2 * look_aboves);
                        }
                        break;
                    }

                    else if (curr_tree > 0)
                    {
                        curr_tree = imputer_tree[curr_tree].parent;
                        look_aboves++;
                    }

                    else /* will only happen if every single value is missing */
                    {
                        break;
                    }
                }
                imputer.cat_weight[col] = std::accumulate(imputer.cat_sum[col].begin(),
                                                          imputer.cat_sum[col].end(),
                                                          (double) 0);
            }
        }
    }

    /* re-adjust the weights according to parameters
       (note that by this point, the weights are a sum) */
    switch(model_params.weigh_imp_rows)
    {
        case Inverse:
        {
            double wsum_div = wsum * std::sqrt(wsum);
            for (double &w : imputer.num_weight)
                w /= wsum_div;

            for (double &w : imputer.cat_weight)
                w /= wsum_div;
            break;
        }

        case Flat:
        {
            for (double &w : imputer.num_weight)
                w /= wsum;
            for (double &w : imputer.cat_weight)
                w /= wsum;
            break;
        }

        default: {}

        /* TODO: maybe divide by nrows for prop */
    }

    double curr_depth_dbl = (double) (curr_depth + 1);
    switch(model_params.depth_imp)
    {
        case Lower:
        {
            for (double &w : imputer.num_weight)
                w /= curr_depth_dbl;
            for (double &w : imputer.cat_weight)
                w /= curr_depth_dbl;
            break;
        }

        case Higher:
        {
            for (double &w : imputer.num_weight)
                w *= curr_depth_dbl;
            for (double &w : imputer.cat_weight)
                w *= curr_depth_dbl;
            break;
        }

        default: {}
    }

    /* now re-adjust sums */
    if (model_params.weigh_imp_rows != Prop || model_params.depth_imp != Same)
    {
        for (size_t col = 0; col < input_data.ncols_numeric; col++)
            imputer.num_sum[col] *= imputer.num_weight[col];

        for (size_t col = 0; col < input_data.ncols_categ; col++)
            for (int cat = 0; cat < input_data.ncat[col]; cat++)
                imputer.cat_sum[col][cat] *= imputer.cat_weight[col];
    }
}


void shrink_impute_node(ImputeNode &imputer)
{
    imputer.num_sum.clear();
    imputer.num_weight.clear();
    imputer.cat_sum.clear();
    imputer.cat_weight.clear();

    imputer.num_sum.shrink_to_fit();
    imputer.num_weight.shrink_to_fit();
    imputer.cat_sum.shrink_to_fit();
    imputer.cat_weight.shrink_to_fit();
}

void drop_nonterminal_imp_node(std::vector<ImputeNode>  &imputer_tree,
                               std::vector<IsoTree>     *trees,
                               std::vector<IsoHPlane>   *hplanes)
{
    if (trees != NULL)
    {
        for (size_t tr = 0; tr < trees->size(); tr++)
        {
            if ((*trees)[tr].tree_left != 0)
            {
                shrink_impute_node(imputer_tree[tr]);
            }

            else
            {
                /* cat_weight is not needed for anything else */
                imputer_tree[tr].cat_weight.clear();
                imputer_tree[tr].cat_weight.shrink_to_fit();
            }
        }
    }

    else
    {
        for (size_t tr = 0; tr < hplanes->size(); tr++)
        {
            if ((*hplanes)[tr].hplane_left != 0)
            {
                shrink_impute_node(imputer_tree[tr]);
            }

            else
            {
                /* cat_weight is not needed for anything else */
                imputer_tree[tr].cat_weight.clear();
                imputer_tree[tr].cat_weight.shrink_to_fit();
            }
        }
    }

    imputer_tree.shrink_to_fit();
}

template <class ImputedData>
void combine_imp_single(ImputedData &restrict imp_addfrom, ImputedData &restrict imp_addto)
{
    size_t col;
    for (size_t ix = 0; ix < imp_addfrom.n_missing_num; ix++)
    {
        imp_addto.num_sum[ix]    += imp_addfrom.num_sum[ix];
        imp_addto.num_weight[ix] += imp_addfrom.num_weight[ix];
    }

    for (size_t ix = 0; ix < imp_addfrom.n_missing_cat; ix++)
    {
        col = imp_addfrom.missing_cat[ix];
        for (size_t cat = 0; cat < imp_addto.cat_sum[col].size(); cat++)
        {
            imp_addto.cat_sum[col][cat] += imp_addfrom.cat_sum[col][cat];
        }
    }

    for (size_t ix = 0; ix < imp_addfrom.n_missing_sp; ix++)
    {
        imp_addto.sp_num_sum[ix]    += imp_addfrom.sp_num_sum[ix];
        imp_addto.sp_num_weight[ix] += imp_addfrom.sp_num_weight[ix];
    }
}

template <class ImputedData, class WorkerMemory>
void combine_tree_imputations(WorkerMemory &workspace,
                              std::vector<ImputedData> &impute_vec,
                              hashed_map<size_t, ImputedData> &impute_map,
                              std::vector<char> &has_missing,
                              int nthreads)
{
    if (workspace.impute_vec.size())
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(has_missing, workspace, impute_vec)
        for (size_t_for row = 0; row < (decltype(row))has_missing.size(); row++)
            if (has_missing[row])
                combine_imp_single(workspace.impute_vec[row], impute_vec[row]);
    }

    else if (workspace.impute_map.size())
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(has_missing, workspace, impute_map)
        for (size_t_for row = 0; row < (decltype(row))has_missing.size(); row++)
            if (has_missing[row])
                combine_imp_single(workspace.impute_map[row], impute_map[row]);
    }
}


template <class ImputedData>
void add_from_impute_node(ImputeNode &imputer, ImputedData &imputed_data, double w)
{
    size_t col;
    for (size_t ix = 0; ix < imputed_data.n_missing_num; ix++)
    {
        col = imputed_data.missing_num[ix];
        imputed_data.num_sum[ix]    += (!is_na_or_inf(imputer.num_sum[col]))? (w * imputer.num_sum[col]) : 0;
        imputed_data.num_weight[ix] += w * imputer.num_weight[ix];
    }

    for (size_t ix = 0; ix < imputed_data.n_missing_sp; ix++)
    {
        col = imputed_data.missing_sp[ix];
        imputed_data.sp_num_sum[ix]    += (!is_na_or_inf(imputer.num_sum[col]))? (w * imputer.num_sum[col]) : 0;
        imputed_data.sp_num_weight[ix] += w * imputer.num_weight[ix];
    }

    for (size_t ix = 0; ix < imputed_data.n_missing_cat; ix++)
    {
        col = imputed_data.missing_cat[ix];
        for (size_t cat = 0; cat < imputer.cat_sum[col].size(); cat++)
            imputed_data.cat_sum[col][cat] += w * imputer.cat_sum[col][cat];
    }
}


template <class InputData, class WorkerMemory>
void add_from_impute_node(ImputeNode &imputer, WorkerMemory &workspace, InputData &input_data)
{
    if (workspace.impute_vec.size())
    {
        if (!workspace.weights_arr.size() && !workspace.weights_map.size())
        {
            for (size_t row = workspace.st; row <= workspace.end;  row++)
                if (input_data.has_missing[workspace.ix_arr[row]])
                    add_from_impute_node(imputer,
                                         workspace.impute_vec[workspace.ix_arr[row]],
                                         (double)1);
        }

        else if (workspace.weights_arr.size())
        {
            for (size_t row = workspace.st; row <= workspace.end;  row++)
                if (input_data.has_missing[workspace.ix_arr[row]])
                    add_from_impute_node(imputer,
                                         workspace.impute_vec[workspace.ix_arr[row]],
                                         workspace.weights_arr[workspace.ix_arr[row]]);
        }

        else
        {
            for (size_t row = workspace.st; row <= workspace.end;  row++)
                if (input_data.has_missing[workspace.ix_arr[row]])
                    add_from_impute_node(imputer,
                                         workspace.impute_vec[workspace.ix_arr[row]],
                                         workspace.weights_map[workspace.ix_arr[row]]);
        }
    }

    else if (workspace.impute_map.size())
    {
        if (!workspace.weights_arr.size() && !workspace.weights_map.size())
        {
            for (size_t row = workspace.st; row <= workspace.end;  row++)
                if (input_data.has_missing[workspace.ix_arr[row]])
                    add_from_impute_node(imputer,
                                         workspace.impute_map[workspace.ix_arr[row]],
                                         (double)1);
        }

        else if (workspace.weights_arr.size())
        {
            for (size_t row = workspace.st; row <= workspace.end;  row++)
                if (input_data.has_missing[workspace.ix_arr[row]])
                    add_from_impute_node(imputer,
                                         workspace.impute_map[workspace.ix_arr[row]],
                                         workspace.weights_arr[workspace.ix_arr[row]]);
        }

        else
        {
            for (size_t row = workspace.st; row <= workspace.end;  row++)
                if (input_data.has_missing[workspace.ix_arr[row]])
                    add_from_impute_node(imputer,
                                         workspace.impute_map[workspace.ix_arr[row]],
                                         workspace.weights_map[workspace.ix_arr[row]]);
        }
    }
}

template <class imp_arr, class InputData>
void apply_imputation_results(imp_arr    &impute_vec,
                              Imputer    &imputer,
                              InputData  &input_data,
                              int        nthreads)
{
    size_t col;

    if (input_data.Xc_indptr != NULL)
    {
        std::vector<size_t> row_pos(input_data.nrows, 0);
        size_t row;

        for (size_t col = 0; col < input_data.ncols_numeric; col++)
        {
            for (auto ix = input_data.Xc_indptr[col]; ix < input_data.Xc_indptr[col + 1]; ix++)
            {
                if (is_na_or_inf(input_data.Xc[ix]))
                {
                    row = input_data.Xc_ind[ix];
                    if (impute_vec[row].sp_num_weight[row_pos[row]] > 0 && !is_na_or_inf(impute_vec[row].sp_num_sum[row_pos[row]]))
                        input_data.Xc[ix]
                            =
                        impute_vec[row].sp_num_sum[row_pos[row]]
                            /
                        impute_vec[row].sp_num_weight[row_pos[row]];
                    else
                        input_data.Xc[ix]
                            =
                        imputer.col_means[col];

                    row_pos[row]++;
                }
            }
        }
    }

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(input_data, impute_vec, imputer) private(col)
    for (size_t_for row = 0; row < (decltype(row))input_data.nrows; row++)
    {
        if (input_data.has_missing[row])
        {
            for (size_t ix = 0; ix < impute_vec[row].n_missing_num; ix++)
            {
                col = impute_vec[row].missing_num[ix];
                if (impute_vec[row].num_weight[ix] > 0 && !is_na_or_inf(impute_vec[row].num_sum[ix]))
                    input_data.numeric_data[row + col * input_data.nrows]
                        =
                    impute_vec[row].num_sum[ix] / impute_vec[row].num_weight[ix];
                else
                    input_data.numeric_data[row + col * input_data.nrows]
                        =
                    imputer.col_means[col];
            }

            for (size_t ix = 0; ix < impute_vec[row].n_missing_cat; ix++)
            {
                col = impute_vec[row].missing_cat[ix];
                input_data.categ_data[row + col * input_data.nrows]
                    =
                std::distance(impute_vec[row].cat_sum[col].begin(),
                              std::max_element(impute_vec[row].cat_sum[col].begin(),
                                                 impute_vec[row].cat_sum[col].end()));

                if (input_data.categ_data[row + col * input_data.nrows] == 0 && impute_vec[row].cat_sum[col][0] <= 0)
                    input_data.categ_data[row + col * input_data.nrows]
                        =
                    imputer.col_modes[col];
            }
        }
    }
}

template <class ImputedData, class InputData>
void apply_imputation_results(std::vector<ImputedData> &impute_vec,
                              hashed_map<size_t, ImputedData> &impute_map,
                              Imputer   &imputer,
                              InputData &input_data,
                              int nthreads)
{
    if (impute_vec.size())
        apply_imputation_results(impute_vec, imputer, input_data, nthreads);
    else if (impute_map.size())
        apply_imputation_results(impute_map, imputer, input_data, nthreads);
}


/* TODO: investigate why in the case of all-missing numeric columns the node weights still
   get filled when using extended model, then remove the workaround that was added here that
   checks if the sum is zero and column is all-nan. Should also modify the earlier code to
   remove these cases from the imputation tracking list when doing the imputations on-the-fly
   as the model is being fit. */
template <class PredictionData, class ImputedData>
void apply_imputation_results(PredictionData  &prediction_data,
                              ImputedData     &imp,
                              Imputer         &imputer,
                              size_t          row)
{
    size_t col;
    size_t pos = 0;
    if (prediction_data.is_col_major)
    {
        for (size_t ix = 0; ix < imp.n_missing_num; ix++)
        {
            col = imp.missing_num[ix];
            if (imp.num_weight[ix] > 0 && !is_na_or_inf(imp.num_sum[ix]) && !(imp.num_sum[ix] == 0 && std::isnan(imputer.col_means[col])))
                prediction_data.numeric_data[row + col * prediction_data.nrows]
                    =
                imp.num_sum[ix] / imp.num_weight[ix];
            else
                prediction_data.numeric_data[row + col * prediction_data.nrows]
                    =
                imputer.col_means[col];
        }
    }

    else
    {
        for (size_t ix = 0; ix < imp.n_missing_num; ix++)
        {
            col = imp.missing_num[ix];
            if (imp.num_weight[ix] > 0 && !is_na_or_inf(imp.num_sum[ix]) && !(imp.num_sum[ix] == 0 && std::isnan(imputer.col_means[col])))
                prediction_data.numeric_data[col + row * imputer.ncols_numeric]
                    =
                imp.num_sum[ix] / imp.num_weight[ix];
            else
                prediction_data.numeric_data[col + row * imputer.ncols_numeric]
                    =
                imputer.col_means[col];
        }
    }

    if (prediction_data.Xr != NULL)
        for (auto ix = prediction_data.Xr_indptr[row]; ix < prediction_data.Xr_indptr[row + 1]; ix++)
        {
            if (is_na_or_inf(prediction_data.Xr[ix]))
            {
                if (imp.sp_num_weight[pos] > 0 && !is_na_or_inf(imp.sp_num_sum[pos]))
                    prediction_data.Xr[ix]
                        =
                    imp.sp_num_sum[pos] / imp.sp_num_weight[pos];
                else
                    prediction_data.Xr[ix]
                        =
                    imputer.col_means[imp.missing_sp[pos]];
                pos++;
            }
        }

    if (prediction_data.is_col_major)
    {
        for (size_t ix = 0; ix < imp.n_missing_cat; ix++)
        {
            col = imp.missing_cat[ix];
            prediction_data.categ_data[row + col * prediction_data.nrows]
                        =
            std::distance(imp.cat_sum[col].begin(),
                          std::max_element(imp.cat_sum[col].begin(), imp.cat_sum[col].end()));

            if (prediction_data.categ_data[row + col * prediction_data.nrows] == 0)
            {
                if (imp.cat_sum.empty() || imp.cat_sum[col].empty())
                {
                    prediction_data.categ_data[row + col * prediction_data.nrows] = -1;
                }

                else if (imp.cat_sum[col][0] <= 0)
                {
                    prediction_data.categ_data[row + col * prediction_data.nrows]
                        =
                    imputer.col_modes[col];
                }
            }
        }
    }

    else
    {
        for (size_t ix = 0; ix < imp.n_missing_cat; ix++)
        {
            col = imp.missing_cat[ix];
            prediction_data.categ_data[col + row * imputer.ncols_categ]
                        =
            std::distance(imp.cat_sum[col].begin(),
                          std::max_element(imp.cat_sum[col].begin(), imp.cat_sum[col].end()));

            if (prediction_data.categ_data[col + row * imputer.ncols_categ] == 0)
            {
                if (imp.cat_sum.empty() || imp.cat_sum[col].empty())
                {
                    prediction_data.categ_data[col + row * imputer.ncols_categ] = -1;
                }
                
                else if (imp.cat_sum[col][0] <= 0)
                {
                    prediction_data.categ_data[col + row * imputer.ncols_categ]
                        =
                    imputer.col_modes[col];
                }
            }
        }
    }
}


template <class ImputedData, class InputData>
void initialize_impute_calc(ImputedData &imp, InputData &input_data, size_t row)
{
    imp.n_missing_num = 0;
    imp.n_missing_cat = 0;
    imp.n_missing_sp  = 0;

    if (input_data.numeric_data != NULL)
    {
        imp.missing_num.resize(input_data.ncols_numeric);
        for (size_t col = 0; col < input_data.ncols_numeric; col++)
            if (is_na_or_inf(input_data.numeric_data[row + col * input_data.nrows]))
                imp.missing_num[imp.n_missing_num++] = col;
        imp.missing_num.resize(imp.n_missing_num);
        imp.num_sum.assign(imp.n_missing_num,    0);
        imp.num_weight.assign(imp.n_missing_num, 0);
    }

    else if (input_data.Xc_indptr != NULL)
    {
        imp.missing_sp.resize(input_data.ncols_numeric);
        decltype(input_data.Xc_indptr) res;
        for (size_t col = 0; col < input_data.ncols_numeric; col++)
        {
            res = std::lower_bound(input_data.Xc_ind + input_data.Xc_indptr[col],
                                   input_data.Xc_ind + input_data.Xc_indptr[col + 1],
                                   row);
            if (
                res != input_data.Xc_ind + input_data.Xc_indptr[col + 1] && 
                *res == static_cast<typename std::remove_pointer<decltype(res)>::type>(row) && 
                is_na_or_inf(input_data.Xc[res - input_data.Xc_ind])
                )
            {
                imp.missing_sp[imp.n_missing_sp++] = col;
            }
        }
        imp.sp_num_sum.assign(imp.n_missing_sp,    0);
        imp.sp_num_weight.assign(imp.n_missing_sp, 0);
    }
    
    if (input_data.categ_data != NULL)
    {
        imp.missing_cat.resize(input_data.ncols_categ);
        for (size_t col = 0; col < input_data.ncols_categ; col++)
            if (input_data.categ_data[row + col * input_data.nrows] < 0)
                imp.missing_cat[imp.n_missing_cat++] = col;
        imp.missing_cat.resize(imp.n_missing_cat);
        imp.cat_weight.assign(imp.n_missing_cat, 0);
        imp.cat_sum.resize(input_data.ncols_categ);
        for (size_t cat = 0; cat < imp.n_missing_cat; cat++)
            imp.cat_sum[imp.missing_cat[cat]].assign(input_data.ncat[imp.missing_cat[cat]], 0);
    }
}

template <class ImputedData, class PredictionData>
void initialize_impute_calc(ImputedData &imp, PredictionData &prediction_data, Imputer &imputer, size_t row)
{
    imp.n_missing_num = 0;
    imp.n_missing_cat = 0;
    imp.n_missing_sp  = 0;

    if (prediction_data.numeric_data != NULL)
    {
        if (!imp.missing_num.size())
            imp.missing_num.resize(imputer.ncols_numeric);

        if (prediction_data.is_col_major)
        {
            for (size_t col = 0; col < imputer.ncols_numeric; col++)
                if (is_na_or_inf(prediction_data.numeric_data[row + col * prediction_data.nrows]))
                    imp.missing_num[imp.n_missing_num++] = col;
        }

        else
        {
            for (size_t col = 0; col < imputer.ncols_numeric; col++)
                if (is_na_or_inf(prediction_data.numeric_data[col + row * imputer.ncols_numeric]))
                    imp.missing_num[imp.n_missing_num++] = col;
        }

        if (!imp.num_sum.size())
        {
            imp.num_sum.resize(imputer.ncols_numeric,    0);
            imp.num_weight.resize(imputer.ncols_numeric, 0);
        }

        else
        {
            std::fill(imp.num_sum.begin(),     imp.num_sum.begin()    + imp.n_missing_num,  0);
            std::fill(imp.num_weight.begin(),  imp.num_weight.begin() + imp.n_missing_num,  0);
        }
    }

    else if (prediction_data.Xr != NULL)
    {
        if (!imp.missing_sp.size())
            imp.missing_sp.resize(imputer.ncols_numeric);
        for (auto ix = prediction_data.Xr_indptr[row]; ix < prediction_data.Xr_indptr[row + 1]; ix++)
            if (is_na_or_inf(prediction_data.Xr[ix]))
                imp.missing_sp[imp.n_missing_sp++] = prediction_data.Xr_ind[ix];

        if (!imp.sp_num_sum.size())
        {
            imp.sp_num_sum.resize(imputer.ncols_numeric,    0);
            imp.sp_num_weight.resize(imputer.ncols_numeric, 0);
        }

        else
        {
            std::fill(imp.sp_num_sum.begin(),     imp.sp_num_sum.begin()    + imp.n_missing_sp,  0);
            std::fill(imp.sp_num_weight.begin(),  imp.sp_num_weight.begin() + imp.n_missing_sp,  0);
        }
    }
    
    if (prediction_data.categ_data != NULL)
    {
        if (!imp.missing_cat.size())
            imp.missing_cat.resize(imputer.ncols_categ);

        if (prediction_data.is_col_major)
        {
            for (size_t col = 0; col < imputer.ncols_categ; col++)
            {
                if (prediction_data.categ_data[row + col * prediction_data.nrows] < 0)
                    imp.missing_cat[imp.n_missing_cat++] = col;
            }
        }

        else
        {
            for (size_t col = 0; col < imputer.ncols_categ; col++)
            {
                if (prediction_data.categ_data[col + row * imputer.ncols_categ] < 0)
                    imp.missing_cat[imp.n_missing_cat++] = col;
            }
        }

        if (!imp.cat_weight.size())
        {
            imp.cat_weight.resize(imputer.ncols_categ, 0);
            imp.cat_sum.resize(imputer.ncols_categ);
            for (size_t col = 0; col < imputer.ncols_categ; col++)
                imp.cat_sum[col].resize(imputer.ncat[col], 0);
        }

        else
        {
            std::fill(imp.cat_weight.begin(), imp.cat_weight.begin() + imp.n_missing_cat, 0);
            for (size_t col = 0; col < imp.n_missing_cat; col++)
                std::fill(imp.cat_sum[imp.missing_cat[col]].begin(),
                          imp.cat_sum[imp.missing_cat[col]].end(),
                          0);
        }
    }
}

// template class ImputedData <class InputData>
// ImputedData::ImputedData(InputData &input_data, size_t row)
// {
//     initialize_impute_calc(*this, input_data, row);
// }

template <class ImputedData, class InputData>
void allocate_imp_vec(std::vector<ImputedData> &impute_vec, InputData &input_data, int nthreads)
{
    impute_vec.resize(input_data.nrows);
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(impute_vec, input_data)
    for (size_t_for row = 0; row < (decltype(row))input_data.nrows; row++)
        if (input_data.has_missing[row])
            initialize_impute_calc(impute_vec[row], input_data, row);
}


template <class ImputedData, class InputData>
void allocate_imp_map(hashed_map<size_t, ImputedData> &impute_map, InputData &input_data)
{
    for (size_t row = 0; row < input_data.nrows; row++)
        if (input_data.has_missing[row])
            impute_map[row] = ImputedData(input_data, row);
}

template <class ImputedData, class InputData>
void allocate_imp(InputData &input_data,
                  std::vector<ImputedData> &impute_vec,
                  hashed_map<size_t, ImputedData> &impute_map,
                  int nthreads)
{
    if (input_data.n_missing == 0)
        return;
    else if (input_data.n_missing <= input_data.nrows / (nthreads * 10))
        allocate_imp_map(impute_map, input_data);
    else
        allocate_imp_vec(impute_vec, input_data, nthreads);
}

template <class ImputedData, class InputData>
void check_for_missing(InputData &input_data,
                       std::vector<ImputedData> &impute_vec,
                       hashed_map<size_t, ImputedData> &impute_map,
                       int nthreads)
{
    input_data.has_missing.assign(input_data.nrows, false);

    if (input_data.Xc_indptr != NULL)
    {
        for (size_t col = 0; col < input_data.ncols_numeric; col++)
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(col, input_data)
            for (size_t_for ix = input_data.Xc_indptr[col]; ix < (decltype(ix))input_data.Xc_indptr[col + 1]; ix++)
                if (is_na_or_inf(input_data.Xc[ix]))
                    input_data.has_missing[input_data.Xc_ind[ix]] = true;
            #pragma omp barrier
    }

    if (input_data.numeric_data != NULL || input_data.categ_data != NULL)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) shared(input_data)
        for (size_t_for row = 0; row < (decltype(row))input_data.nrows; row++)
        {
            if (input_data.Xc_indptr == NULL)
            {
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (is_na_or_inf(input_data.numeric_data[row + col * input_data.nrows]))
                    {
                        input_data.has_missing[row] = true;
                        break;
                    }
                }
            }

            if (!input_data.has_missing[row])
                for (size_t col = 0; col < input_data.ncols_categ; col++)
                {
                    if (input_data.categ_data[row + col * input_data.nrows] < 0)
                    {
                        input_data.has_missing[row] = true;
                        break;
                    }
                }
        }
    }

    input_data.n_missing = std::accumulate(input_data.has_missing.begin(), input_data.has_missing.end(), (size_t)0);
    allocate_imp(input_data, impute_vec, impute_map, nthreads);
}

template <class PredictionData>
size_t check_for_missing(PredictionData  &prediction_data,
                         Imputer         &imputer,
                         size_t          ix_arr[],
                         int             nthreads)
{
    std::vector<char> has_missing(prediction_data.nrows, false);

    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(has_missing, prediction_data, imputer)
    for (size_t_for row = 0; row < (decltype(row))prediction_data.nrows; row++)
    {
        if (prediction_data.numeric_data != NULL)
        {
            if (prediction_data.is_col_major)
            {
                for (size_t col = 0; col < imputer.ncols_numeric; col++)
                {
                    if (is_na_or_inf(prediction_data.numeric_data[row + col * prediction_data.nrows]))
                    {
                        has_missing[row] = true;
                        break;
                    }
                }
            }

            else
            {
                for (size_t col = 0; col < imputer.ncols_numeric; col++)
                {
                    if (is_na_or_inf(prediction_data.numeric_data[col + row * imputer.ncols_numeric]))
                    {
                        has_missing[row] = true;
                        break;
                    }
                }
            }
        }

        else if (prediction_data.Xr != NULL)
        {
            for (auto ix = prediction_data.Xr_indptr[row]; ix < prediction_data.Xr_indptr[row + 1]; ix++)
            {
                if (is_na_or_inf(prediction_data.Xr[ix]))
                {
                    has_missing[row] = true;
                    break;
                }
            }
        }

        if (!has_missing[row])
        {
            if (prediction_data.is_col_major)
            {
                for (size_t col = 0; col < imputer.ncols_categ; col++)
                {
                    if (prediction_data.categ_data[row + col * prediction_data.nrows] < 0)
                    {
                        has_missing[row] = true;
                        break;
                    }
                }
            }

            else
            {
                for (size_t col = 0; col < imputer.ncols_categ; col++)
                {
                    if (prediction_data.categ_data[col + row * imputer.ncols_categ] < 0)
                    {
                        has_missing[row] = true;
                        break;
                    }
                }
            }
        }
    }

    size_t st = 0;
    size_t temp;
    for (size_t row = 0; row < prediction_data.nrows; row++)
    {
        if (has_missing[row])
        {
            temp        = ix_arr[st];
            ix_arr[st]  = ix_arr[row];
            ix_arr[row] = temp;
            st++;
        }
    }

    if (st == 0)
        return 0;

    return st;
}
