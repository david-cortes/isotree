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
*     Copyright (c) 2019-2024, David Cortes
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

/* TODO: should create versions of these functions that would work on the
   serialized raw bytes instead, as it will likely be faster due to better
   cache utilizations and those objects use less memory. */

/* TODO: these trees are all created in a depth-first fashion, which will
   not be cache-friendly when predictions are sent to a right-side branch. In
   order to make predictions faster, could re-arrange the trees after-the-fact
   so that they contain batches of consecutive nodes (parent and children and
   grandchildren) up to some depth - that way these prediction functions would
   run faster. After that, could also do a manual tree leaves unroll within each
   batch with stack-assigned variables for an even faster prediction function. */

/* TODO: add 'const' qualifiers to all data inputs here. */

/* Predict outlier score, average depth, or terminal node numbers
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data for which to make predictions. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data for which to make predictions. May be ordered by rows
*       (i.e. entries 1..n contain row 0, n+1..2n row 1, etc.) - a.k.a. row-major - or by
*       columns (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.) - a.k.a. column-major
*       (see parameter 'is_col_major').
*       Pass NULL if there are no categorical columns.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
* - is_col_major
*       Whether 'numeric_data' and 'categ_data' come in column-major order, like the data to which the
*       model was fit. If passing 'false', will assume they are in row-major order. Note that most of
*       the functions in this library work only with column-major order, but here both are suitable
*       and row-major is preferred. Both arrays must have the same orientation (row/column major).
*       If there is numeric sparse data in combination with categorical dense data and there are many
*       rows, it is recommended to pass the categorical data in column major order, as it will take
*       a faster route.
*       If passing 'is_col_major=true', must also provide 'ld_numeric' and/or 'ld_categ'.
* - ld_numeric
*       Leading dimension of the array 'numeric_data', if it is passed in row-major format.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'numeric_data + n*ld_numeric'). If passing
*       'numeric_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows. This is ignored when passing numeric
*       data in sparse format.
* - ld_categ
*       Leading dimension of the array 'categ_data', if it is passed in row-major format.
*       Typically, this corresponds to the number of columns, but may be larger (the array will
*       be accessed assuming that row 'n' starts at 'categ_data + n*ld_categ'). If passing
*       'categ_data' in column-major order, this is ignored and will be assumed that the
*       leading dimension corresponds to the number of rows.
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds.
*       Must be in sorted order, otherwise results will be incorrect.
*       Pass NULL if there are no sparse numeric columns in CSC format.
* - Xc_indptr[ncols_categ + 1]
*       Pointer to column index pointers that tell at entry [col] where does column 'col'
*       start and at entry [col + 1] where does column 'col' end.
*       Pass NULL if there are no sparse numeric columns in CSC format.
* - Xr[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSR format (row-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'. 
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
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* - standardize
*       Whether to standardize the average depths for each row according to their relative magnitude
*       compared to the expected average, in order to obtain an outlier score. If passing 'false',
*       will output the average depth instead.
*       Ignored when not passing 'output_depths'.
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - output_depths[nrows] (out)
*       Pointer to array where the output average depths or outlier scores will be written into
*       (the return type is controlled according to parameter 'standardize').
*       Should always be passed when calling this function (it is not optional).
* - tree_num[nrows * ntrees] (out)
*       Pointer to array where the output terminal node numbers will be written into.
*       Note that the mapping between tree node and terminal tree node is not stored in
*       the model object for efficiency reasons, so this mapping will be determined on-the-fly
*       when passing this parameter, and as such, there will be some overhead regardless of
*       the actual number of rows. Output will be in column-major order ([nrows, ntrees]).
*       This will not be calculable when using 'ndim==1' alongside with either
*       'missing_action==Divide' or 'new_categ_action=Weighted'.
*       Pass NULL if this type of output is not needed.
* - per_tree_depths[nrows * ntrees] (out)
*       Pointer to array where to output per-tree depths or expected depths for each row.
*       Note that these will not include range penalties ('penalize_range=true').
*       Output will be in row-major order ([nrows, ntrees]).
*       This will not be calculable when using 'ndim==1' alongside with either
*       'missing_action==Divide' or 'new_categ_action=Weighted'.
*       Pass NULL if this type of output is not needed.
* - indexer
*       Pointer to associated tree indexer for the model being used, if it was constructed,
*       which can be used to speed up tree numbers/indices predictions.
*       This is ignored when not passing 'tree_num'.
*       Pass NULL if the indexer has not been constructed.
*/
template <class real_t, class sparse_ix>
void predict_iforest(real_t *restrict numeric_data, int *restrict categ_data,
                     bool is_col_major, size_t ld_numeric, size_t ld_categ,
                     real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     real_t *restrict Xr, sparse_ix *restrict Xr_ind, sparse_ix *restrict Xr_indptr,
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double *restrict output_depths,   sparse_ix *restrict tree_num,
                     double *restrict per_tree_depths,
                     TreesIndexer *indexer)
{
    if (unlikely(!nrows)) return;

    /* put data in a struct for passing it in fewer lines */
    PredictionData<real_t, sparse_ix>
                   prediction_data = {numeric_data, categ_data, nrows,
                                      is_col_major, ld_numeric, ld_categ,
                                      Xc, Xc_ind, Xc_indptr,
                                      Xr, Xr_ind, Xr_indptr};

    int nthreads_orig = nthreads;
    if ((size_t)nthreads > nrows)
        nthreads = nrows;

    /* For batch predictions of sparse CSC, will take a specialized route */
    if (prediction_data.Xc_indptr != NULL && (prediction_data.categ_data == NULL || prediction_data.is_col_major))
    {
        batched_csc_predict(prediction_data, nthreads_orig,
                            model_outputs, model_outputs_ext,
                            output_depths, tree_num,
                            per_tree_depths);
    }

    /* Regular case (no specialized CSC route) */
    else if (model_outputs != NULL)
    {
        if (
            model_outputs->missing_action == Fail &&
            (model_outputs->new_cat_action != Weighted || model_outputs->cat_split_type == SingleCateg || prediction_data.categ_data == NULL) &&
            prediction_data.Xc_indptr == NULL && prediction_data.Xr_indptr == NULL &&
            !model_outputs->has_range_penalty
            )
        {
            if (prediction_data.categ_data == NULL && (nrows == 1 || !prediction_data.is_col_major))
            {
                #pragma omp parallel for if(nrows > 1) schedule(static) num_threads(nthreads) \
                        shared(nrows, model_outputs, prediction_data, output_depths, tree_num, per_tree_depths)
                for (size_t_for row = 0; row < (decltype(row))nrows; row++)
                {
                    double score = 0;
                    for (size_t tree = 0; tree < model_outputs->trees.size(); tree++)
                    {
                        traverse_itree_fast(model_outputs->trees[tree],
                                            *model_outputs,
                                            prediction_data.numeric_data + row * prediction_data.ncols_numeric,
                                            score,
                                            (tree_num == NULL)? NULL : (tree_num + nrows * tree),
                                            (per_tree_depths == NULL)?
                                                NULL : (per_tree_depths + tree + row*model_outputs->trees.size()),
                                            (size_t) row);
                    }
                    output_depths[row] = score;
                }
            }

            else
            {
                #pragma omp parallel for if(nrows > 1) schedule(static) num_threads(nthreads) \
                        shared(nrows, model_outputs, prediction_data, output_depths, tree_num, per_tree_depths)
                for (size_t_for row = 0; row < (decltype(row))nrows; row++)
                {
                    double score = 0;
                    for (size_t tree = 0; tree < model_outputs->trees.size(); tree++)
                    {
                        traverse_itree_no_recurse(model_outputs->trees[tree],
                                                  *model_outputs,
                                                  prediction_data,
                                                  score,
                                                  (tree_num == NULL)? NULL : (tree_num + nrows * tree),
                                                  (per_tree_depths == NULL)?
                                                      NULL : (per_tree_depths + tree + row*model_outputs->trees.size()),
                                                  (size_t) row);
                    }
                    output_depths[row] = score;
                }
            }
        }

        else
        {
            bool threw_exception = false;
            std::exception_ptr ex = NULL;

            #pragma omp parallel for if(nrows > 1) schedule(static) num_threads(nthreads) \
                    shared(nrows, model_outputs, prediction_data, output_depths, tree_num, per_tree_depths, threw_exception, ex)
            for (size_t_for row = 0; row < (decltype(row))nrows; row++)
            {
                if (threw_exception) continue;
                double score = 0;
                try
                {
                    for (size_t tree = 0; tree < model_outputs->trees.size(); tree++)
                    {
                        score += traverse_itree(model_outputs->trees[tree],
                                                *model_outputs,
                                                prediction_data,
                                                (std::vector<ImputeNode>*)NULL,
                                                (ImputedData<sparse_ix, double>*)NULL,
                                                (double)0,
                                                (size_t) row,
                                                (tree_num == NULL)? NULL : (tree_num + nrows * tree),
                                                (per_tree_depths == NULL)?
                                                    NULL : (per_tree_depths + tree + row*model_outputs->trees.size()),
                                                (size_t) 0);
                    }
                    output_depths[row] = score;
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

            if (threw_exception)
                std::rethrow_exception(ex);
        }
    }
    

    else
    {
        if (
            model_outputs_ext->missing_action == Fail &&
            prediction_data.categ_data == NULL &&
            prediction_data.Xc_indptr == NULL &&
            prediction_data.Xr_indptr == NULL &&
            !model_outputs_ext->has_range_penalty
            )
        {
            if (prediction_data.is_col_major && nrows > 1)
            {
                #pragma omp parallel for if(nrows > 1) schedule(static) num_threads(nthreads) \
                        shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num, per_tree_depths)
                for (size_t_for row = 0; row < (decltype(row))nrows; row++)
                {
                    double score = 0;
                    for (size_t tree = 0; tree < model_outputs_ext->hplanes.size(); tree++)
                    {
                        traverse_hplane_fast_colmajor(model_outputs_ext->hplanes[tree],
                                                      *model_outputs_ext,
                                                      prediction_data,
                                                      score,
                                                      (tree_num == NULL)? NULL : (tree_num + nrows * tree),
                                                      (per_tree_depths == NULL)?
                                                            NULL : (per_tree_depths + tree + row*model_outputs_ext->hplanes.size()),
                                                      (size_t) row);
                    }
                    output_depths[row] = score;
                }
            }

            else
            {
                #pragma omp parallel for if(nrows > 1) schedule(static) num_threads(nthreads) \
                        shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num, per_tree_depths)
                for (size_t_for row = 0; row < (decltype(row))nrows; row++)
                {
                    double score = 0;
                    for (size_t tree = 0; tree < model_outputs_ext->hplanes.size(); tree++)
                    {
                        traverse_hplane_fast_rowmajor(model_outputs_ext->hplanes[tree],
                                                      *model_outputs_ext,
                                                      prediction_data.numeric_data + row * prediction_data.ncols_numeric,
                                                      score,
                                                      (tree_num == NULL)? NULL : (tree_num + nrows * tree),
                                                      (per_tree_depths == NULL)?
                                                            NULL : (per_tree_depths + tree + row*model_outputs_ext->hplanes.size()),
                                                      (size_t) row);
                    }
                    output_depths[row] = score;
                }
            }
        }

        else
        {
            #pragma omp parallel for if(nrows > 1) schedule(static) num_threads(nthreads) \
                    shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num, per_tree_depths)
            for (size_t_for row = 0; row < (decltype(row))nrows; row++)
            {
                double score = 0;
                for (size_t tree = 0; tree < model_outputs_ext->hplanes.size(); tree++)
                {
                    traverse_hplane(model_outputs_ext->hplanes[tree],
                                    *model_outputs_ext,
                                    prediction_data,
                                    score,
                                    (std::vector<ImputeNode>*)NULL,
                                    (ImputedData<sparse_ix, double>*)NULL,
                                    (tree_num == NULL)? NULL : (tree_num + nrows * tree),
                                    (per_tree_depths == NULL)?
                                        NULL : (per_tree_depths + tree + row*model_outputs_ext->hplanes.size()),
                                    (size_t) row);
                }
                output_depths[row] = score;
            }
        }
    }

    /* translate sum-of-depths to outlier score */
    double ntrees, depth_divisor;
    if (model_outputs != NULL)
    {
        ntrees = (double) model_outputs->trees.size();
        depth_divisor = ntrees * (model_outputs->exp_avg_depth);
    }

    else
    {
        ntrees = (double) model_outputs_ext->hplanes.size();
        depth_divisor = ntrees * (model_outputs_ext->exp_avg_depth);
    }

    
    /* for density and boxed_ratio, each tree will have 'log(d)'' instead of 'd' */
    bool is_density = (model_outputs != NULL && model_outputs->scoring_metric == Density) ||
                      (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == Density);
    bool is_bratio  = (model_outputs != NULL && model_outputs->scoring_metric == BoxedRatio) ||
                      (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedRatio);
    bool is_bdens   = (model_outputs != NULL && model_outputs->scoring_metric == BoxedDensity) ||
                      (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedDensity);
    bool is_bdens2  = (model_outputs != NULL && model_outputs->scoring_metric == BoxedDensity2) ||
                      (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedDensity2);

    if (standardize)
    {
        if (is_density || is_bdens2)
        {
            ntrees = -ntrees;
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] /= ntrees;
        }

        else if (is_bdens)
        {
            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] = -std::exp(output_depths[row] / ntrees);
        }

        else if (is_bratio)
        {
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] = output_depths[row] / ntrees;
        }

        else
        {
            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] = std::exp2( - output_depths[row] / depth_divisor );
        }
    }

    else
    {
        if (is_density || is_bdens || is_bdens2)
        {
            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] = std::exp(output_depths[row] / ntrees);
        }

        else if (is_bratio)
        {
            ntrees = -ntrees;
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] /= ntrees;
        }

        else
        {
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] /= ntrees;
        }
    }

    if (per_tree_depths != NULL && (is_density || is_bdens || is_bdens2))
    {
        size_t ntrees = (model_outputs != NULL)? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
        #ifndef _WIN32
        #pragma omp simd
        #endif
        for (size_t ix = 0; ix < nrows*ntrees; ix++)
            per_tree_depths[ix] = std::exp(per_tree_depths[ix]);
    }


    /* re-map tree numbers to start at zero (if predicting tree numbers) */
    /* Note: usually this type of 'prediction' is not required,
       thus this mapping is not stored in the model objects so as to
       save memory */
    if (tree_num != NULL)
    {
        if (indexer != NULL && !indexer->indices.empty())
        {
            size_t ntrees = (model_outputs != NULL)? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
            if (model_outputs != NULL)
            {
                if (model_outputs->missing_action == Divide)
                    goto manual_remap;
                if (model_outputs->new_cat_action == Weighted && model_outputs->cat_split_type == SubSet && categ_data != NULL)
                    goto manual_remap;
            }

            for (size_t tree = 0; tree < ntrees; tree++)
            {
                size_t *restrict mapping = indexer->indices[tree].terminal_node_mappings.data();
                for (size_t row = 0; row < nrows; row++)
                {
                    tree_num[row + tree*nrows] = mapping[tree_num[row + tree*nrows]];
                }
            }
        }

        else
        {
            manual_remap:
            remap_terminal_trees(model_outputs, model_outputs_ext,
                                 prediction_data, tree_num, nthreads);
        }
    }
}

template <class real_t, class sparse_ix>
void traverse_itree_fast(std::vector<IsoTree>  &tree,
                         IsoForest             &model_outputs,
                         real_t *restrict      row_numeric_data,
                         double &restrict      output_depth,
                         sparse_ix *restrict   tree_num,
                         double *restrict      tree_depth,
                         size_t                row) noexcept
{
    size_t curr_lev = 0;
    double xval;
    while (true)
    {
        if (unlikely(tree[curr_lev].tree_left == 0))
        {
            output_depth += tree[curr_lev].score;
            if (unlikely(tree_num != NULL))
                tree_num[row] = curr_lev;
            if (unlikely(tree_depth != NULL))
                *tree_depth = tree[curr_lev].score;
            break;
        }

        else
        {
            xval     = row_numeric_data[tree[curr_lev].col_num];
            curr_lev = (xval <= tree[curr_lev].num_split)?
                        tree[curr_lev].tree_left : tree[curr_lev].tree_right;
        }
    }
}

template <class PredictionData, class sparse_ix>
void traverse_itree_no_recurse(std::vector<IsoTree>  &tree,
                               IsoForest             &model_outputs,
                               PredictionData        &prediction_data,
                               double &restrict      output_depth,
                               sparse_ix *restrict   tree_num,
                               double *restrict      tree_depth,
                               size_t                row) noexcept
{
    size_t curr_lev = 0;
    double xval;
    int    cval;
    while (true)
    {
        // if (tree[curr_lev].score > 0)
        if (unlikely(tree[curr_lev].tree_left == 0))
        {
            output_depth += tree[curr_lev].score;
            if (unlikely(tree_num != NULL))
                tree_num[row] = curr_lev;
            if (unlikely(tree_depth != NULL))
                *tree_depth = tree[curr_lev].score;
            break;
        }

        else
        {
            switch (tree[curr_lev].col_type)
            {
                case Numeric:
                {
                    xval =  prediction_data.numeric_data[
                                prediction_data.is_col_major?
                                (row + tree[curr_lev].col_num * prediction_data.nrows)
                                    :
                                (tree[curr_lev].col_num + row * prediction_data.ncols_numeric)
                            ];
                    curr_lev = (xval <= tree[curr_lev].num_split)?
                                tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                    break;
                }

                case Categorical:
                {
                    cval =  prediction_data.categ_data[
                                prediction_data.is_col_major?
                                (row +  tree[curr_lev].col_num * prediction_data.nrows)
                                    :
                                (tree[curr_lev].col_num + row * prediction_data.ncols_categ)
                            ];
                    switch (model_outputs.cat_split_type)
                    {
                        case SubSet:
                        {

                            if (tree[curr_lev].cat_split.empty()) /* this is for binary columns */
                            {
                                if (cval <= 1)
                                {
                                    curr_lev = (cval == 0)?
                                                tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                }

                                else /* can only work with 'Smallest' + no NAs if reaching this point */
                                {
                                    curr_lev =  (tree[curr_lev].pct_tree_left < .5)? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                }
                            }

                            else
                            {

                                switch (model_outputs.new_cat_action)
                                {
                                    case Random:
                                    {
                                        cval = (cval >= (int)tree[curr_lev].cat_split.size())?
                                                (cval % (int)tree[curr_lev].cat_split.size()) : cval;
                                        curr_lev = (tree[curr_lev].cat_split[cval])?
                                                    tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                        break;
                                    }

                                    case Smallest:
                                    {
                                        if (unlikely(cval >= (int)tree[curr_lev].cat_split.size()))
                                        {
                                            curr_lev =  (tree[curr_lev].pct_tree_left < .5)? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                        }

                                        else
                                        {
                                            curr_lev = (tree[curr_lev].cat_split[cval])?
                                                        tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                        }
                                        break;
                                    }

                                    default:
                                    {
                                        assert(0);
                                        break;
                                    }
                                }
                            }
                            break;
                        }

                        case SingleCateg:
                        {
                            curr_lev = (cval == tree[curr_lev].chosen_cat)?
                                        tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                            break;
                        }
                    }
                    break;
                }

                default:
                {
                    assert(0);
                    break;
                }
            }
        }
    }
}

enum NumericConfig {DenseRowMajor, DenseColMajor, SparseCSR, SparseCSC};

template <class PredictionData, class sparse_ix, class ImputedData>
double traverse_itree(std::vector<IsoTree>     &tree,
                      IsoForest                &model_outputs,
                      PredictionData           &prediction_data,
                      std::vector<ImputeNode> *impute_nodes,     /* only when imputing missing */
                      ImputedData             *imputed_data,     /* only when imputing missing */
                      double                   curr_weight,      /* only when imputing missing */
                      size_t                   row,
                      sparse_ix *restrict      tree_num,
                      double *restrict         tree_depth,
                      size_t                   curr_lev)
{
    double xval;
    int    cval;
    double range_penalty = 0;

    NumericConfig numeric_config;
    if (prediction_data.Xr_indptr != NULL)
        numeric_config = SparseCSR;
    else if (prediction_data.Xc_indptr != NULL)
        numeric_config = SparseCSC;
    else if (prediction_data.is_col_major)
        numeric_config = DenseColMajor;
    else
        numeric_config = DenseRowMajor;

    sparse_ix *row_st = NULL, *row_end = NULL;
    if (numeric_config == SparseCSR)
    {
        row_st  = prediction_data.Xr_ind + prediction_data.Xr_indptr[row];
        row_end = prediction_data.Xr_ind + prediction_data.Xr_indptr[row + 1];
    }

    while (true)
    {
        // if (tree[curr_lev].score >= 0.)
        if (unlikely(tree[curr_lev].tree_left == 0))
        {
            if (unlikely(tree_num != NULL))
                tree_num[row] = curr_lev;
            if (unlikely(tree_depth != NULL))
                *tree_depth = tree[curr_lev].score;
            if (unlikely(imputed_data != NULL))
                add_from_impute_node((*impute_nodes)[curr_lev], *imputed_data, curr_weight);

            return tree[curr_lev].score - range_penalty;
        }

        else
        {
            switch(tree[curr_lev].col_type)
            {
                case Numeric:
                {
                    switch(numeric_config)
                    {
                        case DenseRowMajor:
                        {
                            xval = prediction_data.numeric_data[tree[curr_lev].col_num + row * prediction_data.ncols_numeric];
                            break;
                        }

                        case DenseColMajor:
                        {
                            xval = prediction_data.numeric_data[row +  tree[curr_lev].col_num * prediction_data.nrows];
                            break;
                        }

                        case SparseCSR:
                        {
                            xval = extract_spR(prediction_data, row_st, row_end, tree[curr_lev].col_num);
                            break;
                        }

                        case SparseCSC:
                        {
                            xval = extract_spC(prediction_data, row, tree[curr_lev].col_num);
                            break;
                        }
                    }

                    if (unlikely(std::isnan(xval)))
                    {
                        switch(model_outputs.missing_action)
                        {
                            case Divide:
                            {
                                if (tree_num || tree_depth) throw_unsupported_pred_error();
                                return
                                    tree[curr_lev].pct_tree_left
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_left)
                                    + (1. - tree[curr_lev].pct_tree_left)
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_right)
                                    - range_penalty;
                            }

                            case Impute:
                            {
                                curr_lev = (tree[curr_lev].pct_tree_left >= .5)?
                                                tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                break;
                            }

                            case Fail:
                            {
                                return NAN;
                            }
                        }
                    }

                    else
                    {
                        range_penalty += (xval < tree[curr_lev].range_low) || (xval > tree[curr_lev].range_high);
                        curr_lev = (xval <= tree[curr_lev].num_split)?
                                    tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                    }
                    break;
                }

                case Categorical:
                {
                    cval =  prediction_data.categ_data[
                                prediction_data.is_col_major?
                                (row +  tree[curr_lev].col_num * prediction_data.nrows)
                                    :
                                (tree[curr_lev].col_num + row * prediction_data.ncols_categ)
                            ];
                    if (unlikely(cval < 0))
                    {
                        switch(model_outputs.missing_action)
                        {
                            case Divide:
                            {
                                if (tree_num || tree_depth) throw_unsupported_pred_error();
                                return
                                    tree[curr_lev].pct_tree_left
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_left)
                                    + (1. - tree[curr_lev].pct_tree_left)
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_right)
                                    - range_penalty;
                            }

                            case Impute:
                            {
                                curr_lev = (tree[curr_lev].pct_tree_left >= .5)?
                                                tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                break;
                            }

                            case Fail:
                            {
                                return NAN;
                            }
                        }
                    }

                    else
                    {
                        switch(model_outputs.cat_split_type)
                        {
                            case SingleCateg:
                            {
                                curr_lev = (cval == tree[curr_lev].chosen_cat)?
                                            tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                break;
                            }

                            case SubSet:
                            {

                                if (tree[curr_lev].cat_split.empty())
                                {
                                    if (cval <= 1)
                                    {
                                        curr_lev = (cval == 0)?
                                                    tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                    }

                                    else
                                    {
                                        switch(model_outputs.new_cat_action)
                                        {
                                            case Smallest:
                                            {
                                                curr_lev =  (tree[curr_lev].pct_tree_left < .5)? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                                break;
                                            }

                                            case Weighted:
                                            {
                                                if (tree_num || tree_depth) throw_unsupported_pred_error();
                                                return
                                                    tree[curr_lev].pct_tree_left
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_left)
                                                    + (1. - tree[curr_lev].pct_tree_left)
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_right)
                                                    - range_penalty;
                                            }

                                            default:
                                            {
                                                assert(0);
                                                break;
                                            }
                                        }
                                    }
                                }

                                else
                                {
                                    switch(model_outputs.new_cat_action)
                                    {
                                        case Random:
                                        {
                                            cval = (cval >= (int)tree[curr_lev].cat_split.size())?
                                                    (cval % (int)tree[curr_lev].cat_split.size()) : cval;
                                            curr_lev = (tree[curr_lev].cat_split[cval])?
                                                        tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                            break;
                                        }

                                        case Smallest:
                                        {
                                            if (unlikely(cval >= (int)tree[curr_lev].cat_split.size()))
                                            {
                                                curr_lev =  (tree[curr_lev].pct_tree_left < .5)? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                            }

                                            else
                                            {
                                                curr_lev = (tree[curr_lev].cat_split[cval])?
                                                            tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                            }
                                            break;
                                        }

                                        case Weighted:
                                        {
                                            if (cval >= (int)tree[curr_lev].cat_split.size()
                                                    ||
                                                tree[curr_lev].cat_split[cval] == (-1))
                                            {
                                                if (tree_num || tree_depth) throw_unsupported_pred_error();
                                                return
                                                    tree[curr_lev].pct_tree_left
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_left)
                                                    + (1. - tree[curr_lev].pct_tree_left)
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                                         row, (sparse_ix*)NULL, tree_depth, tree[curr_lev].tree_right)
                                                    - range_penalty;
                                            }

                                            else
                                            {
                                                curr_lev = (tree[curr_lev].cat_split[cval])?
                                                            tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                            }
                                            break;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                    break;
                }

                default:
                {
                    assert(0);
                    break;
                }
            }
        }
    }
}

/* this is a simpler version for situations in which there is
   only numeric data in dense arrays, no missing values, no range penalty */
template <class PredictionData, class sparse_ix>
void traverse_hplane_fast_colmajor(std::vector<IsoHPlane>  &hplane,
                                   ExtIsoForest            &model_outputs,
                                   PredictionData          &prediction_data,
                                   double &restrict        output_depth,
                                   sparse_ix *restrict     tree_num,
                                   double *restrict        tree_depth,
                                   size_t                  row) noexcept
{
    size_t  curr_lev = 0;
    double  hval;

    while(true)
    {
        // if (hplane[curr_lev].score > 0)
        if (unlikely(hplane[curr_lev].hplane_left == 0))
        {
            output_depth += hplane[curr_lev].score;
            if (unlikely(tree_num != NULL))
                tree_num[row] = curr_lev;
            if (unlikely(tree_depth != NULL))
                *tree_depth = hplane[curr_lev].score;
            return;
        }

        else
        {
            hval = 0;
            for (size_t col = 0; col < hplane[curr_lev].col_num.size(); col++)
                hval += (prediction_data.numeric_data[row +  hplane[curr_lev].col_num[col] * prediction_data.nrows] 
                         - hplane[curr_lev].mean[col]) * hplane[curr_lev].coef[col];

            curr_lev  = (hval <= hplane[curr_lev].split_point)?
                         hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;

        }
    }
}

template <class real_t, class sparse_ix>
void traverse_hplane_fast_rowmajor(std::vector<IsoHPlane>  &hplane,
                                   ExtIsoForest            &model_outputs,
                                   real_t *restrict        row_numeric_data,
                                   double &restrict        output_depth,
                                   sparse_ix *restrict     tree_num,
                                   double *restrict        tree_depth,
                                   size_t                  row) noexcept
{
    size_t  curr_lev = 0;
    double  hval;

    while(true)
    {
        // if (hplane[curr_lev].score > 0)
        if (unlikely(hplane[curr_lev].hplane_left == 0))
        {
            output_depth += hplane[curr_lev].score;
            if (unlikely(tree_num != NULL))
                tree_num[row] = curr_lev;
            if (unlikely(tree_depth != NULL))
                *tree_depth = hplane[curr_lev].score;
            return;
        }

        else
        {
            hval = 0;
            for (size_t col = 0; col < hplane[curr_lev].col_num.size(); col++)
                hval += (row_numeric_data[hplane[curr_lev].col_num[col]] 
                         - hplane[curr_lev].mean[col]) * hplane[curr_lev].coef[col];

            curr_lev  = (hval <= hplane[curr_lev].split_point)?
                         hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;

        }
    }
}

/* this is the full version that works with potentially missing values, sparse matrices, and categoricals */
template <class PredictionData, class sparse_ix, class ImputedData>
void traverse_hplane(std::vector<IsoHPlane>   &hplane,
                     ExtIsoForest             &model_outputs,
                     PredictionData           &prediction_data,
                     double &restrict         output_depth,
                     std::vector<ImputeNode> *impute_nodes,     /* only when imputing missing */
                     ImputedData             *imputed_data,     /* only when imputing missing */
                     sparse_ix *restrict      tree_num,
                     double *restrict         tree_depth,
                     size_t                   row) noexcept
{
    size_t  curr_lev = 0;
    double  xval;
    int     cval;
    double  hval;

    size_t ncols_numeric, ncols_categ;

    NumericConfig numeric_config;
    if (prediction_data.Xr_indptr != NULL)
        numeric_config = SparseCSR;
    else if (prediction_data.Xc_indptr != NULL)
        numeric_config = SparseCSC;
    else if (prediction_data.is_col_major)
        numeric_config = DenseColMajor;
    else
        numeric_config = DenseRowMajor;

    sparse_ix *row_st = NULL, *row_end = NULL;
    size_t lb, ub;
    if (numeric_config == SparseCSR)
    {
        row_st  = prediction_data.Xr_ind + prediction_data.Xr_indptr[row];
        row_end = prediction_data.Xr_ind + prediction_data.Xr_indptr[row + 1];
        lb = *row_st;
        ub = *(row_end-1);
    }

    while (true)
    {
        // if (hplane[curr_lev].score > 0)
        if (unlikely(hplane[curr_lev].hplane_left == 0))
        {
            output_depth += hplane[curr_lev].score;
            if (unlikely(tree_num != NULL))
                tree_num[row] = curr_lev;
            if (unlikely(tree_depth != NULL))
                *tree_depth = hplane[curr_lev].score;
            if (unlikely(imputed_data != NULL))
            {
                add_from_impute_node((*impute_nodes)[curr_lev], *imputed_data, (double)1);
            }
            return;
        }

        else
        {
            hval = 0;
            ncols_numeric = 0; ncols_categ = 0;
            for (size_t col = 0; col < hplane[curr_lev].col_num.size(); col++)
            {
                switch(hplane[curr_lev].col_type[col])
                {
                    case Numeric:
                    {
                        switch(numeric_config)
                        {
                            case DenseRowMajor:
                            {
                                xval = prediction_data.numeric_data[hplane[curr_lev].col_num[col] + row * prediction_data.ncols_numeric];
                                break;
                            }

                            case DenseColMajor:
                            {
                                xval = prediction_data.numeric_data[row +  hplane[curr_lev].col_num[col] * prediction_data.nrows];
                                break;
                            }

                            case SparseCSR:
                            {
                                xval = extract_spR(prediction_data, row_st, row_end, hplane[curr_lev].col_num[col], lb, ub);
                                break;
                            }

                            case SparseCSC:
                            {
                                xval = extract_spC(prediction_data, row, hplane[curr_lev].col_num[col]);
                                break;
                            }
                        }

                        if (unlikely(is_na_or_inf(xval)))
                        {
                            if (model_outputs.missing_action != Fail)
                            {
                                hval += hplane[curr_lev].fill_val[col];
                            }

                            else
                            {
                                output_depth = NAN;
                                return;
                            }
                        }

                        else
                        {
                            hval += (xval - hplane[curr_lev].mean[ncols_numeric]) * hplane[curr_lev].coef[ncols_numeric];
                        }

                        ncols_numeric++;
                        break;
                    }

                    case Categorical:
                    {
                        cval = prediction_data.categ_data[
                            prediction_data.is_col_major?
                            (row +  hplane[curr_lev].col_num[col] * prediction_data.nrows)
                                :
                            (hplane[curr_lev].col_num[col] + row * prediction_data.ncols_categ)
                        ];
                        if (unlikely(cval < 0))
                        {
                            if (model_outputs.missing_action != Fail)
                            {
                                hval += hplane[curr_lev].fill_val[col];
                            }
                            
                            else
                            {
                                output_depth = NAN;
                                return;
                            }
                        }

                        else
                        {
                            switch(model_outputs.cat_split_type)
                            {
                                case SingleCateg:
                                {
                                    hval += (cval == hplane[curr_lev].chosen_cat[ncols_categ])? hplane[curr_lev].fill_new[ncols_categ] : 0;
                                    break;
                                }

                                case SubSet:
                                {
                                    if (unlikely(cval >= (int)hplane[curr_lev].cat_coef[ncols_categ].size()))
                                    {
                                        if (model_outputs.new_cat_action == Random) {
                                            cval = cval % (int)hplane[curr_lev].cat_coef[ncols_categ].size();
                                            hval += hplane[curr_lev].cat_coef[ncols_categ][cval];
                                        }

                                        else {
                                            hval += hplane[curr_lev].fill_new[ncols_categ];
                                        }
                                    }
                                    
                                    else
                                    {
                                        hval += hplane[curr_lev].cat_coef[ncols_categ][cval];
                                    }
                                    
                                    break;
                                }
                            }
                        }

                        ncols_categ++;
                        break;
                    }

                    default:
                    {
                        assert(0);
                        break;
                    }
                }

            }

            output_depth -= (hval < hplane[curr_lev].range_low) ||
                            (hval > hplane[curr_lev].range_high);
            curr_lev       = (hval <= hplane[curr_lev].split_point)?
                             hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;
        }
    }
}

template <class real_t, class sparse_ix>
void batched_csc_predict(PredictionData<real_t, sparse_ix> &prediction_data, int nthreads,
                         IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                         double *restrict output_depths,   sparse_ix *restrict tree_num,
                         double *restrict per_tree_depths)
{
    #ifdef _OPENMP
    size_t ntrees = (model_outputs != NULL)? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
    if ((size_t)nthreads > ntrees)
        nthreads = ntrees;
    #else
    nthreads = 1;
    #endif
    std::vector<WorkerForPredictCSC> worker_memory(nthreads);

    bool threw_exception = false;
    std::exception_ptr ex = NULL;

    if (model_outputs != NULL)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(worker_memory, model_outputs, prediction_data, tree_num, per_tree_depths, threw_exception, ex)
        for (size_t_for tree = 0; tree < (decltype(tree))model_outputs->trees.size(); tree++)
        {
            if (threw_exception) continue;
            try
            {
                WorkerForPredictCSC *ptr_worker = &worker_memory[omp_get_thread_num()];
                if (!ptr_worker->depths.size())
                {
                    ptr_worker->depths.resize(prediction_data.nrows);
                    ptr_worker->ix_arr.resize(prediction_data.nrows);
                    std::iota(ptr_worker->ix_arr.begin(),
                              ptr_worker->ix_arr.end(),
                              (size_t)0);

                    if (model_outputs->missing_action == Divide ||
                        (model_outputs->new_cat_action == Weighted && model_outputs->cat_split_type == SubSet && prediction_data.categ_data != NULL)
                    ) {
                        ptr_worker->weights_arr.resize(prediction_data.nrows);
                    }
                }

                ptr_worker->st  = 0;
                ptr_worker->end = prediction_data.nrows - 1;
                if (model_outputs->missing_action == Divide)
                    std::fill(ptr_worker->weights_arr.begin(),
                              ptr_worker->weights_arr.end(),
                              (double)1);

                traverse_itree_csc(*ptr_worker,
                                   model_outputs->trees[tree],
                                   *model_outputs,
                                   prediction_data,
                                   (tree_num == NULL)?
                                        ((sparse_ix*)NULL) : (tree_num + tree*prediction_data.nrows),
                                   per_tree_depths,
                                   (size_t)0,
                                   model_outputs->has_range_penalty);
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

    else
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(worker_memory, model_outputs_ext, prediction_data, tree_num, per_tree_depths, threw_exception, ex)
        for (size_t_for tree = 0; tree < (decltype(tree))model_outputs_ext->hplanes.size(); tree++)
        {
            if (threw_exception) continue;
            try
            {
                WorkerForPredictCSC *ptr_worker = &worker_memory[omp_get_thread_num()];
                if (!ptr_worker->depths.size())
                {
                    ptr_worker->depths.resize(prediction_data.nrows);
                    ptr_worker->comb_val.resize(prediction_data.nrows);
                    ptr_worker->ix_arr.resize(prediction_data.nrows);
                    std::iota(ptr_worker->ix_arr.begin(),
                              ptr_worker->ix_arr.end(),
                              (size_t)0);
                }

                ptr_worker->st  = 0;
                ptr_worker->end = prediction_data.nrows - 1;

                traverse_hplane_csc(*ptr_worker,
                                    model_outputs_ext->hplanes[tree],
                                    *model_outputs_ext,
                                    prediction_data,
                                    (tree_num == NULL)?
                                        ((sparse_ix*)NULL) : (tree_num + tree*prediction_data.nrows),
                                    per_tree_depths,
                                    (size_t)0,
                                    model_outputs_ext->has_range_penalty);
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

    #ifdef _OPENMP
    if (nthreads <= 1)
    #endif
    {
        std::copy(worker_memory.front().depths.begin(), worker_memory.front().depths.end(), output_depths);
    }

    #ifdef _OPENMP
    else
    {
        std::fill(output_depths, output_depths + prediction_data.nrows, (double)0);
        for (auto &workspace : worker_memory)
            if (workspace.depths.size())
                #if !defined(_MSC_VER) && !defined(_WIN32)
                #pragma omp simd
                #endif
                for (size_t row = 0; row < prediction_data.nrows; row++)
                    output_depths[row] += workspace.depths[row];
    }
    #endif
}

template <class PredictionData, class sparse_ix>
void traverse_itree_csc(WorkerForPredictCSC   &workspace,
                        std::vector<IsoTree>  &trees,
                        IsoForest             &model_outputs,
                        PredictionData        &prediction_data,
                        sparse_ix *restrict   tree_num,
                        double *restrict      per_tree_depths,
                        size_t                curr_tree,
                        bool                  has_range_penalty)
{
    // if (trees[curr_tree].score >= 0)
    if (unlikely(trees[curr_tree].tree_left == 0))
    {
        if (model_outputs.missing_action != Divide)
            for (size_t row = workspace.st; row <= workspace.end; row++)
                workspace.depths[workspace.ix_arr[row]] += trees[curr_tree].score;
        else
            for (size_t row = workspace.st; row <= workspace.end; row++)
                workspace.depths[workspace.ix_arr[row]] += workspace.weights_arr[workspace.ix_arr[row]] * trees[curr_tree].score;
        if (unlikely(tree_num != NULL))
            for (size_t row = workspace.st; row <= workspace.end; row++)
                tree_num[workspace.ix_arr[row]] = curr_tree;
        if (unlikely(per_tree_depths != NULL))
            for (size_t row = workspace.st; row <= workspace.end; row++)
                per_tree_depths[workspace.ix_arr[row]] = trees[curr_tree].score;
        return;
    }

    /* in this case, the indices are sorted in the csc penalty function */
    if (!(has_range_penalty && model_outputs.missing_action != Divide && curr_tree > 0) && trees[curr_tree].col_type == Numeric)
        std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);

    /* TODO: should mix the splitting function with the range penalty */
    
    /* divide according to tree */
    size_t orig_end = workspace.end;
    size_t st_NA, end_NA, split_ix;
    switch (trees[curr_tree].col_type)
    {
        case Numeric:
        {
            divide_subset_split(workspace.ix_arr.data(), workspace.st, workspace.end, trees[curr_tree].col_num,
                                prediction_data.Xc, prediction_data.Xc_ind, prediction_data.Xc_indptr,
                                trees[curr_tree].num_split, model_outputs.missing_action,
                                st_NA, end_NA, split_ix);
            break;
        }

        case Categorical:
        {
            switch (model_outputs.cat_split_type)
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

        default:
        {
            assert(0);
            break;
        }
    }

    /* continue splitting recursively */
    if (unlikely(model_outputs.new_cat_action == Weighted && model_outputs.cat_split_type == SubSet && prediction_data.categ_data != NULL))
        goto missing_action_divide;
    switch (model_outputs.missing_action)
    {
        case Impute:
        {
            split_ix = (trees[curr_tree].pct_tree_left >= .5)? end_NA : st_NA;
        }

        case Fail:
        {
            if (split_ix > workspace.st)
            {
                workspace.end = split_ix - 1;

                if (has_range_penalty && trees[curr_tree].col_type == Numeric)
                    add_csc_range_penalty(workspace,
                                          prediction_data,
                                          (double*)NULL,
                                          trees[curr_tree].col_num,
                                          trees[curr_tree].range_low,
                                          trees[curr_tree].range_high);

                traverse_itree_csc(workspace,
                                   trees,
                                   model_outputs,
                                   prediction_data,
                                   tree_num,
                                   per_tree_depths,
                                   trees[curr_tree].tree_left,
                                   has_range_penalty);
            }


            if (split_ix <= orig_end)
            {
                workspace.st  = split_ix;
                workspace.end = orig_end;

                if (has_range_penalty && trees[curr_tree].col_type == Numeric)
                    add_csc_range_penalty(workspace,
                                          prediction_data,
                                          (double*)NULL,
                                          trees[curr_tree].col_num,
                                          trees[curr_tree].range_low,
                                          trees[curr_tree].range_high);

                traverse_itree_csc(workspace,
                                   trees,
                                   model_outputs,
                                   prediction_data,
                                   tree_num,
                                   per_tree_depths,
                                   trees[curr_tree].tree_right,
                                   has_range_penalty);
            }
            break;
        }

        case Divide:
        {
            missing_action_divide:
            /* TODO: maybe here it shouldn't copy the whole ix_arr,
               but then it'd need to re-generate it from outside too */
            std::vector<double> weights_arr;
            std::vector<size_t> ix_arr;
            if (end_NA > workspace.st)
            {
                weights_arr.assign(workspace.weights_arr.begin(),
                                   workspace.weights_arr.begin() + end_NA);
                ix_arr.assign(workspace.ix_arr.data(),
                              workspace.ix_arr.data() + end_NA);
            }

            if (has_range_penalty && trees[curr_tree].col_type == Numeric)
            {
                size_t st = workspace.st;
                size_t end = workspace.end;

                if (workspace.st < st_NA)
                {
                    workspace.end = st_NA - 1;
                    add_csc_range_penalty(workspace,
                                          prediction_data,
                                          workspace.weights_arr.data(),
                                          trees[curr_tree].col_num,
                                          trees[curr_tree].range_low,
                                          trees[curr_tree].range_high);
                }

                if (workspace.end >= end_NA)
                {
                    workspace.st = end_NA;
                    workspace.end = end;
                    add_csc_range_penalty(workspace,
                                          prediction_data,
                                          workspace.weights_arr.data(),
                                          trees[curr_tree].col_num,
                                          trees[curr_tree].range_low,
                                          trees[curr_tree].range_high);
                }

                workspace.st = st;
                workspace.end = end;
            }

            if (end_NA > workspace.st)
            {
                if (st_NA < end_NA && (tree_num || per_tree_depths)) throw_unsupported_pred_error();
                workspace.end = end_NA - 1;
                for (size_t row = st_NA; row < end_NA; row++)
                    workspace.weights_arr[workspace.ix_arr[row]] *= trees[curr_tree].pct_tree_left;
                traverse_itree_csc(workspace,
                                   trees,
                                   model_outputs,
                                   prediction_data,
                                   tree_num,
                                   per_tree_depths,
                                   trees[curr_tree].tree_left,
                                   has_range_penalty);
            }

            if (st_NA <= orig_end)
            {
                if (st_NA < end_NA && (tree_num || per_tree_depths)) throw_unsupported_pred_error();
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
                    workspace.weights_arr[workspace.ix_arr[row]] *= (1. - trees[curr_tree].pct_tree_left);
                traverse_itree_csc(workspace,
                                   trees,
                                   model_outputs,
                                   prediction_data,
                                   tree_num,
                                   per_tree_depths,
                                   trees[curr_tree].tree_right,
                                   has_range_penalty);
            }
            break;
        }
    }
}

template <class PredictionData, class sparse_ix>
void traverse_hplane_csc(WorkerForPredictCSC      &workspace,
                         std::vector<IsoHPlane>   &hplanes,
                         ExtIsoForest             &model_outputs,
                         PredictionData           &prediction_data,
                         sparse_ix *restrict      tree_num,
                         double *restrict         per_tree_depths,
                         size_t                   curr_tree,
                         bool                     has_range_penalty)
{
    // if (hplanes[curr_tree].score >= 0)
    if (unlikely(hplanes[curr_tree].hplane_left == 0))
    {
        for (size_t row = workspace.st; row <= workspace.end; row++)
            workspace.depths[workspace.ix_arr[row]] += hplanes[curr_tree].score;
        if (unlikely(tree_num != NULL))
            for (size_t row = workspace.st; row <= workspace.end; row++)
                tree_num[workspace.ix_arr[row]] = curr_tree;
        if (unlikely(per_tree_depths != NULL))
            for (size_t row = workspace.st; row <= workspace.end; row++)
                per_tree_depths[workspace.ix_arr[row]] = hplanes[curr_tree].score;
        return;
    }

    std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);
    std::fill(workspace.comb_val.begin(), workspace.comb_val.begin() + (workspace.end - workspace.st + 1), 0.);
    double unused;

    if (likely(prediction_data.categ_data == NULL))
    {
        for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
            add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end,
                            hplanes[curr_tree].col_num[col], workspace.comb_val.data(),
                            prediction_data.Xc, prediction_data.Xc_ind, prediction_data.Xc_indptr,
                            hplanes[curr_tree].coef[col], (double)0, hplanes[curr_tree].mean[col],
                            (model_outputs.missing_action == Fail)?  unused : hplanes[curr_tree].fill_val[col],
                            model_outputs.missing_action, NULL, NULL, false);
    }

    else
    {
        size_t ncols_numeric = 0;
        size_t ncols_categ = 0;
        for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
        {
            switch (hplanes[curr_tree].col_type[col])
            {
                case Numeric:
                {
                    add_linear_comb(workspace.ix_arr.data(), workspace.st, workspace.end,
                                    hplanes[curr_tree].col_num[col], workspace.comb_val.data(),
                                    prediction_data.Xc, prediction_data.Xc_ind, prediction_data.Xc_indptr,
                                    hplanes[curr_tree].coef[ncols_numeric], (double)0, hplanes[curr_tree].mean[ncols_numeric],
                                    (model_outputs.missing_action == Fail)?  unused : hplanes[curr_tree].fill_val[col],
                                    model_outputs.missing_action, NULL, NULL, false);
                    ncols_numeric++;
                    break;
                }

                case Categorical:
                {
                    add_linear_comb<double>(
                                    workspace.ix_arr.data(), workspace.st, workspace.end, workspace.comb_val.data(),
                                    prediction_data.categ_data + hplanes[curr_tree].col_num[col] * prediction_data.nrows,
                                    (model_outputs.cat_split_type == SubSet)? (int)hplanes[curr_tree].cat_coef[ncols_categ].size() : 0,
                                    (model_outputs.cat_split_type == SubSet)? hplanes[curr_tree].cat_coef[ncols_categ].data() : NULL,
                                    (model_outputs.cat_split_type == SingleCateg)? hplanes[curr_tree].fill_new[ncols_categ] : 0.,
                                    (model_outputs.cat_split_type == SingleCateg)? hplanes[curr_tree].chosen_cat[ncols_categ] : 0,
                                    hplanes[curr_tree].fill_val[col], hplanes[curr_tree].fill_new[ncols_categ], NULL, NULL,
                                    model_outputs.new_cat_action, model_outputs.missing_action, model_outputs.cat_split_type, false);
                    ncols_categ++;
                    break;
                }

                default:
                {
                    assert(0);
                    break;
                }
            }
        }
    }

    if (has_range_penalty)
    {
        for (size_t row = workspace.st; row <= workspace.end; row++)
            workspace.depths[workspace.ix_arr[row]]
                -=
            (workspace.comb_val[row - workspace.st] < hplanes[curr_tree].range_low) ||
            (workspace.comb_val[row - workspace.st] > hplanes[curr_tree].range_high);
    }

    /* divide data */
    size_t split_ix = divide_subset_split(workspace.ix_arr.data(), workspace.comb_val.data(),
                                          workspace.st, workspace.end, hplanes[curr_tree].split_point);

    /* continue splitting recursively */
    size_t orig_end = workspace.end;
    if (split_ix > workspace.st)
    {
        workspace.end = split_ix - 1;
        traverse_hplane_csc(workspace,
                            hplanes,
                            model_outputs,
                            prediction_data,
                            tree_num,
                            per_tree_depths,
                            hplanes[curr_tree].hplane_left,
                            has_range_penalty);
    }

    if (split_ix <= orig_end)
    {
        workspace.st  = split_ix;
        workspace.end = orig_end;
        traverse_hplane_csc(workspace,
                            hplanes,
                            model_outputs,
                            prediction_data,
                            tree_num,
                            per_tree_depths,
                            hplanes[curr_tree].hplane_right,
                            has_range_penalty);
    }
}

template <class PredictionData>
void add_csc_range_penalty(WorkerForPredictCSC     &workspace,
                           const PredictionData    &prediction_data,
                           const double *restrict  weights_arr,
                           size_t                  col_num,
                           double                  range_low,
                           double                  range_high)
{
    std::sort(workspace.ix_arr.begin() + workspace.st, workspace.ix_arr.begin() + workspace.end + 1);

    size_t st_col  = prediction_data.Xc_indptr[col_num];
    size_t end_col = prediction_data.Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = prediction_data.Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(workspace.ix_arr.data() + workspace.st,
                                      workspace.ix_arr.data() + workspace.end + 1,
                                      prediction_data.Xc_ind[st_col]);

    if (range_low <= 0 && range_high >= 0)
    {
        for (size_t *row = ptr_st;
             row != workspace.ix_arr.data() + workspace.end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (prediction_data.Xc_ind[curr_pos] == (decltype(*prediction_data.Xc_ind))(*row))
            {
                if (likely(!std::isnan(prediction_data.Xc[curr_pos])
                               &&
                           (   prediction_data.Xc[curr_pos] < range_low    ||
                               prediction_data.Xc[curr_pos] > range_high   )))
                {
                    workspace.depths[*row] -= (weights_arr == NULL)? 1. : weights_arr[*row];
                }
                
                if (row == workspace.ix_arr.data() + workspace.end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(prediction_data.Xc_ind + curr_pos + 1,
                                            prediction_data.Xc_ind + end_col  + 1,
                                            *(++row))
                                - prediction_data.Xc_ind;
            }

            else
            {
                if (prediction_data.Xc_ind[curr_pos] > (decltype(*prediction_data.Xc_ind))(*row))
                    row = std::lower_bound(row + 1,
                                           workspace.ix_arr.data() + workspace.end + 1,
                                           prediction_data.Xc_ind[curr_pos]);
                else
                    curr_pos = std::lower_bound(prediction_data.Xc_ind + curr_pos + 1,
                                                prediction_data.Xc_ind + end_col  + 1,
                                                *row)
                                    - prediction_data.Xc_ind;
            }
        }
    }

    else
    {
        if (likely(weights_arr == NULL))
            for (size_t row = workspace.st; row <= workspace.end; row++)
                workspace.depths[workspace.ix_arr[row]]--;
        else
            for (size_t row = workspace.st; row <= workspace.end; row++)
                workspace.depths[workspace.ix_arr[row]] -= weights_arr[workspace.ix_arr[row]];


        for (size_t *row = ptr_st;
             row != workspace.ix_arr.data() + workspace.end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (prediction_data.Xc_ind[curr_pos] == (decltype(*prediction_data.Xc_ind))(*row))
            {
                if (likely(std::isnan(prediction_data.Xc[curr_pos])
                               ||
                           (   prediction_data.Xc[curr_pos] >= range_low    &&
                               prediction_data.Xc[curr_pos] <= range_high   )))
                {
                    workspace.depths[*row] += (weights_arr == NULL)? 1. : weights_arr[*row];
                }
                
                if (row == workspace.ix_arr.data() + workspace.end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(prediction_data.Xc_ind + curr_pos + 1,
                                            prediction_data.Xc_ind + end_col  + 1,
                                            *(++row))
                                - prediction_data.Xc_ind;
            }

            else
            {
                if (prediction_data.Xc_ind[curr_pos] > (decltype(*prediction_data.Xc_ind))(*row))
                    row = std::lower_bound(row + 1,
                                           workspace.ix_arr.data() + workspace.end + 1,
                                           prediction_data.Xc_ind[curr_pos]);
                else
                    curr_pos = std::lower_bound(prediction_data.Xc_ind + curr_pos + 1,
                                                prediction_data.Xc_ind + end_col  + 1,
                                                *row)
                                    - prediction_data.Xc_ind;
            }
        }
    }
}

void throw_unsupported_pred_error()
{
    throw std::runtime_error(
        "Cannot predict per-tree depths or indices for missing data with 'missing_action=divide' or new categories with 'new_categ_action=weighted'."
    );
}

template <class PredictionData>
double extract_spC(const PredictionData &prediction_data, size_t row, size_t col_num) noexcept
{
    const decltype(prediction_data.Xc_indptr)
               search_res = std::lower_bound(prediction_data.Xc_ind + prediction_data.Xc_indptr[col_num],
                                             prediction_data.Xc_ind + prediction_data.Xc_indptr[col_num + 1],
                                             row);
    if (
        search_res == (prediction_data.Xc_ind + prediction_data.Xc_indptr[col_num + 1])
            ||
        (*search_res) != static_cast<typename std::remove_pointer<decltype(search_res)>::type>(row)
        )
        return 0.;
    else
        return prediction_data.Xc[search_res - prediction_data.Xc_ind];
}

template <class PredictionData, class sparse_ix>
static inline double extract_spR(const PredictionData &prediction_data,
                                 const sparse_ix *row_st, const sparse_ix *row_end,
                                 size_t col_num, size_t lb, size_t ub) noexcept
{
    if (row_end == row_st || col_num < lb || col_num > ub)
        return 0.;
    const sparse_ix *search_res = std::lower_bound(row_st, row_end, (sparse_ix) col_num);
    if (search_res == row_end || *search_res != (sparse_ix)col_num)
        return 0.;
    else
        return prediction_data.Xr[search_res - prediction_data.Xr_ind];
}

template <class PredictionData, class sparse_ix>
double extract_spR(const PredictionData &prediction_data, const sparse_ix *row_st, const sparse_ix *row_end, size_t col_num) noexcept
{
    if (row_end == row_st)
        return 0.;
    const sparse_ix *search_res = std::lower_bound(row_st, row_end, (sparse_ix) col_num);
    if (search_res == row_end || *search_res != (sparse_ix)col_num)
        return 0.;
    else
        return prediction_data.Xr[search_res - prediction_data.Xr_ind];
}

template <class sparse_ix>
void get_num_nodes(const IsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads) noexcept
{
    std::fill(n_terminal, n_terminal + model_outputs.trees.size(), 0);
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(model_outputs, n_nodes, n_terminal)
    for (size_t_for tree = 0; tree < (decltype(tree))model_outputs.trees.size(); tree++)
    {
        n_nodes[tree] = model_outputs.trees[tree].size();
        for (const IsoTree &node : model_outputs.trees[tree])
        {
            n_terminal[tree] += (node.tree_left == 0);
        }
    }
}

template <class sparse_ix>
void get_num_nodes(const ExtIsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads) noexcept
{
    std::fill(n_terminal, n_terminal + model_outputs.hplanes.size(), 0);
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(model_outputs, n_nodes, n_terminal)
    for (size_t_for hplane = 0; hplane <(decltype(hplane)) model_outputs.hplanes.size(); hplane++)
    {
        n_nodes[hplane] = model_outputs.hplanes[hplane].size();
        for (const IsoHPlane &node : model_outputs.hplanes[hplane])
        {
            n_terminal[hplane] += (node.hplane_left == 0);
        }
    }
}
