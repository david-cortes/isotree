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

/* Predict outlier score, average depth, or terminal node numbers
* 
* Parameters
* ==========
* - numeric_data[nrows * ncols_numeric]
*       Pointer to numeric data for which to make predictions. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.),
*       and the column order must be the same as in the data that was used to fit the model.
*       Pass NULL if there are no dense numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
* - categ_data[nrows * ncols_categ]
*       Pointer to categorical data for which to make predictions. Must be ordered by columns like Fortran,
*       not ordered by rows like C (i.e. entries 1..n contain column 0, n+1..2n column 1, etc.),
*       and the column order must be the same as in the data that was used to fit the model.
*       Pass NULL if there are no categorical columns.
*       Each category should be represented as an integer, and these integers must start at zero and
*       be in consecutive order - i.e. if category '3' is present, category '2' must have also been
*       present when the model was fit (note that they are not treated as being ordinal, this is just
*       an encoding). Missing values should be encoded as negative numbers such as (-1). The encoding
*       must be the same as was used in the data to which the model was fit.
* - Xc[nnz]
*       Pointer to numeric data in sparse numeric matrix in CSC format (column-compressed).
*       Pass NULL if there are no sparse numeric columns.
*       Can only pass one of 'numeric_data', 'Xc' + 'Xc_ind' + 'Xc_indptr', 'Xr' + 'Xr_ind' + 'Xr_indptr'.
* - Xc_ind[nnz]
*       Pointer to row indices to which each non-zero entry in 'Xc' corresponds.
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
*       (the return type is control according to parameter 'standardize').
*       Must already be initialized to zeros. Must also be passed and when the desired output
*       is terminal node numbers.
* - tree_num[nrows * ntrees] (out)
*       Pointer to array where the output terminal node numbers will be written into.
*       Note that the mapping between tree node and terminal tree node is not stored in
*       the model object for efficiency reasons, so this mapping will be determined on-the-fly
*       when passing this parameter, and as such, there will be some overhead regardless of
*       the actual number of rows. Pass NULL if only average depths or outlier scores are desired.
*/
void predict_iforest(double numeric_data[], int categ_data[],
                     double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     double Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[])
{
    /* put data in a struct for passing it in fewer lines */
    PredictionData prediction_data = {numeric_data, categ_data, nrows,
                                      Xc, Xc_ind, Xc_indptr,
                                      Xr, Xr_ind, Xr_indptr};

    if ((size_t)nthreads > nrows)
        nthreads = nrows;

    if (model_outputs != NULL)
    {
        if (
            model_outputs->missing_action == Fail &&
            (model_outputs->new_cat_action != Weighted || prediction_data.categ_data == NULL) &&
            prediction_data.Xc == NULL && prediction_data.Xr == NULL
            )
        {
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(nrows, model_outputs, prediction_data, output_depths, tree_num)
            for (size_t_for row = 0; row < nrows; row++)
            {
                for (std::vector<IsoTree> &tree : model_outputs->trees)
                {
                    traverse_itree_no_recurse(tree,
                                              *model_outputs,
                                              prediction_data,
                                              output_depths[row],
                                              (tree_num == NULL)? NULL : tree_num + nrows * (&tree - &(model_outputs->trees[0])),
                                              (size_t) row);
                }
            }
        }

        else
        {
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(nrows, model_outputs, prediction_data, output_depths, tree_num)
            for (size_t_for row = 0; row < nrows; row++)
            {
                for (std::vector<IsoTree> &tree : model_outputs->trees)
                {
                    output_depths[row] += traverse_itree(tree,
                                                         *model_outputs,
                                                         prediction_data,
                                                         NULL, NULL, 0,
                                                         (size_t) row,
                                                         (tree_num == NULL)? NULL : tree_num + nrows * (&tree - &(model_outputs->trees[0])),
                                                         (size_t) 0);
                }
            }
        }
    }
    

    else
    {
        if (
            model_outputs_ext->missing_action == Fail &&
            prediction_data.categ_data == NULL &&
            prediction_data.Xc == NULL &&
            prediction_data.Xr == NULL
            )
        {
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num)
            for (size_t_for row = 0; row < nrows; row++)
            {
                for (std::vector<IsoHPlane> &hplane : model_outputs_ext->hplanes)
                {
                    traverse_hplane_fast(hplane,
                                         *model_outputs_ext,
                                         prediction_data,
                                         output_depths[row],
                                         (tree_num == NULL)? NULL : tree_num + nrows * (&hplane - &(model_outputs_ext->hplanes[0])),
                                         (size_t) row);
                }
            }
        }

        else
        {
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num)
            for (size_t_for row = 0; row < nrows; row++)
            {
                for (std::vector<IsoHPlane> &hplane : model_outputs_ext->hplanes)
                {
                    traverse_hplane(hplane,
                                    *model_outputs_ext,
                                    prediction_data,
                                    output_depths[row],
                                    NULL, NULL,
                                    (tree_num == NULL)? NULL : tree_num + nrows * (&hplane - &(model_outputs_ext->hplanes[0])),
                                    (size_t) row);
                }
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

    if (standardize)
        #pragma omp parallel for schedule(static) num_threads(nthreads) shared(nrows, output_depths, depth_divisor)
        for (size_t_for row = 0; row < nrows; row++)
            output_depths[row] = exp2( - output_depths[row] / depth_divisor );
    else
        #pragma omp parallel for schedule(static) num_threads(nthreads) shared(nrows, output_depths, ntrees)
        for (size_t_for row = 0; row < nrows; row++)
            output_depths[row] /= ntrees;


    /* re-map tree numbers to start at zero (if predicting tree numbers) */
    /* Note: usually this type of 'prediction' is not required,
       thus this mapping is not stored in the model objects so as to
       save memory */
    if (tree_num != NULL)
        remap_terminal_trees(model_outputs, model_outputs_ext,
                             prediction_data, tree_num, nthreads);
}


void traverse_itree_no_recurse(std::vector<IsoTree>  &tree,
                               IsoForest             &model_outputs,
                               PredictionData        &prediction_data,
                               double                &output_depth,
                               sparse_ix *restrict   tree_num,
                               size_t                row)
{
    size_t curr_lev = 0;
    double xval;
    while (true)
    {
        if (tree[curr_lev].score > 0)
        {
            output_depth += tree[curr_lev].score;
            if (tree_num != NULL)
                tree_num[row] = curr_lev;
            break;
        }

        else
        {
            switch(tree[curr_lev].col_type)
            {
                case Numeric:
                {
                    xval = prediction_data.numeric_data[row +  tree[curr_lev].col_num * prediction_data.nrows];
                    curr_lev = (xval <= tree[curr_lev].num_split)?
                                tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                    output_depth += (xval < tree[curr_lev].range_low) || (xval > tree[curr_lev].range_high);
                    break;
                }

                case Categorical:
                {
                    switch(model_outputs.cat_split_type)
                    {
                        case SubSet:
                        {

                            if (!tree[curr_lev].cat_split.size()) /* this is for binary columns */
                            {
                                if (prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows] <= 1)
                                {
                                    curr_lev = (
                                                prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                    == 0
                                                )?
                                                tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                }

                                else /* can only work with 'Smallest' + no NAs if reaching this point */
                                {
                                    curr_lev =  (tree[curr_lev].pct_tree_left < .5)? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                }
                            }

                            else
                            {

                                switch(model_outputs.new_cat_action)
                                {
                                    case Random:
                                    {
                                        curr_lev = (tree[curr_lev].cat_split[
                                                                prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                                ]
                                                    )?
                                                    tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                        break;
                                    }

                                    case Smallest:
                                    {
                                        if (
                                            prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                >= (int)tree[curr_lev].cat_split.size()
                                            )
                                        {
                                            curr_lev =  (tree[curr_lev].pct_tree_left < .5)? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                        }

                                        else
                                        {
                                            curr_lev = (tree[curr_lev].cat_split[
                                                                    prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                                    ]
                                                        )?
                                                        tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                        }
                                        break;
                                    }
                                }
                            }
                            break;
                        }

                        case SingleCateg:
                        {
                            curr_lev = (
                                        prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                            ==
                                        tree[curr_lev].chosen_cat
                                        )?
                                        tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
}


double traverse_itree(std::vector<IsoTree>     &tree,
                      IsoForest                &model_outputs,
                      PredictionData           &prediction_data,
                      std::vector<ImputeNode> *impute_nodes,     /* only when imputing missing */
                      ImputedData             *imputed_data,     /* only when imputing missing */
                      double                   curr_weight,      /* only when imputing missing */
                      size_t                   row,
                      sparse_ix *restrict      tree_num,
                      size_t                   curr_lev)
{
    double xval;
    double range_penalty = 0;

    sparse_ix *row_st = NULL, *row_end = NULL;
    if (prediction_data.Xr != NULL)
    {
        row_st  = prediction_data.Xr_ind + prediction_data.Xr_indptr[row];
        row_end = prediction_data.Xr_ind + prediction_data.Xr_indptr[row + 1];
    }

    while (true)
    {
        if (tree[curr_lev].score >= 0.)
        {
            if (tree_num != NULL)
                tree_num[row] = curr_lev;
            if (imputed_data != NULL)
                add_from_impute_node((*impute_nodes)[curr_lev], *imputed_data, curr_weight);

            return tree[curr_lev].score + range_penalty;
        }

        else
        {
            switch(tree[curr_lev].col_type)
            {
                case Numeric:
                {

                    if (prediction_data.Xc == NULL && prediction_data.Xr == NULL)
                        xval = prediction_data.numeric_data[row +  tree[curr_lev].col_num * prediction_data.nrows];
                    else if (row_st != NULL)
                        xval = extract_spR(prediction_data, row_st, row_end, tree[curr_lev].col_num);
                    else
                        xval = extract_spC(prediction_data, row, tree[curr_lev].col_num);

                    if (isnan(xval))
                    {
                        switch(model_outputs.missing_action)
                        {
                            case Divide:
                            {
                                return
                                    tree[curr_lev].pct_tree_left
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                         row, NULL, tree[curr_lev].tree_left)
                                    + (1 - tree[curr_lev].pct_tree_left)
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                         row, NULL, tree[curr_lev].tree_right)
                                    + range_penalty;
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
                        curr_lev = (xval <=tree[curr_lev].num_split)?
                                    tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                        range_penalty += (xval < tree[curr_lev].range_low) || (xval > tree[curr_lev].range_high);
                    }
                    break;
                }

                case Categorical:
                {

                    if (prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows] < 0)
                    {
                        switch(model_outputs.missing_action)
                        {
                            case Divide:
                            {
                                return
                                    tree[curr_lev].pct_tree_left
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                         row, NULL, tree[curr_lev].tree_left)
                                    + (1 - tree[curr_lev].pct_tree_left)
                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                         row, NULL, tree[curr_lev].tree_right)
                                    + range_penalty;
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
                                curr_lev = (
                                            prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                ==
                                            tree[curr_lev].chosen_cat
                                            )?
                                            tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                break;
                            }

                            case SubSet:
                            {

                                if (!tree[curr_lev].cat_split.size())
                                {
                                    if (prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows] <= 1)
                                    {
                                        curr_lev = (
                                                    prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                        == 0
                                                    )?
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
                                                return
                                                    tree[curr_lev].pct_tree_left
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                                         row, NULL, tree[curr_lev].tree_left)
                                                    + (1 - tree[curr_lev].pct_tree_left)
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                                         row, NULL, tree[curr_lev].tree_right)
                                                    + range_penalty;
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
                                            curr_lev = (tree[curr_lev].cat_split[
                                                                    prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                                    ]
                                                        )?
                                                        tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                            break;
                                        }

                                        case Smallest:
                                        {
                                            if (
                                                prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                    >= (int)tree[curr_lev].cat_split.size()
                                                )
                                            {
                                                curr_lev =  (tree[curr_lev].pct_tree_left < .5)? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                            }

                                            else
                                            {
                                                curr_lev = (tree[curr_lev].cat_split[
                                                                        prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                                        ]
                                                            )?
                                                            tree[curr_lev].tree_left : tree[curr_lev].tree_right;
                                            }
                                            break;
                                        }

                                        case Weighted:
                                        {
                                            if (
                                                prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                    >= (int)tree[curr_lev].cat_split.size()
                                                ||
                                                tree[curr_lev].cat_split[
                                                            prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                            ]
                                                    == (-1)
                                                )
                                            {
                                                return
                                                    tree[curr_lev].pct_tree_left
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * tree[curr_lev].pct_tree_left,
                                                                         row, NULL, tree[curr_lev].tree_left)
                                                    + (1 - tree[curr_lev].pct_tree_left)
                                                        * traverse_itree(tree, model_outputs, prediction_data,
                                                                         impute_nodes, imputed_data, curr_weight * (1 - tree[curr_lev].pct_tree_left),
                                                                         row, NULL, tree[curr_lev].tree_right)
                                                    + range_penalty;
                                            }

                                            else
                                            {
                                                curr_lev = (tree[curr_lev].cat_split[
                                                                        prediction_data.categ_data[row +  tree[curr_lev].col_num * prediction_data.nrows]
                                                                        ]
                                                            )?
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
            }
        }
    }
}

/* this is a simpler version for situations in which there is
   only numeric data in dense arrays and no missing values */
void traverse_hplane_fast(std::vector<IsoHPlane>  &hplane,
                          ExtIsoForest            &model_outputs,
                          PredictionData          &prediction_data,
                          double                  &output_depth,
                          sparse_ix *restrict     tree_num,
                          size_t                  row)
{
    size_t  curr_lev = 0;
    double  hval;

    while(true)
    {
        if (hplane[curr_lev].score > 0)
        {
            output_depth += hplane[curr_lev].score;
            if (tree_num != NULL)
                tree_num[row] = curr_lev;
            return;
        }

        else
        {
            hval = 0;
            for (size_t col = 0; col < hplane[curr_lev].col_num.size(); col++)
                hval += (prediction_data.numeric_data[row +  hplane[curr_lev].col_num[col] * prediction_data.nrows] 
                         - hplane[curr_lev].mean[col]) * hplane[curr_lev].coef[col];
        }

        output_depth += (hval < hplane[curr_lev].range_low) ||
                        (hval > hplane[curr_lev].range_high);
        curr_lev      = (hval <= hplane[curr_lev].split_point)?
                         hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;
    }
}

/* this is the full version that works with potentially missing values, sparse matrices, and categoricals */
void traverse_hplane(std::vector<IsoHPlane>   &hplane,
                     ExtIsoForest             &model_outputs,
                     PredictionData           &prediction_data,
                     double                   &output_depth,
                     std::vector<ImputeNode> *impute_nodes,     /* only when imputing missing */
                     ImputedData             *imputed_data,     /* only when imputing missing */
                     sparse_ix *restrict      tree_num,
                     size_t                   row)
{
    size_t  curr_lev = 0;
    double  xval;
    int     cval;
    double  hval;

    size_t ncols_numeric, ncols_categ;

    sparse_ix *row_st = NULL, *row_end = NULL;
    if (prediction_data.Xr != NULL)
    {
        row_st  = prediction_data.Xr_ind + prediction_data.Xr_indptr[row];
        row_end = prediction_data.Xr_ind + prediction_data.Xr_indptr[row + 1];
    }

    while(true)
    {
        if (hplane[curr_lev].score > 0)
        {
            output_depth += hplane[curr_lev].score;
            if (tree_num != NULL)
                tree_num[row] = curr_lev;
            if (imputed_data != NULL)
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
                        if (prediction_data.Xc == NULL && prediction_data.Xr == NULL)
                            xval = prediction_data.numeric_data[row +  hplane[curr_lev].col_num[col] * prediction_data.nrows];
                        else if (row_st != NULL)
                            xval = extract_spR(prediction_data, row_st, row_end, hplane[curr_lev].col_num[col]);
                        else
                            xval = extract_spC(prediction_data, row, hplane[curr_lev].col_num[col]);

                        if (is_na_or_inf(xval))
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
                        cval = prediction_data.categ_data[row +  hplane[curr_lev].col_num[col] * prediction_data.nrows];
                        if (cval < 0)
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
                                    if (cval >= (int)hplane[curr_lev].cat_coef[ncols_categ].size())
                                        hval += hplane[curr_lev].fill_new[ncols_categ];
                                    else
                                        hval += hplane[curr_lev].cat_coef[ncols_categ][cval];
                                    break;
                                }
                            }
                        }

                        ncols_categ++;
                        break;
                    }
                }

            }

            output_depth += (hval < hplane[curr_lev].range_low) ||
                            (hval > hplane[curr_lev].range_high);
            curr_lev       = (hval <= hplane[curr_lev].split_point)?
                             hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;
        }
    }
}

double extract_spC(PredictionData &prediction_data, size_t row, size_t col_num)
{
    sparse_ix *search_res = std::lower_bound(prediction_data.Xc_ind + prediction_data.Xc_indptr[col_num],
                                             prediction_data.Xc_ind + prediction_data.Xc_indptr[col_num + 1],
                                             (sparse_ix) row);
    if (
        search_res == (prediction_data.Xc_ind + prediction_data.Xc_indptr[col_num + 1])
            ||
        *search_res != row
        )
        return 0;
    else
        return prediction_data.Xc[search_res - prediction_data.Xc_ind];
}

double extract_spR(PredictionData &prediction_data, sparse_ix *row_st, sparse_ix *row_end, size_t col_num)
{
    sparse_ix *search_res = std::lower_bound(row_st, row_end, (sparse_ix) col_num);
    if (search_res == row_end || *search_res != (sparse_ix)col_num)
        return 0;
    else
        return prediction_data.Xr[search_res - prediction_data.Xr_ind];
}

void get_num_nodes(IsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads)
{
    std::fill(n_terminal, n_terminal + model_outputs.trees.size(), 0);
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(model_outputs, n_nodes, n_terminal)
    for (size_t_for tree = 0; tree < model_outputs.trees.size(); tree++)
    {
        n_nodes[tree] = model_outputs.trees[tree].size();
        for (IsoTree &node : model_outputs.trees[tree])
        {
            n_terminal[tree] += (node.score > 0);
        }
    }
}

void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *restrict n_nodes, sparse_ix *restrict n_terminal, int nthreads)
{
    std::fill(n_terminal, n_terminal + model_outputs.hplanes.size(), 0);
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(model_outputs, n_nodes, n_terminal)
    for (size_t_for hplane = 0; hplane < model_outputs.hplanes.size(); hplane++)
    {
        n_nodes[hplane] = model_outputs.hplanes[hplane].size();
        for (IsoHPlane &node : model_outputs.hplanes[hplane])
        {
            n_terminal[hplane] += (node.score > 0);
        }
    }
}

