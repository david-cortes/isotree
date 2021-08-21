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

template <class sparse_ix__>
bool check_indices_are_sorted(sparse_ix__ indices[], size_t n)
{
    if (n <= 1)
        return true;
    if (indices[n-1] < indices[0])
        return false;
    for (size_t ix = 1; ix < n; ix++)
        if (indices[ix] < indices[ix-1])
            return false;
    return true;
}

template <class real_t__, class sparse_ix__>
void sort_csc_indices(real_t__ *restrict Xc, sparse_ix__ *restrict Xc_ind, sparse_ix__ *restrict Xc_indptr, size_t ncols_numeric)
{
    std::vector<double> buffer_sorted_vals;
    std::vector<sparse_ix__> buffer_sorted_ix;
    std::vector<size_t> argsorted;
    size_t n_this;
    size_t ix1, ix2;
    for (size_t col = 0; col < ncols_numeric; col++)
    {
        ix1 = Xc_indptr[col];
        ix2 = Xc_indptr[col+1];
        n_this = ix2 - ix1;
        if (n_this && !check_indices_are_sorted(Xc_ind + ix1, n_this))
        {
            if (buffer_sorted_vals.size() < n_this)
            {
                buffer_sorted_vals.resize(n_this);
                buffer_sorted_ix.resize(n_this);
                argsorted.resize(n_this);
            }
            std::iota(argsorted.begin(), argsorted.begin() + n_this, ix1);
            std::sort(argsorted.begin(), argsorted.begin() + n_this,
                      [&Xc_ind](const size_t a, const size_t b){return Xc_ind[a] < Xc_ind[b];});
            for (size_t ix = 0; ix < n_this; ix++)
                buffer_sorted_ix[ix] = Xc_ind[argsorted[ix]];
            std::copy(buffer_sorted_ix.begin(), buffer_sorted_ix.begin() + n_this, Xc_ind + ix1);
            for (size_t ix = 0; ix < n_this; ix++)
                buffer_sorted_vals[ix] = Xc[argsorted[ix]];
            std::copy(buffer_sorted_vals.begin(), buffer_sorted_vals.begin() + n_this, Xc + ix1);
        }
    }
}


template <class real_t__, class sparse_ix__>
void reconstruct_csr_sliced
(
    real_t__ *restrict orig_Xr, sparse_ix__ *restrict orig_Xr_indptr,
    real_t__ *restrict rec_Xr, sparse_ix__ *restrict rec_Xr_indptr,
    size_t nrows
)
{
    for (size_t row = 0; row < nrows; row++)
        std::copy(rec_Xr + rec_Xr_indptr[row],
                  rec_Xr + rec_Xr_indptr[row+(size_t)1],
                  orig_Xr + orig_Xr_indptr[row]);
}

#define is_in_set(vv, ss) ((ss).find((vv)) != (ss).end())

template <class real_t__, class sparse_ix__, class size_t_>
void reconstruct_csr_with_categ
(
    real_t__ *restrict orig_Xr, sparse_ix__ *restrict orig_Xr_ind, sparse_ix__ *restrict orig_Xr_indptr,
    real_t__ *restrict rec_Xr, sparse_ix__ *restrict rec_Xr_ind, sparse_ix__ *restrict rec_Xr_indptr,
    int *restrict rec_X_cat, bool is_col_major,
    size_t_ *restrict cols_numeric, size_t_ *restrict cols_categ,
    size_t nrows, size_t ncols, size_t ncols_numeric, size_t ncols_categ
)
{
    /* Check if the numeric columns go first and in the original order */
    bool num_is_seq = false;
    if (ncols_numeric > 0 && check_indices_are_sorted(cols_numeric, ncols_numeric)) {
        if (cols_numeric[0] == 0 && cols_numeric[ncols_numeric-1] == (size_t_)ncols_numeric-1)
            num_is_seq = true;
    }

    hashed_set<size_t> cols_numeric_set;
    hashed_set<size_t> cols_categ_set(cols_categ, cols_categ + ncols_categ);
    hashed_map<size_t, sparse_ix__> orig_to_rec_num;
    hashed_map<size_t, size_t> orig_to_rec_cat;

    sparse_ix__ col_orig;
    sparse_ix__ *restrict col_ptr;
    
    if (num_is_seq)
    {
        reconstruct_csr_sliced(
            orig_Xr, orig_Xr_indptr,
            rec_Xr, rec_Xr_indptr,
            nrows
        );
    }

    else
    {
        if (ncols_numeric)
            cols_numeric_set = hashed_set<size_t>(cols_numeric, cols_numeric + ncols_numeric);
        for (size_t col = 0; col < ncols_numeric; col++)
            orig_to_rec_num[cols_numeric[col]] = col;
    }

    for (size_t col = 0; col < ncols_categ; col++)
        orig_to_rec_cat[cols_categ[col]] = col;


    for (size_t row = 0; row < nrows; row++)
    {
        for (auto col = orig_Xr_indptr[row]; col < orig_Xr_indptr[row+1]; col++)
        {
            if (isnan(orig_Xr[col]))
            {
                col_orig = orig_Xr_ind[col];
                if (is_in_set(col_orig, cols_numeric_set)) {
                    col_ptr = std::lower_bound(rec_Xr_ind + rec_Xr_indptr[row],
                                               rec_Xr_ind + rec_Xr_indptr[row+1],
                                               col_orig);
                    orig_Xr[col] = rec_Xr[std::distance(rec_Xr_ind, col_ptr)];
                }

                else if (is_in_set((size_t)col_orig, cols_categ_set)) {
                    orig_Xr[col] = rec_X_cat[is_col_major?
                                             (row + nrows*orig_to_rec_cat[col_orig])
                                                :
                                             (orig_to_rec_cat[col_orig] + row*ncols_categ)];
                    #ifndef _FOR_R
                    orig_Xr[col] = (orig_Xr[col] < 0)? NAN : orig_Xr[col];
                    #else
                    orig_Xr[col] = (orig_Xr[col] < 0)? NA_REAL : orig_Xr[col];
                    #endif
                }
            }

            else if (orig_Xr[col] < 0)
            {
                col_orig = orig_Xr_ind[col];
                if (is_in_set((size_t)col_orig, cols_categ_set)) {
                    orig_Xr[col] = rec_X_cat[is_col_major?
                                             (row + nrows*orig_to_rec_cat[col_orig])
                                                :
                                             (orig_to_rec_cat[col_orig] + row*ncols_categ)];
                    #ifndef _FOR_R
                    orig_Xr[col] = (orig_Xr[col] < 0)? NAN : orig_Xr[col];
                    #else
                    orig_Xr[col] = (orig_Xr[col] < 0)? NA_REAL : orig_Xr[col];
                    #endif
                }
            }
        }
    }
}
