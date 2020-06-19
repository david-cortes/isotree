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

/* for regular numerical */
void calc_mean_and_sd(size_t ix_arr[], size_t st, size_t end, double *restrict x,
                      MissingAction missing_action, double &x_sd, double &x_mean)
{
    long double m = 0;
    long double s = 0;
    long double m_prev = 0;

    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
        {
            m += (x[ix_arr[row]] - m) / (long double)(row - st + 1);
            s += (x[ix_arr[row]] - m) * (x[ix_arr[row]] - m_prev);
            m_prev = m;
        }

        x_mean = m;
        x_sd   = sqrtl(s / (long double)(end - st + 1));
    }

    else
    {
        size_t cnt = 0;
        for (size_t row = st; row <= end; row++)
        {
            if (!is_na_or_inf(x[ix_arr[row]]))
            {
                cnt++;
                m += (x[ix_arr[row]] - m) / (long double)cnt;
                s += (x[ix_arr[row]] - m) * (x[ix_arr[row]] - m_prev);
                m_prev = m;
            }
        }

        x_mean = m;
        x_sd   = sqrtl(s / (long double)cnt);
    }
}

/* for sparse numerical */
void calc_mean_and_sd(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                      double *restrict Xc, sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                      double &x_sd, double &x_mean)
{
    /* ix_arr must be already sorted beforehand */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
    {
        x_sd   = 0;
        x_mean = 0;
        return;
    }
    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = (size_t) Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, (size_t)Xc_ind[st_col]);

    size_t cnt = end - st + 1;
    long double sum = 0;
    long double sum_sq = 0;

    /* Note: this function will discard NAs regardless of chosen action. If reaching the point of calling
       this function, chances are that the performance gain of not checking for them will not be important */

    for (size_t *row = ptr_st;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == *row)
        {
            if (is_na_or_inf(Xc[curr_pos]))
            {
                cnt--;
            }

            else
            {
                sum    += Xc[curr_pos];
                sum_sq += square(Xc[curr_pos]);
            }

            if (row == ix_arr + end || curr_pos == end_col) break;
            curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
        }

        else
        {
            if (Xc_ind[curr_pos] > *row)
                row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
            else
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
        }
    }

    x_mean = sum / (long double) cnt;
    x_sd   = calc_sd_raw(cnt, sum, sum_sq);
}


/* Note about these functions: they write into an array that does not need to match to 'ix_arr',
   and instead, the index that is stored in ix_arr[n] will have the value in res[n] */

/* for regular numerical */
void add_linear_comb(size_t ix_arr[], size_t st, size_t end, double *restrict res,
                     double *restrict x, double &coef, double x_sd, double x_mean, double &fill_val,
                     MissingAction missing_action, double *restrict buffer_arr,
                     size_t *restrict buffer_NAs, bool first_run)
{
    /* TODO: here don't need the buffer for NAs */

    if (first_run)
        coef /= x_sd;

    size_t cnt = 0;
    size_t cnt_NA = 0;
    double *restrict res_write = res - st;

    if (missing_action == Fail)
    {    
        for (size_t row = st; row <= end; row++)
            res_write[row] += (x[ix_arr[row]] - x_mean) * coef;
    }

    else
    {
        if (first_run)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (!is_na_or_inf(x[ix_arr[row]]))
                {
                    res_write[row]    += (x[ix_arr[row]] - x_mean) * coef;
                    buffer_arr[cnt++]  =  x[ix_arr[row]];
                }

                else
                {
                    buffer_NAs[cnt_NA++] = row;
                }

            }
        }

        else
        {
            for (size_t row = st; row <= end; row++)
            {
                res_write[row] += (is_na_or_inf(x[ix_arr[row]]))? fill_val : ( (x[ix_arr[row]]-x_mean) * coef );
            }
            return;
        }
        
        size_t mid_ceil = cnt / 2;
        std::partial_sort(buffer_arr, buffer_arr + mid_ceil + 1, buffer_arr + cnt);

        if ((cnt % 2) == 0)
            fill_val = (buffer_arr[mid_ceil - 1] + buffer_arr[mid_ceil]) / 2.0;
        else
            fill_val = buffer_arr[mid_ceil];

        fill_val = (fill_val - x_mean) * coef;
        if (cnt_NA)
        {
            for (size_t row = 0; row < cnt_NA; row++)
                res_write[buffer_NAs[row]] += fill_val;
        }

    }
}

/* for sparse numerical */
void add_linear_comb(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, double *restrict res,
                     double *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     double &coef, double x_sd, double x_mean, double &fill_val, MissingAction missing_action,
                     double *restrict buffer_arr, size_t *restrict buffer_NAs, bool first_run)
{
    /* ix_arr must be already sorted beforehand */

    /* if it's all zeros, no need to do anything, but this is not supposed
       to happen while fitting because the range is determined before calling this */
    if (
            Xc_indptr[col_num] == Xc_indptr[col_num + 1] ||
            Xc_ind[Xc_indptr[col_num]] > ix_arr[end] ||
            Xc_ind[Xc_indptr[col_num + 1] - 1] < ix_arr[st]
        )
    {
        if (first_run)
        {
            coef /= x_sd;
            if (missing_action != Fail)
                fill_val = 0;
        }

        double *restrict res_write = res - st;
        double offset = x_mean * coef;
        for (size_t row = st; row <= end; row++)
            res_write[row] -= offset;

        return;
    }

    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, (size_t)Xc_ind[st_col]);

    size_t cnt_non_NA = 0; /* when NAs need to be imputed */
    size_t cnt_NA = 0; /* when NAs need to be imputed */
    size_t n_sample = end - st + 1;
    size_t *ix_arr_plus_st = ix_arr + st;

    if (first_run)
        coef /= x_sd;

    double *restrict res_write = res - st;
    double offset = x_mean * coef;
    for (size_t row = st; row <= end; row++)
        res_write[row] -= offset;

    size_t ind_end_col = Xc_ind[end_col];
    size_t nmatches = 0;

    if (missing_action != Fail)
    {
        if (first_run)
        {
            for (size_t *row = ptr_st;
                 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (Xc_ind[curr_pos] == *row)
                {
                    if (is_na_or_inf(Xc[curr_pos]))
                    {
                        buffer_NAs[cnt_NA++] = row - ix_arr_plus_st;
                    }

                    else
                    {
                        buffer_arr[cnt_non_NA++]   = Xc[curr_pos];
                        res[row - ix_arr_plus_st] += Xc[curr_pos] * coef;
                    }

                    nmatches++;
                    if (row == ix_arr + end || curr_pos == end_col) break;
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
                }

                else
                {
                    if (Xc_ind[curr_pos] > *row)
                        row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
                    else
                        curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
                }
            }
        }

        else
        {
            /* when impute value for missing has already been determined */
            for (size_t *row = ptr_st;
                 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (Xc_ind[curr_pos] == *row)
                {
                    res[row - ix_arr_plus_st] += is_na_or_inf(Xc[curr_pos])?
                                                  (fill_val + offset) : (Xc[curr_pos] * coef);
                    if (row == ix_arr + end) break;
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
                }

                else
                {
                    if (Xc_ind[curr_pos] > *row)
                        row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
                    else
                        curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
                }
            }

            return;
        }

        
        /* Determine imputation value */
        std::sort(buffer_arr, buffer_arr + cnt_non_NA);
        size_t mid_ceil = (n_sample - cnt_NA) / 2;
        size_t nzeros = (end - st + 1) - nmatches;
        if (nzeros > mid_ceil && buffer_arr[0] > 0)
        {
            fill_val = 0;
            return;
        }

        else
        {
            size_t n_neg = (buffer_arr[0] > 0)?
                            0 : ((buffer_arr[cnt_non_NA - 1] < 0)?
                                 cnt_non_NA : std::lower_bound(buffer_arr, buffer_arr + cnt_non_NA, (double)0) - buffer_arr);

            
            if (n_neg < (mid_ceil-1) && n_neg + nzeros > mid_ceil)
            {
                fill_val = 0;
                return;
            }

            else
            {
                /* if the sample size is odd, take the middle, otherwise take a simple average */
                if (((n_sample - cnt_NA) % 2) != 0)
                {
                    if (mid_ceil < n_neg)
                        fill_val = buffer_arr[mid_ceil];
                    else if (mid_ceil < n_neg + nzeros)
                        fill_val = 0;
                    else
                        fill_val = buffer_arr[mid_ceil - nzeros];
                }

                else
                {
                    if (mid_ceil < n_neg)
                    {
                        fill_val = (buffer_arr[mid_ceil - 1] + buffer_arr[mid_ceil]) / 2;
                    }

                    else if (mid_ceil < n_neg + nzeros)
                    {
                        if (mid_ceil == n_neg)
                            fill_val = buffer_arr[mid_ceil - 1] / 2;
                        else
                            fill_val = 0;
                    }

                    else
                    {
                        if (mid_ceil == n_neg + nzeros && nzeros > 0)
                            fill_val = buffer_arr[n_neg] / 2;
                        else
                            fill_val = (buffer_arr[mid_ceil - nzeros - 1] + buffer_arr[mid_ceil - nzeros]) / 2; /* WRONG!!!! */
                    }
                }

                /* fill missing if any */
                fill_val *= coef;
                if (cnt_NA && fill_val)
                    for (size_t ix = 0; ix < cnt_NA; ix++)
                        res[buffer_NAs[ix]] += fill_val; 

                /* next time, it will need to have the offset added */
                fill_val -= offset;
            }
        }
    }

    else /* no NAs */
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == *row)
            {
                res[row - ix_arr_plus_st] += Xc[curr_pos] * coef;
                if (row == ix_arr + end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
            }

            else
            {
                if (Xc_ind[curr_pos] > *row)
                    row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
                else
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
            }
        }
    }
}

/* for categoricals */
void add_linear_comb(size_t *restrict ix_arr, size_t st, size_t end, double *restrict res,
                     int x[], int ncat, double *restrict cat_coef, double single_cat_coef, int chosen_cat,
                     double &fill_val, double &fill_new, size_t *restrict buffer_cnt, size_t *restrict buffer_pos,
                     NewCategAction new_cat_action, MissingAction missing_action, CategSplit cat_split_type, bool first_run)
{
    double *restrict res_write = res - st;
    switch(cat_split_type)
    {
        case SingleCateg:
        {
            /* in this case there's no need to make-up an impute value for new categories, only for NAs */
            switch(missing_action)
            {
                case Fail:
                {
                    for (size_t row = st; row <= end; row++)
                        res_write[row] += (x[ix_arr[row]] == chosen_cat)? single_cat_coef : 0;
                    return;
                }

                case Impute:
                {
                    size_t cnt_NA = 0;
                    size_t cnt_this = 0;
                    size_t cnt = end - st + 1;
                    if (first_run)
                    {
                        for (size_t row = st; row <= end; row++)
                        {
                            if (x[ix_arr[row]] < 0)
                            {
                                cnt_NA++;
                            }

                            else if (x[ix_arr[row]] == chosen_cat)
                            {
                                cnt_this++;
                                res_write[row] += single_cat_coef;
                            }
                        }
                    }

                    else
                    {
                        for (size_t row = st; row <= end; row++)
                            res_write[row] += (x[ix_arr[row]] < 0)? fill_val : ((x[ix_arr[row]] == chosen_cat)? single_cat_coef : 0);
                        return;
                    }

                    fill_val = (cnt_this > (cnt - cnt_NA - cnt_this))? single_cat_coef : 0;
                    if (cnt_NA)
                    {
                        for (size_t row = st; row <= end; row++)
                            if (x[ix_arr[row]] < 0)
                                res_write[row] += fill_val;
                    }
                    return;
                }
            }
        }

        case SubSet:
        {
            /* in this case, since the splits are by more than 1 variable, it's not possible to
               divide missing/new categoricals by assigning weights, so they have to be imputed
               in both cases, unless using random weights for the new ones, in which case they won't
               need to be imputed for new, but sill need it for NA */

            if (new_cat_action == Random && missing_action == Fail)
            {
                for (size_t row = st; row <= end; row++)
                    res_write[row] += cat_coef[x[ix_arr[row]]];
                return;
            }

            if (!first_run)
            {
                if (missing_action == Fail)
                {
                    for (size_t row = st; row <= end; row++)
                        res_write[row] += (x[ix_arr[row]] >= ncat)? fill_new : cat_coef[x[ix_arr[row]]];
                }

                else
                {
                    for (size_t row = st; row <= end; row++)
                        res_write[row] += (x[ix_arr[row]] < 0)? fill_val : ((x[ix_arr[row]] >= ncat)? fill_new : cat_coef[x[ix_arr[row]]]);
                }
                return;
            }

            std::fill(buffer_cnt, buffer_cnt + ncat + 1, 0);
            switch(missing_action)
            {
                case Fail:
                {
                    for (size_t row = st; row <= end; row++)
                    {
                        buffer_cnt[x[ix_arr[row]]]++;
                        res_write[row] += cat_coef[x[ix_arr[row]]];
                    }
                    break;
                }

                default:
                {
                    for (size_t row = st; row <= end; row++)
                    {
                        if (x[ix_arr[row]] >= 0)
                        {
                            buffer_cnt[x[ix_arr[row]]]++;
                            res_write[row] += cat_coef[x[ix_arr[row]]];
                        }

                        else
                        {
                            buffer_cnt[ncat]++;
                        }

                    }
                    break;
                }
            }

            switch(new_cat_action)
            {
                case Smallest:
                {
                    size_t smallest = SIZE_MAX;
                    int cat_smallest;
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[cat] > 0 && buffer_cnt[cat] < smallest)
                        {
                            smallest = buffer_cnt[cat];
                            cat_smallest = cat;
                        }
                    }
                    fill_new = cat_coef[cat_smallest];
                    if (missing_action == Fail) break;
                }

                default:
                {
                    /* Determine imputation value as the category in sorted order that gives 50% + 1 */
                    long double cnt_l = (long double)((end - st + 1) - buffer_cnt[ncat]);
                    std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
                    std::sort(buffer_pos, buffer_pos + ncat, [&cat_coef](const size_t a, const size_t b){return cat_coef[a] < cat_coef[b];});

                    double cumprob = 0;
                    int cat;
                    for (cat = 0; cat < ncat; cat++)
                    {
                        cumprob += (long double)buffer_cnt[buffer_pos[cat]] / cnt_l;
                        if (cumprob >= .5) break;
                    }
                    // cat = std::min(cat, ncat); /* in case it picks the last one */
                    fill_val = cat_coef[buffer_pos[cat]];
                    if (new_cat_action != Smallest)
                        fill_new = fill_val;

                    if (buffer_cnt[ncat] > 0) /* NAs */
                        for (size_t row = st; row <= end; row++)
                            if (x[ix_arr[row]] < 0)
                                res_write[row] += fill_val;
                }
            }

            /* now fill unseen categories */
            if (new_cat_action != Random)
                for (int cat = 0; cat < ncat; cat++)
                    if (!buffer_cnt[cat])
                        cat_coef[cat] = fill_new;

        }
    }
} 
