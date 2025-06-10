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

/* FIXME / TODO: here the calculations of medians do not take weights into account */

#define SD_MIN 1e-10
/* https://www.johndcook.com/blog/standard_deviation/ */

/* for regular numerical */
template <class real_t, class real_t_>
void calc_mean_and_sd_t(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x,
                        MissingAction missing_action, double &restrict x_sd, double &restrict x_mean)
{
    real_t m = 0;
    real_t s = 0;
    real_t m_prev = x[ix_arr[st]];
    real_t xval;

    if (missing_action == Fail)
    {
        m_prev = x[ix_arr[st]];
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            m += (xval - m) / (real_t)(row - st + 1);
            s  = std::fma(xval - m, xval - m_prev, s);
            m_prev = m;
        }

        x_mean = m;
        x_sd   = std::sqrt(s / (real_t)(end - st + 1));
    }

    else
    {
        size_t cnt = 0;
        while (is_na_or_inf(m_prev) && st <= end)
        {
            m_prev = x[ix_arr[++st]];
        }

        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            if (likely(!is_na_or_inf(xval)))
            {
                cnt++;
                m += (xval - m) / (real_t)cnt;
                s  = std::fma(xval - m, xval - m_prev, s);
                m_prev = m;
            }
        }

        x_mean = m;
        x_sd   = std::sqrt(s / (real_t)cnt);
    }
}

template <class real_t_>
double calc_mean_only(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x)
{
    size_t cnt = 0;
    double m = 0;
    real_t_ xval;
    for (size_t row = st; row <= end; row++)
    {
        xval = x[ix_arr[row]];
        if (likely(!is_na_or_inf(xval)))
        {
            cnt++;
            m += (xval - m) / (double)cnt;
        }
    }

    return m;
}

template <class real_t_, class ldouble_safe>
void calc_mean_and_sd(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x,
                      MissingAction missing_action, double &restrict x_sd, double &restrict x_mean)
{
    if (end - st + 1 < THRESHOLD_LONG_DOUBLE)
        calc_mean_and_sd_t<double, real_t_>(ix_arr, st, end, x, missing_action, x_sd, x_mean);
    else
        calc_mean_and_sd_t<ldouble_safe, real_t_>(ix_arr, st, end, x, missing_action, x_sd, x_mean);
    x_sd = std::fmax(x_sd, SD_MIN);
}

template <class real_t_, class mapping, class ldouble_safe>
void calc_mean_and_sd_weighted(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x, mapping &restrict w,
                               MissingAction missing_action, double &restrict x_sd, double &restrict x_mean)
{
    ldouble_safe cnt = 0;
    ldouble_safe w_this;
    ldouble_safe m = 0;
    ldouble_safe s = 0;
    ldouble_safe m_prev = x[ix_arr[st]];
    ldouble_safe xval;
    while (is_na_or_inf(m_prev) && st <= end)
    {
        m_prev = x[ix_arr[++st]];
    }

    for (size_t row = st; row <= end; row++)
    {
        xval = x[ix_arr[row]];
        if (likely(!is_na_or_inf(xval)))
        {
            w_this = w[ix_arr[row]];
            cnt += w_this;
            m = std::fma(w_this, (xval - m) / cnt, m);
            s = std::fma(w_this, (xval - m) * (xval - m_prev), s);
            m_prev = m;
        }
    }

    x_mean = m;
    x_sd   = std::sqrt((ldouble_safe)s / (ldouble_safe)cnt);
}

template <class real_t_, class mapping>
double calc_mean_only_weighted(size_t ix_arr[], size_t st, size_t end, real_t_ *restrict x, mapping &restrict w)
{
    double cnt = 0;
    double w_this;
    double m = 0;
    for (size_t row = st; row <= end; row++)
    {
        if (likely(!is_na_or_inf(x[ix_arr[row]])))
        {
            w_this = w[ix_arr[row]];
            cnt += w_this;
            m = std::fma(w_this, (x[ix_arr[row]] - m) / cnt, m);
        }
    }

    return m;
}

/* for sparse numerical */
template <class real_t_, class sparse_ix, class real_t>
void calc_mean_and_sd_(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                      real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                      double &restrict x_sd, double &restrict x_mean)
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
    size_t added = 0;
    real_t m = 0;
    real_t s = 0;
    real_t m_prev = 0;

    for (size_t *row = ptr_st;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == (sparse_ix)(*row))
        {
            if (unlikely(is_na_or_inf(Xc[curr_pos])))
            {
                cnt--;
            }

            else
            {
                if (added == 0) m_prev = Xc[curr_pos];
                m += (Xc[curr_pos] - m) / (real_t)(++added);
                s  = std::fma(Xc[curr_pos] - m, Xc[curr_pos] - m_prev, s);
                m_prev = m;
            }

            if (row == ix_arr + end || curr_pos == end_col) break;
            curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
        }

        else
        {
            if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
            else
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
        }
    }

    if (added == 0)
    {
        x_mean = 0;
        x_sd = 0;
        return;
    }

    /* Note: up to this point:
         m = sum(x)/nnz
         s = sum(x^2) - (1/nnz)*(sum(x)^2)
       Here the standard deviation is given by:
         sigma = (1/n)*(sum(x^2) - (1/n)*(sum(x)^2))
       The difference can be put to a closed form. */
    if (cnt > added)
    {
        s += square(m) * ((real_t)added * ((real_t)1 - (real_t)added/(real_t)cnt));
        m *= (real_t)added / (real_t)cnt;
    }

    x_mean = m;
    x_sd   = std::sqrt(s / (real_t)cnt);
}

template <class real_t_, class sparse_ix, class ldouble_safe>
void calc_mean_and_sd(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                      real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                      double &restrict x_sd, double &restrict x_mean)
{
    if (end - st + 1 < THRESHOLD_LONG_DOUBLE)
        calc_mean_and_sd_<real_t_, sparse_ix, double>(ix_arr, st, end, col_num, Xc, Xc_ind, Xc_indptr, x_sd, x_mean);
    else
        calc_mean_and_sd_<real_t_, sparse_ix, ldouble_safe>(ix_arr, st, end, col_num, Xc, Xc_ind, Xc_indptr, x_sd, x_mean);
    x_sd = std::fmax(SD_MIN, x_sd);
}

template <class real_t_, class sparse_ix, class ldouble_safe>
double calc_mean_only(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                      real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr)
{
    /* ix_arr must be already sorted beforehand */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return 0;
    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = (size_t) Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, (size_t)Xc_ind[st_col]);

    size_t cnt = end - st + 1;
    size_t added = 0;
    double m = 0;

    for (size_t *row = ptr_st;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == (sparse_ix)(*row))
        {
            if (unlikely(is_na_or_inf(Xc[curr_pos])))
                cnt--;
            else
                m += (Xc[curr_pos] - m) / (double)(++added);

            if (row == ix_arr + end || curr_pos == end_col) break;
            curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
        }

        else
        {
            if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
            else
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
        }
    }

    if (added == 0)
        return 0;

    if (cnt > added)
        m *= ((ldouble_safe)added / (ldouble_safe)cnt);

    return m;
}

template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
void calc_mean_and_sd_weighted(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                               real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                               double &restrict x_sd, double &restrict x_mean, mapping &restrict w)
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

    ldouble_safe cnt = 0.;
    for (size_t row = st; row <= end; row++)
        cnt += w[ix_arr[row]];
    ldouble_safe added = 0;
    ldouble_safe m = 0;
    ldouble_safe s = 0;
    ldouble_safe m_prev = 0;
    ldouble_safe w_this;

    for (size_t *row = ptr_st;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == (sparse_ix)(*row))
        {
            if (unlikely(is_na_or_inf(Xc[curr_pos])))
            {
                cnt -= w[*row];
            }

            else
            {
                w_this = w[*row];
                if (added == 0) m_prev = Xc[curr_pos];
                added += w_this;
                m = std::fma(w_this, (Xc[curr_pos] - m) / added, m);
                s = std::fma(w_this, (Xc[curr_pos] - m) * (Xc[curr_pos] - m_prev), s);
                m_prev = m;
            }

            if (row == ix_arr + end || curr_pos == end_col) break;
            curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
        }

        else
        {
            if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
            else
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
        }
    }

    if (added == 0)
    {
        x_mean = 0;
        x_sd = 0;
        return;
    }

    /* Note: up to this point:
         m = sum(x)/nnz
         s = sum(x^2) - (1/nnz)*(sum(x)^2)
       Here the standard deviation is given by:
         sigma = (1/n)*(sum(x^2) - (1/n)*(sum(x)^2))
       The difference can be put to a closed form. */
    if (cnt > added)
    {
        s += square(m) * (added * ((ldouble_safe)1 - (ldouble_safe)added/(ldouble_safe)cnt));
        m *= added / cnt;
    }

    x_mean = m;
    x_sd   = std::sqrt(s / (ldouble_safe)cnt);
}

template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
double calc_mean_only_weighted(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                               real_t_ *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                               mapping &restrict w)
{
    /* ix_arr must be already sorted beforehand */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return 0;
    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = (size_t) Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, (size_t)Xc_ind[st_col]);

    ldouble_safe cnt = 0.;
    for (size_t row = st; row <= end; row++)
        cnt += w[ix_arr[row]];
    ldouble_safe added = 0;
    ldouble_safe m = 0;
    ldouble_safe w_this;

    for (size_t *row = ptr_st;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == (sparse_ix)(*row))
        {
            if (unlikely(is_na_or_inf(Xc[curr_pos]))) {
                cnt -= w[*row];
            }

            else {
                w_this = w[*row];
                added += w_this;
                m += w_this * (Xc[curr_pos] - m) / added;
            }

            if (row == ix_arr + end || curr_pos == end_col) break;
            curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
        }

        else
        {
            if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
            else
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
        }
    }

    if (added == 0)
        return 0;

    if (cnt > added)
        m *= (ldouble_safe)added / (ldouble_safe)cnt;

    return m;
}

/* Note about these functions: they write into an array that does not need to match to 'ix_arr',
   and instead, the index that is stored in ix_arr[n] will have the value in res[n] */


/* for regular numerical */
template <class real_t_>
void add_linear_comb(const size_t ix_arr[], size_t st, size_t end, double *restrict res,
                     const real_t_ *restrict x, double &coef, double x_sd, double x_mean, double &restrict fill_val,
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
            res_write[row] = std::fma(x[ix_arr[row]] - x_mean, coef, res_write[row]);
    }

    else
    {
        if (first_run)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (likely(!is_na_or_inf(x[ix_arr[row]])))
                {
                    res_write[row]     =  std::fma(x[ix_arr[row]] - x_mean, coef, res_write[row]);
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
            fill_val = buffer_arr[mid_ceil-1] + (buffer_arr[mid_ceil] - buffer_arr[mid_ceil-1]) / 2.0;
        else
            fill_val = buffer_arr[mid_ceil];

        fill_val = (fill_val - x_mean) * coef;
        if (cnt_NA && fill_val)
        {
            for (size_t row = 0; row < cnt_NA; row++)
                res_write[buffer_NAs[row]] += fill_val;
        }

    }
}

/* for regular numerical */
template <class real_t_, class mapping, class ldouble_safe>
void add_linear_comb_weighted(const size_t ix_arr[], size_t st, size_t end, double *restrict res,
                              const real_t_ *restrict x, double &coef, double x_sd, double x_mean, double &restrict fill_val,
                              MissingAction missing_action, double *restrict buffer_arr,
                              size_t *restrict buffer_NAs, bool first_run, mapping &restrict w)
{
    /* TODO: here don't need the buffer for NAs */

    if (first_run)
        coef /= x_sd;

    size_t cnt = 0;
    size_t cnt_NA = 0;
    double *restrict res_write = res - st;
    ldouble_safe cumw = 0;
    double w_this;
    /* TODO: these buffers should be allocated externally */
    std::vector<double> obs_weight;

    if (first_run && missing_action != Fail)
    {
        obs_weight.resize(end - st + 1, 0.);
    }

    if (missing_action == Fail)
    {    
        for (size_t row = st; row <= end; row++)
            res_write[row] = std::fma(x[ix_arr[row]] - x_mean, coef, res_write[row]);
    }

    else
    {
        if (first_run)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (likely(!is_na_or_inf(x[ix_arr[row]])))
                {
                    w_this = w[ix_arr[row]];
                    res_write[row]     = std::fma(x[ix_arr[row]] - x_mean, coef, res_write[row]);
                    obs_weight[cnt]    = w_this;
                    buffer_arr[cnt++]  = x[ix_arr[row]];
                    cumw += w_this;
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


        ldouble_safe mid_point = cumw / (ldouble_safe)2;
        std::vector<size_t> sorted_ix(cnt);
        std::iota(sorted_ix.begin(), sorted_ix.end(), (size_t)0);
        std::sort(sorted_ix.begin(), sorted_ix.end(),
                  [&buffer_arr](const size_t a, const size_t b){return buffer_arr[a] < buffer_arr[b];});
        ldouble_safe currw = 0;
        fill_val = buffer_arr[sorted_ix.back()]; /* <- will overwrite later */
        /* TODO: is this median calculation correct? should it do a weighted interpolation? */
        for (size_t ix = 0; ix < cnt; ix++)
        {
            currw += obs_weight[sorted_ix[ix]];
            if (currw >= mid_point)
            {
                if (currw == mid_point && ix < cnt-1)
                    fill_val = buffer_arr[sorted_ix[ix]] + (buffer_arr[sorted_ix[ix+1]] - buffer_arr[sorted_ix[ix]]) / 2.0;
                else
                    fill_val = buffer_arr[sorted_ix[ix]];
                break;
            }
        }

        fill_val = (fill_val - x_mean) * coef;
        if (cnt_NA && fill_val)
        {
            for (size_t row = 0; row < cnt_NA; row++)
                res_write[buffer_NAs[row]] += fill_val;
        }

    }
}

/* for sparse numerical */
template <class real_t_, class sparse_ix>
void add_linear_comb(const size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, double *restrict res,
                     const real_t_ *restrict Xc, const sparse_ix *restrict Xc_ind, const sparse_ix *restrict Xc_indptr,
                     double &restrict coef, double x_sd, double x_mean, double &restrict fill_val, MissingAction missing_action,
                     double *restrict buffer_arr, size_t *restrict buffer_NAs, bool first_run)
{
    /* ix_arr must be already sorted beforehand */

    /* if it's all zeros, no need to do anything, but this is not supposed
       to happen while fitting because the range is determined before calling this */
    if (
            Xc_indptr[col_num] == Xc_indptr[col_num + 1] ||
            Xc_ind[Xc_indptr[col_num]] > (sparse_ix)ix_arr[end] ||
            Xc_ind[Xc_indptr[col_num + 1] - 1] < (sparse_ix)ix_arr[st]
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
        if (offset)
        {
            for (size_t row = st; row <= end; row++)
                res_write[row] -= offset;
        }

        return;
    }

    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    const size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, (size_t)Xc_ind[st_col]);

    size_t cnt_non_NA = 0; /* when NAs need to be imputed */
    size_t cnt_NA = 0; /* when NAs need to be imputed */
    size_t n_sample = end - st + 1;
    const size_t *ix_arr_plus_st = ix_arr + st;

    if (first_run)
        coef /= x_sd;

    double *restrict res_write = res - st;
    double offset = x_mean * coef;
    if (offset)
    {
        for (size_t row = st; row <= end; row++)
            res_write[row] -= offset;
    }

    size_t ind_end_col = Xc_ind[end_col];
    size_t nmatches = 0;

    if (missing_action != Fail)
    {
        if (first_run)
        {
            for (const size_t *row = ptr_st;
                 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (Xc_ind[curr_pos] == (sparse_ix)(*row))
                {
                    if (unlikely(is_na_or_inf(Xc[curr_pos])))
                    {
                        buffer_NAs[cnt_NA++] = row - ix_arr_plus_st;
                    }

                    else
                    {
                        buffer_arr[cnt_non_NA++]   = Xc[curr_pos];
                        res[row - ix_arr_plus_st]  = std::fma(Xc[curr_pos], coef, res[row - ix_arr_plus_st]);
                    }

                    nmatches++;
                    if (row == ix_arr + end || curr_pos == end_col) break;
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
                }

                else
                {
                    if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                        row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
                    else
                        curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
                }
            }
        }

        else
        {
            /* when impute value for missing has already been determined */
            for (const size_t *row = ptr_st;
                 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (Xc_ind[curr_pos] == (sparse_ix)(*row))
                {
                    res[row - ix_arr_plus_st] += is_na_or_inf(Xc[curr_pos])?
                                                  (fill_val + offset) : (Xc[curr_pos] * coef);
                    if (row == ix_arr + end) break;
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
                }

                else
                {
                    if (Xc_ind[curr_pos] > (sparse_ix)(*row))
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
        for (const size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == (sparse_ix)(*row))
            {
                res[row - ix_arr_plus_st] += Xc[curr_pos] * coef;
                if (row == ix_arr + end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
            }

            else
            {
                if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                    row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
                else
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
            }
        }
    }
}

template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
void add_linear_comb_weighted(const size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, double *restrict res,
                              const real_t_ *restrict Xc, const sparse_ix *restrict Xc_ind, const sparse_ix *restrict Xc_indptr,
                              double &restrict coef, double x_sd, double x_mean, double &restrict fill_val, MissingAction missing_action,
                              double *restrict buffer_arr, size_t *restrict buffer_NAs, bool first_run, mapping &restrict w)
{
    /* TODO: there's likely a better way of doing this directly with sparse inputs.
       Think about some way of doing it efficiently. */
    if (first_run && missing_action != Fail)
    {
        std::vector<double> denseX(end-st+1, 0.);
        todense(ix_arr, st, end,
                col_num, Xc, Xc_ind, Xc_indptr,
                denseX.data());
        std::vector<double> obs_weight(end-st+1);
        for (size_t row = st; row <= end; row++)
            obs_weight[row - st] = w[ix_arr[row]];

        size_t end_new = end - st + 1;
        for (size_t ix = 0; ix < end-st+1; ix++)
        {
            if (unlikely(is_na_or_inf(denseX[ix])))
            {
                std::swap(denseX[ix], denseX[--end_new]);
                std::swap(obs_weight[ix], obs_weight[end_new]);
            }
        }

        ldouble_safe cumw = std::accumulate(obs_weight.begin(), obs_weight.begin() + end_new, (ldouble_safe)0);
        ldouble_safe mid_point = cumw / (ldouble_safe)2;
        std::vector<size_t> sorted_ix(end_new);
        std::iota(sorted_ix.begin(), sorted_ix.end(), (size_t)0);
        std::sort(sorted_ix.begin(), sorted_ix.end(),
                  [&denseX](const size_t a, const size_t b){return denseX[a] < denseX[b];});
        ldouble_safe currw = 0;
        fill_val = denseX[sorted_ix.back()]; /* <- will overwrite later */
        /* TODO: is this median calculation correct? should it do a weighted interpolation? */
        for (size_t ix = 0; ix < end_new; ix++)
        {
            currw += obs_weight[sorted_ix[ix]];
            if (currw >= mid_point)
            {
                if (currw == mid_point && ix < end_new-1)
                    fill_val = denseX[sorted_ix[ix]] + (denseX[sorted_ix[ix+1]] - denseX[sorted_ix[ix]]) / 2.0;
                else
                    fill_val = denseX[sorted_ix[ix]];
                break;
            }
        }

        fill_val = (fill_val - x_mean) * (coef / x_sd);
        denseX.clear();
        obs_weight.clear();
        sorted_ix.clear();
        
        add_linear_comb(ix_arr, st, end, col_num, res,
                        Xc, Xc_ind, Xc_indptr,
                        coef, x_sd, x_mean, fill_val, missing_action,
                        buffer_arr, buffer_NAs, false);
    }

    else
    {
        add_linear_comb(ix_arr, st, end, col_num, res,
                        Xc, Xc_ind, Xc_indptr,
                        coef, x_sd, x_mean, fill_val, missing_action,
                        buffer_arr, buffer_NAs, first_run);
    }
}

/* for categoricals */
template <class ldouble_safe>
void add_linear_comb(const size_t *restrict ix_arr, size_t st, size_t end, double *restrict res,
                     const int x[], int ncat, double *restrict cat_coef, double single_cat_coef, int chosen_cat,
                     double &restrict fill_val, double &restrict fill_new, size_t *restrict buffer_cnt, size_t *restrict buffer_pos,
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
                            if (unlikely(x[ix_arr[row]] < 0))
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
                    if (cnt_NA && fill_val)
                    {
                        for (size_t row = st; row <= end; row++)
                            if (x[ix_arr[row]] < 0)
                                res_write[row] += fill_val;
                    }
                    return;
                }

                default:
                {
                    unexpected_error();
                    break;
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
                    int cat_smallest = 0;
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
                    ldouble_safe cnt_l = (ldouble_safe)((end - st + 1) - buffer_cnt[ncat]);
                    std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
                    std::sort(buffer_pos, buffer_pos + ncat, [&cat_coef](const size_t a, const size_t b){return cat_coef[a] < cat_coef[b];});

                    double cumprob = 0;
                    int cat;
                    for (cat = 0; cat < ncat; cat++)
                    {
                        cumprob += (ldouble_safe)buffer_cnt[buffer_pos[cat]] / cnt_l;
                        if (cumprob >= .5) break;
                    }
                    // cat = std::min(cat, ncat); /* in case it picks the last one */
                    fill_val = cat_coef[buffer_pos[cat]];
                    if (new_cat_action != Smallest)
                        fill_new = fill_val;

                    if (buffer_cnt[ncat] > 0 && fill_val) /* NAs */
                        for (size_t row = st; row <= end; row++)
                            if (unlikely(x[ix_arr[row]] < 0))
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

template <class mapping, class ldouble_safe>
void add_linear_comb_weighted(const size_t *restrict ix_arr, size_t st, size_t end, double *restrict res,
                              const int x[], int ncat, double *restrict cat_coef, double single_cat_coef, int chosen_cat,
                              double &restrict fill_val, double &restrict fill_new, size_t *restrict buffer_pos,
                              NewCategAction new_cat_action, MissingAction missing_action, CategSplit cat_split_type,
                              bool first_run, mapping &restrict w)
{
    double *restrict res_write = res - st;
    /* TODO: this buffer should be allocated externally */

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
                    bool has_NA = false;
                    ldouble_safe cnt_this = 0;
                    ldouble_safe cnt_other = 0;
                    if (first_run)
                    {
                        for (size_t row = st; row <= end; row++)
                        {
                            if (unlikely(x[ix_arr[row]] < 0))
                            {
                                has_NA = true;
                            }

                            else if (x[ix_arr[row]] == chosen_cat)
                            {
                                cnt_this += w[ix_arr[row]];
                                res_write[row] += single_cat_coef;
                            }

                            else
                            {
                                cnt_other += w[ix_arr[row]];
                            }
                        }
                    }

                    else
                    {
                        for (size_t row = st; row <= end; row++)
                            res_write[row] += (x[ix_arr[row]] < 0)? fill_val : ((x[ix_arr[row]] == chosen_cat)? single_cat_coef : 0);
                        return;
                    }

                    fill_val = (cnt_this > cnt_other)? single_cat_coef : 0;
                    if (has_NA && fill_val)
                    {
                        for (size_t row = st; row <= end; row++)
                            if (unlikely(x[ix_arr[row]] < 0))
                                res_write[row] += fill_val;
                    }
                    return;
                }

                default:
                {
                    unexpected_error();
                    break;
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

            /* TODO: this buffer should be allocated externally */
            std::vector<ldouble_safe> buffer_cnt(ncat+1, 0.);
            switch(missing_action)
            {
                case Fail:
                {
                    for (size_t row = st; row <= end; row++)
                    {
                        buffer_cnt[x[ix_arr[row]]] += w[ix_arr[row]];
                        res_write[row] += cat_coef[x[ix_arr[row]]];
                    }
                    break;
                }

                default:
                {
                    for (size_t row = st; row <= end; row++)
                    {
                        if (likely(x[ix_arr[row]] >= 0))
                        {
                            buffer_cnt[x[ix_arr[row]]] += w[ix_arr[row]];
                            res_write[row] += cat_coef[x[ix_arr[row]]];
                        }

                        else
                        {
                            buffer_cnt[ncat] += w[ix_arr[row]];
                        }

                    }
                    break;
                }
            }

            switch(new_cat_action)
            {
                case Smallest:
                {
                    ldouble_safe smallest = std::numeric_limits<ldouble_safe>::infinity();
                    int cat_smallest = 0;
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
                    ldouble_safe cnt_l = std::accumulate(buffer_cnt.begin(), buffer_cnt.begin() + ncat, (ldouble_safe)0);
                    std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
                    std::sort(buffer_pos, buffer_pos + ncat, [&cat_coef](const size_t a, const size_t b){return cat_coef[a] < cat_coef[b];});

                    double cumprob = 0;
                    int cat;
                    for (cat = 0; cat < ncat; cat++)
                    {
                        cumprob += buffer_cnt[buffer_pos[cat]] / cnt_l;
                        if (cumprob >= .5) break;
                    }
                    // cat = std::min(cat, ncat); /* in case it picks the last one */
                    fill_val = cat_coef[buffer_pos[cat]];
                    if (new_cat_action != Smallest)
                        fill_new = fill_val;

                    if (buffer_cnt[ncat] > 0 && fill_val) /* NAs */
                        for (size_t row = st; row <= end; row++)
                            if (unlikely(x[ix_arr[row]] < 0))
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
