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

/* TODO: should the kurtosis calculation impute values when using ndim=1 + missing_action=Impute?
   It should be the theoretically correct approach, but will cause the kurtosis to increase
   significantly if there is a large number of missing values, which would lead to prefer
   splitting on columns with mostly missing values. */

/* TODO: this kurtosis caps the minimum values to zero, but the minimum achievable value is 1,
   see how are imprecise results used outside of the function in the different kind of calculations
   that use kurtosis and maybe change the logic. */

#define pw1(x) ((x))
#define pw2(x) ((x) * (x))
#define pw3(x) ((x) * (x) * (x))
#define pw4(x) ((x) * (x) * (x) * (x))

/* https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Higher-order_statistics */
template <class real_t, class ldouble_safe>
double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, real_t x[], MissingAction missing_action)
{
    ldouble_safe m = 0;
    ldouble_safe M2 = 0, M3 = 0, M4 = 0;
    ldouble_safe delta, delta_s, delta_div;
    ldouble_safe diff, n;
    ldouble_safe out;

    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
        {
            n  =  (ldouble_safe)(row - st + 1);

            delta      =  x[ix_arr[row]] - m;
            delta_div  =  delta / n;
            delta_s    =  delta_div * delta_div;
            diff       =  delta * (delta_div * (ldouble_safe)(row - st));

            m   +=  delta_div;
            M4  +=  diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
            M3  +=  diff * delta_div * (n - 2) - 3 * delta_div * M2;
            M2  +=  diff;
        }

        if (unlikely(!is_na_or_inf(M2) && M2 <= 0))
        {
            if (!check_more_than_two_unique_values(ix_arr, st, end, x, missing_action))
                return -HUGE_VAL;
        }

        out = ( M4 / M2 ) * ( (ldouble_safe)(end - st + 1) / M2 );
        return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
    }

    else
    {
        size_t cnt = 0;
        for (size_t row = st; row <= end; row++)
        {
            if (likely(!is_na_or_inf(x[ix_arr[row]])))
            {
                cnt++;
                n = (ldouble_safe) cnt;

                delta      =  x[ix_arr[row]] - m;
                delta_div  =  delta / n;
                delta_s    =  delta_div * delta_div;
                diff       =  delta * (delta_div * (ldouble_safe)(cnt - 1));

                m   +=  delta_div;
                M4  +=  diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
                M3  +=  diff * delta_div * (n - 2) - 3 * delta_div * M2;
                M2  +=  diff;
            }
        }

        if (unlikely(cnt == 0)) return -HUGE_VAL;
        if (unlikely(!is_na_or_inf(M2) && M2 <= 0))
        {
            if (!check_more_than_two_unique_values(ix_arr, st, end, x, missing_action))
                return -HUGE_VAL;
        }

        out = ( M4 / M2 ) * ( (ldouble_safe)cnt / M2 );
        return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
    }
}

template <class real_t, class ldouble_safe>
double calc_kurtosis(real_t x[], size_t n, MissingAction missing_action)
{
    ldouble_safe m = 0;
    ldouble_safe M2 = 0, M3 = 0, M4 = 0;
    ldouble_safe delta, delta_s, delta_div;
    ldouble_safe diff, n_;
    ldouble_safe out;

    if (missing_action == Fail)
    {
        for (size_t row = 0; row < n; row++)
        {
            n_  =  (ldouble_safe)(row + 1);

            delta      =  x[row] - m;
            delta_div  =  delta / n_;
            delta_s    =  delta_div * delta_div;
            diff       =  delta * (delta_div * (ldouble_safe)row);

            m   +=  delta_div;
            M4  +=  diff * delta_s * (n_ * n_ - 3 * n_ + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
            M3  +=  diff * delta_div * (n_ - 2) - 3 * delta_div * M2;
            M2  +=  diff;
        }

        out = ( M4 / M2 ) * ( (ldouble_safe)n / M2 );
        return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
    }

    else
    {
        size_t cnt = 0;
        for (size_t row = 0; row < n; row++)
        {
            if (likely(!is_na_or_inf(x[row])))
            {
                cnt++;
                n_ = (ldouble_safe) cnt;

                delta      =  x[row] - m;
                delta_div  =  delta / n_;
                delta_s    =  delta_div * delta_div;
                diff       =  delta * (delta_div * (ldouble_safe)(cnt - 1));

                m   +=  delta_div;
                M4  +=  diff * delta_s * (n_ * n_ - 3 * n_ + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
                M3  +=  diff * delta_div * (n_ - 2) - 3 * delta_div * M2;
                M2  +=  diff;
            }
        }

        if (unlikely(cnt == 0)) return -HUGE_VAL;

        out = ( M4 / M2 ) * ( (ldouble_safe)cnt / M2 );
        return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
    }
}

/* TODO: is this algorithm correct? */
template <class real_t, class mapping, class ldouble_safe>
double calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, real_t x[],
                              MissingAction missing_action, mapping &restrict w)
{
    ldouble_safe m = 0;
    ldouble_safe M2 = 0, M3 = 0, M4 = 0;
    ldouble_safe delta, delta_s, delta_div;
    ldouble_safe diff;
    ldouble_safe n = 0;
    ldouble_safe out;
    ldouble_safe n_prev = 0.;
    ldouble_safe w_this;

    for (size_t row = st; row <= end; row++)
    {
        if (likely(!is_na_or_inf(x[ix_arr[row]])))
        {
            w_this = w[ix_arr[row]];
            n += w_this;

            delta      =  x[ix_arr[row]] - m;
            delta_div  =  delta / n;
            delta_s    =  delta_div * delta_div;
            diff       =  delta * (delta_div * n_prev);
            n_prev     =  n;

            m   +=  w_this * (delta_div);
            M4  +=  w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
            M3  +=  w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
            M2  +=  w_this * (diff);
        }
    }

    if (unlikely(n <= 0)) return -HUGE_VAL;
    if (unlikely(!is_na_or_inf(M2) && M2 <= std::numeric_limits<double>::epsilon()))
    {
        if (!check_more_than_two_unique_values(ix_arr, st, end, x, missing_action))
            return -HUGE_VAL;
    }

    out = ( M4 / M2 ) * ( n / M2 );
    return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
}

template <class real_t, class ldouble_safe>
double calc_kurtosis_weighted(real_t *restrict x, size_t n_, MissingAction missing_action, real_t *restrict w)
{
    ldouble_safe m = 0;
    ldouble_safe M2 = 0, M3 = 0, M4 = 0;
    ldouble_safe delta, delta_s, delta_div;
    ldouble_safe diff;
    ldouble_safe n = 0;
    ldouble_safe out;
    ldouble_safe n_prev = 0.;
    ldouble_safe w_this;

    for (size_t row = 0; row < n_; row++)
    {
        if (likely(!is_na_or_inf(x[row])))
        {
            w_this = w[row];
            n += w_this;

            delta      =  x[row] - m;
            delta_div  =  delta / n;
            delta_s    =  delta_div * delta_div;
            diff       =  delta * (delta_div * n_prev);
            n_prev     =  n;

            m   +=  w_this * (delta_div);
            M4  +=  w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
            M3  +=  w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
            M2  +=  w_this * (diff);
        }
    }

    if (unlikely(n <= 0)) return -HUGE_VAL;

    out = ( M4 / M2 ) * ( n / M2 );
    return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
}


/* TODO: make these compensated sums */
/* TODO: can this use the same algorithm as above but with a correction at the end,
   like it was done for the variance? */
template <class real_t, class sparse_ix, class ldouble_safe>
double calc_kurtosis(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                     real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     MissingAction missing_action)
{
    /* ix_arr must be already sorted beforehand */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return -HUGE_VAL;

    ldouble_safe s1 = 0;
    ldouble_safe s2 = 0;
    ldouble_safe s3 = 0;
    ldouble_safe s4 = 0;
    ldouble_safe x_sq;
    size_t cnt = end - st + 1;

    if (unlikely(cnt <= 1)) return -HUGE_VAL;
    
    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);

    ldouble_safe xval;

    if (missing_action != Fail)
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == (sparse_ix)(*row))
            {
                xval = Xc[curr_pos];
                if (unlikely(is_na_or_inf(xval)))
                {
                    cnt--;
                }

                else
                {
                    /* TODO: is it safe to use FMA here? some calculations rely on assuming that
                       some of these 's' are larger than the others. Would this procedure be guaranteed
                       to preserve such differences if done with a mixture of sums and FMAs? */
                    x_sq = square(xval);
                    s1 += xval;
                    s2  = std::fma(xval, xval, s2);
                    s3  = std::fma(x_sq, xval, s3);
                    s4  = std::fma(x_sq, x_sq, s4);
                    // s1 += pw1(xval);
                    // s2 += pw2(xval);
                    // s3 += pw3(xval);
                    // s4 += pw4(xval);
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

        if (unlikely(cnt <= (end - st + 1) - (Xc_indptr[col_num+1] - Xc_indptr[col_num]))) return -HUGE_VAL;
    }

    else
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == (sparse_ix)(*row))
            {
                xval = Xc[curr_pos];
                x_sq = square(xval);
                s1 += xval;
                s2  = std::fma(xval, xval, s2);
                s3  = std::fma(x_sq, xval, s3);
                s4  = std::fma(x_sq, x_sq, s4);
                // s1 += pw1(xval);
                // s2 += pw2(xval);
                // s3 += pw3(xval);
                // s4 += pw4(xval);

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

    if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1))) return -HUGE_VAL;
    ldouble_safe cnt_l = (ldouble_safe) cnt;
    ldouble_safe sn = s1 / cnt_l;
    ldouble_safe v  = s2 / cnt_l - pw2(sn);
    if (unlikely(std::isnan(v))) return -HUGE_VAL;
    if (
        v <= std::numeric_limits<double>::epsilon() &&
        !check_more_than_two_unique_values(ix_arr, st, end, col_num,
                                           Xc_indptr, Xc_ind, Xc,
                                           missing_action)
    )
        return -HUGE_VAL;
    if (unlikely(v <= 0)) return 0.;
    ldouble_safe out =  (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
    return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
}

template <class real_t, class sparse_ix, class ldouble_safe>
double calc_kurtosis(size_t col_num, size_t nrows,
                     real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                     MissingAction missing_action)
{
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return -HUGE_VAL;

    ldouble_safe s1 = 0;
    ldouble_safe s2 = 0;
    ldouble_safe s3 = 0;
    ldouble_safe s4 = 0;
    ldouble_safe x_sq;
    size_t cnt = nrows;

    if (unlikely(cnt <= 1)) return -HUGE_VAL;

    ldouble_safe xval;

    if (missing_action != Fail)
    {
        for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num+1]; ix++)
        {
            xval = Xc[ix];
            if (unlikely(is_na_or_inf(xval)))
            {
                cnt--;
            }

            else
            {
                x_sq = square(xval);
                s1 += xval;
                s2  = std::fma(xval, xval, s2);
                s3  = std::fma(x_sq, xval, s3);
                s4  = std::fma(x_sq, x_sq, s4);
                // s1 += pw1(xval);
                // s2 += pw2(xval);
                // s3 += pw3(xval);
                // s4 += pw4(xval);
            }
        }

        if (cnt <= (nrows) - (Xc_indptr[col_num+1] - Xc_indptr[col_num])) return -HUGE_VAL;
    }

    else
    {
        for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num+1]; ix++)
        {
            xval = Xc[ix];
            x_sq = square(xval);
            s1 += xval;
            s2  = std::fma(xval, xval, s2);
            s3  = std::fma(x_sq, xval, s3);
            s4  = std::fma(x_sq, x_sq, s4);
            // s1 += pw1(xval);
            // s2 += pw2(xval);
            // s3 += pw3(xval);
            // s4 += pw4(xval);
        }
    }

    if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1))) return -HUGE_VAL;
    ldouble_safe cnt_l = (ldouble_safe) cnt;
    ldouble_safe sn = s1 / cnt_l;
    ldouble_safe v  = s2 / cnt_l - pw2(sn);
    if (unlikely(std::isnan(v))) return -HUGE_VAL;
    if (
        v <= std::numeric_limits<double>::epsilon() &&
        !check_more_than_two_unique_values(nrows, col_num,
                                           Xc_indptr, Xc_ind, Xc,
                                           missing_action)
    )
        return -HUGE_VAL;
    if (unlikely(v <= 0)) return 0.;
    ldouble_safe out =  (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
    return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
}


template <class real_t, class sparse_ix, class mapping, class ldouble_safe>
double calc_kurtosis_weighted(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                              real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                              MissingAction missing_action, mapping &restrict w)
{
    /* ix_arr must be already sorted beforehand */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return -HUGE_VAL;

    ldouble_safe s1 = 0;
    ldouble_safe s2 = 0;
    ldouble_safe s3 = 0;
    ldouble_safe s4 = 0;
    ldouble_safe x_sq;
    ldouble_safe w_this;
    ldouble_safe cnt = 0;
    for (size_t row = st; row <= end; row++)
        cnt += w[ix_arr[row]];

    if (unlikely(cnt <= 0)) return -HUGE_VAL;
    
    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);

    ldouble_safe xval;

    if (missing_action != Fail)
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == (sparse_ix)(*row))
            {
                w_this = w[*row];
                xval = Xc[curr_pos];

                if (unlikely(is_na_or_inf(xval)))
                {
                    cnt -= w_this;
                }

                else
                {
                    x_sq = xval * xval;
                    s1 = std::fma(w_this, xval, s1);
                    s2 = std::fma(w_this, x_sq, s2);
                    s3 = std::fma(w_this, x_sq*xval, s3);
                    s4 = std::fma(w_this, x_sq*x_sq, s4);
                    // s1 += w_this * pw1(xval);
                    // s2 += w_this * pw2(xval);
                    // s3 += w_this * pw3(xval);
                    // s4 += w_this * pw4(xval);
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

        if (unlikely(cnt <= 0)) return -HUGE_VAL;
    }

    else
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == (sparse_ix)(*row))
            {
                w_this = w[*row];
                xval = Xc[curr_pos];
                
                x_sq = xval * xval;
                s1 = std::fma(w_this, xval, s1);
                s2 = std::fma(w_this, x_sq, s2);
                s3 = std::fma(w_this, x_sq*xval, s3);
                s4 = std::fma(w_this, x_sq*x_sq, s4);
                // s1 += w_this * pw1(xval);
                // s2 += w_this * pw2(xval);
                // s3 += w_this * pw3(xval);
                // s4 += w_this * pw4(xval);

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

    if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1))) return -HUGE_VAL;
    ldouble_safe sn = s1 / cnt;
    ldouble_safe v  = s2 / cnt - pw2(sn);
    if (unlikely(std::isnan(v))) return -HUGE_VAL;
    if (
        v <= std::numeric_limits<double>::epsilon() &&
        !check_more_than_two_unique_values(ix_arr, st, end, col_num,
                                           Xc_indptr, Xc_ind, Xc,
                                           missing_action)
    )
        return -HUGE_VAL;
    if (v <= 0) return 0.;
    ldouble_safe out =  (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt * pw4(sn)) / (cnt * pw2(v));
    return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
}

template <class real_t, class sparse_ix, class ldouble_safe>
double calc_kurtosis_weighted(size_t col_num, size_t nrows,
                              real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                              MissingAction missing_action, real_t *restrict w)
{
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return -HUGE_VAL;

    ldouble_safe s1 = 0;
    ldouble_safe s2 = 0;
    ldouble_safe s3 = 0;
    ldouble_safe s4 = 0;
    ldouble_safe x_sq;
    ldouble_safe w_this;
    ldouble_safe cnt = nrows - (Xc_indptr[col_num + 1] - Xc_indptr[col_num]);
    for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
        cnt += w[Xc_ind[ix]];

    if (unlikely(cnt <= 0)) return -HUGE_VAL;
    
    ldouble_safe xval;

    if (missing_action != Fail)
    {
        for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
        {
            w_this = w[Xc_ind[ix]];
            xval = Xc[ix];

            if (unlikely(is_na_or_inf(xval)))
            {
                cnt -= w_this;
            }

            else
            {
                x_sq = xval * xval;
                s1 = std::fma(w_this, xval, s1);
                s2 = std::fma(w_this, x_sq, s2);
                s3 = std::fma(w_this, x_sq*xval, s3);
                s4 = std::fma(w_this, x_sq*x_sq, s4);
                // s1 += w_this * pw1(xval);
                // s2 += w_this * pw2(xval);
                // s3 += w_this * pw3(xval);
                // s4 += w_this * pw4(xval);
            }
        }

        if (cnt <= 0) return -HUGE_VAL;
    }

    else
    {
        for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
        {
            w_this = w[Xc_ind[ix]];
            xval = Xc[ix];

            x_sq = xval * xval;
            s1 = std::fma(w_this, xval, s1);
            s2 = std::fma(w_this, x_sq, s2);
            s3 = std::fma(w_this, x_sq*xval, s3);
            s4 = std::fma(w_this, x_sq*x_sq, s4);
            // s1 += w_this * pw1(xval);
            // s2 += w_this * pw2(xval);
            // s3 += w_this * pw3(xval);
            // s4 += w_this * pw4(xval);
        }
    }

    if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1))) return -HUGE_VAL;
    ldouble_safe sn = s1 / cnt;
    ldouble_safe v  = s2 / cnt - pw2(sn);
    if (unlikely(std::isnan(v))) return -HUGE_VAL;
    if (
        v <= std::numeric_limits<double>::epsilon() &&
        !check_more_than_two_unique_values(nrows, col_num,
                                           Xc_indptr, Xc_ind, Xc,
                                           missing_action)
    )
        return -HUGE_VAL;
    if (unlikely(v <= 0)) return -HUGE_VAL;
    ldouble_safe out =  (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt * pw4(sn)) / (cnt * pw2(v));
    return (!is_na_or_inf(out))? std::fmax((double)out, 0.) : (-HUGE_VAL);
}


template <class ldouble_safe>
double calc_kurtosis_internal(size_t cnt, int x[], int ncat, size_t buffer_cnt[], double buffer_prob[],
                              MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator)
{
    /* This calculation proceeds as follows:
        - If splitting by subsets, it will assign a random weight ~Unif(0,1) to
          each category, and approximate kurtosis by sampling from such distribution
          with the same probabilities as given by the current counts.
        - If splitting by isolating one category, will binarize at each categorical level,
          assume the values are zero or one, and output the average assuming each categorical
          level has equal probability of being picked.
        (Note that both are misleading heuristics, but might be better than random)
    */
    double sum_kurt = 0;

    cnt -= buffer_cnt[ncat];
    if (cnt <= 1) return -HUGE_VAL;
    ldouble_safe cnt_l = (ldouble_safe) cnt;
    for (int cat = 0; cat < ncat; cat++)
        buffer_prob[cat] = buffer_cnt[cat] / cnt_l;

    switch (cat_split_type)
    {
        case SubSet:
        {
            ldouble_safe temp_v;
            ldouble_safe s1, s2, s3, s4;
            ldouble_safe coef;
            ldouble_safe coef2;
            ldouble_safe w_this;
            UniformUnitInterval runif(0, 1);
            size_t ntry = 50;
            for (size_t iternum = 0; iternum < 50; iternum++)
            {
                s1 = 0; s2 = 0; s3 = 0; s4 = 0;
                for (int cat = 0; cat < ncat; cat++)
                {
                    coef = runif(rnd_generator);
                    coef2 = coef * coef;
                    w_this = buffer_prob[cat];
                    s1 = std::fma(w_this, coef, s1);
                    s2 = std::fma(w_this, coef2, s2);
                    s3 = std::fma(w_this, coef2*coef, s3);
                    s4 = std::fma(w_this, coef2*coef2, s4);
                    // s1 += buffer_prob[cat] * pw1(coef);
                    // s2 += buffer_prob[cat] * pw2(coef);
                    // s3 += buffer_prob[cat] * pw3(coef);
                    // s4 += buffer_prob[cat] * pw4(coef);
                }
                temp_v = s2 - pw2(s1);
                if (temp_v <= 0)
                    ntry--;
                else
                    sum_kurt += (s4 - 4 * s3 * pw1(s1) + 6 * s2 * pw2(s1) - 4 * s1 * pw3(s1) + pw4(s1)) / pw2(temp_v);
            }
            if (unlikely(!ntry))
                return -HUGE_VAL;
            else if (unlikely(is_na_or_inf(sum_kurt)))
                return -HUGE_VAL;
            else
                return std::fmax(sum_kurt, 0.) / (double)ntry;
        }

        case SingleCateg:
        {
            double p;
            int ncat_present = ncat;
            for (int cat = 0; cat < ncat; cat++)
            {
                p = buffer_prob[cat];
                if (p == 0)
                    ncat_present--;
                else
                    sum_kurt += (p - 4 * p * pw1(p) + 6 * p * pw2(p) - 4 * p * pw3(p) + pw4(p)) / pw2(p - pw2(p));
            }
            if (ncat_present <= 1)
                return -HUGE_VAL;
            else if (unlikely(is_na_or_inf(sum_kurt)))
                return -HUGE_VAL;
            else
                return std::fmax(sum_kurt, 0.) / (double)ncat_present;
        }
    }

    return -1; /* this will never be reached, but CRAN complains otherwise */
}

template <class ldouble_safe>
double calc_kurtosis(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat, size_t *restrict buffer_cnt, double buffer_prob[],
                     MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator)
{
    /* This calculation proceeds as follows:
        - If splitting by subsets, it will assign a random weight ~Unif(0,1) to
          each category, and approximate kurtosis by sampling from such distribution
          with the same probabilities as given by the current counts.
        - If splitting by isolating one category, will binarize at each categorical level,
          assume the values are zero or one, and output the average assuming each categorical
          level has equal probability of being picked.
        (Note that both are misleading heuristics, but might be better than random)
    */
    size_t cnt = end - st + 1;
    std::fill(buffer_cnt, buffer_cnt + ncat + 1, (size_t)0);

    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
            buffer_cnt[x[ix_arr[row]]]++;
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (likely(x[ix_arr[row]] >= 0))
                buffer_cnt[x[ix_arr[row]]]++;
            else
                buffer_cnt[ncat]++;
        }
    }

    return calc_kurtosis_internal<ldouble_safe>(
                                  cnt, x, ncat, buffer_cnt, buffer_prob,
                                  missing_action, cat_split_type, rnd_generator);
}

template <class ldouble_safe>
double calc_kurtosis(size_t nrows, int x[], int ncat, size_t buffer_cnt[], double buffer_prob[],
                     MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator)
{
    size_t cnt = nrows;
    std::fill(buffer_cnt, buffer_cnt + ncat + 1, (size_t)0);

    if (missing_action == Fail)
    {
        for (size_t row = 0; row < nrows; row++)
            buffer_cnt[x[row]]++;
    }

    else
    {
        for (size_t row = 0; row < nrows; row++)
        {
            if (likely(x[row] >= 0))
                buffer_cnt[x[row]]++;
            else
                buffer_cnt[ncat]++;
        }
    }

    return calc_kurtosis_internal<ldouble_safe>(
                                  cnt, x, ncat, buffer_cnt, buffer_prob,
                                  missing_action, cat_split_type, rnd_generator);
}


/* TODO: this one should get a buffer preallocated from outside */
template <class mapping, class ldouble_safe>
double calc_kurtosis_weighted_internal(std::vector<ldouble_safe> &buffer_cnt, int x[], int ncat,
                                       double buffer_prob[], MissingAction missing_action, CategSplit cat_split_type,
                                       RNG_engine &rnd_generator, mapping &restrict w)
{
    double sum_kurt = 0;

    ldouble_safe cnt = std::accumulate(buffer_cnt.begin(), buffer_cnt.end(), (ldouble_safe)0);

    cnt -= buffer_cnt[ncat];
    if (unlikely(cnt <= 1)) return -HUGE_VAL;
    for (int cat = 0; cat < ncat; cat++)
        buffer_prob[cat] = buffer_cnt[cat] / cnt;

    switch (cat_split_type)
    {
        case SubSet:
        {
            ldouble_safe temp_v;
            ldouble_safe s1, s2, s3, s4;
            ldouble_safe coef, coef2;
            ldouble_safe w_this;
            UniformUnitInterval runif(0, 1);
            size_t ntry = 50;
            for (size_t iternum = 0; iternum < 50; iternum++)
            {
                s1 = 0; s2 = 0; s3 = 0; s4 = 0;
                for (int cat = 0; cat < ncat; cat++)
                {
                    coef = runif(rnd_generator);
                    coef2 = coef * coef;
                    w_this = buffer_prob[cat];
                    s1 = std::fma(w_this, coef, s1);
                    s2 = std::fma(w_this, coef2, s2);
                    s3 = std::fma(w_this, coef2*coef, s3);
                    s4 = std::fma(w_this, coef2*coef2, s4);
                    // s1 += buffer_prob[cat] * pw1(coef);
                    // s2 += buffer_prob[cat] * pw2(coef);
                    // s3 += buffer_prob[cat] * pw3(coef);
                    // s4 += buffer_prob[cat] * pw4(coef);
                }
                temp_v = s2 - pw2(s1);
                if (unlikely(temp_v <= 0))
                    ntry--;
                else
                    sum_kurt += (s4 - 4 * s3 * pw1(s1) + 6 * s2 * pw2(s1) - 4 * s1 * pw3(s1) + pw4(s1)) / pw2(temp_v);
            }
            if (unlikely(!ntry))
                return -HUGE_VAL;
            else if (unlikely(is_na_or_inf(sum_kurt)))
                return -HUGE_VAL;
            else
                return std::fmax(sum_kurt, 0.) / (double)ntry;
        }

        case SingleCateg:
        {
            double p;
            int ncat_present = ncat;
            for (int cat = 0; cat < ncat; cat++)
            {
                p = buffer_prob[cat];
                if (p == 0)
                    ncat_present--;
                else
                    sum_kurt += (p - 4 * p * pw1(p) + 6 * p * pw2(p) - 4 * p * pw3(p) + pw4(p)) / pw2(p - pw2(p));
            }
            if (ncat_present <= 1)
                return -HUGE_VAL;
            else if (unlikely(is_na_or_inf(sum_kurt)))
                return -HUGE_VAL;
            else
                return std::fmax(sum_kurt, 0.) / (double)ncat_present;
        }
    }

    return -1; /* this will never be reached, but CRAN complains otherwise */
}

template <class mapping, class ldouble_safe>
double calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, int x[], int ncat, double buffer_prob[],
                              MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator,
                              mapping &restrict w)
{
    std::vector<ldouble_safe> buffer_cnt(ncat+1, 0.);
    ldouble_safe w_this;

    for (size_t row = st; row <= end; row++)
    {
        w_this = w[ix_arr[row]];
        if (likely(x[ix_arr[row]] >= 0))
            buffer_cnt[x[ix_arr[row]]] += w_this;
        else
            buffer_cnt[ncat] += w_this;
    }
    
    return calc_kurtosis_weighted_internal<mapping, ldouble_safe>(
                                           buffer_cnt, x, ncat,
                                           buffer_prob, missing_action, cat_split_type,
                                           rnd_generator, w);
}

template <class real_t, class ldouble_safe>
double calc_kurtosis_weighted(size_t nrows, int x[], int ncat, double *restrict buffer_prob,
                              MissingAction missing_action, CategSplit cat_split_type,
                              RNG_engine &rnd_generator, real_t *restrict w)
{
    std::vector<ldouble_safe> buffer_cnt(ncat+1, 0.);
    ldouble_safe w_this;

    for (size_t row = 0; row < nrows; row++)
    {
        w_this = w[row];
        if (likely(x[row] >= 0))
            buffer_cnt[x[row]] += w_this;
        else
            buffer_cnt[ncat] += w_this;
    }
    
    return calc_kurtosis_weighted_internal<real_t *restrict, ldouble_safe>(
                                           buffer_cnt, x, ncat,
                                           buffer_prob, missing_action, cat_split_type,
                                           rnd_generator, w);
}

template <class int_t, class ldouble_safe>
double expected_sd_cat(double p[], size_t n, int_t pos[])
{
    if (n <= 1) return 0;

    ldouble_safe cum_var = -square(p[pos[0]]) / 3.0 - p[pos[0]] * p[pos[1]] / 2.0 + p[pos[0]] / 3.0  - square(p[pos[1]]) / 3.0 + p[pos[1]] / 3.0;
    for (size_t cat1 = 2; cat1 < n; cat1++)
    {
        cum_var += p[pos[cat1]] / 3.0 - square(p[pos[cat1]]) / 3.0;
        for (size_t cat2 = 0; cat2 < cat1; cat2++)
            cum_var -= p[pos[cat1]] * p[pos[cat2]] / 2.0;
    }
    return std::sqrt(std::fmax(cum_var, (ldouble_safe)0));
}

template <class number, class int_t, class ldouble_safe>
double expected_sd_cat(number *restrict counts, double *restrict p, size_t n, int_t *restrict pos)
{
    if (n <= 1) return 0;

    number tot = std::accumulate(pos, pos + n, (number)0, [&counts](number tot, const size_t ix){return tot + counts[ix];});
    ldouble_safe cnt_div = (ldouble_safe) tot;
    for (size_t cat = 0; cat < n; cat++)
        p[pos[cat]] = (ldouble_safe)counts[pos[cat]] / cnt_div;

    return expected_sd_cat<int_t, ldouble_safe>(p, n, pos);
}

template <class number, class int_t, class ldouble_safe>
double expected_sd_cat_single(number *restrict counts, double *restrict p, size_t n, int_t *restrict pos, size_t cat_exclude, number cnt)
{
    if (cat_exclude == 0)
        return expected_sd_cat<number, int_t, ldouble_safe>(counts, p, n-1, pos + 1);

    else if (cat_exclude == (n-1))
        return expected_sd_cat<number, int_t, ldouble_safe>(counts, p, n-1, pos);

    size_t ix_exclude = pos[cat_exclude];

    ldouble_safe cnt_div = (ldouble_safe) (cnt - counts[ix_exclude]);
    for (size_t cat = 0; cat < n; cat++)
        p[pos[cat]] = (ldouble_safe)counts[pos[cat]] / cnt_div;

    ldouble_safe cum_var;
    if (cat_exclude != 1)
        cum_var = -square(p[pos[0]]) / 3.0 - p[pos[0]] * p[pos[1]] / 2.0 + p[pos[0]] / 3.0  - square(p[pos[1]]) / 3.0 + p[pos[1]] / 3.0;
    else
        cum_var = -square(p[pos[0]]) / 3.0 - p[pos[0]] * p[pos[2]] / 2.0 + p[pos[0]] / 3.0  - square(p[pos[2]]) / 3.0 + p[pos[2]] / 3.0;
    for (size_t cat1 = (cat_exclude == 1)? 3 : 2; cat1 < n; cat1++)
    {
        if (pos[cat1] == ix_exclude) continue;
        cum_var += p[pos[cat1]] / 3.0 - square(p[pos[cat1]]) / 3.0;
        for (size_t cat2 = 0; cat2 < cat1; cat2++)
        {
            if (pos[cat2] == ix_exclude) continue;
            cum_var -= p[pos[cat1]] * p[pos[cat2]] / 2.0;
        }

    }
    return std::sqrt(std::fmax(cum_var, (ldouble_safe)0));
}

template <class number, class int_t, class ldouble_safe>
double expected_sd_cat_internal(int ncat, number *restrict buffer_cnt, ldouble_safe cnt_l,
                                int_t *restrict buffer_pos, double *restrict buffer_prob)
{
    /* move zero-valued to the beginning */
    std::iota(buffer_pos, buffer_pos + ncat, (int_t)0);
    int_t st_pos = 0;
    int ncat_present = 0;
    int_t temp;
    for (int cat = 0; cat < ncat; cat++)
    {
        if (buffer_cnt[cat])
        {
            ncat_present++;
            buffer_prob[cat] = (ldouble_safe) buffer_cnt[cat] / cnt_l;
        }

        else
        {
            temp = buffer_pos[st_pos];
            buffer_pos[st_pos] = buffer_pos[cat];
            buffer_pos[cat] = temp;
            st_pos++;
        }
    }

    if (ncat_present <= 1) return 0;
    return expected_sd_cat<int_t, ldouble_safe>(buffer_prob, ncat_present, buffer_pos + st_pos);
}


template <class int_t, class ldouble_safe>
double expected_sd_cat(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat,
                       MissingAction missing_action,
                       size_t *restrict buffer_cnt, int_t *restrict buffer_pos, double buffer_prob[])
{
    /* generate counts */
    std::fill(buffer_cnt, buffer_cnt + ncat + 1, (size_t)0);
    size_t cnt = end - st + 1;

    if (missing_action != Fail)
    {
        int xval;
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            if (unlikely(xval < 0))
                buffer_cnt[ncat]++;
            else
                buffer_cnt[xval]++;
        }
        cnt -= buffer_cnt[ncat];
        if (cnt == 0) return 0;
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (likely(x[ix_arr[row]] >= 0)) buffer_cnt[x[ix_arr[row]]]++;
        }
    }

    return expected_sd_cat_internal<size_t, int_t, ldouble_safe>(ncat, buffer_cnt, cnt, buffer_pos, buffer_prob);
}

template <class mapping, class int_t, class ldouble_safe>
double expected_sd_cat_weighted(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat,
                                MissingAction missing_action, mapping &restrict w,
                                double *restrict buffer_cnt, int_t *restrict buffer_pos, double *restrict buffer_prob)
{
    /* generate counts */
    std::fill(buffer_cnt, buffer_cnt + ncat + 1, 0.);
    ldouble_safe cnt = 0;

    if (missing_action != Fail)
    {
        int xval;
        double w_this;
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            w_this = w[ix_arr[row]];

            if (unlikely(xval < 0)) {
                buffer_cnt[ncat] += w_this;
            }
            else {
                buffer_cnt[xval] += w_this;
                cnt += w_this;
            }
        }
        if (cnt == 0) return 0;
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (likely(x[ix_arr[row]] >= 0))
            {
                buffer_cnt[x[ix_arr[row]]] += w[ix_arr[row]];
            }
        }
        for (int cat = 0; cat < ncat; cat++)
            cnt += buffer_cnt[cat];
        if (unlikely(cnt == 0)) return 0;
    }

    return expected_sd_cat_internal<double, int_t, ldouble_safe>(ncat, buffer_cnt, cnt, buffer_pos, buffer_prob);
}

/* Note: this isn't exactly comparable to the pooled gain from numeric variables,
   but among all the possible options, this is what happens to end up in the most
   similar scale when considering standardized gain. */
template <class number, class ldouble_safe>
double categ_gain(number cnt_left, number cnt_right,
                  ldouble_safe s_left, ldouble_safe s_right,
                  ldouble_safe base_info, ldouble_safe cnt)
{
    return (
            base_info -
            (((cnt_left  <= 1)? 0 : ((ldouble_safe)cnt_left  * std::log((ldouble_safe)cnt_left)))  - s_left) -
            (((cnt_right <= 1)? 0 : ((ldouble_safe)cnt_right * std::log((ldouble_safe)cnt_right))) - s_right)
            ) / cnt;
}


/*  A couple notes about gain calculation:

    Here one wants to find the best split point, maximizing either:
        (1/sigma) * (sigma - (1/n)*(n_left*sigma_left + n_right*sigma_right))
    or:
        (1/sigma) * (sigma - (1/2)*(sigma_left + sigma_right))
    
    All the algorithms here use the sorted-indices approach, which is
    an exact method (note that there's still room for optimization by adding the
    unsorted approach for small sample sizes and for sparse matrices).

    A naive approach would move observations one at a time from right
    to left using this formula:
        sigma = (ssq - s^2/n) / n
        ssq = sum(x^2)
        s = sum(x)
    But such approach has poor numerical precision, and this library is
    aimed precisely at cases in which there are outliers in the data.
    It's possible to improve the numerical precision by standardizing the
    data beforehand, but this library uses instead a more exact two-pass
    sigma calculation observation-by-observation (from left to right and
    from right to left, keeping the calculations of the first pass in an
    array and calculating gain in the second pass), but there's
    other methods too.

    If one is aiming at maximizing the pooled gain, it's possible to
    simplify either the gain or the increase in gain without involving
    'ssq'. Assuming one already has 'ssq' and 's' calculated for the left and
    right partitions, and one wants to move one ovservation from right to left,
    the following hold:
        s_right = s - s_left
        ssq_right = ssq - ssq_left
        n_right = n - n_left
    If one then moves observation x, these are updated as follows:
        s_left_new = s_left + x
        s_right_new = s - s_left - x
        ssq_left_new = ssq_left + x^2
        ssq_right_new = ssq - ssq_left - x^2
        n_left_new = n_left + 1
        n_right_new = n - n_left - 1
    Gain is then:
        (1/sigma) * (sigma - (1/n)*({ssq_left_new - s_left_new^2/n_left_new} + {ssq_right_new - s_right_new^2/n_right_new}))
    Which simplifies to:
        1 - (1/(sigma*n))(ssq - ( (s_left + x)^2/(n_left+1)  +  (s - (s_left + x))^2/(n - (n_left+1)) ))
    Since 'sigma', n', and 'ssq' are constant, they can be ignored when determining the
    maximum gain - that is, one is interest in finding the point that maximizes:
        (s_left+x)^2/(n_left+1) + (s-(s_left+x))^2/(n-(n_left+1))
    And this calculation will be robust-enough when dealing with numbers that were
    already standardized beforehand, as the extended model does at each step.
    Note however that, when fitting this model, one is usually interested in evaluating
    the actual gain, standardized by the standard deviation, as it will try different
    linear combinations which will give different standard deviations, so this simpler
    formula cannot be applied unless only one linear combination is probed.
        
    One can also look at:
        diff_gain = (1/sigma) * (gain_new - gain)
    Which can be simplified to something that doesn't include sums of squares:
        (1/(sigma*n))*(  -s_left^2/n_left  -  (s-s_left)^2/(n-n_left)  +  (s_left+x)^2/(n_left+1)  +  (s-(s_left+x))^2/(n-(n_left+1))  )
    And this calculation would in theory allow getting the actual standardized gain.
    In practice however, this calculation can have poor numerical precision when the
    sample size is large, so the functions here do not even attempt at calculating it,
    and this is the reason why the two-pass approach is preferred.

    The averaged SD formula unfortunately doesn't reduce to something that would involve
    only sums.
*/

/*  TODO: maybe it's not a good idea to use the two-pass approach with un-standardized
    variables at large sample sizes (ndim=1), considering that they come in sorted order.
    Maybe it should instead use sums of centered squares: sigma = sqrt((x-mean(x))^2/n)
    The sums of centered squares method is also likely to be more precise. */

template <class real_t>
double midpoint(real_t x, real_t y)
{
    real_t m = x + (y-x)/(real_t)2;
    if (likely((double)m < (double)y))
        return m;
    else {
        m = std::nextafter(m, y);
        if (m > x && m < y)
            return m;
        else
            return x;
    }
}

template <class real_t>
double midpoint_with_reorder(real_t x, real_t y)
{
    if (x < y)
        return midpoint(x, y);
    else
        return midpoint(y, x);
}

#define sd_gain(sd, sd_left, sd_right) (1. - ((sd_left) + (sd_right)) / (2. * (sd)))
#define pooled_gain(sd, cnt, sd_left, sd_right, cnt_left, cnt_right) \
    (1. - (1./(sd))*(  ( ((real_t)(cnt_left))/(cnt) )*(sd_left) + ( ((real_t)(cnt_right)/(cnt)) )*(sd_right)  ))


/* TODO: make this a compensated sum */
template <class real_t, class real_t_>
double find_split_rel_gain_t(real_t_ *restrict x, size_t n, double &restrict split_point)
{
    real_t this_gain;
    real_t best_gain = -HUGE_VAL;
    real_t x1 = 0, x2 = 0;
    real_t sum_left = 0, sum_right = 0, sum_tot = 0;
    for (size_t row = 0; row < n; row++)
        sum_tot += x[row];
    for (size_t row = 0; row < n-1; row++)
    {
        sum_left += x[row];
        if (x[row] == x[row+1])
            continue;

        sum_right = sum_tot - sum_left;
        this_gain =   sum_left  * (sum_left  / (real_t)(row+1))
                    + sum_right * (sum_right / (real_t)(n-row-1));
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            x1 = x[row]; x2 = x[row+1];
        }
    }
    
    if (best_gain <= -HUGE_VAL)
        return best_gain;
    split_point = midpoint(x1, x2);
    return std::fmax((double)best_gain, std::numeric_limits<double>::epsilon());
}

template <class real_t_, class ldouble_safe>
double find_split_rel_gain(real_t_ *restrict x, size_t n, double &restrict split_point)
{
    if (n < THRESHOLD_LONG_DOUBLE)
        return find_split_rel_gain_t<double, real_t_>(x, n, split_point);
    else
        return find_split_rel_gain_t<ldouble_safe, real_t_>(x, n, split_point);
}

/* Note: there is no 'weighted' version of 'find_split_rel_gain' with unindexed 'x', because calling it would
   imply having to argsort the 'x' values in order to sort the weights, which is less efficient. */

template <class real_t, class real_t_>
double find_split_rel_gain_t(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &split_point, size_t &restrict split_ix)
{
    real_t this_gain;
    real_t best_gain = -HUGE_VAL;
    split_ix = 0; /* <- avoid out-of-bounds at the end */
    real_t sum_left = 0, sum_right = 0, sum_tot = 0;
    for (size_t row = st; row <= end; row++)
        sum_tot += x[ix_arr[row]] - xmean;
    for (size_t row = st; row < end; row++)
    {
        sum_left += x[ix_arr[row]] - xmean;
        if (x[ix_arr[row]] == x[ix_arr[row+1]])
            continue;

        sum_right = sum_tot - sum_left;
        this_gain =   sum_left  * (sum_left  / (real_t)(row - st + 1))
                    + sum_right * (sum_right / (real_t)(end - row));
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            split_ix = row;
        }
    }

    if (best_gain <= -HUGE_VAL)
        return best_gain;
    split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
    return std::fmax((double)best_gain, std::numeric_limits<double>::epsilon());
}

template <class real_t_, class ldouble_safe>
double find_split_rel_gain(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &restrict split_point, size_t &restrict split_ix)
{
    if ((end-st+1) < THRESHOLD_LONG_DOUBLE)
        return find_split_rel_gain_t<double, real_t_>(x, xmean, ix_arr, st, end, split_point, split_ix);
    else
        return find_split_rel_gain_t<ldouble_safe, real_t_>(x, xmean, ix_arr, st, end, split_point, split_ix);
}

template <class real_t, class real_t_, class mapping>
double find_split_rel_gain_weighted_t(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &split_point, size_t &restrict split_ix, mapping &restrict w)
{
    real_t this_gain;
    real_t best_gain = -HUGE_VAL;
    split_ix = 0; /* <- avoid out-of-bounds at the end */
    real_t sum_left = 0, sum_right = 0, sum_tot = 0, sumw = 0, sumw_tot = 0;

    for (size_t row = st; row <= end; row++)
        sumw_tot += w[ix_arr[row]];
    for (size_t row = st; row <= end; row++)
        sum_tot += x[ix_arr[row]] - xmean;
    for (size_t row = st; row < end; row++)
    {
        sumw += w[ix_arr[row]];
        sum_left += x[ix_arr[row]] - xmean;
        if (x[ix_arr[row]] == x[ix_arr[row+1]])
            continue;

        sum_right = sum_tot - sum_left;
        this_gain =   sum_left  * (sum_left  / sumw)
                    + sum_right * (sum_right / (sumw_tot - sumw));
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            split_ix = row;
        }
    }

    if (best_gain <= -HUGE_VAL)
        return best_gain;
    split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
    return std::fmax((double)best_gain, std::numeric_limits<double>::epsilon());
}

template <class real_t_, class mapping, class ldouble_safe>
double find_split_rel_gain_weighted(real_t_ *restrict x, real_t_ xmean, size_t *restrict ix_arr, size_t st, size_t end, double &restrict split_point, size_t &restrict split_ix, mapping &restrict w)
{
    if ((end-st+1) < THRESHOLD_LONG_DOUBLE)
        return find_split_rel_gain_weighted_t<double, real_t_, mapping>(x, xmean, ix_arr, st, end, split_point, split_ix, w);
    else
        return find_split_rel_gain_weighted_t<ldouble_safe, real_t_, mapping>(x, xmean, ix_arr, st, end, split_point, split_ix, w);
}


template <class real_t, class real_t_>
real_t calc_sd_right_to_left(real_t_ *restrict x, size_t n, double *restrict sd_arr)
{
    real_t running_mean = 0;
    real_t running_ssq = 0;
    real_t mean_prev = x[n-1];
    for (size_t row = 0; row < n-1; row++)
    {
        running_mean   += (x[n-row-1] - running_mean) / (real_t)(row+1);
        running_ssq    += (x[n-row-1] - running_mean) * (x[n-row-1] - mean_prev);
        mean_prev       =  running_mean;
        sd_arr[n-row-1] = (row == 0)? 0. : std::sqrt(running_ssq / (real_t)(row+1));
    }
    running_mean   += (x[0] - running_mean) / (real_t)n;
    running_ssq    += (x[0] - running_mean) * (x[0] - mean_prev);
    return std::sqrt(running_ssq / (real_t)n);
}

template <class real_t_, class ldouble_safe>
ldouble_safe calc_sd_right_to_left_weighted(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                                           double *restrict w, ldouble_safe &cumw, size_t *restrict sorted_ix)
{
    ldouble_safe running_mean = 0;
    ldouble_safe running_ssq = 0;
    ldouble_safe mean_prev = x[sorted_ix[n-1]];
    ldouble_safe cnt = 0;
    double w_this;
    for (size_t row = 0; row < n-1; row++)
    {
        w_this = w[sorted_ix[n-row-1]];
        cnt += w_this;
        running_mean   += w_this * (x[sorted_ix[n-row-1]] - running_mean) / cnt;
        running_ssq    += w_this * ((x[sorted_ix[n-row-1]] - running_mean) * (x[sorted_ix[n-row-1]] - mean_prev));
        mean_prev       =  running_mean;
        sd_arr[n-row-1] = (row == 0)? 0. : std::sqrt(running_ssq / cnt);
    }
    w_this = w[sorted_ix[0]];
    cnt += w_this;
    running_mean   += (x[sorted_ix[0]] - running_mean) / cnt;
    running_ssq    += w_this * ((x[sorted_ix[0]] - running_mean) * (x[sorted_ix[0]] - mean_prev));
    cumw = cnt;
    return std::sqrt(running_ssq / cnt);
}

template <class real_t, class real_t_>
real_t calc_sd_right_to_left(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr)
{
    real_t running_mean = 0;
    real_t running_ssq = 0;
    real_t mean_prev = x[ix_arr[end]] - xmean;
    size_t n = end - st + 1;
    for (size_t row = 0; row < n-1; row++)
    {
        running_mean   += ((x[ix_arr[end-row]] - xmean) - running_mean) / (real_t)(row+1);
        running_ssq    += ((x[ix_arr[end-row]] - xmean) - running_mean) * ((x[ix_arr[end-row]] - xmean) - mean_prev);
        mean_prev       =  running_mean;
        sd_arr[n-row-1] = (row == 0)? 0. : std::sqrt(running_ssq / (real_t)(row+1));
    }
    running_mean   += ((x[ix_arr[st]] - xmean) - running_mean) / (real_t)n;
    running_ssq    += ((x[ix_arr[st]] - xmean) - running_mean) * ((x[ix_arr[st]] - xmean) - mean_prev);
    return std::sqrt(running_ssq / (real_t)n);
}

template <class real_t_, class mapping, class ldouble_safe>
ldouble_safe calc_sd_right_to_left_weighted(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end,
                                           double *restrict sd_arr, mapping &restrict w, ldouble_safe &cumw)
{
    ldouble_safe running_mean = 0;
    ldouble_safe running_ssq = 0;
    real_t_ mean_prev = x[ix_arr[end]] - xmean;
    size_t n = end - st + 1;
    ldouble_safe cnt = 0;
    double w_this;
    for (size_t row = 0; row < n-1; row++)
    {
        w_this = w[ix_arr[end-row]];
        cnt += w_this;
        running_mean   += w_this * ((x[ix_arr[end-row]] - xmean) - running_mean) / cnt;
        running_ssq    += w_this * (((x[ix_arr[end-row]] - xmean) - running_mean) * ((x[ix_arr[end-row]] - xmean) - mean_prev));
        mean_prev       =  running_mean;
        sd_arr[n-row-1] = (row == 0)? 0. : std::sqrt(running_ssq / cnt);
    }
    w_this = w[ix_arr[st]];
    cnt += w_this;
    running_mean   += ((x[ix_arr[st]] - xmean) - running_mean) / cnt;
    running_ssq    += w_this * (((x[ix_arr[st]] - xmean) - running_mean) * ((x[ix_arr[st]] - xmean) - mean_prev));
    cumw = cnt;
    return std::sqrt(running_ssq / cnt);
}

template <class real_t, class real_t_>
double find_split_std_gain_t(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                             GainCriterion criterion, double min_gain, double &restrict split_point)
{
    real_t full_sd = calc_sd_right_to_left<real_t>(x, n, sd_arr);
    real_t running_mean = 0;
    real_t running_ssq = 0;
    real_t mean_prev = x[0];
    real_t best_gain = -HUGE_VAL;
    real_t this_sd, this_gain;
    real_t n_ = (real_t)n;
    size_t best_ix = 0;
    for (size_t row = 0; row < n-1; row++)
    {
        running_mean   += (x[row] - running_mean) / (real_t)(row+1);
        running_ssq    += (x[row] - running_mean) * (x[row] - mean_prev);
        mean_prev       =  running_mean;
        if (x[row] == x[row+1])
            continue;

        this_sd = (row == 0)? 0. : std::sqrt(running_ssq / (real_t)(row+1));
        this_gain = (criterion == Pooled)?
                    pooled_gain(full_sd, n_, this_sd, sd_arr[row+1], row+1, n-row-1)
                        :
                    sd_gain(full_sd, this_sd, sd_arr[row+1]);
        if (this_gain > best_gain && this_gain > min_gain)
        {
            best_gain = this_gain;
            best_ix = row;
        }
    }

    if (best_gain > -HUGE_VAL)
        split_point = midpoint(x[best_ix], x[best_ix+1]);

    return best_gain;
}

template <class real_t_, class ldouble_safe>
double find_split_std_gain(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                           GainCriterion criterion, double min_gain, double &restrict split_point)
{
    if (n < THRESHOLD_LONG_DOUBLE)
        return find_split_std_gain_t<double, real_t_>(x, n, sd_arr, criterion, min_gain, split_point);
    else
        return find_split_std_gain_t<ldouble_safe, real_t_>(x, n, sd_arr, criterion, min_gain, split_point);
}

template <class real_t, class ldouble_safe>
double find_split_std_gain_weighted(real_t *restrict x, size_t n, double *restrict sd_arr,
                                    GainCriterion criterion, double min_gain, double &restrict split_point,
                                    double *restrict w, size_t *restrict sorted_ix)
{
    ldouble_safe cumw;
    double full_sd = calc_sd_right_to_left_weighted(x, n, sd_arr, w, cumw, sorted_ix);
    ldouble_safe running_mean = 0;
    ldouble_safe running_ssq = 0;
    ldouble_safe mean_prev = x[sorted_ix[0]];
    double best_gain = -HUGE_VAL;
    double this_sd, this_gain;
    double w_this;
    ldouble_safe currw = 0;
    size_t best_ix = 0;

    for (size_t row = 0; row < n-1; row++)
    {
        w_this = w[sorted_ix[row]];
        currw += w_this;
        running_mean   += w_this * (x[sorted_ix[row]] - running_mean) / currw;
        running_ssq    += w_this * ((x[sorted_ix[row]] - running_mean) * (x[sorted_ix[row]] - mean_prev));
        mean_prev       =  running_mean;
        if (x[sorted_ix[row]] == x[sorted_ix[row+1]])
            continue;

        this_sd = (row == 0)? 0. : std::sqrt(running_ssq / currw);
        this_gain = (criterion == Pooled)?
                    pooled_gain(full_sd, cumw, this_sd, sd_arr[row+1], currw, cumw-currw)
                        :
                    sd_gain(full_sd, this_sd, sd_arr[row+1]);
        if (this_gain > best_gain && this_gain > min_gain)
        {
            best_gain = this_gain;
            best_ix = row;
        }
    }

    if (best_gain > -HUGE_VAL)
        split_point = midpoint(x[sorted_ix[best_ix]], x[sorted_ix[best_ix+1]]);

    return best_gain;
}

template <class real_t, class real_t_>
double find_split_std_gain_t(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                             GainCriterion criterion, double min_gain, double &restrict split_point, size_t &restrict split_ix)
{
    real_t full_sd = calc_sd_right_to_left<real_t>(x, xmean, ix_arr, st, end, sd_arr);
    real_t running_mean = 0;
    real_t running_ssq = 0;
    real_t mean_prev = x[ix_arr[st]] - xmean;
    real_t best_gain = -HUGE_VAL;
    real_t n = (real_t)(end - st + 1);
    real_t this_sd, this_gain;
    split_ix = st;
    for (size_t row = st; row < end; row++)
    {
        running_mean   += ((x[ix_arr[row]] - xmean) - running_mean) / (real_t)(row-st+1);
        running_ssq    += ((x[ix_arr[row]] - xmean) - running_mean) * ((x[ix_arr[row]] - xmean) - mean_prev);
        mean_prev       =  running_mean;
        if (x[ix_arr[row]] == x[ix_arr[row+1]])
            continue;

        this_sd = (row == st)? 0. : std::sqrt(running_ssq / (real_t)(row-st+1));
        this_gain = (criterion == Pooled)?
                    pooled_gain(full_sd, n, this_sd, sd_arr[row-st+1], row-st+1, end-row)
                        :
                    sd_gain(full_sd, this_sd, sd_arr[row-st+1]);
        if (this_gain > best_gain && this_gain > min_gain)
        {
            best_gain = this_gain;
            split_ix = row;
        }
    }
    
    if (best_gain > -HUGE_VAL)
        split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);

    return best_gain;
}

template <class real_t_, class ldouble_safe>
double find_split_std_gain(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                           GainCriterion criterion, double min_gain, double &restrict split_point, size_t &restrict split_ix)
{
    if ((end-st+1) < THRESHOLD_LONG_DOUBLE)
        return find_split_std_gain_t<double, real_t_>(x, xmean, ix_arr, st, end, sd_arr, criterion, min_gain, split_point, split_ix);
    else
        return find_split_std_gain_t<ldouble_safe, real_t_>(x, xmean, ix_arr, st, end, sd_arr, criterion, min_gain, split_point, split_ix);
}

template <class real_t, class mapping, class ldouble_safe>
double find_split_std_gain_weighted(real_t *restrict x, real_t xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                                    GainCriterion criterion, double min_gain, double &restrict split_point, size_t &restrict split_ix, mapping &restrict w)
{
    ldouble_safe cumw;
    double full_sd = calc_sd_right_to_left_weighted(x, xmean, ix_arr, st, end, sd_arr, w, cumw);
    ldouble_safe running_mean = 0;
    ldouble_safe running_ssq = 0;
    ldouble_safe mean_prev = x[ix_arr[st]] - xmean;
    double best_gain = -HUGE_VAL;
    ldouble_safe currw = 0;
    double this_sd, this_gain;
    double w_this;
    split_ix = st;

    for (size_t row = st; row < end; row++)
    {
        w_this = w[ix_arr[row]];
        currw += w_this;
        running_mean   += w_this * ((x[ix_arr[row]] - xmean) - running_mean) / currw;
        running_ssq    += w_this * (((x[ix_arr[row]] - xmean) - running_mean) * ((x[ix_arr[row]] - xmean) - mean_prev));
        mean_prev       =  running_mean;
        if (x[ix_arr[row]] == x[ix_arr[row+1]])
            continue;

        this_sd = (row == st)? 0. : std::sqrt(running_ssq / currw);
        this_gain = (criterion == Pooled)?
                    pooled_gain(full_sd, cumw, this_sd, sd_arr[row-st+1], currw, cumw-currw)
                        :
                    sd_gain(full_sd, this_sd, sd_arr[row-st+1]);
        if (this_gain > best_gain && this_gain > min_gain)
        {
            best_gain = this_gain;
            split_ix = row;
        }
    }

    if (best_gain > -HUGE_VAL)
        split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);

    return best_gain;
}

#ifndef _FOR_R
    #if defined(__clang__)
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wunknown-attributes"
    #endif
#endif

#ifndef _FOR_R
[[gnu::optimize("Ofast")]]
#endif
static inline void xpy1(double *restrict x, double *restrict y, size_t n)
{
    for (size_t ix = 0; ix < n; ix++) y[ix] += x[ix];
}

#ifndef _FOR_R
[[gnu::optimize("Ofast")]]
#endif
static inline void axpy1(const double a, double *restrict x, double *restrict y, size_t n)
{
    for (size_t ix = 0; ix < n; ix++) y[ix] = std::fma(a, x[ix], y[ix]);
}

#ifndef _FOR_R
[[gnu::optimize("Ofast")]]
#endif
static inline void xpy1(double *restrict xval, size_t ind[], size_t nnz, double *restrict y)
{
    for (size_t ix = 0; ix < nnz; ix++) y[ind[ix]] += xval[ix];
}

#ifndef _FOR_R
[[gnu::optimize("Ofast")]]
#endif
static inline void axpy1(const double a, double *restrict xval, size_t ind[], size_t nnz, double *restrict y)
{
    for (size_t ix = 0; ix < nnz; ix++) y[ind[ix]] = std::fma(a, xval[ix], y[ind[ix]]);
}

#ifndef _FOR_R
    #if defined(__clang__)
        #pragma clang diagnostic pop
    #endif
#endif

template <class real_t, class ldouble_safe>
double find_split_full_gain(real_t *restrict x, size_t st, size_t end, size_t *restrict ix_arr,
                            size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                            double *restrict X_row_major, size_t ncols,
                            double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                            double *restrict buffer_sum_left, double *restrict buffer_sum_tot,
                            size_t &restrict split_ix, double &restrict split_point,
                            bool x_uses_ix_arr)
{
    if (end <= st) return -HUGE_VAL;
    if (cols_use != NULL && ncols_use && (double)ncols_use / (double)ncols < 0.1)
        force_cols_use = true;

    memset(buffer_sum_tot, 0, (force_cols_use? ncols_use : ncols)*sizeof(double));
    if (Xr_indptr == NULL)
    {
        if (force_cols_use)
        {
            double *restrict ptr_row;
            for (size_t row = st; row <= end; row++)
            {
                ptr_row = X_row_major + ix_arr[row]*ncols;
                for (size_t col = 0; col < ncols_use; col++)
                    buffer_sum_tot[col] += ptr_row[cols_use[col]];
            }
        }

        else
        {
            for (size_t row = st; row <= end; row++)
                xpy1(X_row_major + ix_arr[row]*ncols, buffer_sum_tot, ncols);
        }
    }

    else
    {
        if (force_cols_use)
        {
            size_t *curr_begin;
            size_t *row_end;
            size_t *curr_col;
            double *restrict Xr_this;
            size_t *cols_end = cols_use + ncols_use;
            for (size_t row = st; row <= end; row++)
            {
                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
                if (curr_begin == row_end) continue;
                curr_col = cols_use;
                Xr_this = Xr + Xr_indptr[ix_arr[row]];
                
                while (curr_col < cols_end && curr_begin < row_end)
                {
                    if (*curr_begin == *curr_col)
                    {
                        buffer_sum_tot[std::distance(cols_use, curr_col)] += Xr_this[std::distance(curr_begin, row_end)];
                        curr_col++;
                        curr_begin++;
                    }

                    else
                    {
                        if (*curr_begin > *curr_col)
                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
                        else
                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
                    }
                }
            }
        }

        else
        {
            size_t ptr_this;
            for (size_t row = st; row <= end; row++)
            {
                ptr_this = Xr_indptr[ix_arr[row]];
                xpy1(Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_tot);
            }
        }
    }

    double best_gain = -HUGE_VAL;
    double this_gain;
    double sl, sr;
    double dl, dr;
    double vleft, vright;
    memset(buffer_sum_left, 0, (force_cols_use? ncols_use : ncols)*sizeof(double));
    if (Xr_indptr == NULL)
    {
        if (!force_cols_use)
        {
            for (size_t row = st; row < end; row++)
            {
                xpy1(X_row_major + ix_arr[row]*ncols, buffer_sum_left, ncols);
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                dl = (double)(row-st+1);
                dr = (double)(end-row);
                for (size_t col = 0; col < ncols; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / dl);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / dr);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }

        else
        {
            double *restrict ptr_row;
            for (size_t row = st; row < end; row++)
            {
                ptr_row = X_row_major + ix_arr[row]*ncols;
                for (size_t col = 0; col < ncols_use; col++)
                    buffer_sum_left[col] += ptr_row[cols_use[col]];
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                dl = (double)(row-st+1);
                dr = (double)(end-row);
                for (size_t col = 0; col < ncols_use; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / dl);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / dr);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }
    }

    else
    {
        if (!force_cols_use)
        {
            size_t ptr_this;
            for (size_t row = st; row < end; row++)
            {
                ptr_this = Xr_indptr[ix_arr[row]];
                xpy1(Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_left);
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                dl = (double)(row-st+1);
                dr = (double)(end-row);
                for (size_t col = 0; col < ncols; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / dl);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / dr);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }

        else
        {
            size_t *curr_begin;
            size_t *row_end;
            size_t *curr_col;
            double *restrict Xr_this;
            size_t *cols_end = cols_use + ncols_use;
            for (size_t row = st; row < end; row++)
            {
                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
                if (curr_begin == row_end) goto skip_sum;
                curr_col = cols_use;
                Xr_this = Xr + Xr_indptr[ix_arr[row]];
                while (curr_col < cols_end && curr_begin < row_end)
                {
                    if (*curr_begin == *curr_col)
                    {
                        buffer_sum_left[std::distance(cols_use, curr_col)] += Xr_this[std::distance(curr_begin, row_end)];
                        curr_col++;
                        curr_begin++;
                    }

                    else
                    {
                        if (*curr_begin > *curr_col)
                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
                        else
                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
                    }
                }

                skip_sum:
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                dl = (double)(row-st+1);
                dr = (double)(end-row);
                for (size_t col = 0; col < ncols_use; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / dl);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / dr);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }
    }

    if (best_gain <= -HUGE_VAL) return best_gain;

    if (x_uses_ix_arr)
        split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
    else
        split_point = midpoint(x[split_ix], x[split_ix+1]);
    return best_gain / (ldouble_safe)(end - st + 1);
}

template <class real_t, class mapping, class ldouble_safe>
double find_split_full_gain_weighted(real_t *restrict x, size_t st, size_t end, size_t *restrict ix_arr,
                                     size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                     double *restrict X_row_major, size_t ncols,
                                     double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                                     double *restrict buffer_sum_left, double *restrict buffer_sum_tot,
                                     size_t &restrict split_ix, double &restrict split_point,
                                     bool x_uses_ix_arr,
                                     mapping &restrict w)
{
    if (end <= st) return -HUGE_VAL;
    if (cols_use != NULL && ncols_use && (double)ncols_use / (double)ncols < 0.1)
        force_cols_use = true;

    double wtot = 0;
    if (x_uses_ix_arr)
    {
        for (size_t row = st; row <= end; row++)
            wtot += w[ix_arr[row]];
    }

    else
    {
        for (size_t row = st; row <= end; row++)
            wtot += w[row];
    }

    memset(buffer_sum_tot, 0, (force_cols_use? ncols_use : ncols)*sizeof(double));
    if (Xr_indptr == NULL)
    {
        if (!force_cols_use)
        {
            if (x_uses_ix_arr)
            {
                for (size_t row = st; row <= end; row++)
                    axpy1(w[ix_arr[row]], X_row_major + ix_arr[row]*ncols, buffer_sum_tot, ncols);
            }

            else
            {
                for (size_t row = st; row <= end; row++)
                    axpy1(w[row], X_row_major + ix_arr[row]*ncols, buffer_sum_tot, ncols);
            }
        }

        else
        {
            double *restrict ptr_row;
            double w_row;

            if (x_uses_ix_arr)
            {
                for (size_t row = st; row <= end; row++)
                {
                    ptr_row = X_row_major + ix_arr[row]*ncols;
                    w_row = w[ix_arr[row]];
                    for (size_t col = 0; col < ncols_use; col++)
                        buffer_sum_tot[col] = std::fma(w_row, ptr_row[cols_use[col]], buffer_sum_tot[col]);
                }
            }

            else
            {
                for (size_t row = st; row <= end; row++)
                {
                    ptr_row = X_row_major + ix_arr[row]*ncols;
                    w_row = w[row];
                    for (size_t col = 0; col < ncols_use; col++)
                        buffer_sum_tot[col] = std::fma(w_row, ptr_row[cols_use[col]], buffer_sum_tot[col]);
                }
            }
        }
    }

    else
    {
        if (!force_cols_use)
        {
            size_t ptr_this;
            if (x_uses_ix_arr)
            {
                for (size_t row = st; row <= end; row++)
                {
                    ptr_this = Xr_indptr[ix_arr[row]];
                    axpy1(w[ix_arr[row]], Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_tot);
                }
            }

            else
            {
                for (size_t row = st; row <= end; row++)
                {
                    ptr_this = Xr_indptr[ix_arr[row]];
                    axpy1(w[row], Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_tot);
                }
            }
        }

        else
        {
            size_t *curr_begin;
            size_t *row_end;
            size_t *curr_col;
            double *restrict Xr_this;
            size_t *cols_end = cols_use + ncols_use;
            double w_row;
            for (size_t row = st; row <= end; row++)
            {
                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
                if (curr_begin == row_end) continue;
                curr_col = cols_use;
                Xr_this = Xr + Xr_indptr[ix_arr[row]];
                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
                size_t dtemp;
                
                while (curr_col < cols_end && curr_begin < row_end)
                {
                    if (*curr_begin == *curr_col)
                    {
                        dtemp = std::distance(cols_use, curr_col);
                        buffer_sum_tot[dtemp]
                            =
                        std::fma(w_row, Xr_this[std::distance(curr_begin, row_end)], buffer_sum_tot[dtemp]);
                        curr_col++;
                        curr_begin++;
                    }

                    else
                    {
                        if (*curr_begin > *curr_col)
                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
                        else
                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
                    }
                }
            }
        }
    }

    double best_gain = -HUGE_VAL;
    double this_gain;
    double sl, sr;
    double vleft, vright;
    double wleft = 0;
    double w_row;
    double wright;
    memset(buffer_sum_left, 0, (force_cols_use? ncols_use : ncols)*sizeof(double));
    if (Xr_indptr == NULL)
    {
        if (!force_cols_use)
        {
            for (size_t row = st; row < end; row++)
            {
                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
                wleft += w_row;
                axpy1(w_row, X_row_major + ix_arr[row]*ncols, buffer_sum_left, ncols);
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                wright = wtot - wleft;
                for (size_t col = 0; col < ncols; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / wleft);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / wright);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }

        else
        {
            double *restrict ptr_row;
            double w_row;
            for (size_t row = st; row < end; row++)
            {
                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
                wleft += w_row;

                ptr_row = X_row_major + ix_arr[row]*ncols;
                for (size_t col = 0; col < ncols_use; col++)
                    buffer_sum_left[col] = std::fma(w_row, ptr_row[cols_use[col]], buffer_sum_left[col]);
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                wright = wtot - wleft;
                for (size_t col = 0; col < ncols_use; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / wleft);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / wright);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }
    }

    else
    {
        if (!force_cols_use)
        {
            size_t ptr_this;
            double w_row;
            for (size_t row = st; row < end; row++)
            {
                w_row= w[x_uses_ix_arr? ix_arr[row] : row];
                wleft += w_row;
                ptr_this = Xr_indptr[ix_arr[row]];
                axpy1(w_row, Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_left);
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                wright = wtot - wleft;
                for (size_t col = 0; col < ncols; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / wleft);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / wright);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }

        else
        {
            size_t *curr_begin;
            size_t *row_end;
            size_t *curr_col;
            double *restrict Xr_this;
            size_t *cols_end = cols_use + ncols_use;
            double w_row;
            size_t dtemp;
            for (size_t row = st; row < end; row++)
            {
                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
                wleft += w_row;
                
                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
                if (curr_begin == row_end) goto skip_sum;
                curr_col = cols_use;
                Xr_this = Xr + Xr_indptr[ix_arr[row]];
                while (curr_col < cols_end && curr_begin < row_end)
                {
                    if (*curr_begin == *curr_col)
                    {
                        dtemp = std::distance(cols_use, curr_col);
                        buffer_sum_left[dtemp]
                            =
                        std::fma(w_row, Xr_this[std::distance(curr_begin, row_end)], buffer_sum_left[dtemp]);
                        curr_col++;
                        curr_begin++;
                    }

                    else
                    {
                        if (*curr_begin > *curr_col)
                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
                        else
                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
                    }
                }

                skip_sum:
                if (x_uses_ix_arr) {
                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
                }
                else {
                    if (unlikely(x[row] == x[row+1])) continue;
                }

                vleft = 0;
                vright = 0;
                wright = wtot - wleft;
                for (size_t col = 0; col < ncols_use; col++)
                {
                    sl = buffer_sum_left[col];
                    vleft += sl * (sl / wleft);
                    sr = buffer_sum_tot[col] - sl;
                    vright += sr * (sr / wright);
                }

                this_gain = vleft + vright;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_ix = row;
                }
            }
        }
    }

    if (best_gain  <= -HUGE_VAL) return best_gain;
    
    split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
    return best_gain / wtot;
}

template <class real_t_, class real_t>
double find_split_dens_shortform_t(real_t *restrict x, size_t n, double &restrict split_point)
{
    double best_gain = -HUGE_VAL;
    size_t n_minus_one = n - 1;
    real_t_ xmin = x[0];
    real_t_ xmax = x[n-1];
    real_t_ xleft, xright;
    real_t_ xmid;
    double this_gain;
    size_t split_ix = 0;

    for (size_t ix = 0; ix < n_minus_one; ix++)
    {
        if (x[ix] == x[ix+1]) continue;
        xmid = (real_t_)x[ix] + ((real_t_)x[ix+1] - (real_t_)x[ix]) / (real_t_)2;
        xleft = xmid - xmin;
        xright = xmax - xmid;
        if (unlikely(!xleft || !xright)) continue;
        this_gain = (real_t_)square(ix+1) / xleft + (real_t_)square(n_minus_one - ix) / xright;
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            split_ix = ix;
        }
    }

    if (best_gain <= -HUGE_VAL) return best_gain;

    real_t_ xtot = (real_t_)xmax - (real_t_)xmin;
    real_t_ nleft = (real_t_)(split_ix+1);
    real_t_ nright = (real_t_)(n_minus_one - split_ix);
    split_point = midpoint(x[split_ix], x[split_ix+1]);
    real_t_ rpct_left = split_point / xtot;
    rpct_left = std::fmax(rpct_left, std::numeric_limits<double>::min());
    real_t_ rpct_right = (real_t_)1 - rpct_left;
    rpct_right = std::fmax(rpct_right, std::numeric_limits<double>::min());

    real_t_ nl_sq = nleft  / (real_t_)n; nl_sq = square(nl_sq);
    real_t_ nr_sq = nright / (real_t_)n; nl_sq = square(nr_sq);

    return nl_sq / rpct_left + nr_sq / rpct_right;
}

template <class real_t, class ldouble_safe>
double find_split_dens_shortform(real_t *restrict x, size_t n, double &restrict split_point)
{
    if (n < INT32_MAX)
        return find_split_dens_shortform_t<double, real_t>(x, n, split_point);
    else
        return find_split_dens_shortform_t<ldouble_safe, real_t>(x, n, split_point);
}

template <class real_t_, class real_t, class mapping>
double find_split_dens_shortform_weighted_t(real_t *restrict x, size_t n, double &restrict split_point, mapping &restrict w, size_t *restrict buffer_indices)
{
    double best_gain = -HUGE_VAL;
    size_t n_minus_one = n - 1;
    real_t_ xmin = x[buffer_indices[0]];
    real_t_ xmax = x[buffer_indices[n-1]];
    real_t_ xleft, xright;
    real_t_ xmid;
    double this_gain;

    real_t_ wtot = 0;
    for (size_t ix = 0; ix < n; ix++)
        wtot += w[buffer_indices[ix]];
    real_t_ w_left = 0;
    real_t_ w_right;
    real_t_ best_w = 0;
    size_t split_ix = 0;

    for (size_t ix = 0; ix < n_minus_one; ix++)
    {
        w_left += w[buffer_indices[ix]];
        if (x[buffer_indices[ix]] == x[buffer_indices[ix+1]]) continue;
        xmid = (real_t_)x[buffer_indices[ix]] + ((real_t_)x[buffer_indices[ix+1]] - (real_t_)x[buffer_indices[ix]]) / (real_t_)2;
        xleft = xmid - xmin;
        xright = xmax - xmid;
        if (unlikely(!xleft || !xright)) continue;

        w_right = wtot - w_left;
        this_gain = square(w_left) / xleft + square(w_right) / xright;
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            best_w = w_left;
            split_ix = xmid;
        }
    }

    if (best_gain <= -HUGE_VAL) return best_gain;

    real_t_ xtot = xmax - xmin;
    w_left = best_w;
    w_right = wtot - w_left;
    w_left = std::fmax(w_left, std::numeric_limits<double>::min());
    w_right = std::fmax(w_right, std::numeric_limits<double>::min());
    split_point = midpoint(x[buffer_indices[split_ix]], x[buffer_indices[split_ix+1]]);
    real_t_ rpct_left = split_point / xtot;
    rpct_left = std::fmax(rpct_left, std::numeric_limits<double>::min());
    real_t_ rpct_right = (real_t_)1 - rpct_left;
    rpct_right = std::fmax(rpct_right, std::numeric_limits<double>::min());

    real_t_ wl_sq = w_left  / wtot; wl_sq = square(wl_sq);
    real_t_ wr_sq = w_right / wtot; wl_sq = square(wr_sq);

    return wl_sq / rpct_left + wr_sq / rpct_right;
}

template <class real_t, class mapping, class ldouble_safe>
double find_split_dens_shortform_weighted(real_t *restrict x, size_t n, double &restrict split_point, mapping &restrict w, size_t *restrict buffer_indices)
{
    if (n < INT32_MAX)
        return find_split_dens_shortform_weighted_t<double, real_t, mapping>(x, n, split_point, w, buffer_indices);
    else
        return find_split_dens_shortform_weighted_t<ldouble_safe, real_t, mapping>(x, n, split_point, w, buffer_indices);
}

template <class real_t>
double find_split_dens_shortform(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                 double &restrict split_point, size_t &restrict split_ix)
{
    double best_gain = -HUGE_VAL;
    real_t xmin = x[ix_arr[st]];
    real_t xmax = x[ix_arr[end]];
    real_t xleft, xright;
    real_t xmid;
    double this_gain;

    for (size_t row = st; row < end; row++)
    {
        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
        xmid = x[ix_arr[row]] + (x[ix_arr[row+1]] - x[ix_arr[row]]) / (real_t)2;
        xleft = xmid - xmin;
        xright = xmax - xmid;
        if (unlikely(!xleft || !xright)) continue;
        this_gain = square(row-st+1) / xleft + square(end-row) / xright;
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            split_ix = row;
        }
    }

    if (best_gain <= -HUGE_VAL) return best_gain;

    double xtot = (double)xmax - (double)xmin;
    double nleft = (double)(split_ix-st+1);
    double nright = (double)(end - split_ix);
    split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
    double rpct_left = split_point / xtot;
    rpct_left = std::fmax(rpct_left, std::numeric_limits<double>::min());
    double rpct_right = 1. - rpct_left;
    rpct_right = std::fmax(rpct_right, std::numeric_limits<double>::min());
    double ntot = (double)(end - st + 1);

    double nl_sq = nleft  / ntot; nl_sq = square(nl_sq);
    double nr_sq = nright / ntot; nl_sq = square(nr_sq);

    return nl_sq / rpct_left + nr_sq / rpct_right;
}

template <class real_t, class mapping>
double find_split_dens_shortform_weighted(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                          double &restrict split_point, size_t &restrict split_ix, mapping &restrict w)
{
    double best_gain = -HUGE_VAL;
    real_t xmin = x[ix_arr[st]];
    real_t xmax = x[ix_arr[end]];
    real_t xleft, xright;
    real_t xmid;
    double this_gain;

    double wtot = 0;
    for (size_t row = st; row <= end; row++)
        wtot += w[ix_arr[row]];
    double w_left = 0;
    double w_right;
    double best_w = 0;

    for (size_t row = st; row < end; row++)
    {
        w_left += w[ix_arr[row]];
        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
        xmid = x[ix_arr[row]] + (x[ix_arr[row+1]] - x[ix_arr[row]]) / (real_t)2;
        xleft = xmid - xmin;
        xright = xmax - xmid;
        if (unlikely(!xleft || !xright)) continue;
        
        w_right = wtot - w_left;
        this_gain = square(w_left) / xleft + square(w_right) / xright;
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            best_w = w_left;
            split_ix = row;
        }
    }

    if (best_gain <= -HUGE_VAL) return best_gain;

    double xtot = (double)xmax - (double)xmin;
    w_left = best_w;
    w_right = wtot - w_left;
    w_left = std::fmax(w_left, std::numeric_limits<double>::min());
    w_right = std::fmax(w_right, std::numeric_limits<double>::min());
    split_point = midpoint(x[split_ix], x[split_ix+1]);
    double rpct_left = split_point / xtot;
    rpct_left = std::fmax(rpct_left, std::numeric_limits<double>::min());
    double rpct_right = 1. - rpct_left;
    rpct_right = std::fmax(rpct_right, std::numeric_limits<double>::min());

    double wl_sq = w_left  / wtot; wl_sq = square(wl_sq);
    double wr_sq = w_right / wtot; wl_sq = square(wr_sq);

    return wl_sq / rpct_left + wr_sq / rpct_right;
}

/* This is a slower but more numerically-robust form */
template <class real_t, class ldouble_safe>
double find_split_dens_longform(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                double &restrict split_point, size_t &restrict split_ix)
{
    double best_gain = -HUGE_VAL;
    real_t xmin = x[ix_arr[st]];
    real_t xmax = x[ix_arr[end]];
    real_t xleft, xright;
    real_t xmid;
    ldouble_safe pct_left, pct_right;
    ldouble_safe rpct_left, rpct_right;
    ldouble_safe n_tot = end - st + 1;
    ldouble_safe xtot = (ldouble_safe)xmax - (ldouble_safe)xmin;
    ldouble_safe cnt_left;
    double this_gain;

    for (size_t row = st; row < end; row++)
    {
        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
        xmid = midpoint(x[ix_arr[row]], x[ix_arr[row+1]]);
        xleft = xmid - xmin;
        xright = xmax - xmid;
        if (unlikely(!xleft || !xright)) continue;

        cnt_left = (ldouble_safe)(row-st+1);

        xleft = std::fmax(xleft, (real_t)std::numeric_limits<real_t>::min());
        xright = std::fmax(xright, (real_t)std::numeric_limits<real_t>::min());
        pct_left = cnt_left / n_tot;
        pct_right = (ldouble_safe)1 - pct_left;
        rpct_left = (ldouble_safe)xleft / xtot;
        rpct_right = (ldouble_safe)xright / xtot;

        this_gain = square(pct_left) / rpct_left + square(pct_right) / rpct_right;
        if (unlikely(is_na_or_inf(this_gain))) continue;
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            split_point = xmid;
            split_ix = row;
        }
    }

    return best_gain;
}

template <class real_t, class mapping, class ldouble_safe>
double find_split_dens_longform_weighted(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                         double &restrict split_point, size_t &restrict split_ix, mapping &restrict w)
{
    double best_gain = -HUGE_VAL;
    real_t xmin = x[ix_arr[st]];
    real_t xmax = x[ix_arr[end]];
    real_t xleft, xright;
    real_t xmid;
    ldouble_safe pct_left, pct_right;
    ldouble_safe rpct_left, rpct_right;
    ldouble_safe xtot = (ldouble_safe)xmax - (ldouble_safe)xmin;
    double this_gain;

    ldouble_safe wtot = 0;
    for (size_t row = st; row <= end; row++)
        wtot += w[ix_arr[row]];
    ldouble_safe w_left = 0;

    for (size_t row = st; row < end; row++)
    {
        w_left += w[ix_arr[row]];
        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
        xmid = midpoint(x[ix_arr[row]], x[ix_arr[row+1]]);
        xleft = xmid - xmin;
        xright = xmax - xmid;
        if (unlikely(!xleft || !xright)) continue;

        xleft = std::fmax(xleft, (real_t)std::numeric_limits<real_t>::min());
        xright = std::fmax(xright, (real_t)std::numeric_limits<real_t>::min());
        pct_left = w_left / wtot;
        pct_right = (ldouble_safe)1 - pct_left;
        rpct_left = (ldouble_safe)xleft / xtot;
        rpct_right = (ldouble_safe)xright / xtot;

        this_gain = square(pct_left) / rpct_left + square(pct_right) / rpct_right;
        if (unlikely(is_na_or_inf(this_gain))) continue;
        if (this_gain > best_gain)
        {
            best_gain = this_gain;
            split_point = xmid;
            split_ix = row;
        }
    }

    return best_gain;
}

template <class real_t, class ldouble_safe>
double find_split_dens(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                       double &restrict split_point, size_t &restrict split_ix)
{
    if (end - st + 1 < INT32_MAX && x[ix_arr[end]] - x[ix_arr[st]] >= 1)
        return find_split_dens_shortform<real_t>(x, ix_arr, st, end, split_point, split_ix);
    else
        return find_split_dens_longform<real_t, ldouble_safe>(x, ix_arr, st, end, split_point, split_ix);
}

template <class real_t, class mapping, class ldouble_safe>
double find_split_dens_weighted(real_t *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                                double &restrict split_point, size_t &restrict split_ix, mapping &restrict w)
{
    if (end - st + 1 < INT32_MAX && x[ix_arr[end]] - x[ix_arr[st]] >= 1)
        return find_split_dens_shortform_weighted<real_t, mapping>(x, ix_arr, st, end, split_point, split_ix, w);
    else
        return find_split_dens_longform_weighted<real_t, mapping, ldouble_safe>(x, ix_arr, st, end, split_point, split_ix, w);
}

template <class int_t, class ldouble_safe>
double find_split_dens_longform(int *restrict x, int ncat, size_t *restrict ix_arr, size_t st, size_t end,
                                CategSplit cat_split_type, MissingAction missing_action,
                                int &restrict chosen_cat, signed char *restrict split_categ, int *restrict saved_cat_mode,
                                size_t *restrict buffer_cnt, int_t *restrict buffer_indices)
{
    if (st >= end || ncat <= 1) return -HUGE_VAL;
    size_t n_nas = 0;
    int xval;
    
    /* count categories */
    memset(buffer_cnt, 0, sizeof(size_t) * ncat);
    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
            if (likely(x[ix_arr[row]] >= 0))
                buffer_cnt[x[ix_arr[row]]]++;
    }

    else if (missing_action == Impute)
    {
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            if (unlikely(xval < 0))
                n_nas++;
            else
                buffer_cnt[xval]++;
        }

        if (unlikely(n_nas >= end-st)) return -HUGE_VAL;

        if (n_nas)
        {
            auto idxmax = std::max_element(buffer_cnt, buffer_cnt + ncat);
            *idxmax += n_nas;
            *saved_cat_mode = (int)std::distance(buffer_cnt, idxmax);
        }
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            if (likely(xval >= 0)) buffer_cnt[xval]++;
        }
    }

    std::iota(buffer_indices, buffer_indices + ncat, (int_t)0);
    std::sort(buffer_indices, buffer_indices + ncat,
              [&buffer_cnt](const int_t a, const int_t b)
              {return buffer_cnt[a] < buffer_cnt[b];});

    int curr = 0;
    if (split_categ != NULL)
    {
        while (buffer_cnt[buffer_indices[curr]] == 0)
        {
            split_categ[buffer_indices[curr]] = -1;
            curr++;
        }
    }

    else
    {
        while (buffer_cnt[buffer_indices[curr]] == 0) curr++;
    }

    int ncat_present = ncat - curr;
    if (ncat_present <= 1) return -HUGE_VAL;
    if (ncat_present == 2)
    {
        switch (cat_split_type)
        {
            case SingleCateg:
            {
                chosen_cat = buffer_indices[curr];
                break;
            }

            case SubSet:
            {
                split_categ[buffer_indices[curr]] = 1;
                split_categ[buffer_indices[curr+1]] = 0;
                break;
            }
        }

        ldouble_safe pct_left
            =
        (ldouble_safe)buffer_cnt[buffer_indices[curr]]
            /
        (ldouble_safe)(
            buffer_cnt[buffer_indices[curr]]
                +
            buffer_cnt[buffer_indices[curr+1]]
        );

        return  ((ldouble_safe)buffer_cnt[buffer_indices[curr]] * (2. * pct_left)
                     +
                 (ldouble_safe)buffer_cnt[buffer_indices[curr+1]] * (2. - 2.*pct_left))
                    /
                 (ldouble_safe)(buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr+1]]);
    }

    size_t ntot;
    if (missing_action == Impute)
        ntot = end - st + 1;
    else
        ntot = std::accumulate(buffer_cnt, buffer_cnt + ncat, (size_t)0);
    if (unlikely(ntot <= 1)) unexpected_error();
    ldouble_safe ntot_ = (ldouble_safe)ntot;

    switch (cat_split_type)
    {
        case SingleCateg:
        {
            double pct_one_cat = 1. / (double)ncat_present;
            double pct_left_smallest = (ldouble_safe)buffer_cnt[buffer_indices[curr]] / ntot_;
            double gain_smallest
                =
            (ldouble_safe)buffer_cnt[buffer_indices[curr]] * (pct_left_smallest / pct_one_cat)
            +
            (ldouble_safe)(ntot - buffer_cnt[buffer_indices[curr]]) * ((1. - pct_left_smallest) / (1. - pct_one_cat))
            ;

            double pct_left_biggest = (ldouble_safe)buffer_cnt[buffer_indices[ncat-1]] / ntot_;
            double gain_biggest
                =
            (ldouble_safe)buffer_cnt[buffer_indices[ncat-1]] * (pct_left_biggest / pct_one_cat)
            +
            (ldouble_safe)(ntot - buffer_cnt[buffer_indices[ncat-1]]) * ((1. - pct_left_biggest) / (1. - pct_one_cat))
            ;

            if (gain_smallest >= gain_biggest)
            {
                chosen_cat = buffer_indices[curr];
                return gain_smallest / ntot_;
            }

            else
            {
                chosen_cat = buffer_indices[ncat-1];
                return gain_biggest / ntot_;
            }
            break;
        }

        case SubSet:
        {
            size_t cnt_left = 0;
            size_t cnt_right;
            int st_cat = curr - 1;
            double this_gain;
            double best_gain = -HUGE_VAL;
            int best_cat = 0;
            ldouble_safe pct_left;
            double pct_cat_left;
            double ncat_present_ = (double)ncat_present;
            for (; curr < ncat; curr++)
            {
                cnt_left += buffer_cnt[buffer_indices[curr]];
                cnt_right = ntot - cnt_left;
                pct_left = (ldouble_safe)cnt_left / ntot_;
                pct_cat_left = (double)(curr - st_cat) / ncat_present_;
                this_gain
                    =
                (ldouble_safe)cnt_left * (pct_left / pct_cat_left)
                +
                (ldouble_safe)cnt_right * (((ldouble_safe)1 - pct_left) / (1. - pct_cat_left))
                ;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    best_cat = curr;
                }
            }

            if (best_gain <= -HUGE_VAL) return best_gain;
            st_cat++;
            for (; st_cat <= best_cat; st_cat++)
                split_categ[buffer_indices[st_cat]] = 1;
            for (; st_cat < ncat; st_cat++)
                split_categ[buffer_indices[st_cat]] = 0;
            return best_gain / ntot_;
            break;
        }
    }

    /* This will not be reached, but CRAN might complain otherwise */
    return -HUGE_VAL;
}

template <class mapping, class int_t, class ldouble_safe>
double find_split_dens_longform_weighted(int *restrict x, int ncat, size_t *restrict ix_arr, size_t st, size_t end,
                                         CategSplit cat_split_type, MissingAction missing_action,
                                         int &restrict chosen_cat, signed char *restrict split_categ, int *restrict saved_cat_mode,
                                         int_t *restrict buffer_indices, mapping &restrict w)
{
    if (st >= end || ncat <= 1) return -HUGE_VAL;
    ldouble_safe w_missing = 0;
    int xval;
    size_t ix_;

    /* count categories */
    /* TODO: allocate this buffer externally */
    std::vector<ldouble_safe> buffer_cnt(ncat, (ldouble_safe)0);
    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
        {
            ix_ = ix_arr[row];
            if (unlikely(x[ix_]) < 0) continue;
            buffer_cnt[x[ix_]] += w[ix_];
        }
    }

    else if (missing_action == Impute)
    {
        for (size_t row = st; row <= end; row++)
        {
            ix_ = ix_arr[row];
            xval = x[ix_];
            if (unlikely(xval < 0))
                w_missing += w[ix_];
            else
                buffer_cnt[xval] += w[ix_];
        }

        if (w_missing)
        {
            auto idxmax = std::max_element(buffer_cnt.begin(), buffer_cnt.end());
            *idxmax += w_missing;
            *saved_cat_mode = (int)std::distance(buffer_cnt.begin(), idxmax);
        }
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            ix_ = ix_arr[row];
            xval = x[ix_];
            if (likely(xval >= 0)) buffer_cnt[xval] += w[ix_];
        }
    }

    std::iota(buffer_indices, buffer_indices + ncat, (int_t)0);
    std::sort(buffer_indices, buffer_indices + ncat,
              [&buffer_cnt](const int_t a, const int_t b)
              {return buffer_cnt[a] < buffer_cnt[b];});

    int curr = 0;
    if (split_categ != NULL)
    {
        while (buffer_cnt[buffer_indices[curr]] == 0)
        {
            split_categ[buffer_indices[curr]] = -1;
            curr++;
        }
    }

    else
    {
        while (buffer_cnt[buffer_indices[curr]] == 0) curr++;
    }

    int ncat_present = ncat - curr;
    if (ncat_present <= 1) return -HUGE_VAL;
    if (ncat_present == 2)
    {
        switch (cat_split_type)
        {
            case SingleCateg:
            {
                chosen_cat = buffer_indices[curr];
                break;
            }

            case SubSet:
            {
                split_categ[buffer_indices[curr]] = 1;
                split_categ[buffer_indices[curr+1]] = 0;
                break;
            }
        }

        ldouble_safe pct_left
            =
        buffer_cnt[buffer_indices[curr]]
            /
        (
            buffer_cnt[buffer_indices[curr]]
                +
            buffer_cnt[buffer_indices[curr+1]]
        );

        return  (buffer_cnt[buffer_indices[curr]] * (pct_left * 2.)
                     +
                 buffer_cnt[buffer_indices[curr+1]] * (2. - 2.*pct_left))
                    /
                (buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr+1]]);
    }

    ldouble_safe ntot = std::accumulate(buffer_cnt.begin(), buffer_cnt.end(), (ldouble_safe)0);
    if (unlikely(ntot <= 0)) unexpected_error();

    switch (cat_split_type)
    {
        case SingleCateg:
        {
            double pct_one_cat = 1. / (double)ncat_present;
            double pct_left_smallest = buffer_cnt[buffer_indices[curr]] / ntot;
            double gain_smallest
                =
            buffer_cnt[buffer_indices[curr]] * (pct_left_smallest / pct_one_cat)
            +
            (ntot - buffer_cnt[buffer_indices[curr]]) * ((1. - pct_left_smallest) / (1. - pct_one_cat))
            ;

            double pct_left_biggest = buffer_cnt[buffer_indices[ncat-1]] / ntot;
            double gain_biggest
                =
            buffer_cnt[buffer_indices[ncat-1]] * (pct_left_biggest / pct_one_cat)
            +
            (ntot - buffer_cnt[buffer_indices[ncat-1]]) * ((1. - pct_left_biggest) / (1. - pct_one_cat))
            ;

            if (gain_smallest >= gain_biggest)
            {
                chosen_cat = buffer_indices[curr];
                return gain_smallest / ntot;
            }

            else
            {
                chosen_cat = buffer_indices[ncat-1];
                return gain_biggest / ntot;
            }
            break;
        }

        case SubSet:
        {
            ldouble_safe cnt_left = 0;
            ldouble_safe cnt_right;
            int st_cat = curr - 1;
            double this_gain;
            double best_gain = -HUGE_VAL;
            int best_cat = 0;
            ldouble_safe pct_left;
            double pct_cat_left;
            double ncat_present_ = (double)ncat_present;
            for (; curr < ncat; curr++)
            {
                cnt_left += buffer_cnt[buffer_indices[curr]];
                cnt_right = ntot - cnt_left;
                pct_left = cnt_left / ntot;
                pct_cat_left = (double)(curr - st_cat) / ncat_present_;
                this_gain
                    =
                (ldouble_safe)cnt_left * (pct_left / pct_cat_left)
                +
                (ldouble_safe)cnt_right * (((ldouble_safe)1 - pct_left) / (1. - pct_cat_left))
                ;
                if (this_gain > best_gain)
                {
                    best_gain = this_gain;
                    best_cat = curr;
                }
            }

            if (best_gain <= -HUGE_VAL) return best_gain;
            st_cat++;
            for (; st_cat <= best_cat; st_cat++)
                split_categ[buffer_indices[st_cat]] = 1;
            for (; st_cat < ncat; st_cat++)
                split_categ[buffer_indices[st_cat]] = 0;
            return best_gain / ntot;
            break;
        }
    }

    /* This will not be reached, but CRAN might complain otherwise */
    return -HUGE_VAL;
}

/* for split-criterion in hyperplanes (see below for version aimed at single-variable splits) */
template <class ldouble_safe>
double eval_guided_crit(double *restrict x, size_t n, GainCriterion criterion,
                        double min_gain, bool as_relative_gain, double *restrict buffer_sd,
                        double &restrict split_point, double &restrict xmin, double &restrict xmax,
                        size_t *restrict ix_arr_plus_st,
                        size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                        double *restrict X_row_major, size_t ncols,
                        double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr)
{
    /* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
       all numbers are assumed to be small and in the same scale */
    double gain = 0;
    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

    /* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
    if (unlikely(n == 2))
    {
        if (x[0] == x[1]) return -HUGE_VAL;
        split_point = midpoint_with_reorder(x[0], x[1]);
        gain        = 1.;
        if (gain > min_gain)
            return gain;
        else
            return 0.;
    }

    if (criterion == FullGain)
    {
        /* TODO: these buffers should be allocated externally */
        std::vector<size_t> argsorted(n);
        std::iota(argsorted.begin(), argsorted.end(), (size_t)0);
        std::sort(argsorted.begin(), argsorted.end(),
                  [&x](const size_t a, const size_t b){return x[a] < x[b];});
        if (x[argsorted[0]] == x[argsorted[n-1]]) return -HUGE_VAL;
        std::vector<double> temp_buffer(n + mult2(ncols));
        for (size_t ix = 0; ix < n; ix++) temp_buffer[ix] = x[argsorted[ix]];
        for (size_t ix = 0; ix < n; ix++)
            argsorted[ix] = ix_arr_plus_st[argsorted[ix]];
        size_t ignored;
        return find_split_full_gain<double, ldouble_safe>(
                                    temp_buffer.data(), (size_t)0, n-1, argsorted.data(),
                                    cols_use, ncols_use, force_cols_use,
                                    X_row_major, ncols,
                                    Xr, Xr_ind, Xr_indptr,
                                    temp_buffer.data() + n, temp_buffer.data() + n + ncols,
                                    ignored, split_point,
                                    false);
    }

    /* sort in ascending order */
    std::sort(x, x + n);
    xmin = x[0]; xmax = x[n-1];
    if (x[0] == x[n-1]) return -HUGE_VAL;

    if (criterion == Pooled && as_relative_gain && min_gain <= 0)
        gain = find_split_rel_gain<double, ldouble_safe>(x, n, split_point);
    else if (criterion == Pooled || criterion == Averaged)
        gain = find_split_std_gain<double, ldouble_safe>(x, n, buffer_sd, criterion, min_gain, split_point);
    else if (criterion == DensityCrit)
        gain = find_split_dens_shortform<double, ldouble_safe>(x, n, split_point);
    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

template <class ldouble_safe>
double eval_guided_crit_weighted(double *restrict x, size_t n, GainCriterion criterion,
                                 double min_gain, bool as_relative_gain, double *restrict buffer_sd,
                                 double &restrict split_point, double &restrict xmin, double &restrict xmax,
                                 double *restrict w, size_t *restrict buffer_indices,
                                 size_t *restrict ix_arr_plus_st,
                                 size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                 double *restrict X_row_major, size_t ncols,
                                 double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr)
{
    /* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
       all numbers are assumed to be small and in the same scale */
    double gain = 0;
    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

    /* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
    if (unlikely(n == 2))
    {
        if (x[0] == x[1]) return -HUGE_VAL;
        split_point = midpoint_with_reorder(x[0], x[1]);
        gain        = 1.;
        if (gain > min_gain)
            return gain;
        else
            return 0.;
    }

    /* sort in ascending order */
    std::iota(buffer_indices, buffer_indices + n, (size_t)0);
    std::sort(buffer_indices, buffer_indices + n,
              [&x](const size_t a, const size_t b){return x[a] < x[b];});
    xmin = x[buffer_indices[0]]; xmax = x[buffer_indices[n-1]];
    if (xmin == xmax) return -HUGE_VAL;

    if (criterion == Pooled || criterion == Averaged)
        gain = find_split_std_gain_weighted<double, ldouble_safe>(x, n, buffer_sd, criterion, min_gain, split_point, w, buffer_indices);
    else if (criterion == DensityCrit)
        gain = find_split_dens_shortform_weighted<double, double *restrict, ldouble_safe>(x, n, split_point, w, buffer_indices);
    else if (criterion == FullGain)
    {
        std::vector<size_t> argsorted(n);
        std::iota(argsorted.begin(), argsorted.end(), (size_t)0);
        std::sort(argsorted.begin(), argsorted.end(),
                  [&x](const size_t a, const size_t b){return x[a] < x[b];});
        if (x[argsorted[0]] == x[argsorted[n-1]]) return -HUGE_VAL;
        std::vector<double> temp_buffer(n + mult2(ncols));
        for (size_t ix = 0; ix < n; ix++) temp_buffer[ix] = x[argsorted[ix]];
        for (size_t ix = 0; ix < n; ix++)
            argsorted[ix] = ix_arr_plus_st[argsorted[ix]];
        size_t ignored;
        gain = find_split_full_gain_weighted<double, double *restrict, ldouble_safe>(
                                             temp_buffer.data(), (size_t)0, n-1, argsorted.data(),
                                             cols_use, ncols_use, force_cols_use,
                                             X_row_major, ncols,
                                             Xr, Xr_ind, Xr_indptr,
                                             temp_buffer.data() + n, temp_buffer.data() + n + ncols,
                                             ignored, split_point,
                                             false,
                                             w);
    }
    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

/* for split-criterion in single-variable splits */
template <class real_t_, class ldouble_safe>
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, real_t_ *restrict x,
                        double *restrict buffer_sd, bool as_relative_gain,
                        double *restrict buffer_imputed_x, double *restrict saved_xmedian,
                        size_t &split_ix, double &restrict split_point, double &restrict xmin, double &restrict xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action,
                        size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                        double *restrict X_row_major, size_t ncols,
                        double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr)
{
    size_t st_orig = st;
    double gain = 0;
    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

    /* move NAs to the front if there's any, exclude them from calculations */
    if (missing_action != Fail)
        st = move_NAs_to_front(ix_arr, st, end, x);

    if (unlikely(st >= end)) return -HUGE_VAL;
    else if (unlikely(st == (end-1)))
    {
        if (x[ix_arr[st]] == x[ix_arr[end]])
            return -HUGE_VAL;
        split_point = midpoint_with_reorder(x[ix_arr[st]], x[ix_arr[end]]);
        split_ix    = st;
        gain        = 1.;
        if (gain > min_gain)
            return gain;
        else
            return 0.;
    }

    /* sort in ascending order */
    std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});
    if (x[ix_arr[st]] == x[ix_arr[end]]) return -HUGE_VAL;
    xmin = x[ix_arr[st]]; xmax = x[ix_arr[end]];

    /* unlike the previous case for the extended model, the data here has not been centered,
       which could make the standard deviations have poor precision. It's nevertheless not
       necessary for this mean to have good precision, since it's only meant for centering,
       so it can be calculated inexactly with simd instructions. */
    real_t_ xmean = 0;
    if (criterion == Pooled || criterion == Averaged)
    {
        for (size_t ix = st; ix <= end; ix++)
            xmean += x[ix_arr[ix]];
        xmean /= (real_t_)(end - st + 1);
    }

    if (missing_action == Impute && st > st_orig)
    {
        missing_action = Fail;
        fill_NAs_with_median(ix_arr, st_orig, st, end, x, buffer_imputed_x, saved_xmedian);
        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
            gain = find_split_rel_gain<double, ldouble_safe>(buffer_imputed_x, (double)xmean, ix_arr, st_orig, end, split_point, split_ix);
        else if (criterion == Pooled || criterion == Averaged)
            gain = find_split_std_gain<double, ldouble_safe>(buffer_imputed_x, (double)xmean, ix_arr, st_orig, end, buffer_sd, criterion, min_gain, split_point, split_ix);
        else if (criterion == DensityCrit)
            gain = find_split_dens<double, ldouble_safe>(buffer_imputed_x, ix_arr, st_orig, end, split_point, split_ix);
        else if (criterion == FullGain)
        {
            /* TODO: this buffer should be allocated from outside */
            std::vector<double> temp_buffer(mult2(ncols));
            gain = find_split_full_gain<double, ldouble_safe>(
                                        buffer_imputed_x, st_orig, end, ix_arr,
                                        cols_use, ncols_use, force_cols_use,
                                        X_row_major, ncols,
                                        Xr, Xr_ind, Xr_indptr,
                                        temp_buffer.data(), temp_buffer.data() + ncols,
                                        split_ix, split_point, true);
        }

        /* Note: in theory, it should be possible to use a faster version assuming a contiguous array for 'x',
           but such an approach might give inexact split points. Better to avoid such inexactness at the
           expense of more computations. */
    }

    else
    {
        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
            gain = find_split_rel_gain<real_t_, ldouble_safe>(x, xmean, ix_arr, st, end, split_point, split_ix);
        else if (criterion == Pooled || criterion == Averaged)
            gain = find_split_std_gain<real_t_, ldouble_safe>(x, xmean, ix_arr, st, end, buffer_sd, criterion, min_gain, split_point, split_ix);
        else if (criterion == DensityCrit)
            gain = find_split_dens<real_t_, ldouble_safe>(x, ix_arr, st, end, split_point, split_ix);
        else if (criterion == FullGain)
        {
            /* TODO: this buffer should be allocated from outside */
            std::vector<double> temp_buffer(mult2(ncols));
            gain = find_split_full_gain<real_t_, ldouble_safe>(
                                        x, st, end, ix_arr,
                                        cols_use, ncols_use, force_cols_use,
                                        X_row_major, ncols,
                                        Xr, Xr_ind, Xr_indptr,
                                        temp_buffer.data(), temp_buffer.data() + ncols,
                                        split_ix, split_point, true);
        }
    }

    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

template <class real_t_, class mapping, class ldouble_safe>
double eval_guided_crit_weighted(size_t *restrict ix_arr, size_t st, size_t end, real_t_ *restrict x,
                                 double *restrict buffer_sd, bool as_relative_gain,
                                 double *restrict buffer_imputed_x, double *restrict saved_xmedian,
                                 size_t &split_ix, double &restrict split_point, double &restrict xmin, double &restrict xmax,
                                 GainCriterion criterion, double min_gain, MissingAction missing_action,
                                 size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                 double *restrict X_row_major, size_t ncols,
                                 double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                                 mapping &restrict w)
{
    size_t st_orig = st;
    double gain = 0;
    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

    /* move NAs to the front if there's any, exclude them from calculations */
    if (missing_action != Fail)
        st = move_NAs_to_front(ix_arr, st, end, x);

    if (unlikely(st >= end)) return -HUGE_VAL;
    else if (unlikely(st == (end-1)))
    {
        if (x[ix_arr[st]] == x[ix_arr[end]])
            return -HUGE_VAL;
        split_point = midpoint_with_reorder(x[ix_arr[st]], x[ix_arr[end]]);
        split_ix    = st;
        gain        = 1.;
        if (gain > min_gain)
            return gain;
        else
            return 0.;
    }

    /* sort in ascending order */
    std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});
    if (x[ix_arr[st]] == x[ix_arr[end]]) return -HUGE_VAL;
    xmin = x[ix_arr[st]]; xmax = x[ix_arr[end]];

    /* unlike the previous case for the extended model, the data here has not been centered,
       which could make the standard deviations have poor precision. It's nevertheless not
       necessary for this mean to have good precision, since it's only meant for centering,
       so it can be calculated inexactly with simd instructions. */
    real_t_ xmean = 0;
    real_t_ cnt = 0;
    if (criterion == Pooled || criterion == Averaged)
    {
        for (size_t ix = st; ix <= end; ix++)
        {
            xmean += x[ix_arr[ix]];
            cnt += w[ix_arr[ix]];
        }
        xmean /= cnt;
    }

    if (missing_action == Impute && st > st_orig)
    {
        missing_action = Fail;
        fill_NAs_with_median(ix_arr, st_orig, st, end, x, buffer_imputed_x, saved_xmedian);
        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
            gain = find_split_rel_gain_weighted<double, mapping, ldouble_safe>(buffer_imputed_x, (double)xmean, ix_arr, st_orig, end, split_point, split_ix, w);
        else if (criterion == Pooled || criterion == Averaged)
            gain = find_split_std_gain_weighted<double, mapping, ldouble_safe>(buffer_imputed_x, (double)xmean, ix_arr, st_orig, end, buffer_sd, criterion, min_gain, split_point, split_ix, w);
        else if (criterion == DensityCrit)
            gain = find_split_dens_weighted<double, mapping, ldouble_safe>(buffer_imputed_x, ix_arr, st_orig, end, split_point, split_ix, w);
        else if (criterion == FullGain)
        {
            std::vector<double> temp_buffer(mult2(ncols));
            gain = find_split_full_gain_weighted<double, mapping, ldouble_safe>(
                                                 buffer_imputed_x, st_orig, end, ix_arr,
                                                 cols_use, ncols_use, force_cols_use,
                                                 X_row_major, ncols,
                                                 Xr, Xr_ind, Xr_indptr,
                                                 temp_buffer.data(), temp_buffer.data() + ncols,
                                                 split_ix, split_point, true,
                                                 w);
        }
    }

    else
    {
        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
            gain = find_split_rel_gain_weighted<real_t_, mapping, ldouble_safe>(x, xmean, ix_arr, st, end, split_point, split_ix, w);
        else if (criterion == Pooled || criterion == Averaged)
            gain = find_split_std_gain_weighted<real_t_, mapping, ldouble_safe>(x, xmean, ix_arr, st, end, buffer_sd, criterion, min_gain, split_point, split_ix, w);
        else if (criterion == DensityCrit)
            gain = find_split_dens_weighted<real_t_, mapping, ldouble_safe>(x, ix_arr, st, end, split_point, split_ix, w);
        else if (criterion == FullGain)
        {
            std::vector<double> temp_buffer(mult2(ncols));
            gain = find_split_full_gain_weighted<real_t_, mapping, ldouble_safe>(
                                                 x, st, end, ix_arr,
                                                 cols_use, ncols_use, force_cols_use,
                                                 X_row_major, ncols,
                                                 Xr, Xr_ind, Xr_indptr,
                                                 temp_buffer.data(), temp_buffer.data() + ncols,
                                                 split_ix, split_point, true,
                                                 w);
        }
    }

    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

/* TODO: here it should only need to look at the non-zero entries. It can then use the
   same algorithm as above, but putting an extra check to see where do the zeros fit in
   the sorted order of the non-zero entries while calculating gains and SDs, and then
   call the 'divide_subset' function after-the-fact to reach the same end result.
   It should be much faster than this if the non-zero entries are few. */
template <class real_t_, class sparse_ix, class ldouble_safe>
double eval_guided_crit(size_t ix_arr[], size_t st, size_t end,
                        size_t col_num, real_t_ Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                        double buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
                        double *restrict saved_xmedian,
                        double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action,
                        size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                        double *restrict X_row_major, size_t ncols,
                        double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr)
{
    size_t ignored;


    todense(ix_arr, st, end,
            col_num, Xc, Xc_ind, Xc_indptr,
            buffer_arr);
    size_t tot = end - st + 1;
    std::iota(buffer_pos, buffer_pos + tot, (size_t)0);

    if (missing_action == Impute)
    {
        missing_action = Fail;
        for (size_t ix = 0; ix < tot; ix++)
        {
            if (unlikely(is_na_or_inf(buffer_arr[ix])))
            {
                goto fill_missing;
            }
        }
        goto no_nas;

        fill_missing:
        {
            size_t idx_half = div2(tot);
            std::nth_element(buffer_pos, buffer_pos + idx_half, buffer_pos + tot,
                             [&buffer_arr](const size_t a, const size_t b){return buffer_arr[a] < buffer_arr[b];});
            *saved_xmedian = buffer_arr[buffer_pos[idx_half]];

            if ((tot % 2) == 0)
            {
                double xlow = *std::max_element(buffer_pos, buffer_pos + idx_half);
                *saved_xmedian = xlow + ((*saved_xmedian)-xlow)/2.;
            }

            for (size_t ix = 0; ix < tot; ix++)
                buffer_arr[ix] = is_na_or_inf(buffer_arr[ix])? (*saved_xmedian) : buffer_arr[ix];
            std::iota(buffer_pos, buffer_pos + tot, (size_t)0);
        }
    }

    no_nas:
    return eval_guided_crit<double, ldouble_safe>(
                            buffer_pos, 0, end - st, buffer_arr, buffer_arr + tot,
                            as_relative_gain, saved_xmedian, (double*)NULL, ignored, split_point,
                            xmin, xmax, criterion, min_gain, missing_action,
                            cols_use, ncols_use, force_cols_use,
                            X_row_major, ncols,
                            Xr, Xr_ind, Xr_indptr);
}

template <class real_t_, class sparse_ix, class mapping, class ldouble_safe>
double eval_guided_crit_weighted(size_t ix_arr[], size_t st, size_t end,
                                 size_t col_num, real_t_ Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                                 double buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
                                 double *restrict saved_xmedian,
                                 double &restrict split_point, double &restrict xmin, double &restrict xmax,
                                 GainCriterion criterion, double min_gain, MissingAction missing_action,
                                 size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
                                 double *restrict X_row_major, size_t ncols,
                                 double *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
                                 mapping &restrict w)
{
    size_t ignored;


    todense(ix_arr, st, end,
            col_num, Xc, Xc_ind, Xc_indptr,
            buffer_arr);
    size_t tot = end - st + 1;
    std::iota(buffer_pos, buffer_pos + tot, (size_t)0);


    if (missing_action == Impute)
    {
        missing_action = Fail;
        for (size_t ix = 0; ix < tot; ix++)
        {
            if (unlikely(is_na_or_inf(buffer_arr[ix])))
            {
                goto fill_missing;
            }
        }
        goto no_nas;

        fill_missing:
        {
            size_t idx_half = div2(tot);
            std::nth_element(buffer_pos, buffer_pos + idx_half, buffer_pos + tot,
                             [&buffer_arr](const size_t a, const size_t b){return buffer_arr[a] < buffer_arr[b];});
            *saved_xmedian = buffer_arr[buffer_pos[idx_half]];

            if ((tot % 2) == 0)
            {
                double xlow = *std::max_element(buffer_pos, buffer_pos + idx_half);
                *saved_xmedian = xlow + ((*saved_xmedian)-xlow)/2.;
            }

            for (size_t ix = 0; ix < tot; ix++)
                buffer_arr[ix] = is_na_or_inf(buffer_arr[ix])? (*saved_xmedian) : buffer_arr[ix];
            std::iota(buffer_pos, buffer_pos + tot, (size_t)0);
        }
    }


    no_nas:
    /* TODO: allocate this buffer externally */
    std::vector<double> buffer_w(tot);
    for (size_t row = st; row <= end; row++)
        buffer_w[row-st] = w[ix_arr[row]];
    /* TODO: in this case, as the weights match with the order of the indices, could use a faster version
       with a weighted rel_gain function instead (not yet implemented). */
    return eval_guided_crit_weighted<double, std::vector<double>, ldouble_safe>(
                                     buffer_pos, 0, end - st, buffer_arr, buffer_arr + tot,
                                     as_relative_gain, saved_xmedian, (double*)NULL, ignored, split_point,
                                     xmin, xmax, criterion, min_gain, missing_action,
                                     cols_use, ncols_use, force_cols_use,
                                     X_row_major, ncols,
                                     Xr, Xr_ind, Xr_indptr,
                                     buffer_w);
}

/* How this works:
   - For Averaged criterion, will take the expected standard deviation that would be gotten with the category counts
     if each category got assigned a real number at random ~ Unif(0,1) and the data were thus converted to
     numerical. In such case, the best split (highest sd gain) is always putting the second-highest count in one
     branch, so there is no point in doing a full search over other permutations. In order to get more reasonable
     splits, when using the option to split by subsets of categories, it will sort the counts and evaluate only
     splits in which the categories are grouped in sorted order - in such cases it tends to pick either the
     smallest or the largest category to assign to one branch, but sometimes picks groups too.
   - For Pooled criterion, will take shannon entropy, which tends to make a more even split. In the case of splitting
     by a single category, it always puts the largest category in a separate branch. In the case of subsets,
     it can either evaluate possible splits over all permutations (not feasible if there are too many categories),
     or look up for splits in sorted order just like for Averaged criterion.
   Splitting by averaged Gini gain (like with Averaged) also selects always the second-largest category to put in one branch,
   while splitting by weighted Gini (like with Pooled) usually selects the largest category to put in one branch. The
   Gini gain is not easily comparable to that of numerical columns, so it's not offered as an option here.
*/
/* https://math.stackexchange.com/questions/3343384/expected-variance-and-kurtosis-from-pmf-in-which-possible-discrete-values-are-dr */
/* TODO: 'buffer_pos' doesn't need to be 'size_t', 'int' would suffice */
template <class ldouble_safe>
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int ncat,
                        int *restrict saved_cat_mode,
                        size_t *restrict buffer_cnt, size_t *restrict buffer_pos, double *restrict buffer_prob,
                        int &restrict chosen_cat, signed char *restrict split_categ, signed char *restrict buffer_split,
                        GainCriterion criterion, double min_gain, bool all_perm,
                        MissingAction missing_action, CategSplit cat_split_type)
{
    if (criterion == DensityCrit)
        return find_split_dens_longform<size_t, ldouble_safe>(
                                        x, ncat, ix_arr, st, end,
                                        cat_split_type, missing_action,
                                        chosen_cat, split_categ, saved_cat_mode,
                                        buffer_cnt, buffer_pos);
    if (st >= end) return -HUGE_VAL;
    size_t n_nas = 0;
    int xval;
    
    /* count categories */
    memset(buffer_cnt, 0, sizeof(size_t) * ncat);
    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
            if (likely(x[ix_arr[row]] >= 0))
                buffer_cnt[x[ix_arr[row]]]++;
    }

    else if (missing_action == Impute)
    {
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            if (unlikely(xval < 0))
                n_nas++;
            else
                buffer_cnt[xval]++;
        }

        if (unlikely(n_nas >= end-st)) return -HUGE_VAL;

        if (n_nas)
        {
            auto idxmax = std::max_element(buffer_cnt, buffer_cnt + ncat);
            *idxmax += n_nas;
            *saved_cat_mode = (int)std::distance(buffer_cnt, idxmax);
        }
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            if (likely(xval >= 0)) buffer_cnt[xval]++;
        }
    }

    double this_gain = -HUGE_VAL;
    double best_gain = -HUGE_VAL;
    std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
    size_t st_pos = 0;

    switch (cat_split_type)
    {
        case SingleCateg:
        {
            size_t cnt = end - st + 1;
            ldouble_safe cnt_l = (ldouble_safe) cnt;
            size_t ncat_present = 0;

            switch(criterion)
            {
                case Averaged:
                {
                    /* move zero-counts to the beginning */
                    size_t temp;
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[cat])
                        {
                            ncat_present++;
                            buffer_prob[cat] = (ldouble_safe) buffer_cnt[cat] / cnt_l;
                        }

                        else
                        {
                            temp = buffer_pos[st_pos];
                            buffer_pos[st_pos] = buffer_pos[cat];
                            buffer_pos[cat] = temp;
                            st_pos++;
                        }
                    }
                    
                    if (ncat_present <= 1) return -HUGE_VAL;

                    double sd_full = expected_sd_cat<size_t, ldouble_safe>(buffer_prob, ncat_present, buffer_pos + st_pos);

                    /* try isolating each category one at a time */
                    for (size_t pos = st_pos; (int)pos < ncat; pos++)
                    {
                        this_gain = sd_gain(sd_full,
                                            0.0,
                                            (expected_sd_cat_single<size_t, size_t, ldouble_safe>(buffer_cnt, buffer_prob, ncat_present, buffer_pos + st_pos, pos - st_pos, cnt)));
                        if (this_gain > min_gain && this_gain > best_gain)
                        {
                            best_gain = this_gain;
                            chosen_cat = buffer_pos[pos];
                        }
                    }
                    break;
                }

                case Pooled:
                {
                    /* here it will always pick the largest one */
                    size_t ncat_present = 0;
                    size_t cnt_max = 0;
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[cat])
                        {
                            ncat_present++;
                            if (cnt_max < buffer_cnt[cat])
                            {
                                cnt_max = buffer_cnt[cat];
                                chosen_cat = cat;
                            }
                        }
                    }
                    
                    if (ncat_present <= 1) return -HUGE_VAL;

                    ldouble_safe cnt_left = (ldouble_safe)((end - st + 1) - cnt_max);
                    this_gain = (
                                    (ldouble_safe)cnt * std::log((ldouble_safe)cnt)
                                        - cnt_left * std::log(cnt_left)
                                        - (ldouble_safe)cnt_max * std::log((ldouble_safe)cnt_max)
                                ) / cnt;
                    best_gain = (this_gain > min_gain)? this_gain : best_gain;
                    break;
                }

                default:
                {
                    unexpected_error();
                    break;
                }
            }
            break;
        }

        case SubSet:
        {
            /* sort by counts */
            std::sort(buffer_pos, buffer_pos + ncat, [&buffer_cnt](const size_t a, const size_t b){return buffer_cnt[a] < buffer_cnt[b];});

            /* set split as: (1):left (0):right (-1):not_present */
            memset(buffer_split, 0, ncat * sizeof(signed char));

            ldouble_safe cnt = (ldouble_safe)(end - st + 1);

            switch(criterion)
            {
                case Averaged:
                {
                    /* determine first non-zero and convert to probabilities */
                    double sd_full;
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[buffer_pos[cat]])
                        {
                            buffer_prob[buffer_pos[cat]] = (ldouble_safe)buffer_cnt[buffer_pos[cat]] / cnt;
                        }

                        else
                        {
                            buffer_split[buffer_pos[cat]] = -1;
                            st_pos++;
                        }
                    }

                    if ((int)st_pos >= (ncat-1)) return -HUGE_VAL;

                    /* calculate full SD assuming they take values randomly ~Unif(0, 1) */
                    size_t ncat_present = (size_t)ncat - st_pos;
                    sd_full = expected_sd_cat<size_t, ldouble_safe>(buffer_prob, ncat_present, buffer_pos + st_pos);
                    if (ncat_present >= log2ceil(SIZE_MAX)) all_perm = false;

                    /* move categories one at a time */
                    for (size_t pos = st_pos; pos < ((size_t)ncat - st_pos - 1); pos++)
                    {
                        buffer_split[buffer_pos[pos]] = 1;
                        this_gain = sd_gain(sd_full,
                                            (expected_sd_cat<size_t, size_t, ldouble_safe>(buffer_cnt, buffer_prob, pos - st_pos + 1, buffer_pos + st_pos)),
                                            (expected_sd_cat<size_t, size_t, ldouble_safe>(buffer_cnt, buffer_prob, (size_t)ncat - pos - 1, buffer_pos + pos + 1))
                                            );
                        if (this_gain > min_gain && this_gain > best_gain)
                        {
                            best_gain = this_gain;
                            memcpy(split_categ, buffer_split, ncat * sizeof(signed char));
                        }
                    }

                    break;
                }

                case Pooled:
                {
                    ldouble_safe s = 0;

                    /* determine first non-zero and get base info */
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[buffer_pos[cat]])
                        {
                            s += (buffer_cnt[buffer_pos[cat]] <= 1)?
                                  0 : ((ldouble_safe) buffer_cnt[buffer_pos[cat]] * std::log((ldouble_safe)buffer_cnt[buffer_pos[cat]]));
                        }

                        else
                        {
                            buffer_split[buffer_pos[cat]] = -1;
                            st_pos++;
                        }
                    }

                    if ((int)st_pos >= (ncat-1)) return -HUGE_VAL;

                    /* calculate base info */
                    ldouble_safe base_info = cnt * std::log(cnt) - s;

                    if (all_perm)
                    {
                        size_t cnt_left, cnt_right;
                        double s_left, s_right;
                        size_t ncat_present = (size_t)ncat - st_pos;
                        size_t ncomb = pow2(ncat_present) - 1;
                        size_t best_combin;

                        for (size_t combin = 1; combin < ncomb; combin++)
                        {
                            cnt_left = 0; cnt_right = 0;
                            s_left   = 0;   s_right = 0;
                            for (size_t pos = st_pos; (int)pos < ncat; pos++)
                            {
                                if (extract_bit(combin, pos))
                                {
                                    cnt_left += buffer_cnt[buffer_pos[pos]];
                                    s_left   += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                                 0 : ((ldouble_safe) buffer_cnt[buffer_pos[pos]]
                                                       * std::log((ldouble_safe) buffer_cnt[buffer_pos[pos]]));
                                }

                                else
                                {
                                    cnt_right += buffer_cnt[buffer_pos[pos]];
                                    s_right   += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                                  0 : ((ldouble_safe) buffer_cnt[buffer_pos[pos]]
                                                        * std::log((ldouble_safe) buffer_cnt[buffer_pos[pos]]));
                                }
                            }

                            this_gain  = categ_gain<size_t, ldouble_safe>(
                                                    cnt_left, cnt_right,
                                                    s_left, s_right,
                                                    base_info, cnt);

                            if (this_gain > min_gain && this_gain > best_gain)
                            {
                                best_gain = this_gain;
                                best_combin = combin;
                            }

                        }

                        if (best_gain > min_gain)
                            for (size_t pos = 0; pos < ncat_present; pos++)
                                split_categ[buffer_pos[st_pos + pos]] = extract_bit(best_combin, pos);

                    }

                    else
                    {
                        /* try moving the categories one at a time */
                        size_t cnt_left = 0;
                        size_t cnt_right = end - st + 1;
                        double s_left = 0;
                        double s_right = s;

                        for (size_t pos = st_pos; pos < (ncat - st_pos - 1); pos++)
                        {
                            buffer_split[buffer_pos[pos]] = 1;
                            s_left    += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                          0 : ((ldouble_safe)buffer_cnt[buffer_pos[pos]] * std::log((ldouble_safe)buffer_cnt[buffer_pos[pos]]));
                            s_right   -= (buffer_cnt[buffer_pos[pos]] <= 1)?
                                          0 : ((ldouble_safe)buffer_cnt[buffer_pos[pos]] * std::log((ldouble_safe)buffer_cnt[buffer_pos[pos]]));
                            cnt_left  += buffer_cnt[buffer_pos[pos]];
                            cnt_right -= buffer_cnt[buffer_pos[pos]];

                            this_gain  = categ_gain<size_t, ldouble_safe>(
                                                    cnt_left, cnt_right,
                                                    s_left, s_right,
                                                    base_info, cnt);

                            if (this_gain > min_gain && this_gain > best_gain)
                            {
                                best_gain = this_gain;
                                memcpy(split_categ, buffer_split, ncat * sizeof(signed char));
                            }
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
    }

    if (st == (end-1)) return 0;

    if (best_gain <= -HUGE_VAL && this_gain <= min_gain && this_gain > -HUGE_VAL)
        return 0;
    else
        return best_gain;
}


template <class mapping, class ldouble_safe>
double eval_guided_crit_weighted(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int ncat,
                                 int *restrict saved_cat_mode,
                                 size_t *restrict buffer_pos, double *restrict buffer_prob,
                                 int &restrict chosen_cat, signed char *restrict split_categ, signed char *restrict buffer_split,
                                 GainCriterion criterion, double min_gain, bool all_perm,
                                 MissingAction missing_action, CategSplit cat_split_type,
                                 mapping &restrict w)
{
    if (criterion == DensityCrit)
        return find_split_dens_longform_weighted<mapping, size_t, ldouble_safe>(
                                                 x, ncat, ix_arr, st, end,
                                                 cat_split_type, missing_action,
                                                 chosen_cat, split_categ, saved_cat_mode,
                                                 buffer_pos, w);
    if (st >= end) return -HUGE_VAL;
    ldouble_safe w_missing = 0;
    int xval;
    size_t ix_;

    /* count categories */
    /* TODO: allocate this buffer externally */
    std::vector<ldouble_safe> buffer_cnt(ncat, (ldouble_safe)0);
    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
        {
            ix_ = ix_arr[row];
            if (unlikely(x[ix_]) < 0) continue;
            buffer_cnt[x[ix_]] += w[ix_];
        }
    }

    else if (missing_action == Impute)
    {
        for (size_t row = st; row <= end; row++)
        {
            ix_ = ix_arr[row];
            xval = x[ix_];
            if (unlikely(xval < 0))
                w_missing += w[ix_];
            else
                buffer_cnt[xval] += w[ix_];
        }

        if (w_missing)
        {
            auto idxmax = std::max_element(buffer_cnt.begin(), buffer_cnt.end());
            *idxmax += w_missing;
            *saved_cat_mode = (int)std::distance(buffer_cnt.begin(), idxmax);
        }
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            ix_ = ix_arr[row];
            xval = x[ix_];
            if (likely(xval >= 0)) buffer_cnt[xval] += w[ix_];
        }
    }

    ldouble_safe cnt = std::accumulate(buffer_cnt.begin(), buffer_cnt.end(), (ldouble_safe)0);

    double this_gain = -HUGE_VAL;
    double best_gain = -HUGE_VAL;
    std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
    size_t st_pos = 0;

    switch(cat_split_type)
    {
        case SingleCateg:
        {
            size_t ncat_present = 0;

            switch(criterion)
            {
                case Averaged:
                {
                    /* move zero-counts to the beginning */
                    size_t temp;
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[cat])
                        {
                            ncat_present++;
                            buffer_prob[cat] = buffer_cnt[cat] / cnt;
                        }

                        else
                        {
                            temp = buffer_pos[st_pos];
                            buffer_pos[st_pos] = buffer_pos[cat];
                            buffer_pos[cat] = temp;
                            st_pos++;
                        }
                    }
                    
                    if (ncat_present <= 1) return -HUGE_VAL;

                    double sd_full = expected_sd_cat<size_t, ldouble_safe>(buffer_prob, ncat_present, buffer_pos + st_pos);

                    /* try isolating each category one at a time */
                    for (size_t pos = st_pos; (int)pos < ncat; pos++)
                    {
                        this_gain = sd_gain(sd_full,
                                            0.0,
                                            (expected_sd_cat_single<ldouble_safe, size_t, ldouble_safe>(buffer_cnt.data(), buffer_prob, ncat_present, buffer_pos + st_pos, pos - st_pos, cnt))
                                            );
                        if (this_gain > min_gain && this_gain > best_gain)
                        {
                            best_gain = this_gain;
                            chosen_cat = buffer_pos[pos];
                        }
                    }
                    break;
                }

                case Pooled:
                {
                    /* here it will always pick the largest one */
                    size_t ncat_present = 0;
                    ldouble_safe cnt_max = 0;
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[cat])
                        {
                            ncat_present++;
                            if (cnt_max < buffer_cnt[cat])
                            {
                                cnt_max = buffer_cnt[cat];
                                chosen_cat = cat;
                            }
                        }
                    }
                    
                    if (ncat_present <= 1) return -HUGE_VAL;

                    ldouble_safe cnt_left = (ldouble_safe)(cnt - cnt_max);

                    /* TODO: think of a better way of dealing with numbers between zero and one */
                    this_gain = (
                                    std::fmax((ldouble_safe)1, cnt) * std::log(std::fmax((ldouble_safe)1, cnt))
                                        - std::fmax((ldouble_safe)1, cnt_left) * std::log(std::fmax((ldouble_safe)1, cnt_left))
                                        - std::fmax((ldouble_safe)1, cnt_max) * std::log(std::fmax((ldouble_safe)1, cnt_max))
                                ) / std::fmax((ldouble_safe)1, cnt);
                    best_gain = (this_gain > min_gain)? this_gain : best_gain;
                    break;
                }

                default:
                {
                    unexpected_error();
                    break;
                }
            }
            break;
        }

        case SubSet:
        {
            /* sort by counts */
            std::sort(buffer_pos, buffer_pos + ncat, [&buffer_cnt](const size_t a, const size_t b){return buffer_cnt[a] < buffer_cnt[b];});

            /* set split as: (1):left (0):right (-1):not_present */
            memset(buffer_split, 0, ncat * sizeof(signed char));


            switch(criterion)
            {
                case Averaged:
                {
                    /* determine first non-zero and convert to probabilities */
                    double sd_full;
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[buffer_pos[cat]])
                        {
                            buffer_prob[buffer_pos[cat]] = (ldouble_safe)buffer_cnt[buffer_pos[cat]] / cnt;
                        }

                        else
                        {
                            buffer_split[buffer_pos[cat]] = -1;
                            st_pos++;
                        }
                    }

                    if ((int)st_pos >= (ncat-1)) return -HUGE_VAL;

                    /* calculate full SD assuming they take values randomly ~Unif(0, 1) */
                    size_t ncat_present = (size_t)ncat - st_pos;
                    sd_full = expected_sd_cat<size_t, ldouble_safe>(buffer_prob, ncat_present, buffer_pos + st_pos);
                    if (ncat_present >= log2ceil(SIZE_MAX)) all_perm = false;

                    /* move categories one at a time */
                    for (size_t pos = st_pos; pos < ((size_t)ncat - st_pos - 1); pos++)
                    {
                        buffer_split[buffer_pos[pos]] = 1;
                        /* TODO: is this correct? */
                        this_gain = sd_gain(sd_full,
                                            (expected_sd_cat<ldouble_safe, size_t, ldouble_safe>(buffer_cnt.data(), buffer_prob, pos - st_pos + 1, buffer_pos + st_pos)),
                                            (expected_sd_cat<ldouble_safe, size_t, ldouble_safe>(buffer_cnt.data(), buffer_prob, (size_t)ncat - pos - 1, buffer_pos + pos + 1))
                                            );
                        if (this_gain > min_gain && this_gain > best_gain)
                        {
                            best_gain = this_gain;
                            memcpy(split_categ, buffer_split, ncat * sizeof(signed char));
                        }
                    }

                    break;
                }

                case Pooled:
                {
                    ldouble_safe s = 0;

                    /* determine first non-zero and get base info */
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[buffer_pos[cat]])
                        {
                            s += (buffer_cnt[buffer_pos[cat]] <= 1)?
                                  (ldouble_safe)0
                                      :
                                  ((ldouble_safe) buffer_cnt[buffer_pos[cat]] * std::log((ldouble_safe)buffer_cnt[buffer_pos[cat]]));
                        }

                        else
                        {
                            buffer_split[buffer_pos[cat]] = -1;
                            st_pos++;
                        }
                    }

                    if ((int)st_pos >= (ncat-1)) return -HUGE_VAL;

                    /* calculate base info */
                    ldouble_safe base_info = std::fmax((ldouble_safe)1, cnt) * std::log(std::fmax((ldouble_safe)1, cnt)) - s;

                    if (all_perm)
                    {
                        size_t cnt_left, cnt_right;
                        double s_left, s_right;
                        size_t ncat_present = (size_t)ncat - st_pos;
                        size_t ncomb = pow2(ncat_present) - 1;
                        size_t best_combin;

                        for (size_t combin = 1; combin < ncomb; combin++)
                        {
                            cnt_left = 0; cnt_right = 0;
                            s_left   = 0;   s_right = 0;
                            for (size_t pos = st_pos; (int)pos < ncat; pos++)
                            {
                                if (extract_bit(combin, pos))
                                {
                                    cnt_left += buffer_cnt[buffer_pos[pos]];
                                    s_left   += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                                 (ldouble_safe)0
                                                    :
                                                 ((ldouble_safe) buffer_cnt[buffer_pos[pos]]
                                                          * std::log((ldouble_safe) buffer_cnt[buffer_pos[pos]]));
                                }

                                else
                                {
                                    cnt_right += buffer_cnt[buffer_pos[pos]];
                                    s_right   += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                                  (ldouble_safe)0
                                                     :
                                                  ((ldouble_safe) buffer_cnt[buffer_pos[pos]]
                                                           * std::log((ldouble_safe) buffer_cnt[buffer_pos[pos]]));
                                }
                            }

                            this_gain  = categ_gain<size_t, ldouble_safe>(
                                                    cnt_left, cnt_right,
                                                    s_left, s_right,
                                                    base_info, cnt);

                            if (this_gain > min_gain && this_gain > best_gain)
                            {
                                best_gain = this_gain;
                                best_combin = combin;
                            }

                        }

                        if (best_gain > min_gain)
                            for (size_t pos = 0; pos < ncat_present; pos++)
                                split_categ[buffer_pos[st_pos + pos]] = extract_bit(best_combin, pos);

                    }

                    else
                    {
                        /* try moving the categories one at a time */
                        size_t cnt_left = 0;
                        size_t cnt_right = end - st + 1;
                        double s_left = 0;
                        double s_right = s;

                        for (size_t pos = st_pos; pos < (ncat - st_pos - 1); pos++)
                        {
                            buffer_split[buffer_pos[pos]] = 1;
                            s_left    += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                          (ldouble_safe)0
                                             :
                                          ((ldouble_safe)buffer_cnt[buffer_pos[pos]] * std::log((ldouble_safe)   buffer_cnt[buffer_pos[pos]]));
                            s_right   -= (buffer_cnt[buffer_pos[pos]] <= 1)?
                                          (ldouble_safe)0
                                             :
                                          ((ldouble_safe)buffer_cnt[buffer_pos[pos]] * std::log((ldouble_safe)   buffer_cnt[buffer_pos[pos]]));
                            cnt_left  += buffer_cnt[buffer_pos[pos]];
                            cnt_right -= buffer_cnt[buffer_pos[pos]];

                            this_gain  = categ_gain<size_t, ldouble_safe>(
                                                    cnt_left, cnt_right,
                                                    s_left, s_right,
                                                    base_info, cnt);

                            if (this_gain > min_gain && this_gain > best_gain)
                            {
                                best_gain = this_gain;
                                memcpy(split_categ, buffer_split, ncat * sizeof(signed char));
                            }
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
    }

    if (st == (end-1)) return 0;

    if (best_gain <= -HUGE_VAL && this_gain <= min_gain && this_gain > -HUGE_VAL)
        return 0;
    else
        return best_gain;
}
