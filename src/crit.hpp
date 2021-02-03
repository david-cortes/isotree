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

#define pw1(x) ((x))
#define pw2(x) ((x) * (x))
#define pw3(x) ((x) * (x) * (x))
#define pw4(x) ((x) * (x) * (x) * (x))

/* https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Higher-order_statistics */
template <class real_t>
double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, real_t x[], MissingAction missing_action)
{
    long double m = 0;
    long double M2 = 0, M3 = 0, M4 = 0;
    long double delta, delta_s, delta_div;
    long double diff, n;
    long double out;

    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
        {
            n  =  (long double)(row - st + 1);

            delta      =  x[ix_arr[row]] - m;
            delta_div  =  delta / n;
            delta_s    =  delta_div * delta_div;
            diff       =  delta * (delta_div * (long double)(row - st));

            m   +=  delta_div;
            M4  +=  diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
            M3  +=  diff * delta_div * (n - 2) - 3 * delta_div * M2;
            M2  +=  diff;
        }

        out = ( M4 / M2 ) * ( (long double)(end - st + 1) / M2 );
        return (!is_na_or_inf(out) && out > 0.)? out : 0.;
    }

    else
    {
        size_t cnt = 0;
        for (size_t row = st; row <= end; row++)
        {
            if (!is_na_or_inf(x[ix_arr[row]]))
            {
                cnt++;
                n = (long double) cnt;

                delta      =  x[ix_arr[row]] - m;
                delta_div  =  delta / n;
                delta_s    =  delta_div * delta_div;
                diff       =  delta * (delta_div * (long double)(cnt - 1));

                m   +=  delta_div;
                M4  +=  diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
                M3  +=  diff * delta_div * (n - 2) - 3 * delta_div * M2;
                M2  +=  diff;
            }
        }

        out = ( M4 / M2 ) * ( (long double)cnt / M2 );
        return (!is_na_or_inf(out) && out > 0.)? out : 0.;
    }
}

/* TODO: is this algorithm correct? */
template <class real_t, class mapping>
double calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, real_t x[],
                              MissingAction missing_action, mapping w)
{
    long double m = 0;
    long double M2 = 0, M3 = 0, M4 = 0;
    long double delta, delta_s, delta_div;
    long double diff;
    long double n = 0;
    long double out;
    long double n_prev = 0.;
    double w_this;

    for (size_t row = st; row <= end; row++)
    {
        if (!is_na_or_inf(x[ix_arr[row]]))
        {
            w_this = w[ix_arr[row]];
            n += w_this;

            delta      =  x[ix_arr[row]] - m;
            delta_div  =  delta / n;
            delta_s    =  delta_div * delta_div;
            diff       =  delta * (delta_div * n_prev);
            n_prev   =  n;

            m   +=  w_this * (delta_div);
            M4  +=  w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
            M3  +=  w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
            M2  +=  w_this * (diff);
        }
    }

    out = ( M4 / M2 ) * ( n / M2 );
    return (!is_na_or_inf(out) && out > 0.)? out : 0.;
}


/* TODO: make these compensated sums */
/* TODO: can this use the same algorithm as above but with a correction at the end,
   like it was done for the variance? */
template <class real_t, class sparse_ix>
double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     MissingAction missing_action)
{
    /* ix_arr must be already sorted beforehand */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return 0;

    long double s1 = 0;
    long double s2 = 0;
    long double s3 = 0;
    long double s4 = 0;
    size_t cnt = end - st + 1;

    if (cnt <= 1) return 0;
    
    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);

    if (missing_action != Fail)
    {
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
                    s1 += pw1(Xc[curr_pos]);
                    s2 += pw2(Xc[curr_pos]);
                    s3 += pw3(Xc[curr_pos]);
                    s4 += pw4(Xc[curr_pos]);
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
    }

    else
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == *row)
            {
                s1 += pw1(Xc[curr_pos]);
                s2 += pw2(Xc[curr_pos]);
                s3 += pw3(Xc[curr_pos]);
                s4 += pw4(Xc[curr_pos]);

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

    if (cnt <= 1 || s2 == 0 || s2 == pw2(s1)) return 0;
    long double cnt_l = (long double) cnt;
    long double sn = s1 / cnt_l;
    long double v  = s2 / cnt_l - pw2(sn);
    if (v <= 0) return 0.;
    long double out =  (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
    return (!is_na_or_inf(out) && out > 0.)? out : 0.;
}


template <class real_t, class sparse_ix, class mapping>
double calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                              real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                              MissingAction missing_action, mapping w)
{
    /* ix_arr must be already sorted beforehand */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
        return 0;

    long double s1 = 0;
    long double s2 = 0;
    long double s3 = 0;
    long double s4 = 0;
    double w_this;
    long double cnt = 0.;
    for (size_t row = st; row <= end; row++)
        cnt += w[ix_arr[row]];

    if (cnt <= 1) return 0;
    
    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);

    if (missing_action != Fail)
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == *row)
            {
                w_this = w[*row];

                if (is_na_or_inf(Xc[curr_pos]))
                {
                    cnt -= w_this;
                }

                else
                {
                    s1 += w_this * pw1(Xc[curr_pos]);
                    s2 += w_this * pw2(Xc[curr_pos]);
                    s3 += w_this * pw3(Xc[curr_pos]);
                    s4 += w_this * pw4(Xc[curr_pos]);
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
    }

    else
    {
        for (size_t *row = ptr_st;
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == *row)
            {
                s1 += pw1(Xc[curr_pos]);
                s2 += pw2(Xc[curr_pos]);
                s3 += pw3(Xc[curr_pos]);
                s4 += pw4(Xc[curr_pos]);

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

    if (cnt <= 1 || s2 == 0 || s2 == pw2(s1)) return 0;
    long double sn = s1 / cnt;
    long double v  = s2 / cnt - pw2(sn);
    if (v <= 0) return 0.;
    long double out =  (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt * pw4(sn)) / (cnt * pw2(v));
    return (!is_na_or_inf(out) && out > 0.)? out : 0.;
}



double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, int x[], int ncat, size_t buffer_cnt[], double buffer_prob[],
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
    double sum_kurt = 0;

    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
            buffer_cnt[x[ix_arr[row]]]++;
    }

    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] >= 0)
                buffer_cnt[x[ix_arr[row]]]++;
            else
                buffer_cnt[ncat]++;
        }
    }

    cnt -= buffer_cnt[ncat];
    if (cnt <= 1) return 0;
    long double cnt_l = (long double) cnt;
    for (int cat = 0; cat < ncat; cat++)
        buffer_prob[cat] = buffer_cnt[cat] / cnt_l;

    switch(cat_split_type)
    {
        case SubSet:
        {
            long double temp_v;
            long double s1, s2, s3, s4;
            long double coef;
            std::uniform_real_distribution<double> runif(0, 1);
            size_t ntry = 50;
            for (size_t iternum = 0; iternum < 50; iternum++)
            {
                s1 = 0; s2 = 0; s3 = 0; s4 = 0;
                for (int cat = 0; cat < ncat; cat++)
                {
                    coef = runif(rnd_generator);
                    s1 += buffer_prob[cat] * pw1(coef);
                    s2 += buffer_prob[cat] * pw2(coef);
                    s3 += buffer_prob[cat] * pw3(coef);
                    s4 += buffer_prob[cat] * pw4(coef);
                }
                temp_v = s2 - pw2(s1);
                if (temp_v <= 0)
                    ntry--;
                else
                    sum_kurt += (s4 - 4 * s3 * pw1(s1) + 6 * s2 * pw2(s1) - 4 * s1 * pw3(s1) + pw4(s1)) / pw2(temp_v);
            }
            if (!ntry)
                return 0;
            else
                return ((!is_na_or_inf(sum_kurt) && sum_kurt > 0.)? sum_kurt : 0.) / (long double)ntry;
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
                return 0;
            else
                return ((!is_na_or_inf(sum_kurt) && sum_kurt > 0.)? sum_kurt : 0.) / (double) ncat_present;
        }
    }

    return -1; /* this will never be reached, but CRAN complains otherwise */
}


/* TODO: this one should get a buffer preallocated from outside */
template <class mapping>
double calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, int x[], int ncat, double buffer_prob[],
                              MissingAction missing_action, CategSplit cat_split_type, RNG_engine &rnd_generator, mapping w)
{
    long double cnt = 0.;
    std::vector<long double> buffer_cnt(ncat+1, 0.);
    double sum_kurt = 0;
    double w_this;

    for (size_t row = st; row <= end; row++)
    {
        w_this = w[ix_arr[row]];
        if (x[ix_arr[row]] >= 0)
            buffer_cnt[x[ix_arr[row]]] += w_this;
        else
            buffer_cnt[ncat] += w_this;
    }
    cnt = std::accumulate(buffer_cnt.begin(), buffer_cnt.end(), (long double)0.);

    cnt -= buffer_cnt[ncat];
    if (cnt <= 1) return 0;
    for (int cat = 0; cat < ncat; cat++)
        buffer_prob[cat] = buffer_cnt[cat] / cnt;

    switch(cat_split_type)
    {
        case SubSet:
        {
            long double temp_v;
            long double s1, s2, s3, s4;
            long double coef;
            std::uniform_real_distribution<double> runif(0, 1);
            size_t ntry = 50;
            for (size_t iternum = 0; iternum < 50; iternum++)
            {
                s1 = 0; s2 = 0; s3 = 0; s4 = 0;
                for (int cat = 0; cat < ncat; cat++)
                {
                    coef = runif(rnd_generator);
                    s1 += buffer_prob[cat] * pw1(coef);
                    s2 += buffer_prob[cat] * pw2(coef);
                    s3 += buffer_prob[cat] * pw3(coef);
                    s4 += buffer_prob[cat] * pw4(coef);
                }
                temp_v = s2 - pw2(s1);
                if (temp_v <= 0)
                    ntry--;
                else
                    sum_kurt += (s4 - 4 * s3 * pw1(s1) + 6 * s2 * pw2(s1) - 4 * s1 * pw3(s1) + pw4(s1)) / pw2(temp_v);
            }
            if (!ntry)
                return 0;
            else
                return ((!is_na_or_inf(sum_kurt) && sum_kurt > 0.)? sum_kurt : 0.) / (long double)ntry;
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
                return 0;
            else
                return ((!is_na_or_inf(sum_kurt) && sum_kurt > 0.)? sum_kurt : 0.) / (double) ncat_present;
        }
    }

    return -1; /* this will never be reached, but CRAN complains otherwise */
}


double expected_sd_cat(double p[], size_t n, size_t pos[])
{
    if (n <= 1) return 0;

    long double cum_var = -square(p[pos[0]]) / 3.0 - p[pos[0]] * p[pos[1]] / 2.0 + p[pos[0]] / 3.0  - square(p[pos[1]]) / 3.0 + p[pos[1]] / 3.0;
    for (size_t cat1 = 2; cat1 < n; cat1++)
    {
        cum_var += p[pos[cat1]] / 3.0 - square(p[pos[cat1]]) / 3.0;
        for (size_t cat2 = 0; cat2 < cat1; cat2++)
            cum_var -= p[pos[cat1]] * p[pos[cat2]] / 2.0;
    }
    return std::sqrt(std::fmax(cum_var, 0.));
}

template <class number>
double expected_sd_cat(number counts[], double p[], size_t n, size_t pos[])
{
    if (n <= 1) return 0;

    number tot = std::accumulate(pos, pos + n, (number)0, [&counts](number tot, const size_t ix){return tot + counts[ix];});
    long double cnt_div = (long double) tot;
    for (size_t cat = 0; cat < n; cat++)
        p[pos[cat]] = (long double)counts[pos[cat]] / cnt_div;

    return expected_sd_cat(p, n, pos);
}

template <class number>
double expected_sd_cat_single(number counts[], double p[], size_t n, size_t pos[], size_t cat_exclude, number cnt)
{
    if (cat_exclude == 0)
        return expected_sd_cat(counts, p, n-1, pos + 1);

    else if (cat_exclude == (n-1))
        return expected_sd_cat(counts, p, n-1, pos);

    size_t ix_exclude = pos[cat_exclude];

    long double cnt_div = (long double) (cnt - counts[ix_exclude]);
    for (size_t cat = 0; cat < n; cat++)
        p[pos[cat]] = (long double)counts[pos[cat]] / cnt_div;

    double cum_var;
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
    return std::sqrt(std::fmax(cum_var, 0.));
}

/* Note: this isn't exactly comparable to the pooled gain from numeric variables,
   but among all the possible options, this is what happens to end up in the most
   similar scale when considering standardized gain. */
template <class number>
double categ_gain(number cnt_left, number cnt_right,
                  long double s_left, long double s_right,
                  long double base_info, long double cnt)
{
    return (
            base_info -
            (((cnt_left  <= 1)? 0 : ((long double)cnt_left  * logl((long double)cnt_left)))  - s_left) -
            (((cnt_right <= 1)? 0 : ((long double)cnt_right * logl((long double)cnt_right))) - s_right)
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


#define avg_between(a, b) ((a) + ((b)-(a))/2.)
#define sd_gain(sd, sd_left, sd_right) (1. - ((sd_left) + (sd_right)) / (2. * (sd)))
#define pooled_gain(sd, cnt, sd_left, sd_right, cnt_left, cnt_right) \
    (1. - (1./(sd))*(  ( ((real_t)(cnt_left))/(cnt) )*(sd_left) + ( ((real_t)(cnt_right)/(cnt)) )*(sd_right)  ))


/* TODO: these functions would not take into account observation weights if available.
   Need to create a 'weighted' version of each. */


/* TODO: make this a compensated sum */
template <class real_t, class real_t_>
double find_split_rel_gain_t(real_t_ *restrict x, size_t n, double &split_point)
{
    real_t this_gain;
    real_t best_gain = -HUGE_VAL;
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
            split_point = avg_between(x[row], x[row+1]);
        }
    }
    if (best_gain <= -HUGE_VAL)
        return best_gain;
    else
        return std::fmax((double)best_gain, std::numeric_limits<double>::epsilon());
}

template <class real_t_>
double find_split_rel_gain(real_t_ *restrict x, size_t n, double &split_point)
{
    if (n < THRESHOLD_LONG_DOUBLE)
        return find_split_rel_gain_t<double, real_t_>(x, n, split_point);
    else
        return find_split_rel_gain_t<long double, real_t_>(x, n, split_point);
}

template <class real_t, class real_t_>
double find_split_rel_gain_t(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double &split_point, size_t &split_ix)
{
    real_t this_gain;
    real_t best_gain = -HUGE_VAL;
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
            split_point = avg_between(x[ix_arr[row]], x[ix_arr[row+1]]);
            split_ix = row;
        }
    }
    if (best_gain <= -HUGE_VAL)
        return best_gain;
    else
        return std::fmax((double)best_gain, std::numeric_limits<double>::epsilon());
}

template <class real_t_>
double find_split_rel_gain(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double &split_point, size_t &split_ix)
{
    if ((end-st+1) < THRESHOLD_LONG_DOUBLE)
        return find_split_rel_gain_t<double, real_t_>(x, xmean, ix_arr, st, end, split_point, split_ix);
    else
        return find_split_rel_gain_t<long double, real_t_>(x, xmean, ix_arr, st, end, split_point, split_ix);
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

template <class real_t_>
long double calc_sd_right_to_left_weighted(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                                           double *restrict w, long double &cumw, size_t *restrict sorted_ix)
{
    long double running_mean = 0;
    long double running_ssq = 0;
    long double mean_prev = x[sorted_ix[n-1]];
    long double cnt = 0;
    double w_this;
    for (size_t row = 0; row < n-1; row++)
    {
        w_this = w[sorted_ix[n-row-1]];
        cnt += w_this;
        running_mean   += (x[sorted_ix[n-row-1]] - running_mean) / cnt;
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

template <class real_t_, class mapping>
long double calc_sd_right_to_left_weighted(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end,
                                           double *restrict sd_arr, mapping w, long double &cumw)
{
    long double running_mean = 0;
    long double running_ssq = 0;
    real_t_ mean_prev = x[ix_arr[end]] - xmean;
    size_t n = end - st + 1;
    long double cnt = 0;
    double w_this;
    for (size_t row = 0; row < n-1; row++)
    {
        w_this = w[ix_arr[end-row]];
        cnt += w_this;
        running_mean   += ((x[ix_arr[end-row]] - xmean) - running_mean) / cnt;
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
                             GainCriterion criterion, double min_gain, double &split_point)
{
    real_t full_sd = calc_sd_right_to_left<real_t>(x, n, sd_arr);
    real_t running_mean = 0;
    real_t running_ssq = 0;
    real_t mean_prev = x[0];
    real_t best_gain = -HUGE_VAL;
    real_t this_sd, this_gain;
    real_t n_ = (real_t)n;
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
            split_point = avg_between(x[row], x[row+1]);
        }
    }
    return best_gain;
}

template <class real_t_>
double find_split_std_gain(real_t_ *restrict x, size_t n, double *restrict sd_arr,
                           GainCriterion criterion, double min_gain, double &split_point)
{
    if (n < THRESHOLD_LONG_DOUBLE)
        return find_split_std_gain_t<double, real_t_>(x, n, sd_arr, criterion, min_gain, split_point);
    else
        return find_split_std_gain_t<long double, real_t_>(x, n, sd_arr, criterion, min_gain, split_point);
}

template <class real_t>
double find_split_std_gain_weighted(real_t *restrict x, size_t n, double *restrict sd_arr,
                                    GainCriterion criterion, double min_gain, double &split_point,
                                    double *restrict w, size_t *restrict sorted_ix)
{
    long double cumw;
    double full_sd = calc_sd_right_to_left_weighted(x, n, sd_arr, w, cumw, sorted_ix);
    long double running_mean = 0;
    long double running_ssq = 0;
    long double mean_prev = x[sorted_ix[0]];
    double best_gain = -HUGE_VAL;
    double this_sd, this_gain;
    double w_this;
    long double currw = 0;

    for (size_t row = 0; row < n-1; row++)
    {
        w_this = w[row];
        currw += w_this;
        running_mean   += (x[sorted_ix[row]] - running_mean) / currw;
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
            split_point = avg_between(x[sorted_ix[row]], x[sorted_ix[row+1]]);
        }
    }
    return best_gain;
}

template <class real_t, class real_t_>
double find_split_std_gain_t(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                             GainCriterion criterion, double min_gain, double &split_point, size_t &split_ix)
{
    real_t full_sd = calc_sd_right_to_left<real_t>(x, xmean, ix_arr, st, end, sd_arr);
    real_t running_mean = 0;
    real_t running_ssq = 0;
    real_t mean_prev = x[ix_arr[st]] - xmean;
    real_t best_gain = -HUGE_VAL;
    real_t n = (real_t)(end - st + 1);
    real_t this_sd, this_gain;
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
            split_point = avg_between(x[ix_arr[row]], x[ix_arr[row+1]]);
            split_ix = row;
        }
    }
    return best_gain;
}

template <class real_t_>
double find_split_std_gain(real_t_ *restrict x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                           GainCriterion criterion, double min_gain, double &split_point, size_t &split_ix)
{
    if ((end-st+1) < THRESHOLD_LONG_DOUBLE)
        return find_split_std_gain_t<double, real_t_>(x, xmean, ix_arr, st, end, sd_arr, criterion, min_gain, split_point, split_ix);
    else
        return find_split_std_gain_t<long double, real_t_>(x, xmean, ix_arr, st, end, sd_arr, criterion, min_gain, split_point, split_ix);
}

template <class real_t, class mapping>
double find_split_std_gain_weighted(real_t *restrict x, real_t xmean, size_t ix_arr[], size_t st, size_t end, double *restrict sd_arr,
                                    GainCriterion criterion, double min_gain, double &split_point, size_t &split_ix, mapping w)
{
    long double cumw;
    double full_sd = calc_sd_right_to_left_weighted(x, xmean, ix_arr, st, end, sd_arr, w, cumw);
    long double running_mean = 0;
    long double running_ssq = 0;
    long double mean_prev = x[ix_arr[st]] - xmean;
    double best_gain = -HUGE_VAL;
    long double currw = 0;
    double this_sd, this_gain;
    double w_this;
    for (size_t row = st; row < end; row++)
    {
        w_this = w[ix_arr[row]];
        currw += w_this;
        running_mean   += ((x[ix_arr[row]] - xmean) - running_mean) / currw;
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
            split_point = avg_between(x[ix_arr[row]], x[ix_arr[row+1]]);
            split_ix = row;
        }
    }
    return best_gain;
}


/* for split-criterion in hyperplanes (see below for version aimed at single-variable splits) */
double eval_guided_crit(double *restrict x, size_t n, GainCriterion criterion,
                        double min_gain, bool as_relative_gain, double *restrict buffer_sd,
                        double &split_point, double &xmin, double &xmax)
{
    /* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
       all numbers are assumed to be small and in the same scale */
    double gain;

    /* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
    if (n == 2)
    {
        if (x[0] == x[1]) return -HUGE_VAL;
        split_point = avg_between(x[0], x[1]);
        gain        = 1.;
        if (gain > min_gain)
            return gain;
        else
            return 0.;
    }

    /* sort in ascending order */
    std::sort(x, x + n);
    xmin = x[0]; xmax = x[n-1];
    if (x[0] == x[n-1]) return -HUGE_VAL;

    if (criterion == Pooled && as_relative_gain && min_gain <= 0)
        gain = find_split_rel_gain(x, n, split_point);
    else
        gain = find_split_std_gain(x, n, buffer_sd, criterion, min_gain, split_point);
    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

double eval_guided_crit_weighted(double *restrict x, size_t n, GainCriterion criterion,
                                 double min_gain, bool as_relative_gain, double *restrict buffer_sd,
                                 double &split_point, double &xmin, double &xmax,
                                 double *restrict w, size_t *restrict buffer_indices)
{
    /* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
       all numbers are assumed to be small and in the same scale */
    double gain;

    /* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
    if (n == 2)
    {
        if (x[0] == x[1]) return -HUGE_VAL;
        split_point = avg_between(x[0], x[1]);
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
    if (x[0] == x[n-1]) return -HUGE_VAL;

    gain = find_split_std_gain_weighted(x, n, buffer_sd, criterion, min_gain, split_point, w, buffer_indices);
    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

/* for split-criterion in single-variable splits */
template <class real_t_>
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, real_t_ *restrict x,
                        double *restrict buffer_sd, bool as_relative_gain,
                        size_t &split_ix, double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action)
{
    double gain;

    /* move NAs to the front if there's any, exclude them from calculations */
    if (missing_action != Fail)
        st = move_NAs_to_front(ix_arr, st, end, x);

    if (st >= end) return -HUGE_VAL;
    else if (st == (end-1))
    {
        if (x[ix_arr[st]] == x[ix_arr[end]])
            return -HUGE_VAL;
        split_point = avg_between(x[ix_arr[st]], x[ix_arr[end]]);
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
    for (size_t ix = st; ix <= end; ix++)
        xmean += x[ix_arr[ix]];
    xmean /= (real_t_)(end - st + 1);

    if (criterion == Pooled && as_relative_gain && min_gain <= 0)
        gain = find_split_rel_gain(x, xmean, ix_arr, st, end, split_point, split_ix);
    else
        gain = find_split_std_gain(x, xmean, ix_arr, st, end, buffer_sd, criterion, min_gain, split_point, split_ix);
    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

template <class real_t_, class mapping>
double eval_guided_crit_weighted(size_t *restrict ix_arr, size_t st, size_t end, real_t_ *restrict x,
                                 double *restrict buffer_sd, bool as_relative_gain,
                                 size_t &split_ix, double &split_point, double &xmin, double &xmax,
                                 GainCriterion criterion, double min_gain, MissingAction missing_action,
                                 mapping w)
{
    double gain;

    /* move NAs to the front if there's any, exclude them from calculations */
    if (missing_action != Fail)
        st = move_NAs_to_front(ix_arr, st, end, x);

    if (st >= end) return -HUGE_VAL;
    else if (st == (end-1))
    {
        if (x[ix_arr[st]] == x[ix_arr[end]])
            return -HUGE_VAL;
        split_point = avg_between(x[ix_arr[st]], x[ix_arr[end]]);
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
    for (size_t ix = st; ix <= end; ix++)
    {
        xmean += x[ix_arr[ix]];
        cnt += w[ix_arr[ix]];
    }
    xmean /= cnt;

    gain = find_split_std_gain_weighted(x, xmean, ix_arr, st, end, buffer_sd, criterion, min_gain, split_point, split_ix, w);
    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
    return std::fmax(0., gain);
}

/* TODO: here it should only need to look at the non-zero entries. It can then use the
   same algorithm as above, but putting an extra check to see where do the zeros fit in
   the sorted order of the non-zero entries while calculating gains and SDs, and then
   call the 'divide_subset' function after-the-fact to reach the same end result.
   It should be much faster than this if the non-zero entries are few. */
template <class real_t_, class sparse_ix>
double eval_guided_crit(size_t ix_arr[], size_t st, size_t end,
                        size_t col_num, real_t_ Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                        double buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
                        double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action)
{
    todense(ix_arr, st, end,
            col_num, Xc, Xc_ind, Xc_indptr,
            buffer_arr);
    std::iota(buffer_pos, buffer_pos + (end - st + 1), (size_t)0);
    size_t ignored;
    return eval_guided_crit(buffer_pos, 0, end - st, buffer_arr, buffer_arr + (end-st+1),
                            as_relative_gain, ignored, split_point,
                            xmin, xmax, criterion, min_gain, missing_action);
}

template <class real_t_, class sparse_ix, class mapping>
double eval_guided_crit_weighted(size_t ix_arr[], size_t st, size_t end,
                                 size_t col_num, real_t_ Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                                 double buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
                                 double &split_point, double &xmin, double &xmax,
                                 GainCriterion criterion, double min_gain, MissingAction missing_action,
                                 mapping w)
{
    todense(ix_arr, st, end,
            col_num, Xc, Xc_ind, Xc_indptr,
            buffer_arr);
    std::iota(buffer_pos, buffer_pos + (end - st + 1), (size_t)0);
    /* TODO: allocate this buffer externally */
    std::vector<double> buffer_w(end-st+1);
    for (size_t row = st; row <= end; row++)
        buffer_w[row-st] = w[ix_arr[row]];
    size_t ignored;
    return eval_guided_crit_weighted(buffer_pos, 0, end - st, buffer_arr, buffer_arr + (end-st+1),
                                     as_relative_gain, ignored, split_point,
                                     xmin, xmax, criterion, min_gain, missing_action, buffer_w);
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
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int ncat,
                        size_t *restrict buffer_cnt, size_t *restrict buffer_pos, double *restrict buffer_prob,
                        int &chosen_cat, char *restrict split_categ, char *restrict buffer_split,
                        GainCriterion criterion, double min_gain, bool all_perm,
                        MissingAction missing_action, CategSplit cat_split_type)
{
    /* move NAs to the front if there's any, exclude them from calculations */
    if (missing_action != Fail)
        st = move_NAs_to_front(ix_arr, st, end, x);

    if (st >= end) return -HUGE_VAL;

    /* count categories */
    memset(buffer_cnt, 0, sizeof(size_t) * ncat);
    for (size_t row = st; row <= end; row++)
        buffer_cnt[x[ix_arr[row]]]++;

    double this_gain = -HUGE_VAL;
    double best_gain = -HUGE_VAL;
    std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
    size_t st_pos = 0;

    switch(cat_split_type)
    {
        case SingleCateg:
        {
            size_t cnt = end - st + 1;
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
                            buffer_prob[cat] = (long double) buffer_cnt[cat] / (long double) cnt;
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

                    double sd_full = expected_sd_cat(buffer_prob, ncat_present, buffer_pos + st_pos);

                    /* try isolating each category one at a time */
                    for (size_t pos = st_pos; (int)pos < ncat; pos++)
                    {
                        this_gain = sd_gain(sd_full,
                                            0.0,
                                            expected_sd_cat_single(buffer_cnt, buffer_prob, ncat_present, buffer_pos + st_pos, pos - st_pos, cnt)
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

                    long double cnt_left = (long double)((end - st + 1) - cnt_max);
                    this_gain = (
                                    (long double)cnt * logl((long double)cnt)
                                        - cnt_left * logl(cnt_left)
                                        - (long double)cnt_max * logl((long double)cnt_max)
                                ) / cnt;
                    best_gain = (this_gain > min_gain)? this_gain : best_gain;
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
            memset(buffer_split, 0, ncat * sizeof(char));

            long double cnt = (long double)(end - st + 1);

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
                            buffer_prob[buffer_pos[cat]] = (long double)buffer_cnt[buffer_pos[cat]] / cnt;
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
                    sd_full = expected_sd_cat(buffer_prob, ncat_present, buffer_pos + st_pos);
                    if (ncat_present >= log2ceil(SIZE_MAX)) all_perm = false;

                    /* move categories one at a time */
                    for (size_t pos = st_pos; pos < ((size_t)ncat - st_pos - 1); pos++)
                    {
                        buffer_split[buffer_pos[pos]] = 1;
                        this_gain = sd_gain(sd_full,
                                            expected_sd_cat(buffer_cnt, buffer_prob, pos - st_pos + 1,       buffer_pos + st_pos),
                                            expected_sd_cat(buffer_cnt, buffer_prob, (size_t)ncat - pos - 1, buffer_pos + pos + 1)
                                            );
                        if (this_gain > min_gain && this_gain > best_gain)
                        {
                            best_gain = this_gain;
                            memcpy(split_categ, buffer_split, ncat * sizeof(char));
                        }
                    }

                    break;
                }

                case Pooled:
                {
                    long double s = 0;

                    /* determine first non-zero and get base info */
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[buffer_pos[cat]])
                        {
                            s += (buffer_cnt[buffer_pos[cat]] <= 1)?
                                  0 : ((long double) buffer_cnt[buffer_pos[cat]] * logl((long double)buffer_cnt[buffer_pos[cat]]));
                        }

                        else
                        {
                            buffer_split[buffer_pos[cat]] = -1;
                            st_pos++;
                        }
                    }

                    if ((int)st_pos >= (ncat-1)) return -HUGE_VAL;

                    /* calculate base info */
                    long double base_info = cnt * logl(cnt) - s;

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
                                                 0 : ((long double) buffer_cnt[buffer_pos[pos]]
                                                       * logl((long double) buffer_cnt[buffer_pos[pos]]));
                                }

                                else
                                {
                                    cnt_right += buffer_cnt[buffer_pos[pos]];
                                    s_right   += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                                  0 : ((long double) buffer_cnt[buffer_pos[pos]]
                                                        * logl((long double) buffer_cnt[buffer_pos[pos]]));
                                }
                            }

                            this_gain  = categ_gain(cnt_left, cnt_right,
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
                                          0 : ((long double)buffer_cnt[buffer_pos[pos]] * logl((long double)buffer_cnt[buffer_pos[pos]]));
                            s_right   -= (buffer_cnt[buffer_pos[pos]] <= 1)?
                                          0 : ((long double)buffer_cnt[buffer_pos[pos]] * logl((long double)buffer_cnt[buffer_pos[pos]]));
                            cnt_left  += buffer_cnt[buffer_pos[pos]];
                            cnt_right -= buffer_cnt[buffer_pos[pos]];

                            this_gain  = categ_gain(cnt_left, cnt_right,
                                                    s_left, s_right,
                                                    base_info, cnt);

                            if (this_gain > min_gain && this_gain > best_gain)
                            {
                                best_gain = this_gain;
                                memcpy(split_categ, buffer_split, ncat * sizeof(char));
                            }
                        }
                    }

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


template <class mapping>
double eval_guided_crit_weighted(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int ncat,
                                 size_t *restrict buffer_pos, double *restrict buffer_prob,
                                 int &chosen_cat, char *restrict split_categ, char *restrict buffer_split,
                                 GainCriterion criterion, double min_gain, bool all_perm,
                                 MissingAction missing_action, CategSplit cat_split_type,
                                 mapping w)
{
    /* move NAs to the front if there's any, exclude them from calculations */
    if (missing_action != Fail)
        st = move_NAs_to_front(ix_arr, st, end, x);

    if (st >= end) return -HUGE_VAL;

    /* count categories */
    /* TODO: allocate this buffer externally */
    std::vector<long double> buffer_cnt(ncat, 0.);
    for (size_t row = st; row <= end; row++)
        buffer_cnt[x[ix_arr[row]]] += w[ix_arr[row]];
    long double cnt = std::accumulate(buffer_cnt.begin(), buffer_cnt.end(), (long double)0);

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

                    double sd_full = expected_sd_cat(buffer_prob, ncat_present, buffer_pos + st_pos);

                    /* try isolating each category one at a time */
                    for (size_t pos = st_pos; (int)pos < ncat; pos++)
                    {
                        this_gain = sd_gain(sd_full,
                                            0.0,
                                            expected_sd_cat_single(buffer_cnt.data(), buffer_prob, ncat_present, buffer_pos + st_pos, pos - st_pos, cnt)
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
                    long double cnt_max = 0;
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

                    long double cnt_left = (long double)(cnt - cnt_max);

                    /* TODO: think of a better way of dealing with numbers between zero and one */
                    this_gain = (
                                    std::fmax(1., cnt) * logl(std::fmax(1., cnt))
                                        - std::fmax(1., cnt_left) * logl(std::fmax(1., cnt_left))
                                        - std::fmax(1., cnt_max) * logl(std::fmax(1., cnt_max))
                                ) / std::fmax(1., cnt);
                    best_gain = (this_gain > min_gain)? this_gain : best_gain;
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
            memset(buffer_split, 0, ncat * sizeof(char));


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
                            buffer_prob[buffer_pos[cat]] = (long double)buffer_cnt[buffer_pos[cat]] / cnt;
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
                    sd_full = expected_sd_cat(buffer_prob, ncat_present, buffer_pos + st_pos);
                    if (ncat_present >= log2ceil(SIZE_MAX)) all_perm = false;

                    /* move categories one at a time */
                    for (size_t pos = st_pos; pos < ((size_t)ncat - st_pos - 1); pos++)
                    {
                        buffer_split[buffer_pos[pos]] = 1;
                        /* TODO: is this correct? */
                        this_gain = sd_gain(sd_full,
                                            expected_sd_cat(buffer_cnt.data(), buffer_prob, pos - st_pos + 1,       buffer_pos + st_pos),
                                            expected_sd_cat(buffer_cnt.data(), buffer_prob, (size_t)ncat - pos - 1, buffer_pos + pos + 1)
                                            );
                        if (this_gain > min_gain && this_gain > best_gain)
                        {
                            best_gain = this_gain;
                            memcpy(split_categ, buffer_split, ncat * sizeof(char));
                        }
                    }

                    break;
                }

                case Pooled:
                {
                    long double s = 0;

                    /* determine first non-zero and get base info */
                    for (int cat = 0; cat < ncat; cat++)
                    {
                        if (buffer_cnt[buffer_pos[cat]])
                        {
                            s += (buffer_cnt[buffer_pos[cat]] <= 1)?
                                  0 : ((long double) buffer_cnt[buffer_pos[cat]] * logl((long double)buffer_cnt[buffer_pos[cat]]));
                        }

                        else
                        {
                            buffer_split[buffer_pos[cat]] = -1;
                            st_pos++;
                        }
                    }

                    if ((int)st_pos >= (ncat-1)) return -HUGE_VAL;

                    /* calculate base info */
                    long double base_info = std::fmax(1., cnt) * logl(std::fmax(1., cnt)) - s;

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
                                                 0 : ((long double) buffer_cnt[buffer_pos[pos]]
                                                       * logl((long double) buffer_cnt[buffer_pos[pos]]));
                                }

                                else
                                {
                                    cnt_right += buffer_cnt[buffer_pos[pos]];
                                    s_right   += (buffer_cnt[buffer_pos[pos]] <= 1)?
                                                  0 : ((long double) buffer_cnt[buffer_pos[pos]]
                                                        * logl((long double) buffer_cnt[buffer_pos[pos]]));
                                }
                            }

                            this_gain  = categ_gain(cnt_left, cnt_right,
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
                                          0 : ((long double)buffer_cnt[buffer_pos[pos]] * logl((long double)buffer_cnt[buffer_pos[pos]]));
                            s_right   -= (buffer_cnt[buffer_pos[pos]] <= 1)?
                                          0 : ((long double)buffer_cnt[buffer_pos[pos]] * logl((long double)buffer_cnt[buffer_pos[pos]]));
                            cnt_left  += buffer_cnt[buffer_pos[pos]];
                            cnt_right -= buffer_cnt[buffer_pos[pos]];

                            this_gain  = categ_gain(cnt_left, cnt_right,
                                                    s_left, s_right,
                                                    base_info, cnt);

                            if (this_gain > min_gain && this_gain > best_gain)
                            {
                                best_gain = this_gain;
                                memcpy(split_categ, buffer_split, ncat * sizeof(char));
                            }
                        }
                    }

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
