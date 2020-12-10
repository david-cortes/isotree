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

#define pw1(x) ((x))
#define pw2(x) ((x) * (x))
#define pw3(x) ((x) * (x) * (x))
#define pw4(x) ((x) * (x) * (x) * (x))

double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, double x[], MissingAction missing_action)
{
    long double m = 0;
    long double M2 = 0, M3 = 0, M4 = 0;
    long double delta, delta_s, delta_div;
    long double diff, n;

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

        return ( M4 / M2 ) * ( (long double)(end - st + 1) / M2 );
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

        return ( M4 / M2 ) * ( (long double)cnt / M2 );
    }
}


double calc_kurtosis(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                     double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
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
    if (v <= 0) return 0;
    return (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
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
                return sum_kurt / (long double)ntry;
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
                return sum_kurt / (double) ncat_present;
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
    return sqrt(fmax(cum_var, 1e-8));
}

double expected_sd_cat(size_t counts[], double p[], size_t n, size_t pos[])
{
    if (n <= 1) return 0;

    size_t tot = std::accumulate(pos, pos + n, (size_t)0, [&counts](size_t tot, const size_t ix){return tot + counts[ix];});
    long double cnt_div = (long double) tot;
    for (size_t cat = 0; cat < n; cat++)
        p[pos[cat]] = (long double)counts[pos[cat]] / cnt_div;

    return expected_sd_cat(p, n, pos);
}

double expected_sd_cat_single(size_t counts[], double p[], size_t n, size_t pos[], size_t cat_exclude, size_t cnt)
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
    return sqrt(fmax(cum_var, 1e-8));
}

double numeric_gain(size_t cnt_left, size_t cnt_right,
                    long double sum_left, long double sum_right,
                    long double sum_sq_left, long double sum_sq_right,
                    double sd_full, long double cnt)
{
    long double residual =
        (long double) cnt_left  * calc_sd_raw_l(cnt_left,  sum_left,  sum_sq_left) +
        (long double) cnt_right * calc_sd_raw_l(cnt_right, sum_right, sum_sq_right);
    return 1 - residual / (cnt * sd_full);
}

double numeric_gain_no_div(size_t cnt_left, size_t cnt_right,
                           long double sum_left, long double sum_right,
                           long double sum_sq_left, long double sum_sq_right,
                           double sd_full, long double cnt)
{
    long double residual =
        (long double) cnt_left  * calc_sd_raw_l(cnt_left,  sum_left,  sum_sq_left) +
        (long double) cnt_right * calc_sd_raw_l(cnt_right, sum_right, sum_sq_right);
    return sd_full - residual / cnt;
}

double categ_gain(size_t cnt_left, size_t cnt_right,
                  long double s_left, long double s_right,
                  long double base_info, long double cnt)
{
    return (
            base_info -
            (((cnt_left  <= 1)? 0 : ((long double)cnt_left  * logl((long double)cnt_left)))  - s_left) -
            (((cnt_right <= 1)? 0 : ((long double)cnt_right * logl((long double)cnt_right))) - s_right)
            ) / cnt;
}


#define avg_between(a, b) (((a) + (b)) / 2)
#define sd_gain(sd, sd_left, sd_right) (1.0 - ((sd_left) + (sd_right)) / (2.0 * (sd)))

/* for split-criterion in hyperplanes (see below for version aimed at single-variable splits) */
double eval_guided_crit(double *restrict x, size_t n, GainCriterion criterion, double min_gain,
                        double &split_point, double &xmin, double &xmax)
{
    /* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
       all numbers are assumed to be small and in the same scale */

    /* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
    if (n == 2)
    {
        split_point = avg_between(x[0], x[1]);
        return 0;
    }

    /* sort in ascending order */
    std::sort(x, x + n);
    if (x[0] == x[n-1]) return -HUGE_VAL;
    xmin = x[0]; xmax = x[n-1];

    /* compute sum - sum_sq - sd in one pass */
    long double sum = 0;
    long double sum_sq = 0;
    double sd_full;
    for (size_t row = 0; row < n; row++)
    {
        sum    += x[row];
        sum_sq += square(x[row]);
    }
    sd_full = calc_sd_raw(n, sum, sum_sq);

    /* try splits by moving observations one at a time from right to left */
    long double sum_left = 0;
    long double sum_sq_left = 0;
    long double sum_right = sum;
    long double sum_sq_right = sum_sq;
    double this_gain = -HUGE_VAL;
    double best_gain = -HUGE_VAL;

    switch(criterion)
    {
        case Averaged:
        {
            for (size_t row = 0; row < n-1; row++)
            {
                sum_left     += x[row];
                sum_sq_left  += square(x[row]);
                sum_right    -= x[row];
                sum_sq_right -= square(x[row]);

                if (x[row] == x[row + 1]) continue;

                this_gain = sd_gain(sd_full,
                                    calc_sd_raw(row + 1,     sum_left,  sum_sq_left),
                                    calc_sd_raw(n - row - 1, sum_right, sum_sq_right)
                                    );
                if (this_gain > min_gain && this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_point = avg_between(x[row], x[row + 1]);
                }
            }
            break;
        }

        case Pooled:
        {
            long double cnt = (long double) n;
            for (size_t row = 0; row < n-1; row++)
            {
                sum_left     += x[row];
                sum_sq_left  += square(x[row]);
                sum_right    -= x[row];
                sum_sq_right -= square(x[row]);

                if (x[row] == x[row + 1]) continue;

                this_gain = numeric_gain(row + 1, n - row - 1,
                                         sum_left, sum_right,
                                         sum_sq_left, sum_sq_right,
                                         sd_full, cnt
                                        );

                if (this_gain > min_gain && this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_point = avg_between(x[row], x[row + 1]);
                }
            }
            break;
        }
    }

    if (best_gain <= -HUGE_VAL && this_gain <= min_gain && this_gain > -HUGE_VAL)
        return 0;
    else
        return best_gain;
}

/* for split-criterion in single-variable splits */
#define std_val(x, m, sd) ( ((x) - (m)) / (sd)  )
double eval_guided_crit(size_t *restrict ix_arr, size_t st, size_t end, double *restrict x,
                        size_t &split_ix, double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action)
{
    /* move NAs to the front if there's any, exclude them from calculations */
    if (missing_action != Fail)
        st = move_NAs_to_front(ix_arr, st, end, x);

    if (st >= end) return -HUGE_VAL;
    else if (st == (end-1))
    {
        split_point = avg_between(x[ix_arr[st]], x[ix_arr[end]]);
        split_ix    = st;
        return 0;
    }

    /* sort in ascending order */
    std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});
    if (x[ix_arr[st]] == x[ix_arr[end]]) return -HUGE_VAL;
    xmin = x[ix_arr[st]]; xmax = x[ix_arr[end]];

    /* Note: these variables are not standardized beforehand, so a single-pass gain
       calculation for both branches would suffer from numerical instability and perhaps give
       negative standard deviations if the sample size is large or the values have different
       orders of magnitude */

    /* first get mean and sd */
    double x_mean, x_sd;
    calc_mean_and_sd(ix_arr, st, end, x,
                     Fail, x_sd, x_mean);

    /* compute sum - sum_sq - sd in one pass, on the standardized values */
    double zval;
    long double sum = 0;
    long double sum_sq = 0;
    double sd_full;
    for (size_t row = st; row <= end; row++)
    {
        zval    = std_val(x[ix_arr[row]], x_mean, x_sd);
        sum    += zval;
        sum_sq += square(zval);
    }
    sd_full = calc_sd_raw(end - st + 1, sum, sum_sq);

    /* try splits by moving observations one at a time from right to left */
    long double sum_left = 0;
    long double sum_sq_left = 0;
    long double sum_right = sum;
    long double sum_sq_right = sum_sq;
    double this_gain = -HUGE_VAL;
    double best_gain = -HUGE_VAL;

    switch(criterion)
    {
        case Averaged:
        {
            for (size_t row = st; row < end; row++)
            {
                zval          = std_val(x[ix_arr[row]], x_mean, x_sd);
                sum_left     += zval;
                sum_sq_left  += square(zval);
                sum_right    -= zval;
                sum_sq_right -= square(zval);

                if (x[ix_arr[row]] == x[ix_arr[row + 1]]) continue;

                this_gain = sd_gain(sd_full,
                                    calc_sd_raw(row - st + 1, sum_left,  sum_sq_left),
                                    calc_sd_raw(end - row,    sum_right, sum_sq_right)
                                    );
                if (this_gain > min_gain && this_gain > best_gain)
                {
                    best_gain = this_gain;
                    split_point = avg_between(x[ix_arr[row]], x[ix_arr[row + 1]]);
                    split_ix = row;
                }
            }
            break;
        }

        case Pooled:
        {
            long double cnt = (long double)(end - st + 1);
            for (size_t row = st; row < end; row++)
            {
                zval          = std_val(x[ix_arr[row]], x_mean, x_sd);
                sum_left     += zval;
                sum_sq_left  += square(zval);
                sum_right    -= zval;
                sum_sq_right -= square(zval);

                if (x[ix_arr[row]] == x[ix_arr[row + 1]]) continue;

                this_gain = numeric_gain_no_div(row - st + 1, end - row,
                                                sum_left, sum_right,
                                                sum_sq_left, sum_sq_right,
                                                sd_full, cnt
                                               );

                if (this_gain > min_gain && this_gain > best_gain)
                {
                    best_gain   = this_gain;
                    split_point = avg_between(x[ix_arr[row]], x[ix_arr[row + 1]]);
                    split_ix    = row;
                }
            }
            break;
        }
    }

    if (best_gain <= -HUGE_VAL && this_gain <= min_gain && this_gain > -HUGE_VAL)
        return 0;
    else
        return best_gain;
}

double eval_guided_crit(size_t ix_arr[], size_t st, size_t end,
                        size_t col_num, double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                        double buffer_arr[], size_t buffer_pos[],
                        double &split_point, double &xmin, double &xmax,
                        GainCriterion criterion, double min_gain, MissingAction missing_action)
{
    todense(ix_arr, st, end,
            col_num, Xc, Xc_ind, Xc_indptr,
            buffer_arr);
    std::iota(buffer_pos, buffer_pos + (end - st + 1), (size_t)0);
    size_t temp;
    return eval_guided_crit(buffer_pos, 0, end - st, buffer_arr, temp, split_point,
                            xmin, xmax, criterion, min_gain, missing_action);
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
                        GainCriterion criterion, double min_gain, bool all_perm, MissingAction missing_action, CategSplit cat_split_type)
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
