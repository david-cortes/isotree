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

/* ceil(log2(x)) done with bit-wise operations ensures perfect precision (and it's faster too)
   https://stackoverflow.com/questions/2589096/find-most-significant-bit-left-most-that-is-set-in-a-bit-array
   https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers  */
#if SIZE_MAX == UINT32_MAX /* 32-bit systems */
    static const int MultiplyDeBruijnBitPosition[32] =
        {
            0, 9,  1,  10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
            8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6,  26, 5,  4, 31
        };
    size_t log2ceil( size_t v )
    {

        v--;
        v |= v >> 1; // first round down to one less than a power of 2
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;

        return MultiplyDeBruijnBitPosition[( uint32_t )( v * 0x07C4ACDDU ) >> 27] + 1;
    }
#elif SIZE_MAX == UINT64_MAX /* 64-bit systems */
    static const uint64_t tab64[64] = {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5};

    size_t log2ceil(size_t value)
    {
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        value |= value >> 32;
        return tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58] + 1;
    }
#else /* other architectures - might not be entirely precise, and will be slower */
    size_t log2ceil(size_t x) {return (size_t)(ceill(log2l((long double) x)));}
#endif

/* http://fredrik-j.blogspot.com/2009/02/how-not-to-compute-harmonic-numbers.html
   https://en.wikipedia.org/wiki/Harmonic_number */
#define THRESHOLD_EXACT_H 256 /* above this will get approximated */
double harmonic(size_t n)
{
    if (n > THRESHOLD_EXACT_H)
        return logl((long double)n) + (long double)0.5772156649;
    else
        return harmonic_recursive((double)1, (double)(n + 1));
}

double harmonic_recursive(double a, double b)
{
    if (b == a + 1) return 1 / a;
    double m = floor((a + b) / 2);
    return harmonic_recursive(a, m) + harmonic_recursive(m, b);
}

/* https://stats.stackexchange.com/questions/423542/isolation-forest-and-average-expected-depth-formula
   https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottom */
double expected_avg_depth(size_t sample_size)
{
    switch(sample_size)
    {
        case 1: return 0.;
        case 2: return 1.;
        case 3: return 5.0/3.0;
        case 4: return 13.0/6.0;
        case 5: return 77.0/30.0;
        case 6: return 29.0/10.0;
        case 7: return 223.0/70.0;
        case 8: return 481.0/140.0;
        case 9: return 4609.0/1260.0;
        default:
        {
            return 2 * (harmonic(sample_size) - 1);
        }
    }
}

double expected_avg_depth(long double approx_sample_size)
{
    if (approx_sample_size < 1.5)
        return 0;
    else if (approx_sample_size < 2.5)
        return 1;
    else if (approx_sample_size <= THRESHOLD_EXACT_H)
        return expected_avg_depth((size_t) roundl(approx_sample_size));
    else
        return 2 * logl(approx_sample_size) - (long double)1.4227843351;
}

/* https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree */
#define THRESHOLD_EXACT_S 87670 /* difference is <5e-4 */
double expected_separation_depth(size_t n)
{
    switch(n)
    {
        case 0: return 0.;
        case 1: return 0.;
        case 2: return 1.;
        case 3: return 1. + (1./3.);
        case 4: return 1. + (1./3.) + (2./9.);
        case 5: return 1.71666666667;
        case 6: return 1.84;
        case 7: return 1.93809524;
        case 8: return 2.01836735;
        case 9: return 2.08551587;
        case 10: return 2.14268078;
        default:
        {
            if (n >= THRESHOLD_EXACT_S)
                return 3;
            else
                return expected_separation_depth_hotstart((double)2.14268078, (size_t)10, n);
        }
    }
}

double expected_separation_depth_hotstart(double curr, size_t n_curr, size_t n_final)
{
    if (n_final >= 1360)
    {    
        if (n_final >= THRESHOLD_EXACT_S)
            return 3;
        else if (n_final >= 40774)
            return 2.999;
        else if (n_final >= 18844)
            return 2.998;
        else if (n_final >= 11956)
            return 2.997;
        else if (n_final >= 8643)
            return 2.996;
        else if (n_final >= 6713)
            return 2.995;
        else if (n_final >= 4229)
            return 2.9925;
        else if (n_final >= 3040)
            return 2.99;
        else if (n_final >= 2724)
            return 2.989;
        else if (n_final >= 1902)
            return 2.985;
        else if (n_final >= 1360)
            return 2.98;

        /* Note on the chosen precision: when calling it on smaller sample sizes,
           the standard error of the separation depth will be larger, thus it's less
           critical to get it right down to the smallest possible precision, while for
           larger samples the standard error of the separation depth will be smaller */
    }

    for (size_t i = n_curr + 1; i <= n_final; i++)
        curr += (-curr * (double)i + 3. * (double)i - 4.) / ((double)i * ((double)(i-1)));
    return curr;
}

/* linear interpolation */
double expected_separation_depth(long double n)
{
    if (n >= THRESHOLD_EXACT_S)
        return 3;
    double s_l = expected_separation_depth((size_t) floorl(n));
    long double u = ceill(n);
    double s_u = s_l + (-s_l * u + 3. * u - 4.) / (u * (u - 1.));
    double diff = n - floorl(n);
    return s_l + diff * s_u;
}

#define ix_comb(i, j, n, ncomb) (  ((ncomb)  + ((j) - (i))) - 1 - (((n) - (i)) * ((n) - (i) - 1)) / 2  )
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n, double counter[], double exp_remainder)
{
    size_t i, j;
    size_t ncomb = (n * (n - 1)) / 2;
    if (exp_remainder <= 1)
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = std::min(ix_arr[el1], ix_arr[el2]);
                j = std::max(ix_arr[el1], ix_arr[el2]);
                // counter[i * (n - (i+1)/2) + j - i - 1]++; /* beaware integer division */
                counter[ix_comb(i, j, n, ncomb)]++;
            }
        }
    else
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = std::min(ix_arr[el1], ix_arr[el2]);
                j = std::max(ix_arr[el1], ix_arr[el2]);
                counter[ix_comb(i, j, n, ncomb)] += exp_remainder;
            }
        }
}

void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double *restrict counter, double *restrict weights, double exp_remainder)
{
    size_t i, j;
    size_t ncomb = (n * (n - 1)) / 2;
    if (exp_remainder <= 1)
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = std::min(ix_arr[el1], ix_arr[el2]);
                j = std::max(ix_arr[el1], ix_arr[el2]);
                // counter[i * (n - (i+1)/2) + j - i - 1] += weights[i] * weights[j]; /* beaware integer division */
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j];
            }
        }
    else
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = std::min(ix_arr[el1], ix_arr[el2]);
                j = std::max(ix_arr[el1], ix_arr[el2]);
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j] * exp_remainder;
            }
        }
}

/* Note to self: don't try merge this into a template with the one above, as the other one has 'restrict' qualifier */
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double counter[], std::unordered_map<size_t, double> &weights, double exp_remainder)
{
    size_t i, j;
    size_t ncomb = (n * (n - 1)) / 2;
    if (exp_remainder <= 1)
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = std::min(ix_arr[el1], ix_arr[el2]);
                j = std::max(ix_arr[el1], ix_arr[el2]);
                // counter[i * (n - (i+1)/2) + j - i - 1] += weights[i] * weights[j]; /* beaware integer division */
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j];
            }
        }
    else
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = std::min(ix_arr[el1], ix_arr[el2]);
                j = std::max(ix_arr[el1], ix_arr[el2]);
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j] * exp_remainder;
            }
        }
}

void increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end, size_t split_ix, size_t n,
                                     double counter[], double exp_remainder)
{
    size_t n_group = 0;
    for (size_t ix = st; ix <= end; ix++)
        if (ix_arr[ix] < split_ix)
            n_group++;
        else
            break;

    n = n - split_ix;

    if (exp_remainder <= 1)
        for (size_t ix1 = st; ix1 < st + n_group; ix1++)
            for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
                counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix]++;
    else
        for (size_t ix1 = st; ix1 < st + n_group; ix1++)
            for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
                counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix] += exp_remainder;
}

void increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end, size_t split_ix, size_t n,
                                     double *restrict counter, double *restrict weights, double exp_remainder)
{
    size_t n_group = 0;
    for (size_t ix = st; ix <= end; ix++)
        if (ix_arr[ix] < split_ix)
            n_group++;
        else
            break;

    n = n - split_ix;

    if (exp_remainder <= 1)
        for (size_t ix1 = st; ix1 < st + n_group; ix1++)
            for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
                counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix]
                    +=
                weights[ix_arr[ix1]] * weights[ix_arr[ix2]];
    else
        for (size_t ix1 = st; ix1 < st + n_group; ix1++)
            for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
                counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix]
                    +=
                weights[ix_arr[ix1]] * weights[ix_arr[ix2]] * exp_remainder;
}

void tmat_to_dense(double *restrict tmat, double *restrict dmat, size_t n, bool diag_to_one)
{
    size_t ncomb = (n * (n - 1)) / 2;
    for (size_t i = 0; i < (n-1); i++)
    {
        for (size_t j = i + 1; j < n; j++)
        {
            // dmat[i + j * n] = dmat[j + i * n] = tmat[i * (n - (i+1)/2) + j - i - 1];
            dmat[i + j * n] = dmat[j + i * n] = tmat[ix_comb(i, j, n, ncomb)];
        }
    }
    if (diag_to_one)
        for (size_t i = 0; i < n; i++)
            dmat[i + i * n] = 1;
    else
        for (size_t i = 0; i < n; i++)
            dmat[i + i * n] = 0;
}

/* Note: do NOT divide by (n-1) as in some situations it will still need to calculate
   the standard deviation with 1-2 observations only (e.g. when using the extended model
   and some column has many rows but only 2 non-missing values, or when using the non-pooled
   std criterion) */
#define SD_MIN 1e-12
double calc_sd_raw(size_t cnt, long double sum, long double sum_sq)
{
    if (cnt <= 1)
        return 0.;
    else
        return sqrtl(fmax(SD_MIN, (sum_sq - (square(sum) / (long double)cnt)) / (long double)cnt ));
}

long double calc_sd_raw_l(size_t cnt, long double sum, long double sum_sq)
{
    if (cnt <= 1)
        return 0.;
    else
        return sqrtl(fmaxl(SD_MIN, (sum_sq - (square(sum) / (long double)cnt)) / (long double)cnt ));
}

void build_btree_sampler(std::vector<double> &btree_weights, double *restrict sample_weights,
                         size_t nrows, size_t &log2_n, size_t &btree_offset)
{
    /* build a perfectly-balanced binary search tree in which each node will
       hold the sum of the weights of its children */
    log2_n = log2ceil(nrows);
    if (!btree_weights.size())
        btree_weights.resize(pow2(log2_n + 1), 0);
    else
        btree_weights.assign(btree_weights.size(), 0);
    btree_offset = pow2(log2_n) - 1;

    std::copy(sample_weights, sample_weights + nrows, btree_weights.begin() + btree_offset);
    for (size_t ix = btree_weights.size() - 1; ix > 0; ix--)
        btree_weights[ix_parent(ix)] += btree_weights[ix];
    
    if (is_na_or_inf(btree_weights[0]))
    {
        fprintf(stderr, "Numeric precision error with sample weights, will not use them.\n");
        log2_n = 0;
        btree_weights.clear();
        btree_weights.shrink_to_fit();
    }
}

void sample_random_rows(std::vector<size_t> &ix_arr, size_t nrows, bool with_replacement,
                        RNG_engine &rnd_generator, std::vector<size_t> &ix_all,
                        double sample_weights[], std::vector<double> &btree_weights,
                        size_t log2_n, size_t btree_offset, std::vector<bool> &is_repeated)
{
    size_t ntake = ix_arr.size();

    /* if with replacement, just generate random uniform numbers */
    if (with_replacement)
    {
        if (sample_weights == NULL)
        {
            std::uniform_int_distribution<size_t> runif(0, nrows - 1);
            for (size_t &ix : ix_arr)
                ix = runif(rnd_generator);
        }

        else
        {
            std::discrete_distribution<size_t> runif(sample_weights, sample_weights + nrows);
            for (size_t &ix : ix_arr)
                ix = runif(rnd_generator);
        }
    }

    /* if all the elements are needed, don't bother with any sampling */
    else if (ntake == nrows)
    {
        std::iota(ix_arr.begin(), ix_arr.end(), (size_t)0);
    }


    /* if there are sample weights, use binary trees to keep track and update weight
       https://stackoverflow.com/questions/57599509/c-random-non-repeated-integers-with-weights */
    else if (sample_weights != NULL)
    {
        double rnd_subrange, w_left;
        double curr_subrange;
        size_t curr_ix;
        for (size_t &ix : ix_arr)
        {
            /* go down the tree by drawing a random number and
               checking if it falls in the left or right ranges */
            curr_ix = 0;
            curr_subrange = btree_weights[0];
            for (size_t lev = 0; lev < log2_n; lev++)
            {
                rnd_subrange = std::uniform_real_distribution<double>(0, curr_subrange)(rnd_generator);
                w_left = btree_weights[ix_child(curr_ix)];
                curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
                curr_subrange = btree_weights[curr_ix];
            }

            /* finally, determine element to choose in this iteration */
            ix = curr_ix - btree_offset;

            /* now remove the weight of the chosen element */
            btree_weights[curr_ix] = 0;
            for (size_t lev = 0; lev < log2_n; lev++)
            {
                curr_ix = ix_parent(curr_ix);
                btree_weights[curr_ix] =   btree_weights[ix_child(curr_ix)]
                                         + btree_weights[ix_child(curr_ix) + 1];
            }
        }
    }

    /* if no sample weights and not with replacement (most common case expected),
       then use different algorithms depending on the sampled fraction */
    else
    {

        /* if sampling a larger fraction, fill an array enumerating the rows, shuffle, and take first N  */
        if (ntake >= (nrows / 2))
        {

            if (!ix_all.size())
                ix_all.resize(nrows);

            /* in order for random seeds to always be reproducible, don't re-use previous shuffles */
            std::iota(ix_all.begin(), ix_all.end(), (size_t)0);

            /* If the number of sampled elements is large, do a full shuffle, enjoy simd-instructs when copying over */
            if (ntake >= ((nrows * 3)/4))
            {
                std::shuffle(ix_all.begin(), ix_all.end(), rnd_generator);
                ix_arr.assign(ix_all.begin(), ix_all.begin() + ntake);
            }

            /* otherwise, do only a partial shuffle (use Yates algorithm) and copy elements along the way */
            else
            {
                size_t chosen;
                for (size_t i = nrows - 1; i >= nrows - ntake; i--)
                {
                    chosen = std::uniform_int_distribution<size_t>(0, i)(rnd_generator);
                    ix_arr[nrows - i - 1] = ix_all[chosen];
                    ix_all[chosen] = ix_all[i];
                }
            }

        }

        /* If the sample size is small, use Floyd's random sampling algorithm
           https://stackoverflow.com/questions/2394246/algorithm-to-select-a-single-random-combination-of-values */
        else
        {

            size_t candidate;

            /* if the sample size is relatively large, use a temporary boolean vector */
            if (((long double)ntake / (long double)nrows) > (1. / 20.))
            {

                if (!is_repeated.size())
                    is_repeated.resize(nrows, false);
                else
                    is_repeated.assign(is_repeated.size(), false);

                for (size_t rnd_ix = nrows - ntake; rnd_ix < nrows; rnd_ix++)
                {
                    candidate = std::uniform_int_distribution<size_t>(0, rnd_ix)(rnd_generator);
                    if (is_repeated[candidate])
                    {
                        ix_arr[ntake - (nrows - rnd_ix)] = rnd_ix;
                        is_repeated[rnd_ix] = true;
                    }

                    else
                    {
                        ix_arr[ntake - (nrows - rnd_ix)] = candidate;
                        is_repeated[candidate] = true;
                    }
                }

            }

            /* if the sample size is very small, use an unordered set */
            else
            {

                std::unordered_set<size_t> repeated_set;
                for (size_t rnd_ix = nrows - ntake; rnd_ix < nrows; rnd_ix++)
                {
                    candidate = std::uniform_int_distribution<size_t>(0, rnd_ix)(rnd_generator);
                    if (repeated_set.find(candidate) == repeated_set.end()) /* TODO: switch to C++20 'contains' */
                    {
                        ix_arr[ntake - (nrows - rnd_ix)] = candidate;
                        repeated_set.insert(candidate);
                    }

                    else
                    {
                        ix_arr[ntake - (nrows - rnd_ix)] = rnd_ix;
                        repeated_set.insert(rnd_ix);
                    }
                }

            }

        }

    }
}

/* https://stackoverflow.com/questions/57599509/c-random-non-repeated-integers-with-weights */
void weighted_shuffle(size_t *restrict outp, size_t n, double *restrict weights, double *restrict buffer_arr, RNG_engine &rnd_generator)
{
    /* determine smallest power of two that is larger than N */
    size_t tree_levels = log2ceil(n);

    /* initialize vector with place-holders for perfectly-balanced tree */
    std::fill(buffer_arr, buffer_arr + pow2(tree_levels + 1), (double)0);

    /* compute sums for the tree leaves at each node */
    size_t offset = pow2(tree_levels) - 1;
    for (size_t ix = 0; ix < n; ix++) {
        buffer_arr[ix + offset] = weights[ix];
    }
    for (size_t ix = pow2(tree_levels+1) - 1; ix > 0; ix--) {
        buffer_arr[ix_parent(ix)] += buffer_arr[ix];
    }

    /* sample according to uniform distribution */
    double rnd_subrange, w_left;
    double curr_subrange;
    int curr_ix;

    for (size_t el = 0; el < n; el++)
    {
        /* go down the tree by drawing a random number and
           checking if it falls in the left or right sub-ranges */
        curr_ix = 0;
        curr_subrange = buffer_arr[0];
        for (size_t lev = 0; lev < tree_levels; lev++)
        {
            rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
            w_left = buffer_arr[ix_child(curr_ix)];
            curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
            curr_subrange = buffer_arr[curr_ix];
        }

        /* finally, add element from this iteration */
        outp[el] = curr_ix - offset;

        /* now remove the weight of the chosen element */
        buffer_arr[curr_ix] = 0;
        for (size_t lev = 0; lev < tree_levels; lev++)
        {
            curr_ix = ix_parent(curr_ix);
            buffer_arr[curr_ix] =   buffer_arr[ix_child(curr_ix)]
                                  + buffer_arr[ix_child(curr_ix) + 1];
        }
    }

}

/* This one samples with replacement. Algorithm is the same as above but keeps
   the weights after taking each sample. */
void WeightedColSampler::initialize(double weights[], size_t n)
{
    this->n = n;
    this->tree_levels = log2ceil(n);
    if (!this->tree_weights.size())
        this->tree_weights.resize(pow2(this->tree_levels + 1), 0);
    else {
        if (this->tree_weights.size() != pow2(this->tree_levels + 1))
            this->tree_weights.resize(this->tree_levels);
        std::fill(this->tree_weights.begin(), this->tree_weights.end(), 0.);
    }

    /* compute sums for the tree leaves at each node */
    this->offset = pow2(this->tree_levels) - 1;
    for (size_t ix = 0; ix < this->n; ix++) {
        this->tree_weights[ix + this->offset] = weights[ix];
    }
    for (size_t ix = pow2(this->tree_levels+1) - 1; ix > 0; ix--) {
        this->tree_weights[ix_parent(ix)] += tree_weights[ix];
    }
}

void WeightedColSampler::drop_col(size_t col)
{
    if (this->is_initialized())
    {
        size_t curr_ix = col + this->offset;
        if (this->tree_weights[curr_ix] > 0)
        {
            this->tree_weights[curr_ix] = 0.;
            for (size_t lev = 0; lev < this->tree_levels; lev++)
            {
                curr_ix = ix_parent(curr_ix);
                this->tree_weights[curr_ix] =   this->tree_weights[ix_child(curr_ix)]
                                              + this->tree_weights[ix_child(curr_ix) + 1];
            }
        }
    }
}

size_t WeightedColSampler::sample_col(RNG_engine &rnd_generator)
{
    size_t curr_ix = 0;
    double rnd_subrange, w_left;
    double curr_subrange = this->tree_weights[0];
    for (size_t lev = 0; lev < tree_levels; lev++)
    {
        rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
        w_left = this->tree_weights[ix_child(curr_ix)];
        curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
        curr_subrange = this->tree_weights[curr_ix];
    }

    return curr_ix - this->offset;
}

void WeightedColSampler::shuffle_cols(std::vector<size_t> &out, RNG_engine &rnd_generator)
{
    out.reserve(this->n);
    out.resize(this->n);
    std::vector<double> curr_weight = this->tree_weights;
    size_t n_available = 0;
    double rnd_subrange, w_left;
    double curr_subrange;
    int curr_ix;

    for (size_t el = 0; el < this->n; el++)
    {
        /* go down the tree by drawing a random number and
           checking if it falls in the left or right sub-ranges */
        curr_ix = 0;
        curr_subrange = curr_weight[0];
        if (curr_subrange == 0)
            break;

        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
            rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
            w_left = curr_weight[ix_child(curr_ix)];
            curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
            curr_subrange = curr_weight[curr_ix];
        }

        /* finally, add element from this iteration */
        out[el] = curr_ix - this->offset;

        /* now remove the weight of the chosen element */
        curr_weight[curr_ix] = 0;
        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
            curr_ix = ix_parent(curr_ix);
            curr_weight[curr_ix] =   curr_weight[ix_child(curr_ix)]
                                   + curr_weight[ix_child(curr_ix) + 1];
        }

        n_available++;
    }

    out.resize(n_available);
}

bool WeightedColSampler::is_initialized()
{
    return this->tree_weights.size() > 0;
}

/* For hyperplane intersections */
size_t divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point)
{
    size_t temp;
    size_t st_orig = st;
    for (size_t row = st_orig; row <= end; row++)
    {
        if (x[row - st_orig] <= split_point)
        {
            temp        = ix_arr[st];
            ix_arr[st]  = ix_arr[row];
            ix_arr[row] = temp;
            st++;
        }
    }
    return st;
}

/* For numerical columns */
void divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point,
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix)
{
    size_t temp;

    /* if NAs are not to be bothered with, just need to do a single pass */
    if (missing_action == Fail)
    {
        /* move to the left if it's l.e. split point */
        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] <= split_point)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        split_ix = st;
    }

    /* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (!isnan(x[ix_arr[row]]) && x[ix_arr[row]] <= split_point)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        st_NA = st;

        for (size_t row = st; row <= end; row++)
        {
            if (isnan(x[ix_arr[row]]))
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        end_NA = st;
    }
}

/* For sparse numeric columns */
void divide_subset_split(size_t ix_arr[], size_t st, size_t end, size_t col_num,
                         double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[], double split_point,
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix)
{
    /* TODO: this is a mess, needs refactoring */
    /* TODO: when moving zeros, would be better to instead move by '>' (opposite as in here) */
    if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
    {
        if (missing_action == Fail)
        {
            split_ix = (0 <= split_point)? (end+1) : st;
        }

        else
        {
            st_NA  = (0 <= split_point)? (end+1) : st;
            end_NA = (0 <= split_point)? (end+1) : st;
        }

    }

    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    size_t temp;
    bool   move_zeros = 0 <= split_point;
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);

    if (move_zeros && ptr_st > ix_arr + st)
        st = ptr_st - ix_arr;

    if (missing_action == Fail)
    {
        if (move_zeros)
        {
            for (size_t *row = ptr_st;
                 row != ix_arr + end + 1;
                )
            {
                if (curr_pos >= end_col + 1)
                {
                    for (size_t *r = row; r <= ix_arr + end; r++)
                    {
                        temp       = ix_arr[st];
                        ix_arr[st] = *r;
                        *r         = temp;
                        st++;
                    }
                    break;
                }

                if (Xc_ind[curr_pos] == *row)
                {
                    if (Xc[curr_pos] <= split_point)
                    {
                        temp       = ix_arr[st];
                        ix_arr[st] = *row;
                        *row       = temp;
                        st++;
                    }
                    if (curr_pos == end_col && row < ix_arr + end)
                    {
                        for (size_t *r = row + 1; r <= ix_arr + end; r++)
                        {
                            temp       = ix_arr[st];
                            ix_arr[st] = *r;
                            *r         = temp;
                            st++;
                        }
                    }
                    if (row == ix_arr + end || curr_pos == end_col) break;
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
                }

                else
                {
                    if (Xc_ind[curr_pos] > *row)
                    {
                        while (row <= ix_arr + end && Xc_ind[curr_pos] > *row)
                        {
                            temp       = ix_arr[st];
                            ix_arr[st] = *row;
                            *row       = temp;
                            st++; row++;
                        }
                    }

                    else
                        curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
                }
            }
        }

        else /* don't move zeros */
        {
            for (size_t *row = ptr_st;
                 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (Xc_ind[curr_pos] == *row)
                {
                    if (Xc[curr_pos] <= split_point)
                    {
                        temp       = ix_arr[st];
                        ix_arr[st] = *row;
                        *row       = temp;
                        st++;
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

        split_ix = st;
    }

    else /* can have NAs */
    {

        bool has_NAs = false;
        if (move_zeros)
        {
            for (size_t *row = ptr_st;
                 row != ix_arr + end + 1;
                )
            {
                if (curr_pos >= end_col + 1)
                {
                    for (size_t *r = row; r <= ix_arr + end; r++)
                    {
                        temp       = ix_arr[st];
                        ix_arr[st] = *r;
                        *r         = temp;
                        st++;
                    }
                    break;
                }

                if (Xc_ind[curr_pos] == *row)
                {
                    if (isnan(Xc[curr_pos]))
                        has_NAs = true;
                    else if (Xc[curr_pos] <= split_point)
                    {
                        temp       = ix_arr[st];
                        ix_arr[st] = *row;
                        *row       = temp;
                        st++;
                    }
                    if (curr_pos == end_col && row < ix_arr + end)
                        for (size_t *r = row + 1; r <= ix_arr + end; r++)
                        {
                            temp       = ix_arr[st];
                            ix_arr[st] = *r;
                            *r         = temp;
                            st++;
                        }
                    if (row == ix_arr + end || curr_pos == end_col) break;
                    curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
                }

                else
                {
                    if (Xc_ind[curr_pos] > *row)
                    {
                        while (row <= ix_arr + end && Xc_ind[curr_pos] > *row)
                        {
                            temp       = ix_arr[st];
                            ix_arr[st] = *row;
                            *row       = temp;
                            st++; row++;
                        }
                    }

                    else
                    {
                        curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
                    }
                }
            }
        }

        else /* don't move zeros */
        {
            for (size_t *row = ptr_st;
                 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (Xc_ind[curr_pos] == *row)
                {
                    if (isnan(Xc[curr_pos])) has_NAs = true;
                    if (Xc[curr_pos] <= split_point && !isnan(Xc[curr_pos]))
                    {
                        temp       = ix_arr[st];
                        ix_arr[st] = *row;
                        *row       = temp;
                        st++;
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


        st_NA = st;
        if (has_NAs)
        {
            curr_pos = st_col;
            std::sort(ix_arr + st, ix_arr + end + 1);
            for (size_t *row = ix_arr + st;
                 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
                )
            {
                if (Xc_ind[curr_pos] == *row)
                {
                    if (isnan(Xc[curr_pos]))
                    {
                        temp       = ix_arr[st];
                        ix_arr[st] = *row;
                        *row       = temp;
                        st++;
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
        end_NA = st;

    }

}

/* For categorical columns split by subset */
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, char split_categ[],
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix)
{
    size_t temp;

    /* if NAs are not to be bothered with, just need to do a single pass */
    if (missing_action == Fail)
    {
        /* move to the left if it's l.e. than the split point */
        for (size_t row = st; row <= end; row++)
        {
            if (split_categ[ x[ix_arr[row]] ] == 1)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        split_ix = st;
    }

    /* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] >= 0 && split_categ[ x[ix_arr[row]] ] == 1)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        st_NA = st;

        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] < 0)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        end_NA = st;
    }
}

/* For categorical columns split by subset, used at prediction time (with similarity) */
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, char split_categ[],
                         int ncat, MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &st_NA, size_t &end_NA, size_t &split_ix)
{
    size_t temp;

    /* if NAs are not to be bothered with, just need to do a single pass */
    if (missing_action == Fail && new_cat_action != Weighted)
    {
        if (new_cat_action == Smallest && move_new_to_left)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (split_categ[ x[ix_arr[row]] ] == 1 || x[ix_arr[row]] >= ncat)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        else
        {
            for (size_t row = st; row <= end; row++)
            {
                if (split_categ[ x[ix_arr[row]] ] == 1)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        split_ix = st;
    }

    /* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] >= 0 && split_categ[ x[ix_arr[row]] ] == 1)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        st_NA = st;

        if (new_cat_action == Weighted)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] < 0 || split_categ[ x[ix_arr[row]] ] == (-1))
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        else
        {
            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] < 0)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        end_NA = st;
    }
}

/* For categoricals split on a single category */
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, int split_categ,
                         MissingAction missing_action, size_t &st_NA, size_t &end_NA, size_t &split_ix)
{
    size_t temp;

    /* if NAs are not to be bothered with, just need to do a single pass */
    if (missing_action == Fail)
    {
        /* move to the left if it's l.e. than the split point */
        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] == split_categ)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        split_ix = st;
    }

    /* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
    else
    {
        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] == split_categ)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        st_NA = st;

        for (size_t row = st; row <= end; row++)
        {
            if (x[ix_arr[row]] < 0)
            {
                temp        = ix_arr[st];
                ix_arr[st]  = ix_arr[row];
                ix_arr[row] = temp;
                st++;
            }
        }
        end_NA = st;
    }
}

/* For categoricals split on sub-set that turned out to have 2 categories only (prediction-time) */
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end,
                         MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &st_NA, size_t &end_NA, size_t &split_ix)
{
    size_t temp;

    /* if NAs are not to be bothered with, just need to do a single pass */
    if (missing_action == Fail)
    {
        /* move to the left if it's l.e. than the split point */
        if (new_cat_action == Smallest && move_new_to_left)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] == 0 || x[ix_arr[row]] > 1)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        else
        {
            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] == 0)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }
        split_ix = st;
    }

    /* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
    else
    {
        if (new_cat_action == Smallest && move_new_to_left)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] == 0 || x[ix_arr[row]] > 1)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
            st_NA = st;

            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] < 0)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
            end_NA = st;
        }

        else
        {
            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] == 0)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
            st_NA = st;

            for (size_t row = st; row <= end; row++)
            {
                if (x[ix_arr[row]] < 0)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
            end_NA = st;
        }
    }
}

/* for regular numeric columns */
void get_range(size_t ix_arr[], double x[], size_t st, size_t end,
               MissingAction missing_action, double &xmin, double &xmax, bool &unsplittable)
{
    xmin =  HUGE_VAL;
    xmax = -HUGE_VAL;

    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
        {
            xmin = (x[ix_arr[row]] < xmin)? x[ix_arr[row]] : xmin;
            xmax = (x[ix_arr[row]] > xmax)? x[ix_arr[row]] : xmax;
        }
    }


    else
    {
        for (size_t row = st; row <= end; row++)
        {
            xmin = std::fmin(xmin, x[ix_arr[row]]);
            xmax = std::fmax(xmax, x[ix_arr[row]]);
        }
    }

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL);
}

/* for sparse inputs */
void get_range(size_t ix_arr[], size_t st, size_t end, size_t col_num,
               double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
               MissingAction missing_action, double &xmin, double &xmax, bool &unsplittable)
{
    /* ix_arr must already be sorted beforehand */
    xmin =  HUGE_VAL;
    xmax = -HUGE_VAL;

    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1];
    size_t nnz_col = end_col - st_col;
    end_col--;
    size_t curr_pos = st_col;

    if (!nnz_col || 
        Xc_ind[st_col] > ix_arr[end] || 
        ix_arr[st]     > Xc_ind[end_col]
        )
    {
        unsplittable = true;
        return;
    }

    if (nnz_col < end - st + 1 ||
        Xc_ind[st_col]  > ix_arr[st] ||
        Xc_ind[end_col] < ix_arr[end]
        )
    {
        xmin = 0;
        xmax = 0;
    }

    size_t ind_end_col = Xc_ind[end_col];
    size_t nmatches = 0;

    if (missing_action == Fail)
    {
        for (size_t *row = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == *row)
            {
                nmatches++;
                xmin = (Xc[curr_pos] < xmin)? Xc[curr_pos] : xmin;
                xmax = (Xc[curr_pos] > xmax)? Xc[curr_pos] : xmax;
                if (row == ix_arr + end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
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

    else /* can have NAs */
    {
        for (size_t *row = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == *row)
            {
                nmatches++;
                xmin = std::fmin(xmin, Xc[curr_pos]);
                xmax = std::fmax(xmax, Xc[curr_pos]);
                if (row == ix_arr + end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
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

    if (nmatches < (end - st + 1))
    {
        xmin = std::fmin(xmin, 0);
        xmax = std::fmax(xmax, 0);
    }
    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL);

}


void get_categs(size_t ix_arr[], int x[], size_t st, size_t end, int ncat,
                MissingAction missing_action, char categs[], size_t &npresent, bool &unsplittable)
{
    std::fill(categs, categs + ncat, -1);
    npresent = 0;
    for (size_t row = st; row <= end; row++)
        if (x[ix_arr[row]] >= 0)
            categs[x[ix_arr[row]]] = 1;

    npresent = std::accumulate(categs,
                               categs + ncat,
                               (size_t)0,
                               [](const size_t a, const char b){return a + (b > 0);}
                               );

    unsplittable = npresent < 2;
}

long double calculate_sum_weights(std::vector<size_t> &ix_arr, size_t st, size_t end, size_t curr_depth,
                                  std::vector<double> &weights_arr, std::unordered_map<size_t, double> &weights_map)
{
    if (curr_depth > 0 && weights_arr.size())
        return std::accumulate(ix_arr.begin() + st,
                               ix_arr.begin() + end + 1,
                               (long double)0,
                               [&weights_arr](const long double a, const size_t ix){return a + weights_arr[ix];});
    else if (curr_depth > 0 && weights_map.size())
        return std::accumulate(ix_arr.begin() + st,
                               ix_arr.begin() + end + 1,
                               (long double)0,
                               [&weights_map](const long double a, const size_t ix){return a + weights_map[ix];});
    else
        return -HUGE_VAL;
}

size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, double x[])
{
    size_t st_non_na = st;
    size_t temp;

    for (size_t row = st; row <= end; row++)
    {
        if (is_na_or_inf(x[ix_arr[row]]))
        {
            temp = ix_arr[st_non_na];
            ix_arr[st_non_na] = ix_arr[row];
            ix_arr[row] = temp;
            st_non_na++;
        }
    }

    return st_non_na;
}

size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, size_t col_num, double Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[])
{
    size_t st_non_na = st;
    size_t temp;

    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    std::sort(ix_arr + st, ix_arr + end + 1);
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);

    for (size_t *row = ptr_st;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == *row)
        {
            if (is_na_or_inf(Xc[curr_pos]))
            {
                temp = ix_arr[st_non_na];
                ix_arr[st_non_na] = *row;
                *row = temp;
                st_non_na++;
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

    return st_non_na;
}

size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, int x[])
{
    size_t st_non_na = st;
    size_t temp;

    for (size_t row = st; row <= end; row++)
    {
        if (x[ix_arr[row]] < 0)
        {
            temp = ix_arr[st_non_na];
            ix_arr[st_non_na] = ix_arr[row];
            ix_arr[row] = temp;
            st_non_na++;
        }
    }

    return st_non_na;
}

size_t center_NAs(size_t *restrict ix_arr, size_t st_left, size_t st, size_t curr_pos)
{
    size_t temp;
    for (size_t row = st_left; row < st; row++)
    {
        temp = ix_arr[--curr_pos];
        ix_arr[curr_pos] = ix_arr[row];
        ix_arr[row] = temp;
    }

    return curr_pos;
}

void todense(size_t ix_arr[], size_t st, size_t end,
             size_t col_num, double *restrict Xc, sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             double *restrict buffer_arr)
{
    std::fill(buffer_arr, buffer_arr + (end - st + 1), (double)0);

    size_t st_col  = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);

    for (size_t *row = ptr_st;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == *row)
        {
            buffer_arr[row - (ix_arr + st)] = Xc[curr_pos];
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

/* TODO: remove this once SciPy implements it for 'size_t' */
bool check_indices_are_sorted(sparse_ix indices[], size_t n)
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

void sort_csc_indices(double *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr, size_t ncols_numeric)
{
    std::vector<double> buffer_sorted_vals;
    std::vector<sparse_ix> buffer_sorted_ix;
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

/* Function to handle interrupt signals */
void set_interrup_global_variable(int s)
{
    #pragma omp critical
    {
        interrupt_switch = true;
    }
}

void check_interrupt_switch(SignalSwitcher &ss)
{
    if (interrupt_switch)
    {
        ss.restore_handle();
        fprintf(stderr, "Error: procedure was interrupted\n");
        raise(SIGINT);
        #ifdef _FOR_R
        Rcpp::checkUserInterrupt();
        #elif !defined(DONT_THROW_ON_INTERRUPT)
        throw "Error: procedure was interrupted.\n";
        #endif
    }
}

#ifdef _FOR_PYTHON
bool cy_check_interrupt_switch()
{
    return interrupt_switch;
}

void cy_tick_off_interrupt_switch()
{
    interrupt_switch = false;
}
#endif

SignalSwitcher::SignalSwitcher()
{
    #pragma omp critical
    {
        interrupt_switch = false;
        this->old_sig = signal(SIGINT, set_interrup_global_variable);
        this->is_active = true;
    }
}

SignalSwitcher::~SignalSwitcher()
{
    #pragma omp critical
    {
        if (this->is_active)
            signal(SIGINT, this->old_sig);
        this->is_active = false;
        #ifndef _FOR_PYTHON
        interrupt_switch = false;
        #endif
    }
}

void SignalSwitcher::restore_handle()
{
    #pragma omp critical
    {
        if (this->is_active)
        {
            signal(SIGINT, this->old_sig);
            this->is_active = false;
        }
    }
}

/* Return the #def'd constants from standard header. This is in order to determine if the return
   value from the 'fit_model' function is a success or failure within Cython, which does not
   allow importing #def'd macro values. */
int return_EXIT_SUCCESS()
{
    return EXIT_SUCCESS;
}
int return_EXIT_FAILURE()
{
    return EXIT_FAILURE;
}
