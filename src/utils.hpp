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

/* ceil(log2(x)) done with bit-wise operations ensures perfect precision (and it's faster too)
   https://stackoverflow.com/questions/2589096/find-most-significant-bit-left-most-that-is-set-in-a-bit-array
   https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers  */
#if SIZE_MAX == UINT32_MAX /* 32-bit systems */
    constexpr static const uint32_t MultiplyDeBruijnBitPosition[32] =
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
    constexpr static const uint64_t tab64[64] = {
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
#else /* other architectures - might be much slower */
    #if (__cplusplus  >= 202002L)
    #include <bit>
    size_t log2ceil(size_t value)
    {
        size_t out = std::numeric_limits<size_t>::digits - std::countl_zero(value);
        out -= (value == ((size_t)1 << (out-1)));
        return out;
    }
    #else
    size_t log2ceil(size_t value)
    {
        size_t value_ = value;
        size_t out = 0;
        while (value >= 1) {
            value = value >> 1;
            out++;
        }
        out -= (value_ == ((size_t)1 << (out-1)));
        return out;
    }
    #endif
#endif

/* adapted from cephes */
#define EULERS_GAMMA 0.577215664901532860606512
#include "digamma.hpp"

/* http://fredrik-j.blogspot.com/2009/02/how-not-to-compute-harmonic-numbers.html
   https://en.wikipedia.org/wiki/Harmonic_number
   https://github.com/scikit-learn/scikit-learn/pull/19087 */
template <class ldouble_safe>
double harmonic(size_t n)
{
    ldouble_safe temp = (ldouble_safe)1 / square((ldouble_safe)n);
    return  - (ldouble_safe)0.5 * temp * ( (ldouble_safe)1/(ldouble_safe)6  -   temp * ((ldouble_safe)1/(ldouble_safe)60 - ((ldouble_safe)1/(ldouble_safe)126)*temp) )
            + (ldouble_safe)0.5 * ((ldouble_safe)1/(ldouble_safe)n)
            + std::log((ldouble_safe)n) + (ldouble_safe)EULERS_GAMMA;
}

/* usage for getting harmonic(n) is like this: harmonic_recursive((double)1, (double)(n + 1)); */
double harmonic_recursive(double a, double b)
{
    if (b == a + 1) return 1. / a;
    double m = std::floor((a + b) / 2.);
    return harmonic_recursive(a, m) + harmonic_recursive(m, b);
}

/* https://stats.stackexchange.com/questions/423542/isolation-forest-and-average-expected-depth-formula
   https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottom */
#include "exp_depth_table.hpp"
template <class ldouble_safe>
double expected_avg_depth(size_t sample_size)
{
    if (likely(sample_size <= N_PRECALC_EXP_DEPTH)) {
        return exp_depth_table[sample_size - 1];
    }
    return 2. * (harmonic<ldouble_safe>(sample_size) - 1.);
}

/* Note: H(x) = psi(x+1) + gamma */
template <class ldouble_safe>
double expected_avg_depth(ldouble_safe approx_sample_size)
{
    if (approx_sample_size <= 1)
        return 0;
    else if (approx_sample_size < (ldouble_safe)INT32_MAX)
        return 2. * (digamma(approx_sample_size + 1.) + EULERS_GAMMA - 1.);
    else {
        ldouble_safe temp = (ldouble_safe)1 / square(approx_sample_size);
        return (ldouble_safe)2 * std::log(approx_sample_size) + (ldouble_safe)2*((ldouble_safe)EULERS_GAMMA - (ldouble_safe)1)
               + ((ldouble_safe)1/approx_sample_size)
               - temp * ( (ldouble_safe)1/(ldouble_safe)6 -   temp * ((ldouble_safe)1/(ldouble_safe)60 - ((ldouble_safe)1/(ldouble_safe)126)*temp) );
    }
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
template <class ldouble_safe>
double expected_separation_depth(ldouble_safe n)
{
    if (n >= THRESHOLD_EXACT_S)
        return 3;
    double s_l = expected_separation_depth((size_t) std::floor(n));
    ldouble_safe u = std::ceil(n);
    double s_u = s_l + (-s_l * u + 3. * u - 4.) / (u * (u - 1.));
    double diff = n - std::floor(n);
    return s_l + diff * s_u;
}

void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n, double counter[], double exp_remainder)
{
    size_t i, j;
    size_t ncomb = calc_ncomb(n);
    if (exp_remainder <= 1)
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                // counter[i * (n - (i+1)/2) + j - i - 1]++; /* beaware integer division */
                i = ix_arr[el1]; j = ix_arr[el2];
                counter[ix_comb(i, j, n, ncomb)]++;
            }
        }
    else
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = ix_arr[el1]; j = ix_arr[el2];
                counter[ix_comb(i, j, n, ncomb)] += exp_remainder;
            }
        }
}

void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double *restrict counter, double *restrict weights, double exp_remainder)
{
    size_t i, j;
    size_t ncomb = calc_ncomb(n);
    if (exp_remainder <= 1)
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = ix_arr[el1]; j = ix_arr[el2];
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j];
            }
        }
    else
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = ix_arr[el1]; j = ix_arr[el2];
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j] * exp_remainder;
            }
        }
}

/* Note to self: don't try merge this into a template with the one above, as the other one has 'restrict' qualifier */
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
                           double counter[], hashed_map<size_t, double> &weights, double exp_remainder)
{
    size_t i, j;
    size_t ncomb = calc_ncomb(n);
    if (exp_remainder <= 1)
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = ix_arr[el1]; j = ix_arr[el2];
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j];
            }
        }
    else
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = ix_arr[el1]; j = ix_arr[el2];
                counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j] * exp_remainder;
            }
        }
}

void increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end, size_t split_ix, size_t n,
                                     double counter[], double exp_remainder)
{
    size_t *ptr_split_ix = std::lower_bound(ix_arr + st, ix_arr + end + 1, split_ix);
    size_t n_group = std::distance(ix_arr + st, ptr_split_ix);
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
    size_t *ptr_split_ix = std::lower_bound(ix_arr + st, ix_arr + end + 1, split_ix);
    size_t n_group = std::distance(ix_arr + st, ptr_split_ix);
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

void tmat_to_dense(double *restrict tmat, double *restrict dmat, size_t n, double fill_diag)
{
    size_t ncomb = calc_ncomb(n);
    for (size_t i = 0; i < (n-1); i++)
    {
        for (size_t j = i + 1; j < n; j++)
        {
            // dmat[i + j * n] = dmat[j + i * n] = tmat[i * (n - (i+1)/2) + j - i - 1];
            dmat[i + j * n] = dmat[j + i * n] = tmat[ix_comb(i, j, n, ncomb)];
        }
    }
    for (size_t i = 0; i < n; i++)
        dmat[i + i * n] = fill_diag;
}

template <class real_t>
void build_btree_sampler(std::vector<double> &btree_weights, real_t *restrict sample_weights,
                         size_t nrows, size_t &restrict log2_n, size_t &restrict btree_offset)
{
    /* build a perfectly-balanced binary search tree in which each node will
       hold the sum of the weights of its children */
    log2_n = log2ceil(nrows);
    if (btree_weights.empty())
        btree_weights.resize(pow2(log2_n + 1), 0);
    else
        btree_weights.assign(btree_weights.size(), 0);
    btree_offset = pow2(log2_n) - 1;

    for (size_t ix = 0; ix < nrows; ix++)
        btree_weights[ix + btree_offset] = std::fmax(0., sample_weights[ix]);
    for (size_t ix = btree_weights.size() - 1; ix > 0; ix--)
        btree_weights[ix_parent(ix)] += btree_weights[ix];
    
    if (std::isnan(btree_weights[0]) || btree_weights[0] <= 0)
    {
        fprintf(stderr, "Numeric precision error with sample weights, will not use them.\n");
        log2_n = 0;
        btree_weights.clear();
        btree_weights.shrink_to_fit();
    }
}

template <class real_t, class ldouble_safe>
void sample_random_rows(std::vector<size_t> &restrict ix_arr, size_t nrows, bool with_replacement,
                        RNG_engine &rnd_generator, std::vector<size_t> &restrict ix_all,
                        real_t *restrict sample_weights, std::vector<double> &restrict btree_weights,
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
        /* TODO: here could instead generate only 1 random number from zero to the full weight,
           and then subtract from it as it goes down every level. Would have less precision
           but should still work fine. */

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
                rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
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

            if (ix_all.empty())
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
            if (((ldouble_safe)ntake / (ldouble_safe)nrows) > (1. / 50.))
            {

                if (is_repeated.empty())
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

                hashed_set<size_t> repeated_set;
                repeated_set.reserve(ntake);
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
template <class real_t>
void weighted_shuffle(size_t *restrict outp, size_t n, real_t *restrict weights, double *restrict buffer_arr, RNG_engine &rnd_generator)
{
    /* determine smallest power of two that is larger than N */
    size_t tree_levels = log2ceil(n);

    /* initialize vector with place-holders for perfectly-balanced tree */
    std::fill(buffer_arr, buffer_arr + pow2(tree_levels + 1), (double)0);

    /* compute sums for the tree leaves at each node */
    size_t offset = pow2(tree_levels) - 1;
    for (size_t ix = 0; ix < n; ix++) {
        buffer_arr[ix + offset] = std::fmax(0., weights[ix]);
    }
    for (size_t ix = pow2(tree_levels+1) - 1; ix > 0; ix--) {
        buffer_arr[ix_parent(ix)] += buffer_arr[ix];
    }

    /* if the weights are invalid, produce an unweighted shuffle */
    if (std::isnan(buffer_arr[0]) || buffer_arr[0] <= 0)
    {
        std::iota(outp, outp + n, (size_t)0);
        std::shuffle(outp, outp + n, rnd_generator);
        return;
    }

    /* sample according to uniform distribution */
    double rnd_subrange, w_left;
    double curr_subrange;
    size_t curr_ix;

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

double sample_random_uniform(double xmin, double xmax, RNG_engine &rng) noexcept
{
    double out;
    std::uniform_real_distribution<double> runif(xmin, xmax);
    for (int attempt = 0; attempt < 100; attempt++)
    {
        out = runif(rng);
        if (likely(out < xmax)) return out;
    }
    return xmin;
}

template <class ldouble_safe>
template <class other_t>
ColumnSampler<ldouble_safe>& ColumnSampler<ldouble_safe>::operator=(const ColumnSampler<other_t> &other)
{
    this->col_indices = other.col_indices;
    this->tree_weights = other.tree_weights;
    this->curr_pos = other.curr_pos;
    this->curr_col = other.curr_col;
    this->last_given = other.last_given;
    this->n_cols = other.n_cols;
    this->tree_levels = other.tree_levels;
    this->offset = other.offset;
    this->n_dropped = other.n_dropped;
    return *this;
}

/*  This one samples with replacement. When using weights, the algorithm is the
    same as for the row sampler, but keeping the weights after taking each iteration. */
/*  TODO: this column sampler could use coroutines from C++20 once compilers implement them. */
template <class ldouble_safe>
template <class real_t>
void ColumnSampler<ldouble_safe>::initialize(real_t weights[], size_t n_cols)
{
    this->n_cols = n_cols;
    this->tree_levels = log2ceil(n_cols);
    if (this->tree_weights.empty())
        this->tree_weights.resize(pow2(this->tree_levels + 1), 0);
    else {
        if (this->tree_weights.size() != pow2(this->tree_levels + 1))
            this->tree_weights.resize(this->tree_levels);
        std::fill(this->tree_weights.begin(), this->tree_weights.end(), 0.);
    }

    /* compute sums for the tree leaves at each node */
    this->offset = pow2(this->tree_levels) - 1;
    for (size_t ix = 0; ix < this->n_cols; ix++)
        this->tree_weights[ix + this->offset] = std::fmax(0., weights[ix]);
    for (size_t ix = this->tree_weights.size() - 1; ix > 0; ix--)
        this->tree_weights[ix_parent(ix)] += this->tree_weights[ix];

    /* if the weights are invalid, make it an unweighted sampler */
    if (unlikely(std::isnan(this->tree_weights[0]) || this->tree_weights[0] <= 0))
    {
        this->drop_weights();
    }

    this->n_dropped = 0;
}

template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::drop_weights()
{
    this->tree_weights.clear();
    this->tree_weights.shrink_to_fit();
    this->initialize(n_cols);
    this->n_dropped = 0;
}

template <class ldouble_safe>
bool ColumnSampler<ldouble_safe>::has_weights()
{
    return !this->tree_weights.empty();
}

template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::initialize(size_t n_cols)
{
    if (!this->has_weights())
    {
        this->n_cols = n_cols;
        this->curr_pos = n_cols;
        this->col_indices.resize(n_cols);
        std::iota(this->col_indices.begin(), this->col_indices.end(), (size_t)0);
    }
}

/* TODO: this one should instead call the same function for sampling rows,
   and should be done at the time of initialization so as to avoid allocating
   and filling the whole array. That way it'd be faster and use less memory. */
template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::leave_m_cols(size_t m, RNG_engine &rnd_generator)
{
    if (m == 0 || m >= this->n_cols)
        return;

    if (!this->has_weights())
    {
        size_t chosen;
        if (m <= this->n_cols / 4)
        {
            for (this->curr_pos = 0; this->curr_pos < m; this->curr_pos++)
            {
                chosen = std::uniform_int_distribution<size_t>(0, this->n_cols - this->curr_pos - 1)(rnd_generator);
                std::swap(this->col_indices[this->curr_pos + chosen], this->col_indices[this->curr_pos]);
            }
        }

        else if ((ldouble_safe)m >= (ldouble_safe)(3./4.) * (ldouble_safe)this->n_cols)
        {
            for (this->curr_pos = this->n_cols-1; this->curr_pos > this->n_cols - m; this->curr_pos--)
            {
                chosen = std::uniform_int_distribution<size_t>(0, this->curr_pos)(rnd_generator);
                std::swap(this->col_indices[chosen], this->col_indices[this->curr_pos]);
            }
            this->curr_pos = m;
        }

        else
        {
            std::shuffle(this->col_indices.begin(), this->col_indices.end(), rnd_generator);
            this->curr_pos = m;
        }
    }

    else
    {
        std::vector<double> curr_weights = this->tree_weights;
        std::fill(this->tree_weights.begin(), this->tree_weights.end(), 0.);
        double rnd_subrange, w_left;
        double curr_subrange;
        size_t curr_ix;

        for (size_t col = 0; col < m; col++)
        {
            curr_ix = 0;
            curr_subrange = curr_weights[0];
            if (curr_subrange <= 0)
            {
                if (col == 0)
                {
                    this->drop_weights();
                    return;
                }

                else
                {
                    m = col;
                    goto rebuild_tree;
                }
            }

            for (size_t lev = 0; lev < this->tree_levels; lev++)
            {
                rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
                w_left = curr_weights[ix_child(curr_ix)];
                curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
                curr_subrange = curr_weights[curr_ix];
            }

            this->tree_weights[curr_ix] = curr_weights[curr_ix];

            /* now remove the weight of the chosen element */
            curr_weights[curr_ix] = 0;
            for (size_t lev = 0; lev < this->tree_levels; lev++)
            {
                curr_ix = ix_parent(curr_ix);
                curr_weights[curr_ix] =   curr_weights[ix_child(curr_ix)]
                                        + curr_weights[ix_child(curr_ix) + 1];
            }
        }

        /* rebuild the tree after getting new weights */
        rebuild_tree:
        for (size_t ix = this->tree_weights.size() - 1; ix > 0; ix--)
            this->tree_weights[ix_parent(ix)] += this->tree_weights[ix];

        this->n_dropped = this->n_cols - m;
    }
}

template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::drop_col(size_t col, size_t nobs_left)
{
    if (!this->has_weights())
    {
        if (this->col_indices[this->last_given] == col)
        {
            std::swap(this->col_indices[this->last_given], this->col_indices[--this->curr_pos]);
        }

        else if (this->curr_pos > 4*nobs_left)
        {
            return;
        }

        else
        {
            for (size_t ix = 0; ix < this->curr_pos; ix++)
            {
                if (this->col_indices[ix] == col)
                {
                    std::swap(this->col_indices[ix], this->col_indices[--this->curr_pos]);
                    break;
                }
            }
        }

        if (this->curr_col) this->curr_col--;
    }

    else
    {
        this->n_dropped++;
        size_t curr_ix = col + this->offset;
        this->tree_weights[curr_ix] = 0.;
        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
            curr_ix = ix_parent(curr_ix);
            this->tree_weights[curr_ix] =   this->tree_weights[ix_child(curr_ix)]
                                          + this->tree_weights[ix_child(curr_ix) + 1];
        }
    }
}

template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::drop_col(size_t col)
{
    this->drop_col(col, SIZE_MAX);
}

/* to be used exclusively when initializing the density calculator,
   and only when 'col_indices' is a straight range with no dropped columns */
template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::drop_from_tail(size_t col)
{
    std::swap(this->col_indices[col], this->col_indices[--this->curr_pos]);
}

template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::prepare_full_pass()
{
    this->curr_col = 0;

    if (this->has_weights())
    {
        if (this->col_indices.size() < this->n_cols)
            this->col_indices.resize(this->n_cols);
        this->curr_pos = 0;
        for (size_t col = 0; col < this->n_cols; col++)
        {
            if (this->tree_weights[col + this->offset] > 0)
                this->col_indices[this->curr_pos++] = col;
        }
    }
}

template <class ldouble_safe>
bool ColumnSampler<ldouble_safe>::sample_col(size_t &col, RNG_engine &rnd_generator)
{
    if (!this->has_weights())
    {
        switch(this->curr_pos)
        {
            case 0: return false;
            case 1:
            {
                this->last_given = 0;
                col = this->col_indices[0];
                return true;
            }
            default:
            {
                this->last_given = std::uniform_int_distribution<size_t>(0, this->curr_pos-1)(rnd_generator);
                col = this->col_indices[this->last_given];
                return true;
            }
        }
    }

    else
    {
        /* TODO: here could instead generate only 1 random number from zero to the full weight,
           and then subtract from it as it goes down every level. Would have less precision
           but should still work fine. */
        size_t curr_ix = 0;
        double rnd_subrange, w_left;
        double curr_subrange = this->tree_weights[0];
        if (curr_subrange <= 0)
            return false;

        for (size_t lev = 0; lev < tree_levels; lev++)
        {
            rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
            w_left = this->tree_weights[ix_child(curr_ix)];
            curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
            curr_subrange = this->tree_weights[curr_ix];
        }

        col = curr_ix - this->offset;
        return true;
    }
}

template <class ldouble_safe>
bool ColumnSampler<ldouble_safe>::sample_col(size_t &col)
{
    if (this->curr_pos == this->curr_col || this->curr_pos == 0)
        return false;
    this->last_given = this->curr_col;
    col = this->col_indices[this->curr_col++];
    return true;
}

template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::shuffle_remainder(RNG_engine &rnd_generator)
{
    if (!this->has_weights())
    {
        this->prepare_full_pass();
        std::shuffle(this->col_indices.begin(),
                     this->col_indices.begin() + this->curr_pos,
                     rnd_generator);
    }

    else
    {
        if (this->tree_weights[0] <= 0)
            return;
        std::vector<double> curr_weights = this->tree_weights;
        this->curr_pos = 0;
        this->curr_col = 0;

        if (this->col_indices.size() < this->n_cols)
            this->col_indices.resize(this->n_cols);

        double rnd_subrange, w_left;
        double curr_subrange;
        size_t curr_ix;

        for (this->curr_pos = 0; this->curr_pos < this->n_cols; this->curr_pos++)
        {
            curr_ix = 0;
            curr_subrange = curr_weights[0];
            if (curr_subrange <= 0)
                return;

            for (size_t lev = 0; lev < this->tree_levels; lev++)
            {
                rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
                w_left = curr_weights[ix_child(curr_ix)];
                curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
                curr_subrange = curr_weights[curr_ix];
            }

            /* finally, add element from this iteration */
            this->col_indices[this->curr_pos] = curr_ix - this->offset;

            /* now remove the weight of the chosen element */
            curr_weights[curr_ix] = 0;
            for (size_t lev = 0; lev < this->tree_levels; lev++)
            {
                curr_ix = ix_parent(curr_ix);
                curr_weights[curr_ix] =   curr_weights[ix_child(curr_ix)]
                                        + curr_weights[ix_child(curr_ix) + 1];
            }
        }
    }
}

template <class ldouble_safe>
size_t ColumnSampler<ldouble_safe>::get_remaining_cols()
{
    if (!this->has_weights())
        return this->curr_pos;
    else
        return this->n_cols - this->n_dropped;
}

template <class ldouble_safe>
void ColumnSampler<ldouble_safe>::get_array_remaining_cols(std::vector<size_t> &restrict cols)
{
    if (!this->has_weights())
    {
        cols.assign(this->col_indices.begin(), this->col_indices.begin() + this->curr_pos);
        std::sort(cols.begin(), cols.begin() + this->curr_pos);
    }

    else
    {
        size_t n_rem = 0;
        for (size_t col = 0; col < this->n_cols; col++)
        {
            if (this->tree_weights[col + this->offset] > 0)
            {
                cols[n_rem++] = col;
            }
        }
    }
}

template<class ldouble_safe, class real_t>
bool SingleNodeColumnSampler<ldouble_safe, real_t>::initialize
(
    double *restrict weights,
    std::vector<size_t> *col_indices,
    size_t curr_pos,
    size_t n_sample,
    bool backup_weights
)
{
    if (!curr_pos) return false;

    this->col_indices = col_indices->data();
    this->curr_pos = curr_pos;
    this->n_left = this->curr_pos;
    this->weights_orig = weights;
    
    if (n_sample > std::max(log2ceil(this->curr_pos), (size_t)3))
    {
        this->using_tree = true;
        this->backup_weights = false;

        if (this->used_weights.empty()) {
            this->used_weights.reserve(col_indices->size());
            this->mapped_indices.reserve(col_indices->size());
            this->tree_weights.reserve(2 * col_indices->size());
        }

        this->used_weights.resize(this->curr_pos);
        this->mapped_indices.resize(this->curr_pos);

        for (size_t col = 0; col < this->curr_pos; col++) {
            this->mapped_indices[col] = this->col_indices[col];
            this->used_weights[col] = weights[this->col_indices[col]];
            if (!weights[this->col_indices[col]]) this->n_left--;
        }

        this->tree_weights.resize(0);
        build_btree_sampler(this->tree_weights, this->used_weights.data(),
                            this->curr_pos, this->tree_levels, this->offset);

        this->n_inf = 0;
        if (std::isinf(this->tree_weights[0]))
        {
            if (this->mapped_inf_indices.empty())
                this->mapped_inf_indices.resize(this->curr_pos);

            for (size_t col = 0; col < this->curr_pos; col++)
            {
                if (std::isinf(weights[this->col_indices[col]]))
                {
                    this->mapped_inf_indices[this->n_inf++] = this->col_indices[col];
                    weights[this->col_indices[col]] = 0;
                }

                else
                {
                    this->mapped_indices[col - this->n_inf] = this->col_indices[col];
                    this->used_weights[col - this->n_inf] = weights[this->col_indices[col]];
                }
            }

            this->tree_weights.resize(0);
            build_btree_sampler(this->tree_weights, this->used_weights.data(),
                                this->curr_pos - this->n_inf, this->tree_levels, this->offset);
        }

        this->used_weights.resize(0);

        if (this->tree_weights[0] <= 0 && !this->n_inf)
            return false;
    }

    else
    {
        this->using_tree = false;
        this->backup_weights = backup_weights;

        if (this->backup_weights)
        {
            if (this->weights_own.empty())
                this->weights_own.resize(col_indices->size());
            this->weights_own.assign(weights, weights + this->curr_pos);
        }

        this->cumw = 0;
        for (size_t col = 0; col < this->curr_pos; col++) {
            this->cumw += weights[this->col_indices[col]];
            if (!weights[this->col_indices[col]]) this->n_left--;
        }

        if (std::isnan(this->cumw))
            throw std::runtime_error("NAs encountered. Try using a different value for 'missing_action'.\n");

        /* if it's infinite, will choose among columns with infinite weight first */
        this->n_inf = 0;
        if (std::isinf(this->cumw))
        {
            if (this->inifinite_weights.empty())
                this->inifinite_weights.resize(col_indices->size());
            else
                this->inifinite_weights.assign(col_indices->size(), false);

            this->cumw = 0;
            for (size_t col = 0; col < this->curr_pos; col++)
            {
                if (std::isinf(weights[this->col_indices[col]])) {
                    this->n_inf++;
                    this->inifinite_weights[this->col_indices[col]] = true;
                    weights[this->col_indices[col]] = 0;
                }

                else {
                    this->cumw += weights[this->col_indices[col]];
                }
            }
        }

        if (!this->cumw && !this->n_inf) return false;
    }

    return true;
}

template <class ldouble_safe, class real_t>
bool SingleNodeColumnSampler<ldouble_safe, real_t>::sample_col(size_t &col_chosen, RNG_engine &rnd_generator)
{
    if (!this->using_tree)
    {
        if (this->backup_weights)
            this->weights_orig = this->weights_own.data();

        /* if there's infinites, choose uniformly at random from them */
        if (this->n_inf)
        {
            size_t chosen = std::uniform_int_distribution<size_t>(0, this->n_inf-1)(rnd_generator);
            size_t curr = 0;
            for (size_t col = 0; col < this->curr_pos; col++)
            {
                curr += inifinite_weights[this->col_indices[col]];
                if (curr == chosen)
                {
                    col_chosen = this->col_indices[col];
                    this->n_inf--;
                    this->inifinite_weights[col_chosen] = false;
                    this->n_left--;
                    return true;
                }
            }
            assert(0);
        }

        if (!this->n_left) return false;

        /* due to the way this is calculated, there can be large roundoff errors and even negatives */
        if (this->cumw <= 0)
        {
            this->cumw = 0;
            for (size_t col = 0; col < this->curr_pos; col++)
                this->cumw += this->weights_orig[this->col_indices[col]];
            if (unlikely(this->cumw <= 0))
                unexpected_error();
        }

        /* if there are no infinites, choose a column according to weight */
        ldouble_safe chosen = std::uniform_real_distribution<ldouble_safe>((ldouble_safe)0, this->cumw)(rnd_generator);
        ldouble_safe cumw_ = 0;
        for (size_t col = 0; col < this->curr_pos; col++)
        {
            cumw_ += this->weights_orig[this->col_indices[col]];
            if (cumw_ >= chosen)
            {
                col_chosen = this->col_indices[col];
                this->cumw -= this->weights_orig[col_chosen];
                this->weights_orig[col_chosen] = 0;
                this->n_left--;
                return true;
            }
        }
        col_chosen = this->col_indices[this->curr_pos-1];
        this->cumw -= this->weights_orig[col_chosen];
        this->weights_orig[col_chosen] = 0;
        this->n_left--;
        return true;
    }

    else
    {
        /* if there's infinites, choose uniformly at random from them */
        if (this->n_inf)
        {
            size_t chosen = std::uniform_int_distribution<size_t>(0, this->n_inf-1)(rnd_generator);
            col_chosen = this->mapped_inf_indices[chosen];
            std::swap(this->mapped_inf_indices[chosen], this->mapped_inf_indices[--this->n_inf]);
            this->n_left--;
            return true;
        }

        else
        {
            /* TODO: should standardize all these tree traversals into one.
               This one in particular could do with sampling only a single
               random number as it will not typically require exhausting all
               options like the usual column sampler. */
            if (!this->n_left) return false;
            size_t curr_ix = 0;
            double rnd_subrange, w_left;
            double curr_subrange = this->tree_weights[0];
            if (curr_subrange <= 0)
                return false;

            for (size_t lev = 0; lev < tree_levels; lev++)
            {
                rnd_subrange = std::uniform_real_distribution<double>(0., curr_subrange)(rnd_generator);
                w_left = this->tree_weights[ix_child(curr_ix)];
                curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
                curr_subrange = this->tree_weights[curr_ix];
            }

            col_chosen = this->mapped_indices[curr_ix - this->offset];

            this->tree_weights[curr_ix] = 0.;
            for (size_t lev = 0; lev < this->tree_levels; lev++)
            {
                curr_ix = ix_parent(curr_ix);
                this->tree_weights[curr_ix] =   this->tree_weights[ix_child(curr_ix)]
                                              + this->tree_weights[ix_child(curr_ix) + 1];
            }
            
            this->n_left--;
            return true;
        }
    }
}

template <class ldouble_safe, class real_t>
void SingleNodeColumnSampler<ldouble_safe, real_t>::backup(SingleNodeColumnSampler &other, size_t ncols_tot)
{
    other.n_inf = this->n_inf;
    other.n_left = this->n_left;
    other.using_tree = this->using_tree;

    if (this->using_tree)
    {
        if (other.tree_weights.empty())
        {
            other.tree_weights.reserve(ncols_tot);
            other.mapped_inf_indices.reserve(ncols_tot);
        }
        other.tree_weights.assign(this->tree_weights.begin(), this->tree_weights.end());
        other.mapped_inf_indices.assign(this->mapped_inf_indices.begin(), this->mapped_inf_indices.end());
    }

    else
    {
        other.cumw = this->cumw;
        if (this->backup_weights)
        {
            if (other.weights_own.empty())
                other.weights_own.reserve(ncols_tot);
            
            other.weights_own.resize(this->n_left);
            for (size_t col = 0; col < this->n_left; col++)
                other.weights_own[col] = this->weights_own[this->col_indices[col]];
        }

        if (this->inifinite_weights.size())
        {
            if (other.inifinite_weights.empty())
                other.inifinite_weights.reserve(ncols_tot);

            other.inifinite_weights.resize(this->n_left);
            for (size_t col = 0; col < this->n_left; col++)
                other.inifinite_weights[col] = this->inifinite_weights[this->col_indices[col]];
        }
    }
}

template <class ldouble_safe, class real_t>
void SingleNodeColumnSampler<ldouble_safe, real_t>::restore(const SingleNodeColumnSampler &other)
{
    this->n_inf = other.n_inf;
    this->n_left = other.n_left;
    this->using_tree = other.using_tree;

    if (this->using_tree)
    {
        this->tree_weights.assign(other.tree_weights.begin(), other.tree_weights.end());
        this->mapped_inf_indices.assign(other.mapped_inf_indices.begin(), other.mapped_inf_indices.end());
    }

    else
    {
        this->cumw = other.cumw;
        if (this->backup_weights)
        {
            for (size_t col = 0; col < this->n_left; col++)
                this->weights_own[this->col_indices[col]] = other.weights_own[col];
        }

        if (this->inifinite_weights.size())
        {
            for (size_t col = 0; col < this->n_left; col++)
                this->inifinite_weights[this->col_indices[col]] = other.inifinite_weights[col];
        }
    }
}

template <class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::initialize(size_t max_depth, int max_categ, bool reserve_counts, ScoringMetric scoring_metric)
{
    this->multipliers.reserve(max_depth+3);
    this->multipliers.clear();
    if (scoring_metric != AdjDensity)
        this->multipliers.push_back(0);
    else
        this->multipliers.push_back(1);

    if (reserve_counts)
    {
        this->counts.resize(max_categ);
    }
}

template <class ldouble_safe, class real_t>
template <class InputData>
void DensityCalculator<ldouble_safe, real_t>::initialize_bdens(const InputData &input_data,
                                         const ModelParams &model_params,
                                         std::vector<size_t> &ix_arr,
                                         ColumnSampler<ldouble_safe> &col_sampler)
{
    this->fast_bratio = model_params.fast_bratio;
    if (this->fast_bratio)
    {
        this->multipliers.reserve(model_params.max_depth + 3);
        this->multipliers.push_back(0);
    }

    if (input_data.range_low != NULL || input_data.ncat_ != NULL)
    {
        if (input_data.ncols_numeric)
        {
            this->queue_box.reserve(model_params.max_depth+3);
            this->box_low.assign(input_data.range_low, input_data.range_low + input_data.ncols_numeric);
            this->box_high.assign(input_data.range_high, input_data.range_high + input_data.ncols_numeric);
        }

        if (input_data.ncols_categ)
        {
            this->queue_ncat.reserve(model_params.max_depth+2);
            this->ncat.assign(input_data.ncat_, input_data.ncat_ + input_data.ncols_categ);
        }

        if (!this->fast_bratio)
        {
            if (input_data.ncols_numeric)
            {
                this->ranges.resize(input_data.ncols_numeric);
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                    this->ranges[col] = this->box_high[col] - this->box_low[col];
            }

            if (input_data.ncols_categ)
            {
                this->ncat_orig = this->ncat;
            }
        }

        return;
    }

    if (input_data.ncols_numeric)
    {
        this->queue_box.reserve(model_params.max_depth+3);
        this->box_low.resize(input_data.ncols_numeric);
        this->box_high.resize(input_data.ncols_numeric);
        if (!this->fast_bratio)
            this->ranges.resize(input_data.ncols_numeric);
    }
    if (input_data.ncols_categ)
    {
        this->queue_ncat.reserve(model_params.max_depth+2);
    }
    bool unsplittable = false;
    
    size_t npresent = 0;
    std::vector<signed char> categ_present;
    if (input_data.ncols_categ)
    {
        categ_present.resize(input_data.max_categ);
    }


    col_sampler.prepare_full_pass();
    size_t col;
    while (col_sampler.sample_col(col))
    {
        if (col < input_data.ncols_numeric)
        {
            if (input_data.Xc_indptr != NULL)
            {
                get_range((size_t*)ix_arr.data(), (size_t)0, ix_arr.size()-(size_t)1, col,
                          input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                          model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
            }

            else
            {
                get_range((size_t*)ix_arr.data(), input_data.numeric_data + input_data.nrows * col, (size_t)0, ix_arr.size()-(size_t)1,
                          model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
            }


            if (unsplittable)
            {
                this->box_low[col] = 0;
                this->box_high[col] = 0;
                if (!this->fast_bratio)
                    this->ranges[col] = 0;
                col_sampler.drop_col(col);
            }

            if (!this->fast_bratio)
            {
                this->ranges[col] = (ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col];
                this->ranges[col] = std::fmax(this->ranges[col], (ldouble_safe)0);
            }
        }

        else
        {
            get_categs((size_t*)ix_arr.data(),
                       input_data.categ_data + input_data.nrows * (col - input_data.ncols_numeric),
                       (size_t)0, ix_arr.size()-(size_t)1, input_data.ncat[col],
                       model_params.missing_action, categ_present.data(), npresent, unsplittable);

            if (unsplittable)
            {
                this->ncat[col - input_data.ncols_numeric] = 1;
                col_sampler.drop_col(col);
            }

            else
            {
                this->ncat[col - input_data.ncols_numeric] = npresent;
            }
        }
    }

    if (!this->fast_bratio)
        this->ncat_orig = this->ncat;
}

template<class ldouble_safe, class real_t>
template <class InputData>
void DensityCalculator<ldouble_safe, real_t>::initialize_bdens_ext(const InputData &input_data,
                                             const ModelParams &model_params,
                                             std::vector<size_t> &ix_arr,
                                             ColumnSampler<ldouble_safe> &col_sampler,
                                             bool col_sampler_is_fresh)
{
    this->vals_ext_box.reserve(model_params.max_depth + 3);
    this->queue_ext_box.reserve(model_params.max_depth + 3);
    this->vals_ext_box.push_back(0);

    if (input_data.range_low != NULL)
    {
        this->box_low.assign(input_data.range_low, input_data.range_low + input_data.ncols_numeric);
        this->box_high.assign(input_data.range_high, input_data.range_high + input_data.ncols_numeric);
        return;
    }

    this->box_low.resize(input_data.ncols_numeric);
    this->box_high.resize(input_data.ncols_numeric);
    bool unsplittable = false;

    /* TODO: find out if there's an optimal point for choosing one or the other loop
       when using 'leave_m_cols' and when using 'prob_pick_col_by_range', then fill in the
       lines that are commented out. */
    // if (!input_data.ncols_categ || model_params.ncols_per_tree < input_data.ncols_numeric)
    if (input_data.ncols_numeric)
    {
        col_sampler.prepare_full_pass();
        size_t col;
        while (col_sampler.sample_col(col))
        {
            if (col >= input_data.ncols_numeric)
                continue;
            if (input_data.Xc_indptr != NULL)
            {
                get_range((size_t*)ix_arr.data(), (size_t)0, ix_arr.size()-(size_t)1, col,
                          input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                          model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
            }

            else
            {
                get_range((size_t*)ix_arr.data(), input_data.numeric_data + input_data.nrows * col, (size_t)0, ix_arr.size()-(size_t)1,
                          model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
            }

            if (unsplittable)
            {
                this->box_low[col] = 0;
                this->box_high[col] = 0;
                col_sampler.drop_col(col);
            }
        }
    }

    // else if (input_data.ncols_numeric)
    // {
    //     size_t n_unsplittable = 0;
    //     std::vector<size_t> unsplittable_cols;
    //     if (col_sampler_is_fresh && !col_sampler.has_weights())
    //         unsplittable_cols.reserve(input_data.ncols_numeric);

    //     /* TODO: this will do unnecessary calculations when using 'leave_m_cols' */
    //     for (size_t col = 0; col < input_data.ncols_numeric; col++)
    //     {
    //         if (input_data.Xc_indptr != NULL)
    //         {
    //             get_range((size_t*)ix_arr.data(), (size_t)0, ix_arr.size()-(size_t)1, col,
    //                       input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
    //                       model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
    //         }

    //         else
    //         {
    //             get_range((size_t*)ix_arr.data(), input_data.numeric_data + input_data.nrows * col, (size_t)0, ix_arr.size()-(size_t)1,
    //                       model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
    //         }

    //         if (unsplittable)
    //         {
    //             this->box_low[col] = 0;
    //             this->box_high[col] = 0;
    //             n_unsplittable++;
    //             if (col_sampler.has_weights())
    //                 col_sampler.drop_col(col);
    //             else if (col_sampler_is_fresh)
    //                 unsplittable_cols.push_back(col);
    //         }
    //     }

    //     if (n_unsplittable && col_sampler_is_fresh && !col_sampler.has_weights())
    //     {
    //         #if (__cplusplus >= 202002L)
    //         for (auto col : unsplittable_cols | std::views::reverse)
    //             col_sampler.drop_from_tail(col);
    //         #else
    //         for (size_t inv_col = 0; inv_col < unsplittable_cols.size(); inv_col++)
    //         {
    //             size_t col = unsplittable_cols.size() - inv_col - 1;
    //             col_sampler.drop_from_tail(unsplittable_cols[col]);
    //         }
    //         #endif
    //     }

    //     else if (n_unsplittable > model_params.sample_size / 16 && !col_sampler_is_fresh && !col_sampler.has_weights())
    //     {
    //         /* TODO */
    //     }
    // }
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_density(double xmin, double xmax, double split_point)
{
    if (std::isinf(xmax) || std::isinf(xmin) || std::isnan(xmin) || std::isnan(xmax) || std::isnan(split_point))
    {
        this->multipliers.push_back(0);
        return;
    }

    double range = std::fmax(xmax - xmin, std::numeric_limits<double>::min());
    double dleft = std::fmax(split_point - xmin, std::numeric_limits<double>::min());
    double dright = std::fmax(xmax - split_point, std::numeric_limits<double>::min());
    double mult_left = std::log(dleft / range);
    double mult_right = std::log(dright / range);
    while (std::isinf(mult_left))
    {
        dleft = std::nextafter(dleft, (mult_left < 0)? HUGE_VAL : (-HUGE_VAL));
        mult_left = std::log(dleft / range);
    }
    while (std::isinf(mult_right))
    {
        dright = std::nextafter(dright, (mult_right < 0)? HUGE_VAL : (-HUGE_VAL));
        mult_right = std::log(dright / range);
    }

    mult_left = std::isnan(mult_left)? 0 : mult_left;
    mult_right = std::isnan(mult_right)? 0 : mult_right;

    ldouble_safe curr = this->multipliers.back();
    this->multipliers.push_back(curr + mult_right);
    this->multipliers.push_back(curr + mult_left);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_density(int n_left, int n_present)
{
    this->push_density(0., (double)n_present, (double)n_left);
}

/* For single category splits */
template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_density(size_t counts[], int ncat)
{
    /* this one assumes 'categ_present' has entries 0/1 for missing/present */
    int n_present = 0;
    for (int cat = 0; cat < ncat; cat++)
        n_present += counts[cat] > 0;
    this->push_density(0., (double)n_present, 1.);
}

/* For single category splits */
template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_density(int n_present)
{
    this->push_density(0., (double)n_present, 1.);
}

/* For binary categorical splits */
template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_density()
{
    this->multipliers.push_back(0);
    this->multipliers.push_back(0);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_adj(double xmin, double xmax, double split_point, double pct_tree_left, ScoringMetric scoring_metric)
{
    double range = std::fmax(xmax - xmin, std::numeric_limits<double>::min());
    double dleft = std::fmax(split_point - xmin, std::numeric_limits<double>::min());
    double dright = std::fmax(xmax - split_point, std::numeric_limits<double>::min());
    double chunk_left = dleft / range;
    double chunk_right = dright / range;
    if (std::isinf(xmax) || std::isinf(xmin) || std::isnan(xmin) || std::isnan(xmax) || std::isnan(split_point))
    {
        chunk_left = pct_tree_left;
        chunk_right = 1. - pct_tree_left;
        goto add_chunks;
    }

    if (std::isnan(chunk_left) || std::isnan(chunk_right))
    {
        chunk_left = 0.5;
        chunk_right = 0.5;
    }

    chunk_left = pct_tree_left / chunk_left;
    chunk_right = (1. - pct_tree_left) / chunk_right;

    add_chunks:
    chunk_left = 2. / (1. + .5/chunk_left);
    chunk_right = 2. / (1. + .5/chunk_right);
    // chunk_left = 2. / (1. + 1./chunk_left);
    // chunk_right = 2. / (1. + 1./chunk_right);
    // chunk_left = 2. - std::exp2(1. - chunk_left);
    // chunk_right = 2. - std::exp2(1. - chunk_right);

    ldouble_safe curr = this->multipliers.back();
    if (scoring_metric == AdjDepth)
    {
        this->multipliers.push_back(curr + chunk_right);
        this->multipliers.push_back(curr + chunk_left);
    }

    else 
    {
        this->multipliers.push_back(std::fmax(curr * chunk_right, (ldouble_safe)std::numeric_limits<double>::epsilon()));
        this->multipliers.push_back(std::fmax(curr * chunk_left, (ldouble_safe)std::numeric_limits<double>::epsilon()));
    }
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_adj(signed char *restrict categ_present, size_t *restrict counts, int ncat, ScoringMetric scoring_metric)
{
    /* this one assumes 'categ_present' has entries -1/0/1 for missing/right/left */
    int cnt_cat_left = 0;
    int cnt_cat = 0;
    size_t cnt = 0;
    size_t cnt_left = 0;
    for (int cat = 0; cat < ncat; cat++)
    {
        if (counts[cat] > 0)
        {
            cnt += counts[cat];
            cnt_cat_left += categ_present[cat];
            cnt_left += categ_present[cat]? counts[cat] : 0;
            cnt_cat++;
        }
    }

    double pct_tree_left = (ldouble_safe)cnt_left / (ldouble_safe)cnt;
    this->push_adj(0., (double)cnt_cat, (double)cnt_cat_left, pct_tree_left, scoring_metric);
}

/* For single category splits */
template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_adj(size_t *restrict counts, int ncat, int chosen_cat, ScoringMetric scoring_metric)
{
    /* this one assumes 'categ_present' has entries 0/1 for missing/present */
    int cnt_cat = 0;
    size_t cnt = 0;
    for (int cat = 0; cat < ncat; cat++)
    {
        cnt += counts[cat];
        cnt_cat += counts[cat] > 0;
    }

    double pct_tree_left = (ldouble_safe)counts[chosen_cat] / (ldouble_safe)cnt;
    this->push_adj(0., (double)cnt_cat, 1., pct_tree_left, scoring_metric);
}

/* For binary categorical splits */
template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_adj(double pct_tree_left, ScoringMetric scoring_metric)
{
    this->push_adj(0., 1., 0.5, pct_tree_left, scoring_metric);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens(double split_point, size_t col)
{
    if (this->fast_bratio)
        this->push_bdens_fast_route(split_point, col);
    else
        this->push_bdens_internal(split_point, col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens_internal(double split_point, size_t col)
{
    this->queue_box.push_back(this->box_high[col]);
    this->box_high[col] = split_point;
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens_fast_route(double split_point, size_t col)
{
    ldouble_safe curr_range = (ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col];
    ldouble_safe fraction_left  =  ((ldouble_safe)split_point - (ldouble_safe)this->box_low[col]) / curr_range;
    ldouble_safe fraction_right = ((ldouble_safe)this->box_high[col] - (ldouble_safe)split_point) / curr_range;
    fraction_left   = std::fmax(fraction_left, (ldouble_safe)std::numeric_limits<double>::min());
    fraction_left   = std::fmin(fraction_left, (ldouble_safe)(1. - std::numeric_limits<double>::epsilon()));
    fraction_left   = std::log(fraction_left);
    fraction_left  += this->multipliers.back();
    fraction_right  = std::fmax(fraction_right, (ldouble_safe)std::numeric_limits<double>::min());
    fraction_right  = std::fmin(fraction_right, (ldouble_safe)(1. - std::numeric_limits<double>::epsilon()));
    fraction_right  = std::log(fraction_right);
    fraction_right += this->multipliers.back();
    this->multipliers.push_back(fraction_right);
    this->multipliers.push_back(fraction_left);

    this->push_bdens_internal(split_point, col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens(int ncat_branch_left, size_t col)
{
    if (this->fast_bratio)
        this->push_bdens_fast_route(ncat_branch_left, col);
    else
        this->push_bdens_internal(ncat_branch_left, col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens_internal(int ncat_branch_left, size_t col)
{
    this->queue_ncat.push_back(this->ncat[col]);
    this->ncat[col] = ncat_branch_left;
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens_fast_route(int ncat_branch_left, size_t col)
{
    double fraction_left = std::log((double)ncat_branch_left / this->ncat[col]);
    double fraction_right = std::log((double)(this->ncat[col] - ncat_branch_left) / this->ncat[col]);
    ldouble_safe curr = this->multipliers.back();
    this->multipliers.push_back(curr + fraction_right);
    this->multipliers.push_back(curr + fraction_left);

    this->push_bdens_internal(ncat_branch_left, col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens(const std::vector<signed char> &cat_split, size_t col)
{
    if (this->fast_bratio)
        this->push_bdens_fast_route(cat_split, col);
    else
        this->push_bdens_internal(cat_split, col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens_internal(const std::vector<signed char> &cat_split, size_t col)
{
    int ncat_branch_left = 0;
    for (auto el : cat_split)
        ncat_branch_left += el == 1;
    this->push_bdens_internal(ncat_branch_left, col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens_fast_route(const std::vector<signed char> &cat_split, size_t col)
{
    int ncat_branch_left = 0;
    for (auto el : cat_split)
        ncat_branch_left += el == 1;
    this->push_bdens_fast_route(ncat_branch_left, col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::push_bdens_ext(const IsoHPlane &hplane, const ModelParams &model_params)
{
    double x1, x2;
    double xlow = 0, xhigh = 0;
    size_t col;
    size_t col_num = 0;
    size_t col_cat = 0;

    for (size_t col_outer = 0; col_outer < hplane.col_num.size(); col_outer++)
    {
        switch (hplane.col_type[col_outer])
        {
            case Numeric:
            {
                col = hplane.col_num[col_outer];
                x1 = hplane.coef[col_num] * (this->box_low[col] - hplane.mean[col_num]);
                x2 = hplane.coef[col_num] * (this->box_high[col] - hplane.mean[col_num]);
                xlow += std::fmin(x1, x2);
                xhigh += std::fmax(x1, x2);
                break;
            }

            case Categorical:
            {
                switch (model_params.cat_split_type)
                {
                    case SingleCateg:
                    {
                        xlow += std::fmin(hplane.fill_new[col_cat], 0.);
                        xhigh += std::fmax(hplane.fill_new[col_cat], 0.);
                        break;
                    }

                    case SubSet:
                    {
                        xlow += *std::min_element(hplane.cat_coef[col_cat].begin(), hplane.cat_coef[col_cat].end());
                        xhigh += *std::max_element(hplane.cat_coef[col_cat].begin(), hplane.cat_coef[col_cat].end());
                        break;
                    }
                }
                break;
            }

            default:
            {
                assert(0);
            }
        }
    }

    double chunk_left;
    double chunk_right;
    double xdiff = xhigh - xlow;

    if (model_params.scoring_metric != BoxedDensity)
    {
        chunk_left = (hplane.split_point - xlow) / xdiff;
        chunk_right = (xhigh - hplane.split_point) / xdiff;
        chunk_left = std::fmin(chunk_left, std::numeric_limits<double>::min());
        chunk_left = std::fmax(chunk_left, 1.-std::numeric_limits<double>::epsilon());
        chunk_right = std::fmin(chunk_right, std::numeric_limits<double>::min());
        chunk_right = std::fmax(chunk_right, 1.-std::numeric_limits<double>::epsilon());
    }

    else
    {
        chunk_left = xdiff / (hplane.split_point - xlow);
        chunk_right = xdiff / (xhigh - hplane.split_point);
        chunk_left = std::fmin(chunk_left, 1.);
        chunk_right = std::fmin(chunk_right, 1.);
    }

    this->queue_ext_box.push_back(std::log(chunk_right) + this->vals_ext_box.back());
    this->vals_ext_box.push_back(std::log(chunk_left) + this->vals_ext_box.back());
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop()
{
    this->multipliers.pop_back();
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_right()
{
    this->multipliers.pop_back();
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens(size_t col)
{
    if (this->fast_bratio)
        this->pop_bdens_fast_route(col);
    else
        this->pop_bdens_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_internal(size_t col)
{
    double old_high = this->queue_box.back();
    this->queue_box.pop_back();
    this->queue_box.push_back(this->box_low[col]);
    this->box_low[col] = this->box_high[col];
    this->box_high[col] = old_high;
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_fast_route(size_t col)
{
    this->multipliers.pop_back();
    this->pop_bdens_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_right(size_t col)
{
    if (this->fast_bratio)
        this->pop_bdens_right_fast_route(col);
    else
        this->pop_bdens_right_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_right_internal(size_t col)
{
    double old_low = this->queue_box.back();
    this->queue_box.pop_back();
    this->box_low[col] = old_low;
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_right_fast_route(size_t col)
{
    this->multipliers.pop_back();
    this->pop_bdens_right_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_cat(size_t col)
{
    if (this->fast_bratio)
        this->pop_bdens_cat_fast_route(col);
    else
        this->pop_bdens_cat_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_cat_internal(size_t col)
{
    int old_ncat = this->queue_ncat.back();
    this->ncat[col] = old_ncat - this->ncat[col];
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_cat_fast_route(size_t col)
{
    this->multipliers.pop_back();
    this->pop_bdens_cat_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_cat_right(size_t col)
{
    if (this->fast_bratio)
        this->pop_bdens_cat_right_fast_route(col);
    else
        this->pop_bdens_cat_right_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_cat_right_internal(size_t col)
{
    int old_ncat = this->queue_ncat.back();
    this->queue_ncat.pop_back();
    this->ncat[col] = old_ncat;
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_cat_right_fast_route(size_t col)
{
    this->multipliers.pop_back();
    this->pop_bdens_cat_right_internal(col);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_ext()
{
    this->vals_ext_box.pop_back();
    this->vals_ext_box.push_back(this->queue_ext_box.back());
    this->queue_ext_box.pop_back();
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::pop_bdens_ext_right()
{
    this->vals_ext_box.pop_back();
}

/* this outputs the logarithm of the density */
template<class ldouble_safe, class real_t>
double DensityCalculator<ldouble_safe, real_t>::calc_density(ldouble_safe remainder, size_t sample_size)
{
    return std::log(remainder) - std::log((ldouble_safe)sample_size) - this->multipliers.back();
}

template<class ldouble_safe, class real_t>
ldouble_safe DensityCalculator<ldouble_safe, real_t>::calc_adj_depth()
{
    ldouble_safe out = this->multipliers.back();
    return std::fmax(out, (ldouble_safe)std::numeric_limits<double>::min());
}

template<class ldouble_safe, class real_t>
double DensityCalculator<ldouble_safe, real_t>::calc_adj_density()
{
    return this->multipliers.back();
}

/* this outputs the logarithm of the density */
template<class ldouble_safe, class real_t>
ldouble_safe DensityCalculator<ldouble_safe, real_t>::calc_bratio_inv_log()
{
    if (!this->multipliers.empty())
        return -this->multipliers.back();

    ldouble_safe sum_log_switdh = 0;
    ldouble_safe ratio_col;
    for (size_t col = 0; col < this->ranges.size(); col++)
    {
        if (!this->ranges[col]) continue;
        ratio_col = this->ranges[col] / ((ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col]);
        ratio_col = std::fmax(ratio_col, (ldouble_safe)1);
        sum_log_switdh += std::log(ratio_col);
    }

    for (size_t col = 0; col < this->ncat.size(); col++)
    {
        if (this->ncat_orig[col] <= 1) continue;
        sum_log_switdh += std::log((double)this->ncat_orig[col] / (double)this->ncat[col]);
    }

    return sum_log_switdh;
}

template<class ldouble_safe, class real_t>
ldouble_safe DensityCalculator<ldouble_safe, real_t>::calc_bratio_log()
{
    if (!this->multipliers.empty())
        return this->multipliers.back();

    ldouble_safe sum_log_switdh = 0;
    ldouble_safe ratio_col;
    for (size_t col = 0; col < this->ranges.size(); col++)
    {
        if (!this->ranges[col]) continue;
        ratio_col = ((ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col]) / this->ranges[col];
        ratio_col = std::fmax(ratio_col, (ldouble_safe)std::numeric_limits<double>::min());
        ratio_col = std::fmin(ratio_col, (ldouble_safe)(1. - std::numeric_limits<double>::epsilon()));
        sum_log_switdh += std::log(ratio_col);
    }

    for (size_t col = 0; col < this->ncat.size(); col++)
    {
        if (this->ncat_orig[col] <= 1) continue;
        sum_log_switdh += std::log((double)this->ncat[col] / (double)this->ncat_orig[col]);
    }

    return sum_log_switdh;
}

/* this does NOT output the logarithm of the density */
template<class ldouble_safe, class real_t>
double DensityCalculator<ldouble_safe, real_t>::calc_bratio()
{
    return std::exp(this->calc_bratio_log());
}

const double MIN_DENS = std::log(std::numeric_limits<double>::min());

/* this outputs the logarithm of the density */
template<class ldouble_safe, class real_t>
double DensityCalculator<ldouble_safe, real_t>::calc_bdens(ldouble_safe remainder, size_t sample_size)
{
    double out = std::log(remainder) - std::log((ldouble_safe)sample_size) - this->calc_bratio_inv_log();
    return std::fmax(out, MIN_DENS);
}

/* this outputs the logarithm of the density */
template<class ldouble_safe, class real_t>
double DensityCalculator<ldouble_safe, real_t>::calc_bdens2(ldouble_safe remainder, size_t sample_size)
{
    double out = std::log(remainder) - std::log((ldouble_safe)sample_size) - this->calc_bratio_log();
    return std::fmax(out, MIN_DENS);
}

/* this outputs the logarithm of the density */
template<class ldouble_safe, class real_t>
ldouble_safe DensityCalculator<ldouble_safe, real_t>::calc_bratio_log_ext()
{
    return this->vals_ext_box.back();
}

template<class ldouble_safe, class real_t>
double DensityCalculator<ldouble_safe, real_t>::calc_bratio_ext()
{
    double out = std::exp(this->calc_bratio_log_ext());
    return std::fmax(out, std::numeric_limits<double>::min());
}

/* this outputs the logarithm of the density */
template<class ldouble_safe, class real_t>
double DensityCalculator<ldouble_safe, real_t>::calc_bdens_ext(ldouble_safe remainder, size_t sample_size)
{
    double out = std::log(remainder) - std::log((ldouble_safe)sample_size) - this->calc_bratio_log_ext();
    return std::fmax(out, MIN_DENS);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::save_range(double xmin, double xmax)
{
    this->xmin = xmin;
    this->xmax = xmax;
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::restore_range(double &restrict xmin, double &restrict xmax)
{
    xmin = this->xmin;
    xmax = this->xmax;
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::save_counts(size_t *restrict cat_counts, int ncat)
{
    this->counts.assign(cat_counts, cat_counts + ncat);
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::save_n_present_and_left(signed char *restrict split_left, int ncat)
{
    this->n_present = 0;
    this->n_left = 0;
    for (int cat = 0; cat < ncat; cat++)
    {
        this->n_present += split_left[cat] >= 0;
        this->n_left += split_left[cat] == 1;
    }
}

template<class ldouble_safe, class real_t>
void DensityCalculator<ldouble_safe, real_t>::save_n_present(size_t *restrict cat_counts, int ncat)
{
    this->n_present = 0;
    for (int cat = 0; cat < ncat; cat++)
        this->n_present +=  cat_counts[cat] > 0;
}

/* For hyperplane intersections */
size_t divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point) noexcept
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
template <class real_t>
void divide_subset_split(size_t *restrict ix_arr, real_t x[], size_t st, size_t end, double split_point,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept
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
            if (!std::isnan(x[ix_arr[row]]) && x[ix_arr[row]] <= split_point)
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
            if (unlikely(std::isnan(x[ix_arr[row]])))
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
template <class real_t, class sparse_ix>
void divide_subset_split(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                         real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr, double split_point,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept
{
    /* TODO: this is a mess, needs refactoring */
    /* TODO: when moving zeros, would be better to instead move by '>' (opposite as in here) */
    /* TODO: should create an extra version to go along with 'predict' that would
       add the range penalty right here to spare operations. */
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

                if (Xc_ind[curr_pos] == (sparse_ix)(*row))
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
                    if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                    {
                        while (row <= ix_arr + end && Xc_ind[curr_pos] > (sparse_ix)(*row))
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
                if (Xc_ind[curr_pos] == (sparse_ix)(*row))
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
                    if (Xc_ind[curr_pos] > (sparse_ix)(*row))
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

                if (Xc_ind[curr_pos] == (sparse_ix)(*row))
                {
                    if (unlikely(std::isnan(Xc[curr_pos])))
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
                    if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                    {
                        while (row <= ix_arr + end && Xc_ind[curr_pos] > (sparse_ix)(*row))
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
                if (Xc_ind[curr_pos] == (sparse_ix)(*row))
                {
                    if (unlikely(std::isnan(Xc[curr_pos]))) has_NAs = true;
                    if (!std::isnan(Xc[curr_pos]) && Xc[curr_pos] <= split_point)
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
                    if (Xc_ind[curr_pos] > (sparse_ix)(*row))
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
                if (Xc_ind[curr_pos] == (sparse_ix)(*row))
                {
                    if (unlikely(std::isnan(Xc[curr_pos])))
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
                    if (Xc_ind[curr_pos] > (sparse_ix)(*row))
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
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end, signed char split_categ[],
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept
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
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end, signed char split_categ[],
                         int ncat, MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept
{
    size_t temp;
    int cval;

    /* if NAs are not to be bothered with, just need to do a single pass */
    if (missing_action == Fail && new_cat_action != Weighted)
    {
        /* in this case, will need to fill 'split_ix', otherwise need to fill 'st_NA' and 'end_NA' */
        if (new_cat_action == Smallest && move_new_to_left)
        {
            for (size_t row = st; row <= end; row++)
            {
                cval = x[ix_arr[row]];
                if (cval >= ncat || split_categ[cval] == 1 || split_categ[cval] == (-1))
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        else if (new_cat_action == Random)
        {
            for (size_t row = st; row <= end; row++)
            {
                cval = x[ix_arr[row]];
                cval = (cval >= ncat)? (cval % ncat) : cval;
                if (split_categ[cval] == 1)
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
                cval = x[ix_arr[row]];
                if (cval < ncat && split_categ[cval] == 1)
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

    /* if there are new categories, and their direction was decided at random,
       can just reuse what was randomly decided for previous columns by taking
       a remainder w.r.t. the number of previous columns. Note however that this
       will not be an unbiased decision if the model used a gain criterion. */
    else if (new_cat_action == Random)
    {
        if (missing_action == Impute && !move_new_to_left)
        {
            for (size_t row = st; row <= end; row++)
            {
                cval = x[ix_arr[row]];
                cval = (cval >= ncat)? (cval % ncat) : cval;
                if (cval < 0 || split_categ[cval] == 1)
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
                cval = x[ix_arr[row]];
                cval = (cval >= ncat)? (cval % ncat) : cval;
                if (cval >= 0 && split_categ[cval] == 1)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }
        st_NA = st;

        if (!(missing_action == Impute && !move_new_to_left))
        {
            for (size_t row = st; row <= end; row++)
            {
                if (unlikely(x[ix_arr[row]] < 0))
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

    /* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
    else
    {
        /* Note: if having 'new_cat_action'='Smallest' and 'missing_action'='Impute', missing values
           and new categories will necessarily go into different branches, thus it's possible to do
           all the movements in one pass if certain conditions match. */

        if (new_cat_action == Smallest && move_new_to_left)
        {
            for (size_t row = st; row <= end; row++)
            {
                cval = x[ix_arr[row]];
                if (cval >= 0 && (cval >= ncat || split_categ[cval] == 1 || split_categ[cval] == (-1)))
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        else if (missing_action == Impute && !move_new_to_left)
        {
            for (size_t row = st; row <= end; row++)
            {
                cval = x[ix_arr[row]];
                if (cval < ncat && (cval < 0 || split_categ[cval] == 1))
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
                cval = x[ix_arr[row]];
                if (cval >= 0 && cval < ncat && split_categ[cval] == 1)
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        st_NA = st;

        if (new_cat_action == Weighted && missing_action == Divide)
        {
            for (size_t row = st; row <= end; row++)
            {
                cval = x[ix_arr[row]];
                if (cval < 0 || cval >= ncat || split_categ[cval] == (-1))
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        else if (new_cat_action == Weighted)
        {
            for (size_t row = st; row <= end; row++)
            {
                cval = x[ix_arr[row]];
                if (cval >= 0 && (cval >= ncat || split_categ[cval] == (-1)))
                {
                    temp        = ix_arr[st];
                    ix_arr[st]  = ix_arr[row];
                    ix_arr[row] = temp;
                    st++;
                }
            }
        }

        else if (missing_action == Divide)
        {
            for (size_t row = st; row <= end; row++)
            {
                if (unlikely(x[ix_arr[row]] < 0))
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
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end, int split_categ,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept
{
    size_t temp;

    /* if NAs are not to be bothered with, just need to do a single pass */
    if (missing_action == Fail)
    {
        /* move to the left if it's equal to the chosen category */
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

    /* otherwise, first put to the left all equal to chosen and not NA, then all NAs to the end of the left */
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
            if (unlikely(x[ix_arr[row]] < 0))
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
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end,
                         MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix) noexcept
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
                if (unlikely(x[ix_arr[row]] < 0))
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
                if (unlikely(x[ix_arr[row]] < 0))
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
template <class real_t>
void get_range(size_t ix_arr[], real_t *restrict x, size_t st, size_t end,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable) noexcept
{
    xmin =  HUGE_VAL;
    xmax = -HUGE_VAL;
    double xval;

    if (missing_action == Fail)
    {
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            xmin = (xval < xmin)? xval : xmin;
            xmax = (xval > xmax)? xval : xmax;
        }
    }


    else
    {
        for (size_t row = st; row <= end; row++)
        {
            xval = x[ix_arr[row]];
            xmin = std::fmin(xmin, xval);
            xmax = std::fmax(xmax, xval);
        }
    }

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
}

template <class real_t>
void get_range(real_t *restrict x, size_t n,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable) noexcept
{
    xmin =  HUGE_VAL;
    xmax = -HUGE_VAL;

    if (missing_action == Fail)
    {
        for (size_t row = 0; row < n; row++)
        {
            xmin = (x[row] < xmin)? x[row] : xmin;
            xmax = (x[row] > xmax)? x[row] : xmax;
        }
    }


    else
    {
        for (size_t row = 0; row < n; row++)
        {
            xmin = std::fmin(xmin, x[row]);
            xmax = std::fmax(xmax, x[row]);
        }
    }

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
}

/* for sparse inputs */
template <class real_t, class sparse_ix>
void get_range(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
               real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable) noexcept
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
        Xc_ind[st_col]         >   (sparse_ix)ix_arr[end] || 
        (sparse_ix)ix_arr[st]  >   Xc_ind[end_col]
        )
    {
        unsplittable = true;
        return;
    }

    if (nnz_col < end - st + 1 ||
        Xc_ind[st_col]  > (sparse_ix)ix_arr[st] ||
        Xc_ind[end_col] < (sparse_ix)ix_arr[end]
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
            if (Xc_ind[curr_pos] == (sparse_ix)(*row))
            {
                nmatches++;
                xmin = (Xc[curr_pos] < xmin)? Xc[curr_pos] : xmin;
                xmax = (Xc[curr_pos] > xmax)? Xc[curr_pos] : xmax;
                if (row == ix_arr + end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
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

    else /* can have NAs */
    {
        for (size_t *row = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);
             row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
            )
        {
            if (Xc_ind[curr_pos] == (sparse_ix)(*row))
            {
                nmatches++;
                xmin = std::fmin(xmin, Xc[curr_pos]);
                xmax = std::fmax(xmax, Xc[curr_pos]);
                if (row == ix_arr + end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
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

    if (nmatches < (end - st + 1))
    {
        xmin = std::fmin(xmin, 0);
        xmax = std::fmax(xmax, 0);
    }

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
}

template <class real_t, class sparse_ix>
void get_range(size_t col_num, size_t nrows,
               real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable) noexcept
{
    xmin =  HUGE_VAL;
    xmax = -HUGE_VAL;

    if ((size_t)(Xc_indptr[col_num+1] - Xc_indptr[col_num]) < nrows)
    {
        xmin = 0;
        xmax = 0;
    }

    if (missing_action == Fail)
    {
        for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num+1]; ix++)
        {
            xmin = (Xc[ix] < xmin)? Xc[ix] : xmin;
            xmax = (Xc[ix] > xmax)? Xc[ix] : xmax;
        }
    }

    else
    {
        for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num+1]; ix++)
        {
            if (unlikely(std::isinf(Xc[ix]))) continue;
            xmin = std::fmin(xmin, Xc[ix]);
            xmax = std::fmax(xmax, Xc[ix]);
        }
    }

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
}


void get_categs(size_t *restrict ix_arr, int x[], size_t st, size_t end, int ncat,
                MissingAction missing_action, signed char categs[], size_t &restrict npresent, bool &unsplittable) noexcept
{
    std::fill(categs, categs + ncat, -1);
    npresent = 0;
    for (size_t row = st; row <= end; row++)
        if (likely(x[ix_arr[row]] >= 0))
            categs[x[ix_arr[row]]] = 1;

    npresent = std::accumulate(categs,
                               categs + ncat,
                               (size_t)0,
                               [](const size_t a, const signed char b){return a + (b > 0);}
                               );

    unsplittable = npresent < 2;
}

template <class real_t>
bool check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end, real_t x[], MissingAction missing_action)
{
    if (end - st <= 1) return false;

    if (missing_action == Fail)
    {
        real_t x0 = x[ix_arr[st]];
        for (size_t ix = st+1; ix <= end; ix++)
        {
            if (x[ix_arr[ix]] != x0) return true;
        }
    }

    else
    {
        real_t x0;
        size_t ix;
        for (ix = st; ix <= end; ix++)
        {
            if (likely(!is_na_or_inf(x[ix_arr[ix]])))
            {
                x0 = x[ix_arr[ix]];
                ix++;
                break;
            }
        }

        for (; ix <= end; ix++)
        {
            if (!is_na_or_inf(x[ix_arr[ix]]) && x[ix_arr[ix]] != x0)
                return true;
        }
    }

    return false;
}

bool check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end, int x[], MissingAction missing_action)
{
    if (end - st <= 1) return false;

    if (missing_action == Fail)
    {
        int x0 = x[ix_arr[st]];
        for (size_t ix = st+1; ix <= end; ix++)
        {
            if (x[ix_arr[ix]] != x0) return true;
        }
    }

    else
    {
        int x0;
        size_t ix;
        for (ix = st; ix <= end; ix++)
        {
            if (x[ix_arr[ix]] >= 0)
            {
                x0 = x[ix_arr[ix]];
                ix++;
                break;
            }
        }

        for (; ix <= end; ix++)
        {
            if (x[ix_arr[ix]] >= 0 && x[ix_arr[ix]] != x0)
                return true;
        }
    }

    return false;
}

template <class real_t, class sparse_ix>
bool check_more_than_two_unique_values(size_t *restrict ix_arr, size_t st, size_t end, size_t col,
                                       sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_ind, real_t *restrict Xc,
                                       MissingAction missing_action)
{
    if (end - st <= 1) return false;
    if (Xc_indptr[col+1] == Xc_indptr[col]) return false;
    bool has_zeros = (end - st + 1) > (size_t)(Xc_indptr[col+1] - Xc_indptr[col]);
    if (has_zeros && !is_na_or_inf(Xc[Xc_indptr[col]]) && Xc[Xc_indptr[col]] != 0) return true;

    size_t st_col  = Xc_indptr[col];
    size_t end_col = Xc_indptr[col + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];

    /* 'ix_arr' should be sorted beforehand */
    /* TODO: refactor this */
    real_t x0 = 0;
    size_t *row;
    for (row = std::lower_bound(ix_arr + st, ix_arr + end + 1, Xc_ind[st_col]);
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == (sparse_ix)(*row))
        {
            if (is_na_or_inf(Xc[curr_pos]) || (has_zeros && Xc[curr_pos] == 0))
            {
                if (row == ix_arr + end || curr_pos == end_col) return false;
                curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
            }

            x0 = Xc[curr_pos];
            if (has_zeros) return true;
            else if (x0 == 0) has_zeros = true;
            if (row == ix_arr + end || curr_pos == end_col) return false;
            curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
            break;
        }

        else
        {
            if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
            else
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
        }
    }

    for (;
         row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;
        )
    {
        if (Xc_ind[curr_pos] == (sparse_ix)(*row))
        {
            if (is_na_or_inf(Xc[curr_pos]) || (has_zeros && Xc[curr_pos] == 0))
            {
                if (row == ix_arr + end || curr_pos == end_col) break;
                curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
            }

            else if (Xc[curr_pos] != x0)
            {
                return true;
            }

            if (row == ix_arr + end || curr_pos == end_col) break;
            curr_pos = std::lower_bound(Xc_ind + curr_pos, Xc_ind + end_col + 1, *(++row)) - Xc_ind;
        }

        else
        {
            if (Xc_ind[curr_pos] > (sparse_ix)(*row))
                row = std::lower_bound(row + 1, ix_arr + end + 1, Xc_ind[curr_pos]);
            else
                curr_pos = std::lower_bound(Xc_ind + curr_pos + 1, Xc_ind + end_col + 1, *row) - Xc_ind;
        }
    }

    return false;
}

template <class real_t, class sparse_ix>
bool check_more_than_two_unique_values(size_t nrows, size_t col,
                                       sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_ind, real_t *restrict Xc,
                                       MissingAction missing_action)
{
    if (nrows <= 1) return false;
    if (Xc_indptr[col+1] == Xc_indptr[col]) return false;
    bool has_zeros = nrows > (size_t)(Xc_indptr[col+1] - Xc_indptr[col]);
    if (has_zeros && !is_na_or_inf(Xc[Xc_indptr[col]]) && Xc[Xc_indptr[col]] != 0) return true;

    real_t x0 = 0;
    sparse_ix ix;
    for (ix = Xc_indptr[col]; ix < Xc_indptr[col+1]; ix++)
    {
        if (!is_na_or_inf(Xc[ix]))
        {
            if (has_zeros && Xc[ix] == 0) continue;
            if (has_zeros) return true;
            else if (Xc[ix] == 0) has_zeros = true;
            x0 = Xc[ix];
            ix++;
            break;
        }
    }

    for (ix = Xc_indptr[col]; ix < Xc_indptr[col+1]; ix++)
    {
        if (!is_na_or_inf(Xc[ix]))
        {
            if (has_zeros && Xc[ix] == 0) continue;
            if (Xc[ix] != x0) return true;
        }
    }

    return false;
}

void count_categs(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat, size_t *restrict counts)
{
    std::fill(counts, counts + ncat, (size_t)0);
    for (size_t row = st; row <= end; row++)
        if (likely(x[ix_arr[row]] >= 0))
            counts[x[ix_arr[row]]]++;
}

int count_ncateg_in_col(const int x[], const size_t n, const int ncat, unsigned char buffer[])
{
    memset(buffer, 0, ncat*sizeof(char));
    for (size_t ix = 0; ix < n; ix++)
    {
        if (likely(x[ix] >= 0)) buffer[x[ix]] = true;
    }

    int ncat_present = 0;
    for (int cat = 0; cat < ncat; cat++)
        ncat_present += buffer[cat];
    return ncat_present;
}

template <class ldouble_safe>
ldouble_safe calculate_sum_weights(std::vector<size_t> &ix_arr, size_t st, size_t end, size_t curr_depth,
                                   std::vector<double> &weights_arr, hashed_map<size_t, double> &weights_map)
{
    if (curr_depth > 0 && !weights_arr.empty())
        return std::accumulate(ix_arr.begin() + st,
                               ix_arr.begin() + end + 1,
                               (ldouble_safe)0,
                               [&weights_arr](const ldouble_safe a, const size_t ix){return a + weights_arr[ix];});
    else if (curr_depth > 0 && !weights_map.empty())
        return std::accumulate(ix_arr.begin() + st,
                               ix_arr.begin() + end + 1,
                               (ldouble_safe)0,
                               [&weights_map](const ldouble_safe a, const size_t ix){return a + weights_map[ix];});
    else
        return -HUGE_VAL;
}

template <class real_t>
size_t move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, real_t x[])
{
    size_t st_non_na = st;
    size_t temp;

    for (size_t row = st; row <= end; row++)
    {
        if (unlikely(is_na_or_inf(x[ix_arr[row]])))
        {
            temp = ix_arr[st_non_na];
            ix_arr[st_non_na] = ix_arr[row];
            ix_arr[row] = temp;
            st_non_na++;
        }
    }

    return st_non_na;
}

template <class real_t, class sparse_ix>
size_t move_NAs_to_front(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num, real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr)
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
            if (unlikely(is_na_or_inf(Xc[curr_pos])))
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
        if (unlikely(x[ix_arr[row]] < 0))
        {
            temp = ix_arr[st_non_na];
            ix_arr[st_non_na] = ix_arr[row];
            ix_arr[row] = temp;
            st_non_na++;
        }
    }

    return st_non_na;
}

size_t center_NAs(size_t ix_arr[], size_t st_left, size_t st, size_t curr_pos)
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

/* FIXME / TODO: this calculation would not take weight into account */
/* Here:
   - 'ix_arr' should be partitioned putting the NAs and Infs at the beginning: [st_orig, st)
   - the rest of the range [st, end] should be sorted in ascending order
   The output should have a filled-in 'x' with median values, plus a re-sorted 'ix_arr'
   taking into account that now the median values are in the middle. */
template <class real_t>
void fill_NAs_with_median(size_t *restrict ix_arr, size_t st_orig, size_t st, size_t end, real_t *restrict x,
                          double *restrict buffer_imputed_x, double *restrict xmedian)
{
    size_t tot = end - st + 1;
    size_t idx_half = st + div2(tot);
    bool is_odd = (tot % 2) != 0;
    
    if (is_odd)
    {
        *xmedian = x[ix_arr[idx_half]];
        idx_half--;
    }

    else
    {
        idx_half--;
        double xlow = x[ix_arr[idx_half]];
        double xhigh = x[ix_arr[idx_half+(size_t)1]];
        *xmedian = xlow + (xhigh-xlow)/2.;
    }

    for (size_t ix = st_orig; ix < st; ix++)
        buffer_imputed_x[ix_arr[ix]] = (*xmedian);
    for (size_t ix = st; ix <= end; ix++)
        buffer_imputed_x[ix_arr[ix]] = x[ix_arr[ix]];

    /* 'ix_arr' can be resorted in-place, but the logic is a bit complex */
    /* step 1: move all NAs to their place by swapping them with the lower-half
       in ascending order (after this, the lower half will be unordered).
       along the way, copy the indices that claim the places where earlier
       there were missing values. these copied indices will be sorted in
       descending order at the end, as they were inserted in reverse order. */
    size_t end_pointer = idx_half;
    size_t n_move = std::min(st-st_orig, idx_half-st+1);
    size_t temp;
    for (size_t ix = st_orig; ix < st_orig + n_move; ix++)
    {
        temp = ix_arr[end_pointer];
        ix_arr[end_pointer] = ix_arr[ix];
        ix_arr[ix] = temp;
        end_pointer--;
    }

    /* step 2: reverse the indices that were moved to the beginning so
       as to maintain the sorting order */
    std::reverse(ix_arr + st_orig, ix_arr + st_orig + n_move);
    /* step 3: rotate the total number of elements by the number of moved elements */
    size_t n_unmoved = (idx_half - st + 1) - n_move;
    std::rotate(ix_arr + st_orig,
                ix_arr + st_orig + n_move,
                ix_arr + st_orig + n_move + n_unmoved);
}

template <class real_t, class sparse_ix>
void todense(size_t *restrict ix_arr, size_t st, size_t end,
             size_t col_num, real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
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
        if (Xc_ind[curr_pos] == (sparse_ix)(*row))
        {
            buffer_arr[row - (ix_arr + st)] = Xc[curr_pos];
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


template <class real_t>
void colmajor_to_rowmajor(real_t *restrict X, size_t nrows, size_t ncols, std::vector<double> &X_row_major)
{
    X_row_major.resize(nrows * ncols);
    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X_row_major[row + col*nrows] = X[col + row*ncols];
}

template <class real_t, class sparse_ix>
void colmajor_to_rowmajor(real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
                          size_t nrows, size_t ncols,
                          std::vector<double> &Xr, std::vector<size_t> &Xr_ind, std::vector<size_t> &Xr_indptr)
{
    /* First convert to COO */
    size_t nnz = Xc_indptr[ncols];
    std::vector<size_t> row_indices(nnz);
    for (size_t col = 0; col < ncols; col++)
    {
        for (sparse_ix ix = Xc_indptr[col]; ix < Xc_indptr[col+1]; ix++)
        {
            row_indices[ix] = Xc_ind[ix];
        }
    }

    /* Then copy the data argsorted by rows */
    std::vector<size_t> argsorted_indices(nnz);
    std::iota(argsorted_indices.begin(), argsorted_indices.end(), (size_t)0);
    std::stable_sort(argsorted_indices.begin(), argsorted_indices.end(),
                     [&row_indices](const size_t a, const size_t b)
                     {return row_indices[a] < row_indices[b];});
    Xr.resize(nnz);
    Xr_ind.resize(nnz);
    for (size_t ix = 0; ix < nnz; ix++)
    {
        Xr[ix] = Xc[argsorted_indices[ix]];
        Xr_ind[ix] = Xc_ind[argsorted_indices[ix]];
    }

    /* Now build the index pointer */
    Xr_indptr.resize(nrows+1);
    size_t curr_row = 0;
    size_t curr_n = 0;
    for (size_t ix = 0; ix < nnz; ix++)
    {
        if (row_indices[argsorted_indices[ix]] != curr_row)
        {
            Xr_indptr[curr_row+1] = curr_n;
            curr_n = 0;
            curr_row = row_indices[argsorted_indices[ix]];
        }

        else
        {
            curr_n++;
        }
    }
    for (size_t row = 1; row < nrows; row++)
        Xr_indptr[row+1] += Xr_indptr[row];
}


bool interrupt_switch = false;
bool handle_is_locked = false;

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
        throw std::runtime_error("Error: procedure was interrupted.\n");
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
        if (!handle_is_locked)
        {
            handle_is_locked = true;
            interrupt_switch = false;
            this->old_sig = signal(SIGINT, set_interrup_global_variable);
            this->is_active = true;
        }

        else {
            this->is_active = false;
        }
    }
}

SignalSwitcher::~SignalSwitcher()
{
    #ifndef _FOR_PYTHON
    #pragma omp critical
    {
        if (this->is_active && handle_is_locked)
            interrupt_switch = false;
    }
    #endif
    this->restore_handle();
}

void SignalSwitcher::restore_handle()
{
    #pragma omp critical
    {
        if (this->is_active && handle_is_locked)
        {
            signal(SIGINT, this->old_sig);
            this->is_active = false;
            handle_is_locked = false;
        }
    }
}

bool has_long_double()
{
    #ifndef NO_LONG_DOUBLE
    return sizeof(long double) > sizeof(double);
    #else
    return false;
    #endif
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
