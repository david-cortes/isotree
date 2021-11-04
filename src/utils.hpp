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

/* ceil(log2(x)) done with bit-wise operations ensures perfect precision (and it's faster too)
   https://stackoverflow.com/questions/2589096/find-most-significant-bit-left-most-that-is-set-in-a-bit-array
   https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers  */
#if SIZE_MAX == UINT32_MAX /* 32-bit systems */
    constexpr static const int MultiplyDeBruijnBitPosition[32] =
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
    #if __cplusplus  >= 202002L
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

#define THRESHOLD_EXACT_H 256 /* above this will get approximated */

/* adapted from cephes */
#define EULERS_GAMMA 0.577215664901532860606512
double digamma(double x)
{
    double y, z, z2;

    /* check for positive integer up to 10 */
    if( (x <= THRESHOLD_EXACT_H) && (x == std::floor(x)) )
        return harmonic(x - 1) - EULERS_GAMMA;

    if( x < 1.0e17 )
    {
        z = 1.0/(x * x);
        z2 = square(z);
        y = z * ( 8.33333333333333333333E-2
                 -8.33333333333333333333E-3*z
                 +3.96825396825396825397E-3*z2
                 -4.16666666666666666667E-3*z2*z
                 +7.57575757575757575758E-3*square(z2)
                 -2.10927960927960927961E-2*square(z2)*z
                 +8.33333333333333333333E-2*square(z2)*z2);
    }
    else {
        y = 0.0;
    }

    y = std::log(x)  -  (0.5/x)  -  y;
    return y;
}

/* http://fredrik-j.blogspot.com/2009/02/how-not-to-compute-harmonic-numbers.html
   https://en.wikipedia.org/wiki/Harmonic_number
   https://github.com/scikit-learn/scikit-learn/pull/19087 */
double harmonic(size_t n)
{
    if (n > THRESHOLD_EXACT_H)
        return std::log((long double)n) + (long double)EULERS_GAMMA
                + 0.5 * (1./(long double)n)
                - 0.5 * (1./square((long double)n))
                      * ( 1./6. -   (1./square((long double)n))
                                  * (1./60. - (1./126.)*(1./square((long double)n))) );
    else
        return harmonic_recursive((double)1, (double)(n + 1));
}

double harmonic_recursive(double a, double b)
{
    if (b == a + 1) return 1. / a;
    double m = std::floor((a + b) / 2.);
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
            return 2. * (harmonic(sample_size) - 1.);
        }
    }
}

/* Note: H(x) = psi(x+1) + gamma */
double expected_avg_depth(long double approx_sample_size)
{
    if (approx_sample_size <= 1)
        return 0;
    else if (approx_sample_size < (long double)INT32_MAX)
        return 2. * (digamma(approx_sample_size + 1.) + EULERS_GAMMA - 1.);
    else
        return 2. * std::log(approx_sample_size) + 2.*((long double)EULERS_GAMMA - 1.)
                + (1./approx_sample_size)
                - (1./square(approx_sample_size))
                   * ( 1./6. -   (1./square(approx_sample_size))
                               * (1./60. - (1./126.)*(1./square(approx_sample_size))) );
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
    double s_l = expected_separation_depth((size_t) std::floor(n));
    long double u = std::ceil(n);
    double s_u = s_l + (-s_l * u + 3. * u - 4.) / (u * (u - 1.));
    double diff = n - std::floor(n);
    return s_l + diff * s_u;
}

#define ix_comb_(i, j, n, ncomb) (  ((ncomb)  + ((j) - (i))) - (size_t)1 - div2(((n) - (i)) * ((n) - (i) - (size_t)1))  )
#define ix_comb(i, j, n, ncomb) (  ((i) < (j))? ix_comb_(i, j, n, ncomb) : ix_comb_(j, i, n, ncomb)  )
void increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n, double counter[], double exp_remainder)
{
    size_t i, j;
    size_t ncomb = ((n % 2) == 0)? (div2(n) * (n-(size_t)1)) : (n * div2(n-(size_t)1));
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
    size_t ncomb = ((n % 2) == 0)? (div2(n) * (n-(size_t)1)) : (n * div2(n-(size_t)1));
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
    size_t ncomb = ((n % 2) == 0)? (div2(n) * (n-(size_t)1)) : (n * div2(n-(size_t)1));
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
    size_t ncomb = ((n % 2) == 0)? (div2(n) * (n-(size_t)1)) : (n * div2(n-(size_t)1));
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
    
    if (isnan(btree_weights[0]) || btree_weights[0] <= 0)
    {
        fprintf(stderr, "Numeric precision error with sample weights, will not use them.\n");
        log2_n = 0;
        btree_weights.clear();
        btree_weights.shrink_to_fit();
    }
}

template <class real_t>
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
            if (((long double)ntake / (long double)nrows) > (1. / 50.))
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
    if (isnan(buffer_arr[0]) || buffer_arr[0] <= 0)
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

double sample_random_uniform(double xmin, double xmax, RNG_engine &rng)
{
    double out;
    std::uniform_real_distribution<double> runif(xmin, xmax);
    for (int attempt = 0; attempt < 100; attempt++)
    {
        out = runif(rng);
        if (out < xmax) return out;
    }
    return xmin;
}

/*  This one samples with replacement. When using weights, the algorithm is the
    same as for the row sampler, but keeping the weights after taking each iteration. */
/*  TODO: this column sampler could use coroutines from C++20 once compilers implement them. */
template <class real_t>
void ColumnSampler::initialize(real_t weights[], size_t n_cols)
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
    if (isnan(this->tree_weights[0]) || this->tree_weights[0] <= 0)
    {
        this->drop_weights();
    }

    this->n_dropped = 0;
}

void ColumnSampler::drop_weights()
{
    this->tree_weights.clear();
    this->tree_weights.shrink_to_fit();
    this->initialize(n_cols);
    this->n_dropped = 0;
}

bool ColumnSampler::has_weights()
{
    return !this->tree_weights.empty();
}

void ColumnSampler::initialize(size_t n_cols)
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
void ColumnSampler::leave_m_cols(size_t m, RNG_engine &rnd_generator)
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

        else if ((long double)m >= (long double)(3./4.) * (long double)this->n_cols)
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

void ColumnSampler::drop_col(size_t col)
{
    if (!this->has_weights())
    {
        std::swap(this->col_indices[this->last_given], this->col_indices[--this->curr_pos]);
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

void ColumnSampler::prepare_full_pass()
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

bool ColumnSampler::sample_col(size_t &col, RNG_engine &rnd_generator)
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

bool ColumnSampler::sample_col(size_t &col)
{
    if (this->curr_pos == this->curr_col || this->curr_pos == 0)
        return false;
    this->last_given = this->curr_col;
    col = this->col_indices[this->curr_col++];
    return true;
}

void ColumnSampler::shuffle_remainder(RNG_engine &rnd_generator)
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

size_t ColumnSampler::get_remaining_cols()
{
    if (!this->has_weights())
        return this->curr_pos;
    else
        return this->n_cols - this->n_dropped;
}

bool SingleNodeColumnSampler::initialize
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
        for (size_t col = 0; col < this->curr_pos; col++)
            this->cumw += weights[this->col_indices[col]];

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

bool SingleNodeColumnSampler::sample_col(size_t &col_chosen, RNG_engine &rnd_generator)
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
            for (size_t col = 0; col < this->curr_pos; col++)
                this->weights_orig[this->col_indices[col]] = this->weights_own[this->col_indices[col]]? 1 : 0;
        }

        /* if there are no infinites, choose a column according to weight */
        ldouble_safe chosen = std::uniform_real_distribution<ldouble_safe>((ldouble_safe)0, this->cumw)(rnd_generator);
        ldouble_safe cumw = 0;
        for (size_t col = 0; col < this->curr_pos; col++)
        {
            cumw += this->weights_orig[this->col_indices[col]];
            if (cumw >= chosen)
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

void SingleNodeColumnSampler::backup(SingleNodeColumnSampler &other, size_t ncols_tot)
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

void SingleNodeColumnSampler::restore(const SingleNodeColumnSampler &other)
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


void DensityCalculator::initialize(size_t max_depth, int max_categ, bool reserve_counts, ScoringMetric scoring_metric)
{
    this->multipliers.reserve(max_depth+1);
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

void DensityCalculator::push_density(double xmin, double xmax, double split_point)
{
    if (std::isinf(xmax) || std::isinf(xmin))
    {
        this->multipliers.push_back(0);
        return;
    }

    double range = xmax - xmin;
    if (range <= 0) range = std::numeric_limits<double>::min();
    double mult_left = std::log((split_point - xmin) / range);
    double mult_right = std::log((xmax - split_point) / range);
    if (std::isinf(mult_left))
    {
        double diff = split_point - xmin;
        do {
            diff = std::nextafter(diff, HUGE_VAL);
            mult_left = std::log(diff / range);
        } while (std::isinf(mult_left));
    }

    mult_left = std::isnan(mult_left)? 0 : mult_left;
    mult_right = std::isnan(mult_right)? 0 : mult_right;

    ldouble_safe curr = this->multipliers.back();
    this->multipliers.back() = curr + mult_right;
    this->multipliers.push_back(curr + mult_left);
}

void DensityCalculator::push_density(int n_left, int n_present)
{
    this->push_density(0., (double)n_present, (double)n_left);
}

/* For single category splits */
void DensityCalculator::push_density(size_t counts[], int ncat)
{
    /* this one assumes 'categ_present' has entries 0/1 for missing/present */
    int n_present = 0;
    for (size_t cat = 0; cat < ncat; cat++)
        n_present += counts[cat] > 0;
    this->push_density(0., (double)n_present, 1.);
}

/* For single category splits */
void DensityCalculator::push_density(int n_present)
{
    this->push_density(0., (double)n_present, 1.);
}

/* For binary categorical splits */
void DensityCalculator::push_density()
{
    this->multipliers.push_back(0);
}

void DensityCalculator::push_adj(double xmin, double xmax, double split_point, double pct_tree_left, ScoringMetric scoring_metric)
{
    double range = xmax - xmin;
    double chunk_left = (split_point - xmin) / range;
    double chunk_right = (xmax - split_point) / range;
    if (std::isinf(xmax) || std::isinf(xmin))
    {
        chunk_left = pct_tree_left;
        chunk_right = 1. - pct_tree_left;
        goto add_chunks;
    }

    if (range <= 0) range = std::numeric_limits<double>::min();

    if (!std::isnan(chunk_left) && chunk_left <= 0)
    {
        double diff = split_point - xmin;
        do {
            diff = std::nextafter(diff, HUGE_VAL);
            chunk_left = diff / range;
        } while (chunk_left == 0);
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
        this->multipliers.back() = curr + chunk_right;
        this->multipliers.push_back(curr + chunk_left);
    }

    else 
    {
        this->multipliers.back() = std::fmax(curr * chunk_right, (ldouble_safe)std::numeric_limits<double>::epsilon());
        this->multipliers.push_back(std::fmax(curr * chunk_left, (ldouble_safe)std::numeric_limits<double>::epsilon()));
    }
}

void DensityCalculator::push_adj(signed char *restrict categ_present, size_t *restrict counts, int ncat, ScoringMetric scoring_metric)
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

    double pct_tree_left = (long double)cnt_left / (long double)cnt;
    this->push_adj(0., (double)cnt_cat, (double)cnt_cat_left, pct_tree_left, scoring_metric);
}

/* For single category splits */
void DensityCalculator::push_adj(size_t *restrict counts, int ncat, int chosen_cat, ScoringMetric scoring_metric)
{
    /* this one assumes 'categ_present' has entries 0/1 for missing/present */
    int cnt_cat = 0;
    size_t cnt = 0;
    for (int cat = 0; cat < ncat; cat++)
    {
        cnt += counts[cat];
        cnt_cat += counts[cat] > 0;
    }

    double pct_tree_left = (long double)counts[chosen_cat] / (long double)cnt;
    this->push_adj(0., (double)cnt_cat, 1., pct_tree_left, scoring_metric);
}

/* For binary categorical splits */
void DensityCalculator::push_adj(double pct_tree_left, ScoringMetric scoring_metric)
{
    this->push_adj(0., 1., 0.5, pct_tree_left, scoring_metric);
}

void DensityCalculator::pop()
{
    this->multipliers.pop_back();
}

double DensityCalculator::calc_density(long double remainder, size_t sample_size)
{
    long double res = std::exp( std::log(remainder) - std::log((long double)sample_size) - this->multipliers.back() );
    return std::fmin(res, std::log2((long double)sample_size));
    // return std::fmin(res, expected_avg_depth(sample_size));
    // return res;
    // return std::fmin(res, 2.0l*std::log2((long double)sample_size));
}

ldouble_safe DensityCalculator::calc_adj_depth()
{
    ldouble_safe out = this->multipliers.back();
    return std::fmax(out, (ldouble_safe)std::numeric_limits<double>::min());
}

double DensityCalculator::calc_adj_density()
{
    return this->multipliers.back();
}

void DensityCalculator::save_range(double xmin, double xmax)
{
    this->xmin = xmin;
    this->xmax = xmax;
}

void DensityCalculator::restore_range(double &restrict xmin, double &restrict xmax)
{
    xmin = this->xmin;
    xmax = this->xmax;
}

void DensityCalculator::save_counts(size_t *restrict cat_counts, int ncat)
{
    this->counts.assign(cat_counts, cat_counts + ncat);
}

void DensityCalculator::save_n_present_and_left(signed char *restrict split_left, int ncat)
{
    this->n_present = 0;
    this->n_left = 0;
    for (int cat = 0; cat < ncat; cat++)
    {
        this->n_present += split_left[cat] >= 0;
        this->n_left += split_left[cat] == 1;
    }
}

void DensityCalculator::save_n_present(size_t *restrict cat_counts, int ncat)
{
    this->n_present = 0;
    for (int cat = 0; cat < ncat; cat++)
        this->n_present +=  cat_counts[cat] > 0;
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
template <class real_t>
void divide_subset_split(size_t *restrict ix_arr, real_t x[], size_t st, size_t end, double split_point,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix)
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
template <class real_t, class sparse_ix>
void divide_subset_split(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
                         real_t Xc[], sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr, double split_point,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix)
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
                    if (isnan(Xc[curr_pos])) has_NAs = true;
                    if (!isnan(Xc[curr_pos]) && Xc[curr_pos] <= split_point)
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
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix)
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
                         bool move_new_to_left, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix)
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
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end, int split_categ,
                         MissingAction missing_action, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix)
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
void divide_subset_split(size_t *restrict ix_arr, int x[], size_t st, size_t end,
                         MissingAction missing_action, NewCategAction new_cat_action,
                         bool move_new_to_left, size_t &restrict st_NA, size_t &restrict end_NA, size_t &restrict split_ix)
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
template <class real_t>
void get_range(size_t ix_arr[], real_t *restrict x, size_t st, size_t end,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable)
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

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || isnan(xmin) || isnan(xmax);
}

template <class real_t>
void get_range(real_t *restrict x, size_t n,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable)
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

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || isnan(xmin) || isnan(xmax);
}

/* for sparse inputs */
template <class real_t, class sparse_ix>
void get_range(size_t *restrict ix_arr, size_t st, size_t end, size_t col_num,
               real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable)
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

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || isnan(xmin) || isnan(xmax);
}

template <class real_t, class sparse_ix>
void get_range(size_t col_num, size_t nrows,
               real_t *restrict Xc, sparse_ix *restrict Xc_ind, sparse_ix *restrict Xc_indptr,
               MissingAction missing_action, double &restrict xmin, double &restrict xmax, bool &unsplittable)
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
            if (std::isinf(Xc[ix])) continue;
            xmin = std::fmin(xmin, Xc[ix]);
            xmax = std::fmax(xmax, Xc[ix]);
        }
    }

    unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || isnan(xmin) || isnan(xmax);
}


void get_categs(size_t *restrict ix_arr, int x[], size_t st, size_t end, int ncat,
                MissingAction missing_action, signed char categs[], size_t &restrict npresent, bool &unsplittable)
{
    std::fill(categs, categs + ncat, -1);
    npresent = 0;
    for (size_t row = st; row <= end; row++)
        if (x[ix_arr[row]] >= 0)
            categs[x[ix_arr[row]]] = 1;

    npresent = std::accumulate(categs,
                               categs + ncat,
                               (size_t)0,
                               [](const size_t a, const signed char b){return a + (b > 0);}
                               );

    unsplittable = npresent < 2;
}

void count_categs(size_t *restrict ix_arr, size_t st, size_t end, int x[], int ncat, size_t *restrict counts)
{
    std::fill(counts, counts + ncat, (size_t)0);
    for (size_t row = st; row <= end; row++)
        if (x[ix_arr[row]] >= 0)
            counts[x[ix_arr[row]]]++;
}

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
