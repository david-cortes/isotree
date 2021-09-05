#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "isotree_c.h"

/*  To compile this example, first build the package through the cmake system:
      mkdir build
      cd build
      cmake ..
      make
      sudo make install
      sudo ldconfig
   Then compile this single file and link to the shared library:
     gcc -o test isotree_c_ex.c -lisotree -std=c99 -lm

   Or to link against it without a system install, assuming the cmake system
   has already built the library under ./build and this command is called from
   the root folder:
     gcc -o test example/isotree_c_ex.c -std=c99 -lm -I./include -l:libisotree.so -L./build -Wl,-rpath,./build

   Then run with './test'
*/

void fill_random_normal(double array[], size_t n, uint64_t seed);
size_t get_idx_max(double array[], size_t n);
void throw_oom();
int main()
{
    /* Random data from a standard normal distribution
       (100 points generated randomly, plus 1 outlier added manually)
       Library assumes it is passed as a single-dimensional pointer,
       following column-major order (like Fortran) */
    int nrow = 101;
    int ncol = 2;
    double *X = (double*)malloc(nrow * ncol * sizeof(double));
    if (!X) throw_oom();
    uint64_t seed = 123;
    fill_random_normal(X, nrow * ncol, seed);

    /* Now add obvious outlier point (3,3) */
    #define get_ix(row, col) (row + col*nrow)
    X[get_ix(100, 0)] = 3.;
    X[get_ix(100, 1)] = 3.;

    /* Fit a small isolation forest model
       (see 'fit_model.cpp' for the documentation) */
    isotree_parameters_t params = allocate_default_isotree_parameters();
    if (!params) throw_oom();
    isotree_model_t model = isotree_fit(
        params,
        nrow,
        X,
        ncol,
        NULL,
        0,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
    );
    if (!model) throw_oom();

    /* Check which row has the highest outlier score
       (see file 'predict.cpp' for the documentation) */
    double *outlier_scores = (double*)malloc(nrow * sizeof(double));
    if (!outlier_scores) throw_oom();
    isotree_predict(
        model,
        outlier_scores,
        NULL,
        true,
        nrow,
        true,
        X,
        0,
        NULL,
        0,
        false,
        NULL,
        NULL,
        NULL
    );

    size_t row_highest = get_idx_max(outlier_scores, nrow);
    printf("Point with highest outlier score: [");
    printf("%.2f, %.2f]\n", X[get_ix(row_highest, 0)], X[get_ix(row_highest, 1)]);
    fflush(stdout);

    /* Free all the objects that were allocated */
    free(X);
    free(outlier_scores);
    delete_isotree_parameters(params);
    delete_isotree_model(model);
    return EXIT_SUCCESS;
}

/* Helpers used in this example */
void throw_oom()
{
    fprintf(stderr, "Out of memory.\n");
    exit(1);
}

size_t get_idx_max(double array[], size_t n)
{
    double xmax = -HUGE_VAL;
    size_t idx_max = SIZE_MAX;
    for (size_t ix = 0; ix < n; ix++) {
        if (array[ix] > xmax) {
            xmax = array[ix];
            idx_max = ix;
        }
    }
    return idx_max;
}

/* Draws a random uniform number in the interval [0, 2^64),
   using the splitmix64 algorithm: https://prng.di.unimi.it/splitmix64.c */
uint64_t gen_rand64(uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/* Converts a random number within the 'uint64_t' range to a uniformly-random
   'double' in the open unit interval. */
double bits_to_random_floating(uint64_t bits)
{
    return ((double)(bits >> 12) + 0.5) * 0x1.0p-52;
}

/* Applies the box-muller transform to an array of random uniform floating
   point numbers in the open unit interval in order to obtain normally-distributed
   random numbers. */
#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif
void apply_box_muller_transform(double array[], size_t n)
{
    const double twoPI = 2. * M_PI;
    double u, v;
    for (size_t ix = 0; ix < n / 2; ix++) {
        u = sqrt(-2. * log(array[2 * ix]));
        v = array[2 * ix + 1];
        array[2 * ix] = cos(twoPI * v) * u;
        array[2 * ix + 1] = sin(twoPI * v) * u;
    }
}

void fill_random_normal(double array[], size_t n, uint64_t seed)
{
    for (size_t ix = 0; ix < n; ix++)
        array[ix] = bits_to_random_floating(gen_rand64(&seed));
    apply_box_muller_transform(array, n);
}
