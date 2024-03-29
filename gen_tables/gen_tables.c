#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EULERS_GAMMA 0.577215664901532860606512
#define square(x) ((x) * (x))
double harmonic_recursive(double a, double b)
{
    if (b == a + 1) return 1. / a;
    double m = floor((a + b) / 2.);
    return harmonic_recursive(a, m) + harmonic_recursive(m, b);
}

double harmonic(int n)
{
    if (n > 256)
    {
        double temp = 1.0 / square((double)n);
        return  - 0.5 * temp * ( 1./6.  -   temp * (1./60. - (1./126.)*temp) )
                + 0.5 * (1./(double)n)
                + log((double)n) + EULERS_GAMMA;
    }
    
    else
    {
        return harmonic_recursive((double)1, (double)(n + 1));
    }
}

double expected_avg_depth(int sample_size)
{
    return 2. * (harmonic(sample_size) - 1.);
}

#define N_PRECALC 512

int main()
{
    FILE *fout = fopen("../src/exp_depth_table.hpp", "w");
    fprintf(fout, "/* This file is auto-generated by 'gen_tables.c', do not edit by hand. */\n");
    fprintf(fout, "#define N_PRECALC_EXP_DEPTH %d\n", N_PRECALC);
    fprintf(fout, "\n\nconstexpr static const double exp_depth_table[%d] = {\n", N_PRECALC);
    
    double exp_d[N_PRECALC];
    for (int ix = 0; ix < N_PRECALC; ix++) {
        exp_d[ix] = expected_avg_depth(ix+1);
    }

    for (int ln = 0; ln < N_PRECALC-4; ln += 4) {
        fprintf(fout, "    %.17f, %.17f, %.17f, %.17f,\n",
                exp_d[ln], exp_d[ln+1], exp_d[ln+2], exp_d[ln+3]);
    }
    fprintf(fout, "    %.17f, %.17f, %.17f, %.17f\n",
            exp_d[N_PRECALC-4], exp_d[N_PRECALC-3], exp_d[N_PRECALC-2], exp_d[N_PRECALC-1]);

    fprintf(fout, "};\n");
    fclose(fout);
    return EXIT_SUCCESS;    
}
