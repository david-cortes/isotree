/* Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Distributed under 3-clause BSD license with permission from the author,
see https://lists.debian.org/debian-legal/2004/12/msg00295.html

------------------------------------------------------------------

Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1995, 2000 by Stephen L. Moshier

This software is derived from the Cephes Math Library and is
incorporated herein by permission of the author.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

static inline void poly6_twice_return_p7
(
    const double x,
    const double *restrict coefs1,
    const double *restrict coefs2,
    double &restrict res1,
    double &restrict res2,
    double &restrict w7
)
{
    double w[6];
    w[0] = 1.;
    w[1] = x;
    w[2] = x*x;
    w[3] = x*w[2];
    w[4] = w[2]*w[2];
    w[5] = w[2]*w[3];
    w7 = w[3]*w[3];
    res1 = 0.;
    res2 = 0.;
    for (int ix = 0; ix < 6; ix++)
    {
        res1 = std::fma(coefs1[ix], w[ix], res1);
        res2 = std::fma(coefs2[ix], w[ix], res2);
    }
}

static inline double poly7(const double x, const double *restrict coefs)
{
    double w[7];
    w[0] = 1.;
    w[1] = x;
    w[2] = x*x;
    w[3] = x*w[2];
    w[4] = w[2]*w[2];
    w[5] = w[2]*w[3];
    w[6] = w[3]*w[3];

    double out = 0.;
    for (int ix = 0; ix < 7; ix++)
        out = std::fma(w[ix], coefs[ix], out);
    return out;
}

static const double coefs_12_m[6] = {
    0.25479851061131551,
   -0.32555031186804491,
   -0.65031853770896507,
   -0.28919126444774784,
   -0.045251321448739056,
   -0.0020713321167745952
};

static const double coefs_12_d[6] = {
    1.0,
    2.0767117023730469,
    1.4606242909763515,
    0.43593529692665969,
    0.054151797245674225,
    0.0021284987017821144
};

static double coefs_asy[] = {
    8.33333333333333333333E-2,
    -8.33333333333333333333E-3,
    3.96825396825396825397E-3,
    -4.16666666666666666667E-3,
    7.57575757575757575758E-3,
    -2.10927960927960927961E-2,
    8.33333333333333333333E-2
};

/* This is implemented only for positive non-integer inputs */
double digamma(double x)
{
    /* check for positive integer up to 64 */
    if (unlikely((x <= 64) && (x == std::floor(x)))) {
        return harmonic_recursive(1.0, (double)x) - EULERS_GAMMA;
    }

    double y = 0.;

    /* use the recurrence relation to move x into [1, 2] */
    if (x < 1.) {
        y -= 1. / x;
        x += 1.;
    }
    else if (x < 10.) {
        while (x > 2.) {
            x -= 1.;
            y += 1. / x;
        }
    }

    if (x < 1. || x > 2.) {
        double z = 1. / (x*x);
        return y + std::log(x) - 0.5/x - z*poly7(z, coefs_asy);
    }

    const double r1 = 1.46163214463740587234;
    const double r2 = 0.00000000033095646883;
    const double r3 = 0.9016312093258695918615325266959189453125e-19;
    const double Y = 0.99558162689208984;
    double m, d, p7;
    poly6_twice_return_p7(
        x - 1.,
        coefs_12_m,
        coefs_12_d,
        m,
        d,
        p7
    );
    double r = m / std::fma(p7, -0.55789841321675513e-6, d);
    double g = x - r1;
    g -= r2;
    g -= r3;
    return y + g*Y + g*r;
}
