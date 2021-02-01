/*  Note: the R and Python versions calls the 'sort_csc_indices' templated function,
    so it's not enough to just include 'isotree_exportable.hpp' under them and let
    this same file instantiate all supported templated types.
    Also, Cython makes it hard to use overloaded functions since they have to
    be declared multiple times. */

#if !defined(_FOR_R) && !defined(_FOR_PYTHON)

#include "model_joined.hpp"

#define real_t double
#define sparse_ix int
#include "instantiate_model.hpp"
#undef real_t
#undef sparse_ix

#define real_t double
#define sparse_ix int64_t
#include "instantiate_model.hpp"
#undef real_t
#undef sparse_ix

#define real_t double
#define sparse_ix size_t
#include "instantiate_model.hpp"
#undef real_t
#undef sparse_ix

#define _NO_REAL_T

#define real_t float
#define sparse_ix int
#include "instantiate_model.hpp"
#undef real_t
#undef sparse_ix

#define real_t float
#define sparse_ix int64_t
#include "instantiate_model.hpp"
#undef real_t
#undef sparse_ix

#define real_t float
#define sparse_ix size_t
#include "instantiate_model.hpp"
#undef real_t
#undef sparse_ix

#undef _NO_REAL_T

#endif
