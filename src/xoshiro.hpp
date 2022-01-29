/* This file was taken from here:
   https://prng.di.unimi.it
   And adapted as needed for inclusion into IsoTree */

/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */
#include <cstdint>
#include <cstring>
#if (__cplusplus  >= 202002L)
#include <bit>
#endif
using std::uint8_t;
using std::uint32_t;
using std::uint64_t;
using std::memcpy;

#ifndef _FOR_R
    #if defined(__clang__)
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wunknown-attributes"
    #endif
#endif

#if (__cplusplus >= 201703L) || (__cplusplus >= 201402L && (defined(__GNUC__) || defined(_MSC_VER)))
    #define SUPPORTS_HEXFLOAT
#endif

namespace Xoshiro {

#if (__cplusplus  >= 202002L)
#define rotl64(x, k) std::rotl(x, k)
#define rotl32(x, k) std::rotl(x, k)
#else
static inline uint64_t rotl64(const uint64_t x, const int k) noexcept {
    return (x << k) | (x >> (64 - k));
}
static inline uint32_t rotl32(const uint32_t x, const int k) noexcept {
    return (x << k) | (x >> (32 - k));
}
#endif

/* these are in order to avoid gcc warnings about 'strict aliasing rules' */
static inline uint32_t extract_32bits_from64_left(const uint64_t x) noexcept
{
    uint32_t out;
    memcpy(reinterpret_cast<char*>(&out),
           reinterpret_cast<const char*>(&x),
           sizeof(uint32_t));
    return out;
}

static inline uint32_t extract_32bits_from64_right(const uint64_t x) noexcept
{
    uint32_t out;
    memcpy(reinterpret_cast<char*>(&out),
           reinterpret_cast<const char*>(&x) + sizeof(uint32_t),
           sizeof(uint32_t));
    return out;
}

static inline void assign_32bits_to64_left(uint64_t &assign_to, const uint32_t take_from) noexcept
{
    memcpy(reinterpret_cast<char*>(&assign_to),
           reinterpret_cast<const char*>(&take_from),
           sizeof(uint32_t));
}

static inline void assign_32bits_to64_right(uint64_t &assign_to, const uint32_t take_from) noexcept
{
    memcpy(reinterpret_cast<char*>(&assign_to) + sizeof(uint32_t),
           reinterpret_cast<const char*>(&take_from),
           sizeof(uint32_t));
}

/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and 
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */
static inline uint64_t splitmix64(const uint64_t seed) noexcept
{
    uint64_t z = (seed + 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */
class Xoshiro256PP
{
public:
    using result_type = uint64_t;
    uint64_t state[4];

    constexpr static result_type min() noexcept
    {
        return 0;
    }

    constexpr static result_type max() noexcept
    {
        return UINT64_MAX;
    }

    Xoshiro256PP() noexcept = default;

    inline void seed(const uint64_t seed) noexcept
    {
        this->state[0] = splitmix64(splitmix64(seed));
        this->state[1] = splitmix64(this->state[0]);
        this->state[2] = splitmix64(this->state[1]);
        this->state[3] = splitmix64(this->state[2]);
    }

    template <class integer>
    inline void seed(const integer seed) noexcept
    {
        this->seed((uint64_t)seed);
    }

    Xoshiro256PP(const uint64_t seed) noexcept
    {
        this->seed(seed);
    }

    template <class integer>
    Xoshiro256PP(const integer seed) noexcept
    {
        this->seed((uint64_t)seed);
    }

    inline result_type operator()() noexcept
    {
        const uint64_t result = rotl64(this->state[0] + this->state[3], 23) + this->state[0];
        const uint64_t t = this->state[1] << 17;
        this->state[2] ^= this->state[0];
        this->state[3] ^= this->state[1];
        this->state[1] ^= this->state[2];
        this->state[0] ^= this->state[3];
        this->state[2] ^= t;
        this->state[3] = rotl64(this->state[3], 45);
        return result;
    }
};

/* This is xoshiro128++ 1.0, one of our 32-bit all-purpose, rock-solid
   generators. It has excellent speed, a state size (128 bits) that is
   large enough for mild parallelism, and it passes all tests we are aware
   of.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero. */
class Xoshiro128PP
{
public:
    using result_type = uint32_t;
    uint32_t state[4];

    constexpr static result_type min() noexcept
    {
        return 0;
    }

    constexpr static result_type max() noexcept
    {
        return UINT32_MAX;
    }

    Xoshiro128PP() noexcept = default;


    inline void seed(const uint64_t seed) noexcept
    {
        const auto t1 = splitmix64(seed);
        const auto t2 = splitmix64(t1);
        this->state[0] = extract_32bits_from64_left(t1);
        this->state[1] = extract_32bits_from64_right(t1);
        this->state[2] = extract_32bits_from64_left(t2);
        this->state[3] = extract_32bits_from64_right(t2);
    }

    inline void seed(const uint32_t seed) noexcept
    {
        uint64_t temp;
        assign_32bits_to64_left(temp, seed);
        assign_32bits_to64_right(temp, seed);
        this->seed(temp);
    }


    template <class integer>
    inline void seed(const integer seed) noexcept
    {
        this->seed((uint64_t)seed);
    }

    Xoshiro128PP(const uint32_t seed) noexcept
    {
        this->seed(seed);
    }

    Xoshiro128PP(const uint64_t seed) noexcept
    {
        this->seed(seed);
    }

    template <class integer>
    Xoshiro128PP(const integer seed) noexcept
    {
        this->seed((uint64_t)seed);
    }

    inline result_type operator()() noexcept
    {
        const uint32_t result = rotl32(this->state[0] + this->state[3], 7) + this->state[0];
        const uint32_t t = this->state[1] << 9;
        this->state[2] ^= this->state[0];
        this->state[3] ^= this->state[1];
        this->state[1] ^= this->state[2];
        this->state[0] ^= this->state[3];
        this->state[2] ^= t;
        this->state[3] = rotl32(this->state[3], 11);
        return result;
    }
};

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

constexpr static const uint64_t two53_i = (UINT64_C(1) << 53) - UINT64_C(1);
constexpr static const int64_t two53_ii = (INT64_C(1) << 53);
constexpr static const uint64_t two54_i = (UINT64_C(1) << 54) - UINT64_C(1);
constexpr static const uint64_t two52i = (UINT64_C(1) << 52) - UINT64_C(1);
constexpr static const uint32_t two22_i = (UINT32_C(1) << 22) - UINT32_C(1);
constexpr static const uint32_t two21_i = (UINT32_C(1) << 21) - UINT32_C(1);
constexpr static const uint32_t two20_i = (UINT32_C(1) << 20) - UINT32_C(1);
constexpr static const double ui64_d = (double)UINT64_MAX;
constexpr static const double i64_d = (double)INT64_MAX;
constexpr static const double twoPI = 2. * M_PI;

[[gnu::flatten, gnu::always_inline]]
static inline uint64_t gen_bits(Xoshiro256PP &rng) noexcept
{
    return rng();
}

[[gnu::flatten, gnu::always_inline]]
static inline uint64_t gen_bits(Xoshiro128PP &rng) noexcept
{
    uint64_t bits;
    assign_32bits_to64_left(bits, rng());
    assign_32bits_to64_right(bits, rng());
    return bits;
}


/* Note: the way in which endian-ness detection is handled here looks
   inefficient at a first glance. Nevertheless, do NOT try to optimize
   any further as GCC9 has a bug in which it optimizes away some 'if's'
   but with the *wrong* bit ending if done as ternary operators or if
   declaring pointer variables outside of the braces in what comes below. */
static inline bool get_is_little_endian() noexcept
{
    const uint32_t ONE = 1;
    return (*(reinterpret_cast<const unsigned char*>(&ONE)) != 0);
}
static const bool is_little_endian = get_is_little_endian();

/* ~Uniform([0,1))
   Be aware that the compilers headers may:
   - Produce a non-uniform distribution as they divide
     by the maximum value of the generator (not all numbers
     between zero and one are representable).
   - Draw from a closed interval [0,1] (infinitesimal chance
     that something will go wrong, but better not take it).
   (For example, GCC4 had bugs like those)
   Hence this replacement. It is not too much slower
   than what the compiler's header use. */
class UniformUnitInterval
{
public:
    UniformUnitInterval() noexcept = default;

    template <class A, class B>
    UniformUnitInterval(A a, B b) noexcept {}
    
    template <class XoshiroRNG>
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double operator()(XoshiroRNG &rng) noexcept
    {
        #if SIZE_MAX >= UINT64_MAX
        #   ifdef SUPPORTS_HEXFLOAT
        return (double)(gen_bits(rng) >> 11) * 0x1.0p-53;
        #   else
        return std::ldexp(gen_bits(rng) >> 11, -53);
        #   endif
        #else
        uint64_t bits = gen_bits(rng);
        char *rbits_ = reinterpret_cast<char*>(&bits);
        if (is_little_endian) rbits_ += sizeof(uint32_t);
        uint32_t rbits;
        memcpy(&rbits, rbits_, sizeof(uint32_t));
        rbits = rbits & two21_i;
        memcpy(rbits_, &rbits, sizeof(uint32_t));
        #   ifdef SUPPORTS_HEXFLOAT
        return (double)bits * 0x1.0p-53;
        #   else
        return std::ldexp(bits, -53);
        #endif
        #endif
    }
};

/* Note: this should sample in an open interval [-1,1].
   It's however quite hard to sample uniformly in an open
   interval with floating point numbers, since it'd require
   drawing a random number up to a power of 2 plus one, which
   does not translate into the required precision with
   increments of 2^n that IEEE754 floats have around the unit
   interval. Nevertheless, since it'd be less than ideal to
   output zero from here (that is, it would mean not taking
   a column when creating a random hyperplane), it instead
   will transform exact zeros into exact ones. */
class UniformMinusOneToOne
{
public:
    UniformMinusOneToOne() noexcept = default;

    template <class A, class B>
    UniformMinusOneToOne(A a, B b) noexcept {}

    template <class XoshiroRNG>
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double operator()(XoshiroRNG &rng) noexcept
    {
        #if SIZE_MAX >= UINT64_MAX
        #   ifdef SUPPORTS_HEXFLOAT
        double out = (double)((int64_t)(gen_bits(rng)  >> 10) - two53_ii) * 0x1.0p-53;
        #   else
        double out = std::ldexp((int64_t)(gen_bits(rng) >> 10) - two53_ii, -53);
        #endif
        if (unlikely(out == 0)) out = 1;
        return out;
        #else
        uint64_t bits = gen_bits(rng);
        char *rbits_ = reinterpret_cast<char*>(&bits);
        if (is_little_endian) rbits_ += sizeof(uint32_t);
        uint32_t rbits;
        memcpy(&rbits, rbits_, sizeof(uint32_t));
        rbits = rbits & two22_i;
        memcpy(rbits_, &rbits, sizeof(uint32_t));
        #   ifdef SUPPORTS_HEXFLOAT
        double out = (double)((int64_t)bits - two53_ii) * 0x1.0p-53;
        #   else
        double out = std::ldexp((int64_t)bits - two53_ii, -53);
        #endif
        if (unlikely(out == 0)) out = 1;
        return out;
        #endif
    }
};

/* Normal distribution sampled from uniform numbers using ziggurat method. */
#include "ziggurat.hpp"
class StandardNormalDistr
{
public:
    StandardNormalDistr() noexcept = default;

    template <class A, class B>
    StandardNormalDistr(A a, B b) noexcept {}

    template <class XoshiroRNG>
    #ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
    #endif
    double operator()(XoshiroRNG &rng) noexcept
    {
        repeat_draw:
        uint64_t rnd = gen_bits(rng);
        uint8_t rectangle = rnd & 255; /* <- number of rectangles (took 8 bits) */
        rnd >>= 8;
        uint8_t sign = rnd & 1; /* <- took 1 bit */
        /* there's currently 56 bits left, already used 1 for the sign, need to
           take 52 for for the uniform draw, so can chop off 3 more than what
           was taken to get there faster. */
        rnd >>= 4;
        double rnorm = rnd * wi_double[rectangle];
        if (likely(rnd < ki_double[rectangle]))
        {
            return sign? rnorm : -rnorm;
        }

        else
        {
            if (likely(rectangle != 0))
            {
                rnd = gen_bits(rng);
                #ifdef SUPPORTS_HEXFLOAT
                double runif = ((double)(rnd  >> 12) + 0.5) * 0x1.0p-52;
                #else
                double runif = ((double)(rnd >> 12) + 0.5);
                runif = std::ldexp(runif, -52);
                #endif
                if (runif * (fi_double[rectangle-1] - fi_double[rectangle])
                        <
                    std::exp(-0.5 * rnorm * rnorm) - fi_double[rectangle])
                {
                    return sign? rnorm : -rnorm;
                }
                goto repeat_draw;
            }

            else
            {
                double runif, runif2;
                double a_by_d;
                while (true)
                {
                    #ifdef SUPPORTS_HEXFLOAT
                    runif = ((double)(gen_bits(rng) >> 12) + 0.5) * 0x1.0p-52;
                    runif2 = ((double)(gen_bits(rng) >> 12) + 0.5) * 0x1.0p-52;
                    #else
                    runif = std::ldexp((double)(gen_bits(rng) >> 12) + 0.5, -52);
                    runif2 = std::ldexp((double)(gen_bits(rng) >> 12) + 0.5, -52);
                    #endif
                    a_by_d = -ziggurat_nor_inv_r * std::log(runif);
                    if (-2.0 * std::log(runif2) > a_by_d * a_by_d)
                    {
                        rnorm = ziggurat_nor_r + a_by_d;
                        return sign? rnorm : -rnorm;
                    }
                }
            }
        }
    }
};

}

#ifndef _FOR_R
    #if defined(__clang__)
        #pragma clang diagnostic pop
    #endif
#endif
