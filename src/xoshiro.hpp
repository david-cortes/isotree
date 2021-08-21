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
using std::uint32_t;
using std::uint64_t;
using std::memcpy;

namespace Xoshiro {

[[gnu::hot]]
static inline uint64_t rotl64(const uint64_t x, const int k) {
    return (x << k) | (x >> (64 - k));
}

[[gnu::hot]]
static inline uint32_t rotl32(const uint32_t x, const int k) {
    return (x << k) | (x >> (32 - k));
}

/* these are in order to avoid gcc warnings about 'strict aliasing rules' */
static inline uint32_t extract_32bits_from64_left(const uint64_t x)
{
    uint32_t out;
    memcpy(reinterpret_cast<char*>(&out),
           reinterpret_cast<const char*>(&x),
           sizeof(uint32_t));
    return out;
}

static inline uint32_t extract_32bits_from64_right(const uint64_t x)
{
    uint32_t out;
    memcpy(reinterpret_cast<char*>(&out),
           reinterpret_cast<const char*>(&x) + sizeof(uint32_t),
           sizeof(uint32_t));
    return out;
}

static inline void assign_32bits_to64_left(uint64_t &assign_to, const uint32_t take_from)
{
    memcpy(reinterpret_cast<char*>(&assign_to),
           reinterpret_cast<const char*>(&take_from),
           sizeof(uint32_t));
}

static inline void assign_32bits_to64_right(uint64_t &assign_to, const uint32_t take_from)
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
[[gnu::hot]]
static inline uint64_t splitmix64(const uint64_t seed)
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

    constexpr static result_type min()
    {
        return 0;
    }

    constexpr static result_type max()
    {
        return UINT64_MAX;
    }

    Xoshiro256PP() = default;

    inline void seed(const uint64_t seed)
    {
        this->state[0] = splitmix64(splitmix64(seed));
        this->state[1] = splitmix64(this->state[0]);
        this->state[2] = splitmix64(this->state[1]);
        this->state[3] = splitmix64(this->state[2]);
    }

    template <class integer>
    inline void seed(const integer seed)
    {
        this->seed((uint64_t)seed);
    }

    Xoshiro256PP(const uint64_t seed)
    {
        this->seed(seed);
    }

    template <class integer>
    Xoshiro256PP(const integer seed)
    {
        this->seed((uint64_t)seed);
    }

    [[gnu::hot]]
    inline result_type operator()()
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

    constexpr static result_type min()
    {
        return 0;
    }

    constexpr static result_type max()
    {
        return UINT32_MAX;
    }

    Xoshiro128PP() = default;


    inline void seed(const uint64_t seed)
    {
        const auto t1 = splitmix64(seed);
        const auto t2 = splitmix64(t1);
        this->state[0] = extract_32bits_from64_left(t1);
        this->state[1] = extract_32bits_from64_right(t1);
        this->state[2] = extract_32bits_from64_left(t2);
        this->state[3] = extract_32bits_from64_right(t2);
    }

    inline void seed(const uint32_t seed)
    {
        uint64_t temp;
        assign_32bits_to64_left(temp, seed);
        assign_32bits_to64_right(temp, seed);
        this->seed(temp);
    }


    template <class integer>
    inline void seed(const integer seed)
    {
        this->seed((uint64_t)seed);
    }

    Xoshiro128PP(const uint32_t seed)
    {
        this->seed(seed);
    }

    Xoshiro128PP(const uint64_t seed)
    {
        this->seed(seed);
    }

    template <class integer>
    Xoshiro128PP(const integer seed)
    {
        this->seed((uint64_t)seed);
    }

    [[gnu::hot]]
    inline result_type operator()()
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

static inline uint64_t gen_bits(Xoshiro256PP &rng)
{
    return rng();
}

static inline uint64_t gen_bits(Xoshiro128PP &rng)
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
static inline bool get_is_little_endian()
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
    UniformUnitInterval() = default;

    template <class A, class B>
    UniformUnitInterval(A a, B b) {}
    
    template <class XoshiroRNG>
    [[gnu::hot]]
    double operator()(XoshiroRNG &rng)
    {
        #if SIZE_MAX >= UINT64_MAX
        return std::ldexp((double)(gen_bits(rng) & two53_i), -53);
        #else
        uint64_t bits = gen_bits(rng);
        char *rbits_ = reinterpret_cast<char*>(&bits);
        if (is_little_endian) rbits_ += sizeof(uint32_t);
        uint32_t rbits;
        memcpy(&rbits, rbits_, sizeof(uint32_t));
        rbits = rbits & two21_i;
        memcpy(rbits_, &rbits, sizeof(uint32_t));
        return std::ldexp((double)bits, -53);
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
    UniformMinusOneToOne() = default;

    template <class A, class B>
    UniformMinusOneToOne(A a, B b) {}

    template <class XoshiroRNG>
    [[gnu::hot]]
    double operator()(XoshiroRNG &rng)
    {
        #if SIZE_MAX >= UINT64_MAX
        return std::ldexp((double)((int64_t)(gen_bits(rng) & two54_i) - two53_ii), -53);
        #else
        uint64_t bits = gen_bits(rng);
        char *rbits_ = reinterpret_cast<char*>(&bits);
        if (is_little_endian) rbits_ += sizeof(uint32_t);
        uint32_t rbits;
        memcpy(&rbits, rbits_, sizeof(uint32_t));
        rbits = rbits & two22_i;
        memcpy(rbits_, &rbits, sizeof(uint32_t));
        return std::ldexp((double)((int64_t)bits - two53_ii), -53);
        #endif
    }
};

/* Normal distribution using Box-Muller transform in raw form.
   This usually gives better results than the polar form, but
   it's slightly slower. Note that it requires drawing random
   uniform numbers in an open interval rather than half open.

   About the math:
   - It produces a uniform number [0,2^64-1]
   - Ignores the first 12 bits, setting it to [0,2^52-1]
   - Adds +0.5, leaving [2^-1, 2^52-2^-1]
   - Divides by 2^52, leaving [2^-53, 1-2^-53]
   Which is how it reaches an unbiased open uniform distribution. */
class StandardNormalDistr
{
public:
    double reserve;
    double has_reserve = false;

    StandardNormalDistr() = default;

    template <class A, class B>
    StandardNormalDistr(A a, B b) : has_reserve(false) {}

    template <class XoshiroRNG>
    [[gnu::hot]]
    double operator()(XoshiroRNG &rng)
    {
        double res;
        if (has_reserve) {
            res = this->reserve;
        }
        
        else {
            #if SIZE_MAX >= UINT64_MAX
            double rnd1 = std::ldexp(((double)(gen_bits(rng) & two52i) + 0.5), -52);
            double rnd2 = std::ldexp(((double)(gen_bits(rng) & two52i) + 0.5), -52);
            #else
            double rnd1, rnd2;
            uint64_t bits1 = gen_bits(rng);
            uint64_t bits2 = gen_bits(rng);
            char *rbits1_ = reinterpret_cast<char*>(&bits1);
            char *rbits2_ = reinterpret_cast<char*>(&bits2);
            if (is_little_endian) {
                rbits1_ += sizeof(uint32_t);
                rbits2_ += sizeof(uint32_t);
            }
            uint32_t rbits1, rbits2;
            memcpy(&rbits1, rbits1_, sizeof(uint32_t));
            rbits1 = rbits1 & two20_i;
            memcpy(rbits1_, &rbits1, sizeof(uint32_t));
            memcpy(&rbits2, rbits2_, sizeof(uint32_t));
            rbits2 = rbits2 & two20_i;
            memcpy(rbits2_, &rbits2, sizeof(uint32_t));
            rnd1 = std::ldexp((double)bits1 + 0.5, -52);
            rnd2 = std::ldexp((double)bits2 + 0.5, -52);
            #endif

            rnd1 = std::sqrt(-2. * std::log(rnd1));
            res = std::cos(twoPI * rnd2) * rnd1;
            this->reserve = std::sin(twoPI * rnd2) * rnd1;
        }

        this->has_reserve = !this->has_reserve;
        if (!res) return this->operator()(rng);
        return res;
    }
};

}
