/* This file was taken from here:
   https://prng.di.unimi.it
   And adapted as needed for inclusion into IsoTree */

/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */
namespace Xorshiro {

#include <cstdint>
#include <cstring>
using std::uint32_t;
using std::uint64_t;
using std::memcpy;

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
    memcpy(&out, reinterpret_cast<const uint32_t*>(&x), sizeof(uint32_t));
    return out;
}

static inline uint32_t extract_32bits_from64_right(const uint64_t x)
{
    uint32_t out;
    memcpy(&out, reinterpret_cast<const uint32_t*>(&x) + 1, sizeof(uint32_t));
    return out;
}

static inline void assign_32bits_to64_left(uint64_t assign_to, const uint32_t take_from)
{
    memcpy(reinterpret_cast<uint32_t*>(&assign_to), &take_from, sizeof(uint32_t));
}

static inline void assign_32bits_to64_right(uint64_t assign_to, const uint32_t take_from)
{
    memcpy(reinterpret_cast<uint32_t*>(&assign_to) + 1, &take_from, sizeof(uint32_t));
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

    constexpr uint64_t min()
    {
        return 0;
    }

    constexpr uint64_t max()
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

    constexpr uint32_t min()
    {
        return 0;
    }

    constexpr uint32_t max()
    {
        return UINT32_MAX;
    }

    Xoshiro128PP() = default;


    inline void seed(const uint64_t seed)
    {
        const auto t1 = splitmix64(seed);
        const auto t2 = splitmix64(t1);
        this->state[0] = splitmix64(extract_32bits_from64_left(t1));
        this->state[1] = splitmix64(extract_32bits_from64_right(t1));
        this->state[2] = splitmix64(extract_32bits_from64_left(t2));
        this->state[3] = splitmix64(extract_32bits_from64_right(t2));
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

}
