#ifndef BITOPS_HPP_
#define BITOPS_HPP_
#include <iostream>

namespace common
{
#include <immintrin.h> // for BMI2 instructions

    static inline uint64_t //__attribute__((target("bmi2")))
    pdep(uint64_t x, uint64_t y)
    {
        return _pdep_u64(x, y);
    }

    static inline uint64_t //__attribute__((target("bmi2")))
    pext(uint64_t x, uint64_t y)
    {
        return _pext_u64(x, y);
    }

    static inline int
    popcount(uint64_t x)
    {
        return __builtin_popcountll(x);
    }

    static inline int
    clz(uint64_t x)
    {
        return __builtin_clzll(x);
    }

    static inline __uint128_t //__attribute__((target("bmi2")))
    pdep(uint64_t x, __uint128_t y)
    {
        uint64_t ylow = y & 0xffffffffffffffff;

        int shift = __builtin_popcountll(ylow);
        __uint128_t res = _pdep_u64(x >> shift, y >> 64);
        res = _pdep_u64(x, ylow) | (res << 64);
        return res;
    }

    static inline __uint128_t //__attribute__((target("bmi2")))
    pext(__uint128_t x, __uint128_t y)
    {
        uint64_t ylow = y & 0xffffffffffffffff;
        uint64_t xlow = x & 0xffffffffffffffff;

        int shift = __builtin_popcountll(ylow);
        return (static_cast<__uint128_t>(_pext_u64(x >> 64, y >> 64)) << shift) | _pext_u64(xlow, ylow);
    }
    static inline int
    popcount(__uint128_t x)
    {
        return __builtin_popcountll(x >> 64) + __builtin_popcountll(x & 0xffffffffffffffff);
    }

    static inline int
    clz(__uint128_t x)
    {
        uint64_t xhi = x >> 64;
        if (!xhi)
            return 64 + __builtin_clzll(x & 0xffffffffffffffff);
        return __builtin_clzll(xhi);
    }

    static inline uint64_t
    lhalf(__uint128_t x)
    {
        uint64_t y = x & 0xffffffffffffffff;
        return y;
    }

    static inline uint64_t
    uhalf(__uint128_t x)
    {
        // uint64_t y = (x >> 64) & 0xffffffffffffffff;
        return (x > ((__uint128_t(1) << 64) - 1)) ? (x >> 64) : 0;

        // return y;
    }

} // namespace common

#endif
