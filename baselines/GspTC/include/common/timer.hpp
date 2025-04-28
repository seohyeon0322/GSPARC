#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <algorithm>

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <float.h>
#include <sys/mman.h>
#ifdef USE_MKL_
#include <mkl.h>
#endif
#ifdef USE_ESSL_
#include <essl.h>
#include <cblas.h>
#include <lapacke.h>
#endif
#ifdef USE_OPENBLAS_
#include <cblas.h>
#include <lapacke.h>
#endif
#include <omp.h>

namespace common
{

#define MAP_HUGE_SHIFT 26
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
// set to 1 to use 1G hugepages and to 0 if you want to use 2M hugepages
#define USE_1G 0
    // set the PRE_ALLOC[_ALTO] definitions in main.c and alto.c accordingly for the general
    // usage of hugepages

    typedef unsigned long long IType;
#if 1
    typedef double FType;
#define POTRF dpotrf_
#define POTRS dpotrs_
#define GELSY dgelsy_
#define SYRK cblas_dsyrk
#else
    typedef float FType;
#define POTRF spotrf
#define POTRS spotrs
#define GELSY sgelsy
#define SYRK cblas_ssyrk
#endif

#define CACHELINE 64

#define ROW 1

    inline uint64_t ReadTSC(void)
    {
#if defined(__i386__)

        uint64_t x;
        __asm__ __volatile__(".byte 0x0f, 0x31" : "=A"(x));
        return x;

#elif defined(__x86_64__)

        uint32_t hi, lo;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)lo) | (((uint64_t)hi) << 32);

#elif defined(__powerpc__)

        uint64_t result = 0;
        uint64_t upper, lower, tmp;
        __asm__ __volatile__("0:                  \n"
                             "\tmftbu   %0           \n"
                             "\tmftb    %1           \n"
                             "\tmftbu   %2           \n"
                             "\tcmpw    %2,%0        \n"
                             "\tbne     0b         \n" : "=r"(upper), "=r"(lower),
                                                         "=r"(tmp));
        result = upper;
        result = result << 32;
        result = result | lower;
        return result;
#else
        return 0ULL;
#endif // defined(__i386__)
    }

#ifndef TIME
#define TIME 0
#endif
    static inline void BEGIN_TIMER(
        uint64_t *ticks)
    {
#if TIME
        *ticks = ReadTSC();
#endif
    }

    static inline void END_TIMER(
        uint64_t *ticks)
    {
#if TIME
        *ticks = ReadTSC();
#endif
    }

    void ELAPSED_TIME_ACCUMULATE(
        uint64_t start,
        uint64_t end,
        double *t_elapsed);

    void ELAPSED_TIME(
        uint64_t start,
        uint64_t end,
        double *t_elapsed);

    void PRINT_TIMER(
        const char *message,
        double t);

    void ASMTrace(char *str);

    void InitTSC(void);

    double ElapsedTime(uint64_t ticks);

}

#endif // COMMON_HPP_
