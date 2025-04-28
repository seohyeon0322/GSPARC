#ifndef _SIZE_H__
#define _SIZE_H__

#include <cstddef>
#include <inttypes.h>

namespace common
{
    template <typename T = size_t>
    constexpr T KiB(size_t const i)
    {
        return i << 10;
    }

    template <typename T = size_t>
    constexpr T MiB(size_t const i)
    {
        return i << 20;
    }

    template <typename T = size_t>
    constexpr T GiB(size_t const i)
    {
        return i << 30;
    }

    template <typename T = size_t>
    constexpr T TiB(size_t const i)
    {
        return i << 40;
    }

    template <typename T1, typename T2>
    constexpr T1 aligned_size(const T1 size, const T2 align)
    {
        return align * ((size + align - 1) / align);
    }

    inline char *byteToString(uint64_t const bytes)
    {
        double size = (double)bytes;
        int suff = 0;
        const char *suffix[5] = {"B", "KiB", "MiB", "GiB", "TiB"};
        while (size > 1024 && suff < 4)
        {
            size /= 1024.;
            ++suff;
        }
        char *ret = NULL;
        if (asprintf(&ret, "%0.2f %s", size, suffix[suff]) == -1)
        {
            fprintf(stderr, "SPT: asprintf failed with %" PRIu64 " bytes.\n", bytes);
            ret = NULL;
        }
        return ret;
    }

}
#endif // _SIZE_H__