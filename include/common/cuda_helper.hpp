#ifndef CUDA_HELPER_HPP_
#define CUDA_HELPER_HPP_

#include <cuda_runtime.h>

namespace common
{
    namespace cuda
    {

#define _ERROR_TO_STRING(err) \
    case err:                 \
        return #err;
        const char *_cuda_get_error_enum(cudaError_t err);

        void _cuda_check(cudaError_t result, char *const func, const char *const file, int const line);
#define _checkCudaErrors(val) ::common::cuda::_cuda_check((val), #val, __FILE__, __LINE__)

        void *device_malloc(size_t size);
        void device_free(void *ptr);
        void *pinned_malloc(size_t size);
        void pinned_free(void *ptr);

        size_t get_available_device_memory();

        /* Copy */
        void h2dcpy(void *dst, const void *src, size_t size);
        void h2dcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream);
        void d2hcpy(void *dst, const void *src, size_t size);
        void d2hcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream);
        void h2dcpy_symbol(void *symbol, const void *src, size_t size);

        void device_memset(void *dst, int value, size_t size);

        void stream_sync(cudaStream_t const &stream);
        void device_sync();

        void set_device(int device);

        void destory_streams(cudaStream_t *streams, size_t count);

        /* Timer */
        // void start_timer(cudaEvent_t *start, cudaEvent_t *stop);
        // void end_timer(cudaEvent_t *start, cudaEvent_t *stop);
        // void end_timer_with_msg(cudaEvent_t *start, cudaEvent_t *stop, const char *msg);

    } // namespace cuda
} // namespace common

#endif