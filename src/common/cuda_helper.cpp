#include <iostream>
#include <cuda_runtime.h>

#include "common/cuda_helper.hpp"
#include "common/size.hpp"

namespace common
{
    namespace cuda
    {

        const char *_cuda_get_error_enum(cudaError_t err)
        {
            switch (err)
            {
                _ERROR_TO_STRING(cudaSuccess)
                _ERROR_TO_STRING(cudaErrorMissingConfiguration)
                _ERROR_TO_STRING(cudaErrorMemoryAllocation)
                _ERROR_TO_STRING(cudaErrorInitializationError)
                _ERROR_TO_STRING(cudaErrorLaunchFailure)
                _ERROR_TO_STRING(cudaErrorPriorLaunchFailure)
                _ERROR_TO_STRING(cudaErrorLaunchTimeout)
                _ERROR_TO_STRING(cudaErrorLaunchOutOfResources)
                _ERROR_TO_STRING(cudaErrorInvalidDeviceFunction)
                _ERROR_TO_STRING(cudaErrorInvalidConfiguration)
                _ERROR_TO_STRING(cudaErrorInvalidDevice)
                _ERROR_TO_STRING(cudaErrorInvalidValue)
                _ERROR_TO_STRING(cudaErrorInvalidPitchValue)
                _ERROR_TO_STRING(cudaErrorInvalidSymbol)
                _ERROR_TO_STRING(cudaErrorMapBufferObjectFailed)
                _ERROR_TO_STRING(cudaErrorUnmapBufferObjectFailed)
                _ERROR_TO_STRING(cudaErrorInvalidHostPointer)
                _ERROR_TO_STRING(cudaErrorInvalidDevicePointer)
                _ERROR_TO_STRING(cudaErrorInvalidTexture)
                _ERROR_TO_STRING(cudaErrorInvalidTextureBinding)
                _ERROR_TO_STRING(cudaErrorInvalidChannelDescriptor)
                _ERROR_TO_STRING(cudaErrorInvalidMemcpyDirection)
                _ERROR_TO_STRING(cudaErrorAddressOfConstant)
                _ERROR_TO_STRING(cudaErrorTextureFetchFailed)
                _ERROR_TO_STRING(cudaErrorTextureNotBound)
                _ERROR_TO_STRING(cudaErrorSynchronizationError)
                _ERROR_TO_STRING(cudaErrorInvalidFilterSetting)
                _ERROR_TO_STRING(cudaErrorInvalidNormSetting)
                _ERROR_TO_STRING(cudaErrorMixedDeviceExecution)
                _ERROR_TO_STRING(cudaErrorCudartUnloading)
                _ERROR_TO_STRING(cudaErrorUnknown)
                _ERROR_TO_STRING(cudaErrorNotYetImplemented)
                _ERROR_TO_STRING(cudaErrorMemoryValueTooLarge)
                _ERROR_TO_STRING(cudaErrorInvalidResourceHandle)
                _ERROR_TO_STRING(cudaErrorNotReady)
                _ERROR_TO_STRING(cudaErrorInsufficientDriver)
                _ERROR_TO_STRING(cudaErrorSetOnActiveProcess)
                _ERROR_TO_STRING(cudaErrorInvalidSurface)
                _ERROR_TO_STRING(cudaErrorNoDevice)
                _ERROR_TO_STRING(cudaErrorECCUncorrectable)
                _ERROR_TO_STRING(cudaErrorSharedObjectSymbolNotFound)
                _ERROR_TO_STRING(cudaErrorSharedObjectInitFailed)
                _ERROR_TO_STRING(cudaErrorUnsupportedLimit)
                _ERROR_TO_STRING(cudaErrorDuplicateVariableName)
                _ERROR_TO_STRING(cudaErrorDuplicateTextureName)
                _ERROR_TO_STRING(cudaErrorDuplicateSurfaceName)
                _ERROR_TO_STRING(cudaErrorDevicesUnavailable)
                _ERROR_TO_STRING(cudaErrorInvalidKernelImage)
                _ERROR_TO_STRING(cudaErrorNoKernelImageForDevice)
                _ERROR_TO_STRING(cudaErrorIncompatibleDriverContext)
                _ERROR_TO_STRING(cudaErrorPeerAccessAlreadyEnabled)
                _ERROR_TO_STRING(cudaErrorPeerAccessNotEnabled)
                _ERROR_TO_STRING(cudaErrorDeviceAlreadyInUse)
                _ERROR_TO_STRING(cudaErrorProfilerDisabled)
                _ERROR_TO_STRING(cudaErrorProfilerNotInitialized)
                _ERROR_TO_STRING(cudaErrorProfilerAlreadyStarted)
                _ERROR_TO_STRING(cudaErrorProfilerAlreadyStopped)
                _ERROR_TO_STRING(cudaErrorAssert)
                _ERROR_TO_STRING(cudaErrorTooManyPeers)
                _ERROR_TO_STRING(cudaErrorHostMemoryAlreadyRegistered)
                _ERROR_TO_STRING(cudaErrorHostMemoryNotRegistered)
                _ERROR_TO_STRING(cudaErrorStartupFailure)
                _ERROR_TO_STRING(cudaErrorApiFailureBase)
            }
            return "<unknown>";
        }

        void _cuda_check(cudaError_t result, char *const func, const char *const file, int const line)
        {
            if (result)
            {
                std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                             file, line, static_cast<unsigned int>(result), _cuda_get_error_enum(result), func);
                cudaDeviceReset();
                std::exit(EXIT_FAILURE);
            }
        }

        void *device_malloc(size_t size)
        {
            void *ptr;
            _checkCudaErrors(cudaMalloc(&ptr, size));
            return ptr;
        }
        void device_free(void *ptr)
        {
            if (ptr == nullptr)
                throw std::runtime_error(std::string("Memory Deallocation ERROR \n\t [ptr != NULL]"));

            _checkCudaErrors(cudaFree(ptr));
        }

        void *pinned_malloc(size_t size)
        {
            void *ptr;
            _checkCudaErrors(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
            return ptr;
        }

        void pinned_free(void *ptr)
        {
            if (ptr == nullptr)
                throw std::runtime_error(std::string("Memory Deallocation ERROR \n\t [ptr != NULL]"));

            _checkCudaErrors(cudaFreeHost(ptr));
        }

        size_t get_available_device_memory()
        {
            size_t avail, total, used;
            cudaSetDevice(0);
            _checkCudaErrors(cudaMemGetInfo(&avail, &total));
            used = total - avail;
            std::cout << "Device memory:\n\t used " << common::byteToString(used)
                      << "\n\t available " << common::byteToString(avail)
                      << "\n\t total " << common::byteToString(total) << std::endl;
            return avail - common::MiB(128);
        }

        /* Copy */
        void h2dcpy(void *dst, const void *src, size_t size)
        {
            _checkCudaErrors(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
        }
        void h2dcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream)
        {
            _checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
        }
        void d2hcpy(void *dst, const void *src, size_t size)
        {
            _checkCudaErrors(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
        }
        void d2hcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream)
        {
            _checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
        }
        void h2dcpy_symbol(void *symbol, const void *src, size_t size)
        {
            _checkCudaErrors(cudaMemcpyToSymbol(symbol, src, size, 0, cudaMemcpyHostToDevice));
        }

        /* Memset */
        void device_memset(void *dst, int value, size_t size)
        {
            _checkCudaErrors(cudaMemset(dst, value, size));
        }

        /* Synchronization */
        void stream_sync(cudaStream_t const &stream)
        {
            _checkCudaErrors(cudaStreamSynchronize(stream));
        }
        void device_sync()
        {
            _checkCudaErrors(cudaDeviceSynchronize());
        }

        /* Device */
        void set_device(int device)
        {
            _checkCudaErrors(cudaSetDevice(device));
        }

        void destory_streams(cudaStream_t *streams, size_t count)
        {
            for (size_t i = 0; i < count; i++)
            {
                _checkCudaErrors(cudaStreamDestroy(streams[i]));
            }
        }

        /* Timer */
        void start_timer(cudaEvent_t *start, cudaEvent_t *stop)
        {
            _checkCudaErrors(cudaEventCreate(start));
            _checkCudaErrors(cudaEventCreate(stop));
            _checkCudaErrors(cudaEventRecord(*start, 0));
        }

        void end_timer(cudaEvent_t *start, cudaEvent_t *stop)
        {
            _checkCudaErrors(cudaEventRecord(*stop, 0));
            _checkCudaErrors(cudaEventSynchronize(*stop));
            float elapsedTime;
            _checkCudaErrors(cudaEventElapsedTime(&elapsedTime, *start, *stop));
            std::cout << "Elapsed time: " << elapsedTime/1000.0f << " s" << std::endl;
        }

        void end_timer_with_msg(cudaEvent_t *start, cudaEvent_t *stop, const char *msg)
        {
            _checkCudaErrors(cudaEventRecord(*stop, 0));
            _checkCudaErrors(cudaEventSynchronize(*stop));
            float elapsedTime;
            _checkCudaErrors(cudaEventElapsedTime(&elapsedTime, *start, *stop));
            std::cout << msg << ": " << elapsedTime/1000.0f << " s" << std::endl;
        }

    }
}