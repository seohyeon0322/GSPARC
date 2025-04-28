#ifndef SORT_CUH_
#define SORT_CUH_

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include "common/cuda_helper.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/cuda_memory_allocator.hpp"
#include "gsparc/timer.hpp"

namespace gsparc
{

    template <typename IndexType, typename UIndexType, typename LIndexType>
    struct IndirectComparator
    {
        UIndexType *uindices;
        LIndexType *indices;

        __host__ __device__ bool operator()(const IndexType &i, const IndexType &j) const
        {
            if (uindices[i] < uindices[j])
                return true;
            if (uindices[i] > uindices[j])
                return false;
            return indices[i] < indices[j];
        }
    };

    template <typename SLITOMType>
    void sort_64(SLITOMType *SX, SLITOMType *SY, CudaMemoryPool **pools, int gpu_count, Timer *timer)
    {
        using slitom_t = SLITOMType;
        using ulindex_t = typename slitom_t::ulindex_t;
        using lindex_t = typename slitom_t::lindex_t;
        using value_t = typename slitom_t::value_t;

        uint64_t nnzX = SX->nnz;
        uint64_t nnzY = SY->nnz;

        // SX->d_indices = gsparc::allocate<lindex_t *>(gpu_count);
        // SX->d_values = gsparc::allocate<value_t *>(gpu_count);
        // SY->d_indices = gsparc::allocate<lindex_t *>(gpu_count);
        // SY->d_values = gsparc::allocate<value_t *>(gpu_count);

        printf("test\n");
        if (SX->indices == nullptr || SX->values == nullptr)
        {
            printf("SX->indices or SX->values is null\n");
            return;
        }
        timer->start();
#pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
        for (int g = 0; g < gpu_count; ++g)
        {
            CudaMemoryPool *pool = pools[g];
            checkCudaErrors(cudaSetDevice(g));

            // if (g == 0)
            // {
            cudaStream_t stream1, stream2;
            checkCudaErrors(cudaStreamCreate(&stream1));
            checkCudaErrors(cudaStreamCreate(&stream2));

            lindex_t *d_indices_X = pool->allocate<lindex_t>(nnzX);
            value_t *d_values_X = pool->allocate<value_t>(nnzX);

            lindex_t *d_indices_Y = pool->allocate<lindex_t>(nnzY);
            value_t *d_values_Y = pool->allocate<value_t>(nnzY);

            common::cuda::h2dcpy_async(d_indices_X, SX->indices, nnzX * sizeof(lindex_t), stream1);
            common::cuda::h2dcpy_async(d_values_X, SX->values, nnzX * sizeof(value_t), stream1);

            void *d_temp_storage_X = nullptr;
            size_t temp_storage_bytes_X = 0;
            void *d_temp_storage_Y = nullptr;
            size_t temp_storage_bytes_Y = 0;
            void *d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            cub::DeviceRadixSort::SortPairs(d_temp_storage_X, temp_storage_bytes_X, d_indices_X, d_indices_X, d_values_X, d_values_X, nnzX, 0, sizeof(lindex_t) * 8, stream1);
            cub::DeviceRadixSort::SortPairs(d_temp_storage_Y, temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY);

            temp_storage_bytes = std::max(temp_storage_bytes_X, temp_storage_bytes_Y);
            d_temp_storage = pool->allocate<void>(temp_storage_bytes);

            // common::cuda::stream_sync(stream1);

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_indices_X, d_indices_X, d_values_X, d_values_X, nnzX, 0, sizeof(lindex_t) * 8, stream1);

            common::cuda::d2hcpy_async(SX->indices, d_indices_X, nnzX * sizeof(lindex_t), stream1);
            common::cuda::d2hcpy_async(SX->values, d_values_X, nnzX * sizeof(value_t), stream1);

            common::cuda::h2dcpy_async(d_indices_Y, SY->indices, nnzY * sizeof(lindex_t), stream2);
            common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY * sizeof(value_t), stream2);

            common::cuda::stream_sync(stream2);

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY);
            common::cuda::device_sync();
            common::cuda::d2hcpy(SY->indices, d_indices_Y, nnzY * sizeof(lindex_t));
            common::cuda::d2hcpy(SY->values, d_values_Y, nnzY * sizeof(value_t));

            // if (SX->nprtn == 1 && gpu_count == 1)
            // {
                SX->d_indices[g] = d_indices_X;
                SX->d_values[g] = d_values_X;
            // }
            // if (SY->nprtn == 1)
            // {
                SY->d_indices[g] = d_indices_Y;
                SY->d_values[g] = d_values_Y;
            // }

            common::cuda::stream_sync(stream1);
            common::cuda::stream_sync(stream2);

            checkCudaErrors(cudaStreamDestroy(stream1));
            checkCudaErrors(cudaStreamDestroy(stream2));
            // }
            // else
            // {
            //     cudaStream_t stream1;
            //     checkCudaErrors(cudaStreamCreate(&stream1));
            //     lindex_t *d_indices_Y = pool->allocate<lindex_t>(nnzY);
            //     value_t *d_values_Y = pool->allocate<value_t>(nnzY);

            //     common::cuda::h2dcpy_async(d_indices_Y, SY->indices, nnzY * sizeof(lindex_t), stream1);
            //     common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY * sizeof(value_t), stream1);

            //     void *d_temp_storage_Y = nullptr;
            //     size_t temp_storage_bytes_Y = 0;

            //     cub::DeviceRadixSort::SortPairs(d_temp_storage_Y, temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY, 0, sizeof(lindex_t) * 8, stream1);

            //     d_temp_storage_Y = pool->allocate<void>(temp_storage_bytes_Y);

            //     cub::DeviceRadixSort::SortPairs(d_temp_storage_Y, temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY, 0, sizeof(lindex_t) * 8, stream1);

            //     if (SY->nprtn == 1)
            //     {
            //         SY->d_indices[g] = d_indices_Y;
            //         SY->d_values[g] = d_values_Y;
            //     }
            // }
        }
        timer->stop();
        timer->printElapsed("Sort Tensors");
        // common::end_timer_with_msg(&timer, "Sort Tensors");
    }

    //     template <typename SLITOMType>
    //     void sort_one_64(SLITOMType *SX, CudaMemoryPool **pools, int gpu_count)
    //     {
    //         using slitom_t = SLITOMType;
    //         using ulindex_t = typename slitom_t::ulindex_t;
    //         using lindex_t = typename slitom_t::lindex_t;
    //         using value_t = typename slitom_t::value_t;

    //         uint64_t nnzX = SX->nnz;

    //         SX->d_indices = static_cast<lindex_t **>(common::cuda::pinned_malloc(gpu_count * sizeof(lindex_t *)));
    //         SX->d_values = static_cast<value_t **>(common::cuda::pinned_malloc(gpu_count * sizeof(value_t *)));

    //         double timer;

    //         common::start_timer(&timer);
    // #pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
    //         for (int g = 0; g < gpu_count; ++g)
    //         {
    //             CudaMemoryPool *pool = pools[g];
    //             checkCudaErrors(cudaSetDevice(g));

    //             if (g == 0)
    //             {
    //                 cudaStream_t stream1, stream2;
    //                 checkCudaErrors(cudaStreamCreate(&stream1));

    //                 lindex_t *d_indices_X = pool->allocate<lindex_t>(nnzX);
    //                 value_t *d_values_X = pool->allocate<value_t>(nnzX);

    //                 common::cuda::h2dcpy_async(d_indices_X, SX->indices, nnzX * sizeof(lindex_t), stream1);
    //                 common::cuda::h2dcpy_async(d_values_X, SX->values, nnzX * sizeof(value_t), stream1);

    //                 void *d_temp_storage_X = nullptr;
    //                 size_t temp_storage_bytes_X = 0;
    //                 void *d_temp_storage_Y = nullptr;
    //                 size_t temp_storage_bytes_Y = 0;
    //                 void *d_temp_storage = nullptr;
    //                 size_t temp_storage_bytes = 0;

    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage_X, temp_storage_bytes_X, d_indices_X, d_indices_X, d_values_X, d_values_X, nnzX, 0, sizeof(lindex_t) * 8, stream1);
    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y, temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY);

    //                 temp_storage_bytes = std::max(temp_storage_bytes_X, temp_storage_bytes_Y);
    //                 d_temp_storage = pool->allocate<void>(temp_storage_bytes);

    //                 // common::cuda::stream_sync(stream1);

    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_indices_X, d_indices_X, d_values_X, d_values_X, nnzX, 0, sizeof(lindex_t) * 8, stream1);

    //                 common::cuda::d2hcpy_async(SX->indices, d_indices_X, nnzX * sizeof(lindex_t), stream1);
    //                 common::cuda::d2hcpy_async(SX->values, d_values_X, nnzX * sizeof(value_t), stream1);

    //                 SX->d_indices[g] = d_indices_X;
    //                 SX->d_values[g] = d_values_X;

    //                 common::cuda::stream_sync(stream1);

    //                 checkCudaErrors(cudaStreamDestroy(stream1));
    //             }
    //         }
    //         common::end_timer_with_msg(&timer, "Sort Tensors");
    //     }

    //     template <typename SLITOMType, typename IndexType>
    //     void sort_128(SLITOMType *SX, SLITOMType *SY, CudaMemoryPool **pools, int gpu_count)
    //     {
    //         using slitom_t = SLITOMType;
    //         using ulindex_t = typename slitom_t::ulindex_t;
    //         using lindex_t = typename slitom_t::lindex_t;
    //         using value_t = typename slitom_t::value_t;
    //         using index_t = IndexType;

    //         uint64_t nnzX = SX->nnz;
    //         uint64_t nnzY = SY->nnz;

    //         SX->d_indices = static_cast<lindex_t **>(common::cuda::pinned_malloc(gpu_count * sizeof(lindex_t *)));
    //         SX->d_values = static_cast<value_t **>(common::cuda::pinned_malloc(gpu_count * sizeof(value_t *)));
    //         SY->d_indices = static_cast<lindex_t **>(common::cuda::pinned_malloc(gpu_count * sizeof(lindex_t *)));
    //         SY->d_values = static_cast<value_t **>(common::cuda::pinned_malloc(gpu_count * sizeof(value_t *)));

    //         __uint128_t *X_indices = static_cast<__uint128_t *>(common::cuda::pinned_malloc(SX->nnz * sizeof(__uint128_t)));
    //         __uint128_t *Y_indices = static_cast<__uint128_t *>(common::cuda::pinned_malloc(SY->nnz * sizeof(__uint128_t)));

    // #pragma omp parallel for
    //         for (uint64_t i = 0; i < SX->nnz; ++i)
    //         {
    //             X_indices[i] = static_cast<__uint128_t>(SX->uindices[i]) << 64 | static_cast<__uint128_t>(SX->indices[i]);
    //         }

    // #pragma omp parallel for
    //         for (uint64_t i = 0; i < SY->nnz; ++i)
    //         {
    //             Y_indices[i] = static_cast<__uint128_t>(SY->uindices[i]) << 64 | static_cast<__uint128_t>(SY->indices[i]);
    //         }

    //         double timer;

    //         common::start_timer(&timer);
    // #pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
    //         for (int g = 0; g < gpu_count; ++g)
    //         {
    //             CudaMemoryPool *pool = pools[g];
    //             checkCudaErrors(cudaSetDevice(g));

    //             if (g == 0)
    //             {
    //                 cudaStream_t stream1, stream2;
    //                 checkCudaErrors(cudaStreamCreate(&stream1));
    //                 checkCudaErrors(cudaStreamCreate(&stream2));

    //                 __uint128_t *d_indices_X = pool->allocate<__uint128_t>(nnzX);
    //                 value_t *d_values_X = pool->allocate<value_t>(nnzX);

    //                 __uint128_t *d_indices_Y = pool->allocate<__uint128_t>(nnzY);
    //                 value_t *d_values_Y = pool->allocate<value_t>(nnzY);

    //                 common::cuda::h2dcpy_async(d_indices_X, X_indices, nnzX * sizeof(__uint128_t), stream1);
    //                 common::cuda::h2dcpy_async(d_values_X, SX->values, nnzX * sizeof(value_t), stream1);

    //                 void *d_temp_storage_X = nullptr;
    //                 size_t temp_storage_bytes_X = 0;
    //                 void *d_temp_storage_Y = nullptr;
    //                 size_t temp_storage_bytes_Y = 0;
    //                 void *d_temp_storage = nullptr;
    //                 size_t temp_storage_bytes = 0;

    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage_X, temp_storage_bytes_X, d_indices_X, d_indices_X, d_values_X, d_values_X, nnzX, 0, sizeof(__uint128_t) * 8, stream1);
    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y, temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY);

    //                 temp_storage_bytes = std::max(temp_storage_bytes_X, temp_storage_bytes_Y);
    //                 d_temp_storage = pool->allocate<void>(temp_storage_bytes);

    //                 common::cuda::stream_sync(stream1);

    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_indices_X, d_indices_X, d_values_X, d_values_X, nnzX, 0, sizeof(__uint128_t) * 8, stream1);
    //                 printf("flag0\n");
    //                 common::cuda::d2hcpy_async(X_indices, d_indices_X, nnzX * sizeof(__uint128_t), stream1);
    //                 common::cuda::d2hcpy_async(SX->values, d_values_X, nnzX * sizeof(value_t), stream1);
    //                 printf("flag1\n");

    //                 common::cuda::h2dcpy_async(d_indices_Y, Y_indices, nnzY * sizeof(__uint128_t), stream2);
    //                 common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY * sizeof(value_t), stream2);
    //                 printf("flag2\n");
    //                 common::cuda::stream_sync(stream2);
    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY);
    //                 common::cuda::device_sync();
    //                 printf("flag3\n");
    //                 common::cuda::d2hcpy(Y_indices, d_indices_Y, nnzY * sizeof(lindex_t));
    //                 printf("flag3-5\n");
    //                 common::cuda::d2hcpy(SY->values, d_values_Y, nnzY * sizeof(value_t));
    //                 printf("flag4\n");
    //                 // SX->d_indices[g] = d_indices_X;
    //                 // SX->d_values[g] = d_values_X;

    //                 // SY->d_indices[g] = d_indices_Y;
    //                 // SY->d_values[g] = d_values_Y;
    //                 common::cuda::stream_sync(stream1);
    //                 common::cuda::stream_sync(stream2);

    //                 pool->deallocate<__uint128_t>(d_indices_X, nnzX);
    //                 pool->deallocate<value_t>(d_values_X, nnzX);
    //                 pool->deallocate<__uint128_t>(d_indices_Y, nnzY);
    //                 pool->deallocate<value_t>(d_values_Y, nnzY);

    //                 checkCudaErrors(cudaStreamDestroy(stream1));
    //                 checkCudaErrors(cudaStreamDestroy(stream2));
    //             }
    //             else
    //             {
    //                 cudaStream_t stream1;
    //                 checkCudaErrors(cudaStreamCreate(&stream1));
    //                 lindex_t *d_indices_Y = pool->allocate<lindex_t>(nnzY);
    //                 value_t *d_values_Y = pool->allocate<value_t>(nnzY);

    //                 common::cuda::h2dcpy_async(d_indices_Y, SY->indices, nnzY * sizeof(lindex_t), stream1);
    //                 common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY * sizeof(value_t), stream1);

    //                 void *d_temp_storage_Y = nullptr;
    //                 size_t temp_storage_bytes_Y = 0;

    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y, temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY, 0, sizeof(lindex_t) * 8, stream1);

    //                 d_temp_storage_Y = pool->allocate<void>(temp_storage_bytes_Y);

    //                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y, temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y, d_values_Y, nnzY, 0, sizeof(lindex_t) * 8, stream1);

    //                 SY->d_indices[g] = d_indices_Y;
    //                 SY->d_values[g] = d_values_Y;
    //             }

    // #pragma omp parallel for
    //             for (uint64_t i = 0; i < SX->nnz; ++i)
    //             {
    //                 SX->uindices[i] = common::uhalf(X_indices[i]);
    //                 SX->indices[i] = common::lhalf(X_indices[i]);
    //             }

    // #pragma omp parallel for
    //             for (uint64_t i = 0; i < SY->nnz; ++i)
    //             {
    //                 SY->uindices[i] = common::uhalf(Y_indices[i]);
    //                 SY->indices[i] = common::lhalf(Y_indices[i]);
    //             }
    //         }
    //         common::end_timer_with_msg(&timer, "Sort Tensors");

    //         common::cuda::pinned_free(X_indices);
    //         common::cuda::pinned_free(Y_indices);

    //     }
    template <typename SLITOMType, typename IndexType>
    void sort_128(SLITOMType *SX, SLITOMType *SY, CudaMemoryPool **pools, int gpu_count, Timer *timer)
    {
        using slitom_t = SLITOMType;
        using ulindex_t = typename slitom_t::ulindex_t;
        using lindex_t = typename slitom_t::lindex_t;
        using value_t = typename slitom_t::value_t;
        using index_t = IndexType;

        uint64_t nnzX = SX->nnz;
        uint64_t nnzY = SY->nnz;

        // common::start_timer(&timer);
        timer->start();

#pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
        for (int g = 0; g < gpu_count; ++g)
        {
            CudaMemoryPool *pool = pools[g];
            // SX->d_uindices = gsparc::allocate<ulindex_t *>(gpu_count);
            // SX->d_indices = gsparc::allocate<lindex_t *>(gpu_count);
            // SX->d_values = gsparc::allocate<value_t *>(gpu_count);
            // SY->d_uindices = gsparc::allocate<ulindex_t *>(gpu_count);
            // SY->d_indices = gsparc::allocate<lindex_t *>(gpu_count);
            // SY->d_values = gsparc::allocate<value_t *>(gpu_count);

            checkCudaErrors(cudaSetDevice(g));

            if (g == 0)
            {
                cudaStream_t stream1, stream2;
                checkCudaErrors(cudaStreamCreate(&stream1));
                checkCudaErrors(cudaStreamCreate(&stream2));

                // ---- X 데이터 할당 및 host -> device 복사 ----
                ulindex_t *d_uindices_X = pool->allocate<ulindex_t>(nnzX);
                lindex_t *d_indices_X = pool->allocate<lindex_t>(nnzX);
                value_t *d_values_X = pool->allocate<value_t>(nnzX);

                common::cuda::h2dcpy_async(d_uindices_X, SX->uindices, nnzX * sizeof(ulindex_t), stream1);
                common::cuda::h2dcpy_async(d_indices_X, SX->indices, nnzX * sizeof(lindex_t), stream1);
                common::cuda::h2dcpy_async(d_values_X, SX->values, nnzX * sizeof(value_t), stream1);

                // ---- Y 데이터 할당 및 host -> device 복사 ----
                ulindex_t *d_uindices_Y = pool->allocate<ulindex_t>(nnzY);
                lindex_t *d_indices_Y = pool->allocate<lindex_t>(nnzY);
                value_t *d_values_Y = pool->allocate<value_t>(nnzY);

                common::cuda::h2dcpy_async(d_uindices_Y, SY->uindices, nnzY * sizeof(ulindex_t), stream2);
                common::cuda::h2dcpy_async(d_indices_Y, SY->indices, nnzY * sizeof(lindex_t), stream2);
                common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY * sizeof(value_t), stream2);

                gsparc::CudaMemoryPoolAllocator<index_t> alloc(pool);
                auto exec_policy = thrust::cuda::par(alloc);

                // ==== X 배열 처리 ====
                // 정렬을 위한 인덱스 배열 할당 및 초기화
                index_t *sorted_idx_X = pool->allocate<index_t>(nnzX);
                auto idx_X_begin = thrust::device_pointer_cast(sorted_idx_X);
                auto idx_X_end = idx_X_begin + nnzX;
                thrust::sequence(exec_policy, idx_X_begin, idx_X_end);

                // X 배열에 대해 비교 연산자 생성 후 인덱스 정렬
                IndirectComparator<index_t, ulindex_t, lindex_t> comp_X{d_uindices_X, d_indices_X};
                thrust::sort(exec_policy, idx_X_begin, idx_X_end, comp_X);

                // gather를 위한 임시 버퍼 할당
                ulindex_t *d_uindices_temp_X = pool->allocate<ulindex_t>(nnzX);
                lindex_t *d_indices_temp_X = pool->allocate<lindex_t>(nnzX);
                value_t *d_values_temp_X = pool->allocate<value_t>(nnzX);

                auto d_uindices_ptr_X = thrust::device_pointer_cast(d_uindices_X);
                auto d_indices_ptr_X = thrust::device_pointer_cast(d_indices_X);
                auto d_values_ptr_X = thrust::device_pointer_cast(d_values_X);

                // sorted_idx_X에 따른 gather 수행 -> 임시 버퍼에 저장
                thrust::gather(exec_policy, idx_X_begin, idx_X_end,
                               d_uindices_ptr_X,
                               thrust::device_pointer_cast(d_uindices_temp_X));
                thrust::gather(exec_policy, idx_X_begin, idx_X_end,
                               d_indices_ptr_X,
                               thrust::device_pointer_cast(d_indices_temp_X));
                thrust::gather(exec_policy, idx_X_begin, idx_X_end,
                               d_values_ptr_X,
                               thrust::device_pointer_cast(d_values_temp_X));

                // 임시 버퍼의 결과를 host 메모리로 복사 (비동기 방식)
                common::cuda::d2hcpy_async(SX->uindices, d_uindices_temp_X, nnzX * sizeof(ulindex_t), stream1);
                common::cuda::d2hcpy_async(SX->indices, d_indices_temp_X, nnzX * sizeof(lindex_t), stream1);
                common::cuda::d2hcpy_async(SX->values, d_values_temp_X, nnzX * sizeof(value_t), stream1);

                // ==== Y 배열 처리 ====
                index_t *sorted_idx_Y = pool->allocate<index_t>(nnzY);
                auto idx_Y_begin = thrust::device_pointer_cast(sorted_idx_Y);
                auto idx_Y_end = idx_Y_begin + nnzY;
                thrust::sequence(exec_policy, idx_Y_begin, idx_Y_end);

                IndirectComparator<index_t, ulindex_t, lindex_t> comp_Y{d_uindices_Y, d_indices_Y};
                thrust::sort(exec_policy, idx_Y_begin, idx_Y_end, comp_Y);

                // gather를 위한 임시 버퍼 할당
                ulindex_t *d_uindices_temp_Y = pool->allocate<ulindex_t>(nnzY);
                lindex_t *d_indices_temp_Y = pool->allocate<lindex_t>(nnzY);
                value_t *d_values_temp_Y = pool->allocate<value_t>(nnzY);

                auto d_uindices_ptr_Y = thrust::device_pointer_cast(d_uindices_Y);
                auto d_indices_ptr_Y = thrust::device_pointer_cast(d_indices_Y);
                auto d_values_ptr_Y = thrust::device_pointer_cast(d_values_Y);

                // sorted_idx_Y에 따른 gather 수행 -> 임시 버퍼에 저장
                thrust::gather(exec_policy, idx_Y_begin, idx_Y_end,
                               d_uindices_ptr_Y,
                               thrust::device_pointer_cast(d_uindices_temp_Y));
                thrust::gather(exec_policy, idx_Y_begin, idx_Y_end,
                               d_indices_ptr_Y,
                               thrust::device_pointer_cast(d_indices_temp_Y));
                thrust::gather(exec_policy, idx_Y_begin, idx_Y_end,
                               d_values_ptr_Y,
                               thrust::device_pointer_cast(d_values_temp_Y));

                // 임시 버퍼의 결과를 host 메모리로 복사 (비동기 방식)
                common::cuda::d2hcpy_async(SY->uindices, d_uindices_temp_Y, nnzY * sizeof(ulindex_t), stream1);
                common::cuda::d2hcpy_async(SY->indices, d_indices_temp_Y, nnzY * sizeof(lindex_t), stream1);
                common::cuda::d2hcpy_async(SY->values, d_values_temp_Y, nnzY * sizeof(value_t), stream1);

                // 스트림 동기화 후 스트림 제거
                common::cuda::stream_sync(stream1);
                common::cuda::stream_sync(stream2);
                cudaStreamDestroy(stream1);
                cudaStreamDestroy(stream2);

                // device 메모리 포인터 저장 (추후 필요 시 사용)
                SX->d_uindices[g] = d_uindices_temp_X;
                SX->d_indices[g] = d_indices_temp_X;
                SX->d_values[g] = d_values_temp_X;
                SY->d_uindices[g] = d_uindices_temp_Y;
                SY->d_indices[g] = d_indices_temp_Y;
                SY->d_values[g] = d_values_temp_Y;

                // 할당한 임시 버퍼 해제
                pool->deallocate<index_t>(sorted_idx_X, nnzX);
                pool->deallocate<ulindex_t>(d_uindices_X, nnzX);
                pool->deallocate<lindex_t>(d_indices_X, nnzX);
                pool->deallocate<value_t>(d_values_X, nnzX);

                pool->deallocate<index_t>(sorted_idx_Y, nnzY);
                pool->deallocate<ulindex_t>(d_uindices_Y, nnzY);
                pool->deallocate<lindex_t>(d_indices_Y, nnzY);
                pool->deallocate<value_t>(d_values_Y, nnzY);
            }
        }
        timer->stop();
        timer->printElapsed("Sort Tensors");
        // common::end_timer_with_msg(&timer, "Sort Tensors");
    }
}

#endif