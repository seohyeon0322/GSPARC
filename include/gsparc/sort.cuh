#ifndef SORT_CUH_
#define SORT_CUH_

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/tuple.h>

#include "common/cuda_helper.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/cuda_memory_allocator.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/timer.hpp"

namespace gsparc {

template <typename IndexType, typename UIndexType, typename LIndexType>
struct IndirectComparator {
  UIndexType *uindices;
  LIndexType *indices;

  __host__ __device__ bool operator()(const IndexType &i,
                                      const IndexType &j) const {
    if (uindices[i] < uindices[j])
      return true;
    if (uindices[i] > uindices[j])
      return false;
    return indices[i] < indices[j];
  }
};

template <typename SLITOMType>
void sort_64(SLITOMType *SX, SLITOMType *SY, CudaMemoryPool **pools,
             int gpu_count, Timer *timer) {
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
  if (SX->indices == nullptr || SX->values == nullptr) {
    printf("SX->indices or SX->values is null\n");
    return;
  }
  timer->start();
#pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
  for (int g = 0; g < gpu_count; ++g) {
    CudaMemoryPool *pool = pools[g];
    checkCudaErrors(cudaSetDevice(g));

    cudaStream_t stream1;
    checkCudaErrors(cudaStreamCreate(&stream1));

    int sort_bits_X = SX->nbits;
    int sort_bits_Y = SY->nbits;

    // ---- Sort X ----
    lindex_t *d_indices_X = pool->allocate<lindex_t>(nnzX);
    value_t *d_values_X = pool->allocate<value_t>(nnzX);
    lindex_t *d_indices_X_out = pool->allocate<lindex_t>(nnzX);
    value_t *d_values_X_out = pool->allocate<value_t>(nnzX);

    common::cuda::h2dcpy_async(d_indices_X, SX->indices,
                               nnzX * sizeof(lindex_t), stream1);
    common::cuda::h2dcpy_async(d_values_X, SX->values, nnzX * sizeof(value_t),
                               stream1);

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_indices_X, d_indices_X_out,
        d_values_X, d_values_X_out, nnzX, 0, sort_bits_X, stream1);
    d_temp = pool->allocate<void>(temp_bytes);
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_indices_X, d_indices_X_out,
        d_values_X, d_values_X_out, nnzX, 0, sort_bits_X, stream1);

    common::cuda::stream_sync(stream1);

    // D2H for X
    common::cuda::d2hcpy(SX->indices, d_indices_X_out, nnzX * sizeof(lindex_t));
    common::cuda::d2hcpy(SX->values, d_values_X_out, nnzX * sizeof(value_t));

    // Free X input buffers and temp
    pool->deallocate<void>(d_temp, temp_bytes);
    pool->deallocate<lindex_t>(d_indices_X, nnzX);
    pool->deallocate<value_t>(d_values_X, nnzX);

    SX->d_indices[g] = d_indices_X_out;
    SX->d_values[g] = d_values_X_out;

    // ---- Sort Y (reuse freed memory) ----
    lindex_t *d_indices_Y = pool->allocate<lindex_t>(nnzY);
    value_t *d_values_Y = pool->allocate<value_t>(nnzY);
    lindex_t *d_indices_Y_out = pool->allocate<lindex_t>(nnzY);
    value_t *d_values_Y_out = pool->allocate<value_t>(nnzY);

    common::cuda::h2dcpy_async(d_indices_Y, SY->indices,
                               nnzY * sizeof(lindex_t), stream1);
    common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY * sizeof(value_t),
                               stream1);

    d_temp = nullptr;
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_indices_Y, d_indices_Y_out,
        d_values_Y, d_values_Y_out, nnzY, 0, sort_bits_Y, stream1);
    d_temp = pool->allocate<void>(temp_bytes);
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_indices_Y, d_indices_Y_out,
        d_values_Y, d_values_Y_out, nnzY, 0, sort_bits_Y, stream1);

    common::cuda::stream_sync(stream1);

    // D2H for Y
    common::cuda::d2hcpy(SY->indices, d_indices_Y_out, nnzY * sizeof(lindex_t));
    common::cuda::d2hcpy(SY->values, d_values_Y_out, nnzY * sizeof(value_t));

    // Free Y input buffers and temp
    pool->deallocate<void>(d_temp, temp_bytes);
    pool->deallocate<lindex_t>(d_indices_Y, nnzY);
    pool->deallocate<value_t>(d_values_Y, nnzY);

    SY->d_indices[g] = d_indices_Y_out;
    SY->d_values[g] = d_values_Y_out;

    checkCudaErrors(cudaStreamDestroy(stream1));
  }
  timer->stop();
  timer->printElapsed("Sort Tensors");
}

//     template <typename SLITOMType>
//     void sort_one_64(SLITOMType *SX, CudaMemoryPool **pools, int gpu_count)
//     {
//         using slitom_t = SLITOMType;
//         using ulindex_t = typename slitom_t::ulindex_t;
//         using lindex_t = typename slitom_t::lindex_t;
//         using value_t = typename slitom_t::value_t;

//         uint64_t nnzX = SX->nnz;

//         SX->d_indices = static_cast<lindex_t
//         **>(common::cuda::pinned_malloc(gpu_count * sizeof(lindex_t *)));
//         SX->d_values = static_cast<value_t
//         **>(common::cuda::pinned_malloc(gpu_count * sizeof(value_t *)));

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

//                 common::cuda::h2dcpy_async(d_indices_X, SX->indices, nnzX *
//                 sizeof(lindex_t), stream1);
//                 common::cuda::h2dcpy_async(d_values_X, SX->values, nnzX *
//                 sizeof(value_t), stream1);

//                 void *d_temp_storage_X = nullptr;
//                 size_t temp_storage_bytes_X = 0;
//                 void *d_temp_storage_Y = nullptr;
//                 size_t temp_storage_bytes_Y = 0;
//                 void *d_temp_storage = nullptr;
//                 size_t temp_storage_bytes = 0;

//                 cub::DeviceRadixSort::SortPairs(d_temp_storage_X,
//                 temp_storage_bytes_X, d_indices_X, d_indices_X, d_values_X,
//                 d_values_X, nnzX, 0, sizeof(lindex_t) * 8, stream1);
//                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y,
//                 temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y,
//                 d_values_Y, nnzY);

//                 temp_storage_bytes = std::max(temp_storage_bytes_X,
//                 temp_storage_bytes_Y); d_temp_storage =
//                 pool->allocate<void>(temp_storage_bytes);

//                 // common::cuda::stream_sync(stream1);

//                 cub::DeviceRadixSort::SortPairs(d_temp_storage,
//                 temp_storage_bytes, d_indices_X, d_indices_X, d_values_X,
//                 d_values_X, nnzX, 0, sizeof(lindex_t) * 8, stream1);

//                 common::cuda::d2hcpy_async(SX->indices, d_indices_X, nnzX *
//                 sizeof(lindex_t), stream1);
//                 common::cuda::d2hcpy_async(SX->values, d_values_X, nnzX *
//                 sizeof(value_t), stream1);

//                 SX->d_indices[g] = d_indices_X;
//                 SX->d_values[g] = d_values_X;

//                 common::cuda::stream_sync(stream1);

//                 checkCudaErrors(cudaStreamDestroy(stream1));
//             }
//         }
//         common::end_timer_with_msg(&timer, "Sort Tensors");
//     }

//     template <typename SLITOMType, typename IndexType>
//     void sort_128(SLITOMType *SX, SLITOMType *SY, CudaMemoryPool **pools, int
//     gpu_count)
//     {
//         using slitom_t = SLITOMType;
//         using ulindex_t = typename slitom_t::ulindex_t;
//         using lindex_t = typename slitom_t::lindex_t;
//         using value_t = typename slitom_t::value_t;
//         using index_t = IndexType;

//         uint64_t nnzX = SX->nnz;
//         uint64_t nnzY = SY->nnz;

//         SX->d_indices = static_cast<lindex_t
//         **>(common::cuda::pinned_malloc(gpu_count * sizeof(lindex_t *)));
//         SX->d_values = static_cast<value_t
//         **>(common::cuda::pinned_malloc(gpu_count * sizeof(value_t *)));
//         SY->d_indices = static_cast<lindex_t
//         **>(common::cuda::pinned_malloc(gpu_count * sizeof(lindex_t *)));
//         SY->d_values = static_cast<value_t
//         **>(common::cuda::pinned_malloc(gpu_count * sizeof(value_t *)));

//         __uint128_t *X_indices = static_cast<__uint128_t
//         *>(common::cuda::pinned_malloc(SX->nnz * sizeof(__uint128_t)));
//         __uint128_t *Y_indices = static_cast<__uint128_t
//         *>(common::cuda::pinned_malloc(SY->nnz * sizeof(__uint128_t)));

// #pragma omp parallel for
//         for (uint64_t i = 0; i < SX->nnz; ++i)
//         {
//             X_indices[i] = static_cast<__uint128_t>(SX->uindices[i]) << 64 |
//             static_cast<__uint128_t>(SX->indices[i]);
//         }

// #pragma omp parallel for
//         for (uint64_t i = 0; i < SY->nnz; ++i)
//         {
//             Y_indices[i] = static_cast<__uint128_t>(SY->uindices[i]) << 64 |
//             static_cast<__uint128_t>(SY->indices[i]);
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

//                 common::cuda::h2dcpy_async(d_indices_X, X_indices, nnzX *
//                 sizeof(__uint128_t), stream1);
//                 common::cuda::h2dcpy_async(d_values_X, SX->values, nnzX *
//                 sizeof(value_t), stream1);

//                 void *d_temp_storage_X = nullptr;
//                 size_t temp_storage_bytes_X = 0;
//                 void *d_temp_storage_Y = nullptr;
//                 size_t temp_storage_bytes_Y = 0;
//                 void *d_temp_storage = nullptr;
//                 size_t temp_storage_bytes = 0;

//                 cub::DeviceRadixSort::SortPairs(d_temp_storage_X,
//                 temp_storage_bytes_X, d_indices_X, d_indices_X, d_values_X,
//                 d_values_X, nnzX, 0, sizeof(__uint128_t) * 8, stream1);
//                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y,
//                 temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y,
//                 d_values_Y, nnzY);

//                 temp_storage_bytes = std::max(temp_storage_bytes_X,
//                 temp_storage_bytes_Y); d_temp_storage =
//                 pool->allocate<void>(temp_storage_bytes);

//                 common::cuda::stream_sync(stream1);

//                 cub::DeviceRadixSort::SortPairs(d_temp_storage,
//                 temp_storage_bytes, d_indices_X, d_indices_X, d_values_X,
//                 d_values_X, nnzX, 0, sizeof(__uint128_t) * 8, stream1);
//                 printf("flag0\n");
//                 common::cuda::d2hcpy_async(X_indices, d_indices_X, nnzX *
//                 sizeof(__uint128_t), stream1);
//                 common::cuda::d2hcpy_async(SX->values, d_values_X, nnzX *
//                 sizeof(value_t), stream1); printf("flag1\n");

//                 common::cuda::h2dcpy_async(d_indices_Y, Y_indices, nnzY *
//                 sizeof(__uint128_t), stream2);
//                 common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY *
//                 sizeof(value_t), stream2); printf("flag2\n");
//                 common::cuda::stream_sync(stream2);
//                 cub::DeviceRadixSort::SortPairs(d_temp_storage,
//                 temp_storage_bytes, d_indices_Y, d_indices_Y, d_values_Y,
//                 d_values_Y, nnzY); common::cuda::device_sync();
//                 printf("flag3\n");
//                 common::cuda::d2hcpy(Y_indices, d_indices_Y, nnzY *
//                 sizeof(lindex_t)); printf("flag3-5\n");
//                 common::cuda::d2hcpy(SY->values, d_values_Y, nnzY *
//                 sizeof(value_t)); printf("flag4\n");
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

//                 common::cuda::h2dcpy_async(d_indices_Y, SY->indices, nnzY *
//                 sizeof(lindex_t), stream1);
//                 common::cuda::h2dcpy_async(d_values_Y, SY->values, nnzY *
//                 sizeof(value_t), stream1);

//                 void *d_temp_storage_Y = nullptr;
//                 size_t temp_storage_bytes_Y = 0;

//                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y,
//                 temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y,
//                 d_values_Y, nnzY, 0, sizeof(lindex_t) * 8, stream1);

//                 d_temp_storage_Y =
//                 pool->allocate<void>(temp_storage_bytes_Y);

//                 cub::DeviceRadixSort::SortPairs(d_temp_storage_Y,
//                 temp_storage_bytes_Y, d_indices_Y, d_indices_Y, d_values_Y,
//                 d_values_Y, nnzY, 0, sizeof(lindex_t) * 8, stream1);

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
// Helper: 2-pass CUB RadixSort for 128-bit keys stored as (uindices, indices) pairs.
// CUB RadixSort is stable, so: sort by lower 64-bit first, then by upper bits.
// Uses index permutation to carry all 3 arrays (uindices, indices, values) through sorts.
template <typename IndexType, typename ULIndexType, typename LIndexType,
          typename ValueType>
void radix_sort_128_single(ULIndexType *d_uindices, LIndexType *d_indices,
                           ValueType *d_values, uint64_t nnz, int nbits,
                           CudaMemoryPool *pool, cudaStream_t stream) {
  using index_t = IndexType;
  using ulindex_t = ULIndexType;
  using lindex_t = LIndexType;
  using value_t = ValueType;

  int lower_bits = std::min(nbits, 64);
  int upper_bits = nbits > 64 ? nbits - 64 : 0;

  // Allocate double buffers for indices (used as sort values to carry permutation)
  lindex_t *d_indices_out = pool->allocate<lindex_t>(nnz);
  ulindex_t *d_uindices_out = pool->allocate<ulindex_t>(nnz);
  value_t *d_values_out = pool->allocate<value_t>(nnz);

  // ---- Pass 1: Sort by lower 64-bit (indices), carrying uindices as values ----
  // We need to sort indices and permute uindices+values together.
  // Use CUB to sort (indices -> indices_out) with (uindices -> uindices_out) as satellite.
  // Then separately gather values using the same permutation.
  // Unfortunately CUB only supports one satellite value per sort.
  // Strategy: sort by indices carrying a sequence permutation, then gather all arrays.

  index_t *d_perm = pool->allocate<index_t>(nnz);
  index_t *d_perm_out = pool->allocate<index_t>(nnz);

  // Initialize permutation to identity [0, 1, 2, ..., nnz-1]
  {
    thrust::device_ptr<index_t> perm_ptr(d_perm);
    thrust::sequence(thrust::cuda::par.on(stream), perm_ptr, perm_ptr + nnz);
  }

  // Sort by lower 64-bit: d_indices as keys, d_perm as values
  lindex_t *d_indices_keys_out = pool->allocate<lindex_t>(nnz);
  {
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_indices,
                                    d_indices_keys_out, d_perm, d_perm_out,
                                    nnz, 0, lower_bits, stream);
    d_temp = pool->allocate<void>(temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_indices,
                                    d_indices_keys_out, d_perm, d_perm_out,
                                    nnz, 0, lower_bits, stream);
    common::cuda::stream_sync(stream);
    pool->deallocate<void>(d_temp, temp_bytes);
  }
  pool->deallocate<lindex_t>(d_indices_keys_out, nnz);

  // After pass 1, d_perm_out contains the permutation that sorts by lower 64-bit.
  // Now sort by upper bits (if any) using the permuted uindices as keys.
  if (upper_bits > 0) {
    // Gather uindices according to d_perm_out
    ulindex_t *d_uindices_permuted = pool->allocate<ulindex_t>(nnz);
    {
      thrust::device_ptr<index_t> perm_out_ptr(d_perm_out);
      thrust::device_ptr<ulindex_t> uidx_ptr(d_uindices);
      thrust::device_ptr<ulindex_t> uidx_perm_ptr(d_uindices_permuted);
      thrust::gather(thrust::cuda::par.on(stream), perm_out_ptr,
                     perm_out_ptr + nnz, uidx_ptr, uidx_perm_ptr);
    }

    // Pass 2: Sort by upper bits, carrying d_perm_out as satellite
    ulindex_t *d_uindices_keys_out = pool->allocate<ulindex_t>(nnz);
    index_t *d_perm_final = pool->allocate<index_t>(nnz);
    {
      void *d_temp = nullptr;
      size_t temp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_uindices_permuted,
                                      d_uindices_keys_out, d_perm_out,
                                      d_perm_final, nnz, 0, upper_bits, stream);
      d_temp = pool->allocate<void>(temp_bytes);
      cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_uindices_permuted,
                                      d_uindices_keys_out, d_perm_out,
                                      d_perm_final, nnz, 0, upper_bits, stream);
      common::cuda::stream_sync(stream);
      pool->deallocate<void>(d_temp, temp_bytes);
    }
    pool->deallocate<ulindex_t>(d_uindices_permuted, nnz);
    pool->deallocate<ulindex_t>(d_uindices_keys_out, nnz);
    pool->deallocate<index_t>(d_perm_out, nnz);

    // d_perm_final is the final permutation
    d_perm_out = d_perm_final;
  }

  // Gather all arrays using final permutation
  {
    thrust::device_ptr<index_t> perm_ptr(d_perm_out);
    thrust::device_ptr<ulindex_t> uidx_in(d_uindices);
    thrust::device_ptr<lindex_t> idx_in(d_indices);
    thrust::device_ptr<value_t> val_in(d_values);
    thrust::device_ptr<ulindex_t> uidx_out(d_uindices_out);
    thrust::device_ptr<lindex_t> idx_out(d_indices_out);
    thrust::device_ptr<value_t> val_out(d_values_out);

    thrust::gather(thrust::cuda::par.on(stream), perm_ptr, perm_ptr + nnz,
                   uidx_in, uidx_out);
    thrust::gather(thrust::cuda::par.on(stream), perm_ptr, perm_ptr + nnz,
                   idx_in, idx_out);
    thrust::gather(thrust::cuda::par.on(stream), perm_ptr, perm_ptr + nnz,
                   val_in, val_out);
  }
  common::cuda::stream_sync(stream);

  // Copy sorted results back over originals
  cudaMemcpyAsync(d_uindices, d_uindices_out, nnz * sizeof(ulindex_t),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(d_indices, d_indices_out, nnz * sizeof(lindex_t),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(d_values, d_values_out, nnz * sizeof(value_t),
                  cudaMemcpyDeviceToDevice, stream);
  common::cuda::stream_sync(stream);

  pool->deallocate<index_t>(d_perm, nnz);
  pool->deallocate<index_t>(d_perm_out, nnz);
  pool->deallocate<lindex_t>(d_indices_out, nnz);
  pool->deallocate<ulindex_t>(d_uindices_out, nnz);
  pool->deallocate<value_t>(d_values_out, nnz);
}

template <typename SLITOMType, typename IndexType>
void sort_128(SLITOMType *SX, SLITOMType *SY, CudaMemoryPool **pools,
              int gpu_count, Timer *timer) {
  using slitom_t = SLITOMType;
  using ulindex_t = typename slitom_t::ulindex_t;
  using lindex_t = typename slitom_t::lindex_t;
  using value_t = typename slitom_t::value_t;
  using index_t = IndexType;

  uint64_t nnzX = SX->nnz;
  uint64_t nnzY = SY->nnz;
  int nbitsX = SX->nbits;
  int nbitsY = SY->nbits;

  timer->start();

#pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
  for (int g = 0; g < gpu_count; ++g) {
    CudaMemoryPool *pool = pools[g];

    checkCudaErrors(cudaSetDevice(g));

    if (g == 0) {
      cudaStream_t stream1;
      checkCudaErrors(cudaStreamCreate(&stream1));

      // ---- Sort X first ----
      ulindex_t *d_uindices_X = pool->allocate<ulindex_t>(nnzX);
      lindex_t *d_indices_X = pool->allocate<lindex_t>(nnzX);
      value_t *d_values_X = pool->allocate<value_t>(nnzX);

      common::cuda::h2dcpy_async(d_uindices_X, SX->uindices,
                                 nnzX * sizeof(ulindex_t), stream1);
      common::cuda::h2dcpy_async(d_indices_X, SX->indices,
                                 nnzX * sizeof(lindex_t), stream1);
      common::cuda::h2dcpy_async(d_values_X, SX->values,
                                 nnzX * sizeof(value_t), stream1);
      common::cuda::stream_sync(stream1);

      radix_sort_128_single<index_t>(d_uindices_X, d_indices_X, d_values_X,
                                     nnzX, nbitsX, pool, stream1);

      // D2H for X
      common::cuda::d2hcpy(SX->uindices, d_uindices_X, nnzX * sizeof(ulindex_t));
      common::cuda::d2hcpy(SX->indices, d_indices_X, nnzX * sizeof(lindex_t));
      common::cuda::d2hcpy(SX->values, d_values_X, nnzX * sizeof(value_t));

      SX->d_uindices[g] = d_uindices_X;
      SX->d_indices[g] = d_indices_X;
      SX->d_values[g] = d_values_X;

      // ---- Sort Y (after X frees intermediate buffers inside radix_sort_128_single) ----
      ulindex_t *d_uindices_Y = pool->allocate<ulindex_t>(nnzY);
      lindex_t *d_indices_Y = pool->allocate<lindex_t>(nnzY);
      value_t *d_values_Y = pool->allocate<value_t>(nnzY);

      common::cuda::h2dcpy_async(d_uindices_Y, SY->uindices,
                                 nnzY * sizeof(ulindex_t), stream1);
      common::cuda::h2dcpy_async(d_indices_Y, SY->indices,
                                 nnzY * sizeof(lindex_t), stream1);
      common::cuda::h2dcpy_async(d_values_Y, SY->values,
                                 nnzY * sizeof(value_t), stream1);
      common::cuda::stream_sync(stream1);

      radix_sort_128_single<index_t>(d_uindices_Y, d_indices_Y, d_values_Y,
                                     nnzY, nbitsY, pool, stream1);

      // D2H for Y
      common::cuda::d2hcpy(SY->uindices, d_uindices_Y, nnzY * sizeof(ulindex_t));
      common::cuda::d2hcpy(SY->indices, d_indices_Y, nnzY * sizeof(lindex_t));
      common::cuda::d2hcpy(SY->values, d_values_Y, nnzY * sizeof(value_t));

      SY->d_uindices[g] = d_uindices_Y;
      SY->d_indices[g] = d_indices_Y;
      SY->d_values[g] = d_values_Y;

      checkCudaErrors(cudaStreamDestroy(stream1));
    }
  }
  timer->stop();
  timer->printElapsed("Sort Tensors");
}
} // namespace gsparc

#endif