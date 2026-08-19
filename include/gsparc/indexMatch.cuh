#ifndef INDEXMATCH_CUH
#define INDEXMATCH_CUH

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "common/cuda_helper.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/helper.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/timer.hpp"

namespace gsparc {
template <typename IndexType, typename LIndexType>
__device__ __forceinline__ bool
binary_search_first_occurrence(const LIndexType *array, IndexType size,
                               IndexType target, int fbits,
                               IndexType *first_occurrence) {
  IndexType left = 0, right = size; // [left, right)
  while (left < right) {
    IndexType mid = left + (right - left) / 2;
    LIndexType y_cidx = array[mid] >> fbits;
    if (y_cidx < target)
      left = mid + 1;
    else
      right = mid;
  }
  if (left < size && ((array[left] >> fbits) == target)) {
    *first_occurrence = left;
    return true;
  }
  return false;
}

// Binary search for first index where (array[idx] >> fbits) > target (upper bound)
template <typename IndexType, typename LIndexType>
__device__ __forceinline__ IndexType
binary_search_upper_bound(const LIndexType *array, IndexType size,
                          IndexType target, int fbits, IndexType start) {
  IndexType left = start, right = size; // [left, right)
  while (left < right) {
    IndexType mid = left + (right - left) / 2;
    LIndexType y_cidx = array[mid] >> fbits;
    if (y_cidx <= target)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}

template <typename IndexType, typename LIndexType>
__global__ void
compute_result_size_cuda(const LIndexType *AX_idx, const LIndexType *AY_idx,
                         IndexType X_nnz_B, IndexType Y_nnz_B, IndexType *mCnt,
                         IndexType *mPos, int cbits, int fbits) {
  using index_t = IndexType;
  using lindex_t = LIndexType;

  index_t xidx = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;

  lindex_t mask = (static_cast<lindex_t>(1) << cbits) - 1;

  while (xidx < X_nnz_B) {
    lindex_t idx = AX_idx[xidx];
    lindex_t cidx = (idx & mask);

    index_t first_occurrence;
    bool found = binary_search_first_occurrence<index_t, lindex_t>(
        AY_idx, Y_nnz_B, cidx, fbits, &first_occurrence);
    if (found) {
      mPos[xidx] = first_occurrence;
      // Use binary search for upper bound instead of linear scan
      index_t last_exclusive = binary_search_upper_bound<index_t, lindex_t>(
          AY_idx, Y_nnz_B, cidx, fbits, first_occurrence);
      mCnt[xidx] = last_exclusive - first_occurrence;
    }
    xidx += stride;
  }
}

template <typename IndexType, typename ULIndexType, typename LIndexType>
__device__ bool
binary_search_first_occurrence_extra(const ULIndexType *extra_array,
                                     const LIndexType *array, IndexType size,
                                     __uint128_t target, int lower_cbits,
                                     int fbits, IndexType *first_occurrence) {
  using ulindex_t = ULIndexType;
  using lindex_t = LIndexType;
  using index_t = IndexType;

  index_t left = 0, right = size - 1; // [left, right)
  bool found = false;

  while (left <= right) {
    index_t mid = left + (right - left) / 2;
    __uint128_t y_idx =
        static_cast<__uint128_t>(extra_array[mid]) << 64 | array[mid];
    __uint128_t y_cidx = y_idx >> fbits;
    if (y_cidx < target)
      left = mid + 1;
    else if (y_cidx > target) {
      right = mid - 1;
    } else {
      *first_occurrence = mid;

      if (mid == 0)
        return true;
      right = mid - 1;
      found = true;
    }
    if (left >= size || right >= size) {
      break;
    }
  }

  return found;
}

template <typename IndexType, typename ULIndexType, typename LIndexType>
__global__ void compute_result_size_cuda_extra(
    const ULIndexType *AX_uidx, const LIndexType *AX_idx,
    const ULIndexType *AY_uidx, const LIndexType *AY_idx, IndexType X_nnz_B,
    IndexType Y_nnz_B, IndexType *mCnt, IndexType *mPos, int cbits,
    int X_upper_cbits, int X_lower_cbits, int Y_lower_cbits, int fbits) {
  using index_t = IndexType;
  using lindex_t = LIndexType;
  using ulindex_t = ULIndexType;

  index_t xidx = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;

  lindex_t mask = (static_cast<lindex_t>(1) << cbits) - 1;
  lindex_t upper_mask = (static_cast<lindex_t>(1) << X_upper_cbits) - 1;

  while (xidx < X_nnz_B) {
    __uint128_t idx =
        static_cast<__uint128_t>(AX_uidx[xidx]) << 64 | AX_idx[xidx];
    __uint128_t cidx = idx & (static_cast<__uint128_t>(1) << cbits) - 1;
    index_t first_occurrence;
    bool found =
        binary_search_first_occurrence_extra<index_t, ulindex_t, lindex_t>(
            AY_uidx, AY_idx, Y_nnz_B, cidx, Y_lower_cbits, fbits,
            &first_occurrence);

    if (found) {
      mPos[xidx] = first_occurrence;
      // Binary search for upper bound instead of linear scan
      index_t left_ub = first_occurrence, right_ub = Y_nnz_B;
      while (left_ub < right_ub) {
        index_t mid = left_ub + (right_ub - left_ub) / 2;
        __uint128_t y_idx =
            static_cast<__uint128_t>(AY_uidx[mid]) << 64 | AY_idx[mid];
        __uint128_t y_cidx = y_idx >> fbits;
        if (y_cidx <= cidx)
          left_ub = mid + 1;
        else
          right_ub = mid;
      }
      mCnt[xidx] = left_ub - first_occurrence;
    } else {
      mCnt[xidx] = 0;
    }
    xidx += stride;
  }
}

template <typename LIndexType, typename IndexType>
void IndexMatch(LIndexType *X_indices, uint64_t X_nnz, LIndexType *Y_indices,
                u_int64_t Y_nnz, int cbits, int fbits, IndexType *d_mPos,
                IndexType *d_mCnt, IndexType *d_mCntPrefix,
                IndexType *mCntPrefix, IndexType *ir_size, cudaStream_t stream,
                CudaMemoryPool *memory_pool, Timer *timer, int gpu_count) {
  using index_t = IndexType;
  using lindex_t = LIndexType;

  index_t block_size = 1024;
  index_t grid_size = (X_nnz + block_size - 1) / block_size;
  dim3 blocks_per_grid(grid_size, 1, 1);
  dim3 threads_per_block(block_size, 1, 1);
  cudaEvent_t start, end;
  common::cuda::device_sync();

  gsparc::compute_result_size_cuda<index_t, lindex_t>
      <<<blocks_per_grid, threads_per_block, 0, stream>>>(
          X_indices, Y_indices, X_nnz, Y_nnz, d_mCnt, d_mPos, cbits, fbits);

  common::cuda::stream_sync(stream);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // timer->stop();

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mCnt,
                                d_mCntPrefix + 1, X_nnz, stream);
  common::cuda::stream_sync(stream);
  d_temp_storage = memory_pool->allocate<void>(temp_storage_bytes);

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mCnt,
                                d_mCntPrefix + 1, X_nnz, stream);
  common::cuda::d2hcpy_async(mCntPrefix + 1, d_mCntPrefix + 1,
                             X_nnz * sizeof(index_t), stream);
  common::cuda::stream_sync(stream);
  // common::cuda::end_timer_with_msg(&start, &end, "IndexMatch");

  mCntPrefix[0] = 0;
  *ir_size = mCntPrefix[X_nnz];
  memory_pool->deallocate<void>(d_temp_storage, temp_storage_bytes);
}

template <typename ULIndexType, typename LIndexType, typename IndexType>
void IndexMatch_extra(ULIndexType *X_uindices, LIndexType *X_indices,
                      uint64_t X_nnz, ULIndexType *Y_uindices,
                      LIndexType *Y_indices, u_int64_t Y_nnz, int cbits,
                      int fbits, IndexType *d_mPos, IndexType *d_mCnt,
                      IndexType *d_mCntPrefix, IndexType *mCntPrefix,
                      IndexType *ir_size, cudaStream_t stream,
                      CudaMemoryPool *memory_pool, int gpu_count) {
  using index_t = IndexType;
  using ulindex_t = ULIndexType;
  using lindex_t = LIndexType;

  index_t block_size = 1024;
  index_t grid_size = (X_nnz + block_size - 1) / block_size;
  dim3 blocks_per_grid(grid_size, 1, 1);
  dim3 threads_per_block(block_size, 1, 1);
  cudaEvent_t start, end;
  common::cuda::device_sync();
  int X_extra_cbits = cbits - 64 > 0 ? cbits - 64 : 0;
  int Y_extra_cbits = (cbits + fbits) - 64 > 0 ? (cbits + fbits) - 64 : 0;
  if (fbits > 64) {
    fprintf(stderr, "Error: fbits should be less than or equal to 64\n");
    exit(1);
  }

  gsparc::compute_result_size_cuda_extra<index_t, ulindex_t, lindex_t>
      <<<blocks_per_grid, threads_per_block, 0, stream>>>(
          X_uindices, X_indices, Y_uindices, Y_indices, X_nnz, Y_nnz, d_mCnt,
          d_mPos, cbits, X_extra_cbits, cbits - X_extra_cbits,
          cbits - Y_extra_cbits, fbits);

  common::cuda::stream_sync(stream);
  // common::cuda::end_timer_with_msg(&start, &end, "IndexMatch");
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mCnt,
                                d_mCntPrefix + 1, X_nnz, stream);
  common::cuda::stream_sync(stream);
  d_temp_storage = memory_pool->allocate<void>(temp_storage_bytes);

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mCnt,
                                d_mCntPrefix + 1, X_nnz, stream);
  common::cuda::d2hcpy_async(mCntPrefix + 1, d_mCntPrefix + 1,
                             X_nnz * sizeof(index_t), stream);
  common::cuda::stream_sync(stream);

  mCntPrefix[0] = 0;
  *ir_size = mCntPrefix[X_nnz];
  memory_pool->deallocate<void>(d_temp_storage, temp_storage_bytes);
}
} // namespace gsparc

#endif