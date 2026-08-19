#ifndef INDEXMATCH_COO_CUH
#define INDEXMATCH_COO_CUH

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/timer.hpp"

namespace gsparc {

// Binary search for first occurrence where coo_cmodes_Y[0][idx] == target_c0
template <typename IndexType>
__device__ __forceinline__ bool
coo_binary_search_first(const uint32_t *y_mode0, IndexType size,
                        uint32_t target, IndexType *first_occurrence) {
  IndexType left = 0, right = size;
  while (left < right) {
    IndexType mid = left + (right - left) / 2;
    if (y_mode0[mid] < target)
      left = mid + 1;
    else
      right = mid;
  }
  if (left < size && y_mode0[left] == target) {
    *first_occurrence = left;
    return true;
  }
  return false;
}

// Upper bound: first index where coo_cmodes_Y[0][idx] > target
template <typename IndexType>
__device__ __forceinline__ IndexType
coo_binary_search_upper(const uint32_t *y_mode0, IndexType size,
                        uint32_t target, IndexType start) {
  IndexType left = start, right = size;
  while (left < right) {
    IndexType mid = left + (right - left) / 2;
    if (y_mode0[mid] <= target)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}

// COO index matching kernel: per-mode tuple comparison
// Y is sorted by SLITOM packed key, so contraction modes are in LSB order.
// coo_cmodes[0] corresponds to the most significant contraction mode (last in cpos),
// but since SLITOM sorts by c-index with LSB=first cpos mode, the sort order
// ensures that coo_cmodes values are ordered lexicographically by the packed c-index.
//
// Strategy: binary search on the combined tuple by using the first mode (mode 0
// of the coo arrays, which is the most significant in the packed key) for initial
// search, then verify remaining modes linearly.
template <typename IndexType>
__global__ void
compute_result_size_coo(const uint32_t *const *d_coo_X,
                        const uint32_t *const *d_coo_Y,
                        IndexType X_nnz, IndexType Y_nnz,
                        IndexType *mCnt, IndexType *mPos,
                        int cnmodes) {
  using index_t = IndexType;

  index_t xidx = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;

  while (xidx < X_nnz) {
    // Read X contraction mode values
    uint32_t x_c[8]; // max 8 contraction modes
    for (int c = 0; c < cnmodes; ++c) {
      x_c[c] = d_coo_X[c][xidx];
    }

    // Binary search on first contraction mode (most significant)
    index_t first_occ;
    bool found = coo_binary_search_first<index_t>(
        d_coo_Y[0], Y_nnz, x_c[0], &first_occ);

    if (found) {
      // Find upper bound for first mode
      index_t upper = coo_binary_search_upper<index_t>(
          d_coo_Y[0], Y_nnz, x_c[0], first_occ);

      // Within [first_occ, upper), find exact tuple matches
      index_t match_start = Y_nnz; // sentinel
      index_t match_count = 0;

      for (index_t y = first_occ; y < upper; ++y) {
        bool all_match = true;
        for (int c = 1; c < cnmodes; ++c) {
          if (d_coo_Y[c][y] != x_c[c]) {
            all_match = false;
            break;
          }
        }
        if (all_match) {
          if (match_count == 0)
            match_start = y;
          match_count++;
        }
      }

      mPos[xidx] = match_start;
      mCnt[xidx] = match_count;
    }

    xidx += stride;
  }
}

template <typename IndexType>
void IndexMatch_COO(uint32_t **d_coo_X, uint32_t **d_coo_Y,
                    uint64_t X_nnz, uint64_t Y_nnz,
                    int cnmodes,
                    IndexType *d_mPos, IndexType *d_mCnt,
                    IndexType *d_mCntPrefix, IndexType *mCntPrefix,
                    IndexType *ir_size,
                    cudaStream_t stream, CudaMemoryPool *memory_pool,
                    int gpu_count) {
  using index_t = IndexType;

  // Copy pointer array to device
  uint32_t **d_coo_X_ptrs;
  uint32_t **d_coo_Y_ptrs;
  cudaMalloc(&d_coo_X_ptrs, sizeof(uint32_t*) * cnmodes);
  cudaMalloc(&d_coo_Y_ptrs, sizeof(uint32_t*) * cnmodes);
  cudaMemcpyAsync(d_coo_X_ptrs, d_coo_X, sizeof(uint32_t*) * cnmodes,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_coo_Y_ptrs, d_coo_Y, sizeof(uint32_t*) * cnmodes,
                  cudaMemcpyHostToDevice, stream);
  common::cuda::stream_sync(stream);

  index_t block_size = 1024;
  index_t grid_size = (X_nnz + block_size - 1) / block_size;
  dim3 blocks_per_grid(grid_size, 1, 1);
  dim3 threads_per_block(block_size, 1, 1);

  common::cuda::device_sync();

  compute_result_size_coo<index_t>
      <<<blocks_per_grid, threads_per_block, 0, stream>>>(
          d_coo_X_ptrs, d_coo_Y_ptrs, X_nnz, Y_nnz,
          d_mCnt, d_mPos, cnmodes);
  common::cuda::stream_sync(stream);

  // Prefix sum
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_mCnt, d_mCntPrefix + 1, X_nnz, stream);
  common::cuda::stream_sync(stream);
  d_temp_storage = memory_pool->allocate<void>(temp_storage_bytes);
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_mCnt, d_mCntPrefix + 1, X_nnz, stream);
  common::cuda::d2hcpy_async(mCntPrefix + 1, d_mCntPrefix + 1,
                             X_nnz * sizeof(index_t), stream);
  common::cuda::stream_sync(stream);

  mCntPrefix[0] = 0;
  *ir_size = mCntPrefix[X_nnz];
  memory_pool->deallocate<void>(d_temp_storage, temp_storage_bytes);

  cudaFree(d_coo_X_ptrs);
  cudaFree(d_coo_Y_ptrs);
}

} // namespace gsparc

#endif
