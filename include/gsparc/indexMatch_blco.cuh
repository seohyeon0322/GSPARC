#ifndef INDEXMATCH_BLCO_CUH
#define INDEXMATCH_BLCO_CUH

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/timer.hpp"

namespace gsparc {

#define BLCO_MAX_CMODES 8

// Device-side struct for contraction mode extraction info
struct BLCOCmodeInfo {
    int offsets[BLCO_MAX_CMODES];
    int widths[BLCO_MAX_CMODES];
    int ncmodes;
};

// Extract contraction-only key from full BLCO key (all modes packed in natural order)
__device__ __forceinline__ uint64_t
blco_extract_ckey(uint64_t blco_key, const BLCOCmodeInfo &info) {
    uint64_t ckey = 0;
    int shift = 0;
    for (int c = 0; c < info.ncmodes; ++c) {
        uint64_t cval = (blco_key >> info.offsets[c]) & ((1ULL << info.widths[c]) - 1);
        ckey |= (cval << shift);
        shift += info.widths[c];
    }
    return ckey;
}

// BLCO index matching kernel: linear scan
// BLCO keys are sorted by full key (all modes in natural order),
// so contraction values are NOT contiguous → binary search impossible.
// Each X thread scans all Y elements, comparing extracted contraction values.
template <typename IndexType>
__global__ void
compute_result_size_blco(const uint64_t *blco_X, const uint64_t *blco_Y,
                         IndexType X_nnz, IndexType Y_nnz,
                         IndexType *mCnt, IndexType *mPos,
                         BLCOCmodeInfo info) {
  using index_t = IndexType;

  index_t xidx = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;

  while (xidx < X_nnz) {
    uint64_t x_ckey = blco_extract_ckey(blco_X[xidx], info);

    index_t count = 0;
    index_t first_pos = 0;
    bool found_first = false;

    for (index_t yidx = 0; yidx < Y_nnz; ++yidx) {
      uint64_t y_ckey = blco_extract_ckey(blco_Y[yidx], info);
      if (y_ckey == x_ckey) {
        if (!found_first) {
          first_pos = yidx;
          found_first = true;
        }
        count++;
      }
    }

    if (count > 0) {
      mPos[xidx] = first_pos;
      mCnt[xidx] = count;
    }

    xidx += stride;
  }
}

template <typename IndexType>
void IndexMatch_BLCO(uint64_t *d_blco_X, uint64_t *d_blco_Y,
                     uint64_t X_nnz, uint64_t Y_nnz,
                     IndexType *d_mPos, IndexType *d_mCnt,
                     IndexType *d_mCntPrefix, IndexType *mCntPrefix,
                     IndexType *ir_size,
                     cudaStream_t stream, CudaMemoryPool *memory_pool,
                     int gpu_count,
                     int ncmodes, const int *cmode_offsets, const int *cmode_widths) {
  using index_t = IndexType;

  // Build device-side info struct
  BLCOCmodeInfo info;
  info.ncmodes = ncmodes;
  for (int c = 0; c < ncmodes; ++c) {
    info.offsets[c] = cmode_offsets[c];
    info.widths[c] = cmode_widths[c];
  }

  index_t block_size = 1024;
  index_t grid_size = (X_nnz + block_size - 1) / block_size;
  dim3 blocks_per_grid(grid_size, 1, 1);
  dim3 threads_per_block(block_size, 1, 1);

  common::cuda::device_sync();

  compute_result_size_blco<index_t>
      <<<blocks_per_grid, threads_per_block, 0, stream>>>(
          d_blco_X, d_blco_Y, X_nnz, Y_nnz, d_mCnt, d_mPos, info);
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
}

} // namespace gsparc

#endif
