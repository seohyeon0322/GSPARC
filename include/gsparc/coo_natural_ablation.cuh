#ifndef COO_NATURAL_ABLATION_CUH
#define COO_NATURAL_ABLATION_CUH

// Natural-sorted COO IndexMatch Ablation
// - COO per-mode arrays in natural tensor order (unsorted w.r.t. c-tuple)
// - Since c-tuples are NOT contiguous, binary search is impossible
// - IndexMatch falls back to O(X*Y) linear scan (each X scans all Y)
// - Purpose: show that natural-layout COO has the same asymptotic cost as BLCO

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "gsparc/coo_ablation.cuh"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/timer.hpp"

namespace gsparc {
namespace coo_natural_ablation {

// Linear scan kernel: each X thread compares its c-tuple against all Y.
template <typename IndexType>
__global__ void natcoo_match_linear_scan(uint32_t *const *coo_X_cmodes,
                                         uint32_t *const *coo_Y_cmodes,
                                         int cnmodes, IndexType X_nnz,
                                         IndexType Y_nnz, IndexType *mCnt,
                                         IndexType *mPos) {
  IndexType xidx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType stride = blockDim.x * gridDim.x;

  while (xidx < X_nnz) {
    uint32_t x_c[8];
    for (int c = 0; c < cnmodes; ++c) {
      x_c[c] = coo_X_cmodes[c][xidx];
    }

    IndexType count = 0;
    IndexType first_pos = 0;
    bool found_first = false;

    for (IndexType yidx = 0; yidx < Y_nnz; ++yidx) {
      bool match = true;
      for (int c = 0; c < cnmodes; ++c) {
        if (coo_Y_cmodes[c][yidx] != x_c[c]) {
          match = false;
          break;
        }
      }
      if (match) {
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

// Main entry point
template <typename tensor_t>
void RunCOONaturalAblation(tensor_t *tensor_x, tensor_t *tensor_y,
                           const int *cpos_x, const int *cpos_y, int cnmodes,
                           CudaMemoryPool **pool, int gpu_count,
                           int num_iter) {
  using index_t = typename tensor_t::index_t;
  using lindex_t = uint64_t;

  printf("=== Natural-sorted COO IndexMatch Ablation (GPU, O(X*Y) scan) ===\n");
  printf("X_nnz=%llu  Y_nnz=%llu  cnmodes=%d\n",
         (unsigned long long)tensor_x->nnz,
         (unsigned long long)tensor_y->nnz, cnmodes);

  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint64_t x_nnz = tensor_x->nnz;
  uint64_t y_nnz = tensor_y->nnz;

  for (int iter = 0; iter < num_iter; ++iter) {
    printf("===============iteration %d/%d================\n", iter + 1,
           num_iter);
    pool[0]->reset();

    // Build c-mode arrays (natural order, i.e., order they appear in .tns)
    // No sort: rows with same c-tuple are NOT contiguous in natural order.
    uint32_t *d_coo_X[8], *d_coo_Y[8];
    for (int c = 0; c < cnmodes; ++c) {
      d_coo_X[c] = pool[0]->allocate<uint32_t>(x_nnz);
      coo_ablation::build_one_cmode<index_t>(
          d_coo_X[c], tensor_x->indices[cpos_x[c]], x_nnz, pool[0], stream);
    }
    for (int c = 0; c < cnmodes; ++c) {
      d_coo_Y[c] = pool[0]->allocate<uint32_t>(y_nnz);
      coo_ablation::build_one_cmode<index_t>(
          d_coo_Y[c], tensor_y->indices[cpos_y[c]], y_nnz, pool[0], stream);
    }
    common::cuda::stream_sync(stream);

    // Device-side pointer arrays
    uint32_t **d_coo_X_ptrs = nullptr;
    uint32_t **d_coo_Y_ptrs = nullptr;
    cudaMalloc(&d_coo_X_ptrs, sizeof(uint32_t *) * cnmodes);
    cudaMalloc(&d_coo_Y_ptrs, sizeof(uint32_t *) * cnmodes);
    cudaMemcpyAsync(d_coo_X_ptrs, d_coo_X, sizeof(uint32_t *) * cnmodes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_coo_Y_ptrs, d_coo_Y, sizeof(uint32_t *) * cnmodes,
                    cudaMemcpyHostToDevice, stream);
    common::cuda::stream_sync(stream);

    // IndexMatch auxiliary buffers
    lindex_t *d_mPos = pool[0]->allocate<lindex_t>(x_nnz);
    lindex_t *d_mCnt = pool[0]->allocate<lindex_t>(x_nnz);
    lindex_t *d_mCntPrefix = pool[0]->allocate<lindex_t>(x_nnz + 1);
    cudaMemsetAsync(d_mCnt, 0, x_nnz * sizeof(lindex_t), stream);
    cudaMemsetAsync(d_mPos, 0, x_nnz * sizeof(lindex_t), stream);
    common::cuda::stream_sync(stream);

    // Launch config
    uint64_t block = 256;
    uint64_t grid = (x_nnz + block - 1) / block;
    if (grid > 65535) grid = 65535;

    Timer match_timer;
    match_timer.start();

    natcoo_match_linear_scan<lindex_t><<<grid, block, 0, stream>>>(
        d_coo_X_ptrs, d_coo_Y_ptrs, cnmodes, x_nnz, y_nnz, d_mCnt, d_mPos);
    common::cuda::stream_sync(stream);

    // Prefix sum to compute ir_size
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_mCnt, d_mCntPrefix + 1,
                                  x_nnz, stream);
    d_temp = pool[0]->allocate(temp_bytes);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_mCnt, d_mCntPrefix + 1,
                                  x_nnz, stream);
    common::cuda::stream_sync(stream);
    match_timer.stop();

    // Get ir_size
    lindex_t ir_size = 0;
    cudaMemcpyAsync(&ir_size, d_mCntPrefix + x_nnz, sizeof(lindex_t),
                    cudaMemcpyDeviceToHost, stream);
    common::cuda::stream_sync(stream);

    printf("IndexMatch: %f s\n", match_timer.getTotalTime());
    printf("ir_size: %llu\n", (unsigned long long)ir_size);

    // Cleanup
    pool[0]->deallocate(d_temp, temp_bytes);
    pool[0]->deallocate<lindex_t>(d_mCntPrefix, x_nnz + 1);
    pool[0]->deallocate<lindex_t>(d_mCnt, x_nnz);
    pool[0]->deallocate<lindex_t>(d_mPos, x_nnz);
    cudaFree(d_coo_Y_ptrs);
    cudaFree(d_coo_X_ptrs);
    for (int c = cnmodes - 1; c >= 0; --c) {
      pool[0]->deallocate<uint32_t>(d_coo_Y[c], y_nnz);
    }
    for (int c = cnmodes - 1; c >= 0; --c) {
      pool[0]->deallocate<uint32_t>(d_coo_X[c], x_nnz);
    }
  }

  cudaStreamDestroy(stream);
}

} // namespace coo_natural_ablation
} // namespace gsparc

#endif // COO_NATURAL_ABLATION_CUH
