#ifndef COO_ABLATION_CUH
#define COO_ABLATION_CUH

// Pure COO IndexMatch Ablation
// - Builds COO per-mode device arrays directly from raw tensor->indices
//   (does NOT go through SLITOM ConvertTensor/Sort/Partition)
// - Single GPU partition by default; nnz-equal X-chunking fallback on OOM
// - Stops after IndexMatch kernel (no accumulation)

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>
#include <omp.h>
#include <parallel/algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/indexMatch_coo.cuh"
#include "gsparc/timer.hpp"

namespace gsparc {
namespace coo_ablation {

// ----- kernels -----------------------------------------------------------

template <typename SrcType>
__global__ void cast_truncate_kernel(uint32_t *__restrict__ dst,
                                     const SrcType *__restrict__ src,
                                     uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    dst[i] = static_cast<uint32_t>(src[i]);
  }
}

__global__ void init_perm_kernel(uint64_t *__restrict__ perm, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    perm[i] = i;
  }
}

// key[i] = OR_c (coo[c][i] << bit_offsets[c]); mode 0 placed at MSB.
__global__ void build_composite_key_kernel(uint64_t *__restrict__ keys,
                                           uint32_t *const *coo_arrays,
                                           const int *bit_offsets,
                                           int cnmodes, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    uint64_t k = 0;
    for (int c = 0; c < cnmodes; ++c) {
      k |= (uint64_t)(coo_arrays[c][i]) << bit_offsets[c];
    }
    keys[i] = k;
  }
}

__global__ void gather_u32_kernel(uint32_t *__restrict__ dst,
                                  const uint32_t *__restrict__ src,
                                  const uint64_t *__restrict__ perm,
                                  uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    dst[i] = src[perm[i]];
  }
}

// ----- helpers -----------------------------------------------------------

inline void launch_cfg(uint64_t n, uint64_t &grid, uint64_t &block) {
  block = 256;
  grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;
}

// H2D copy (with uint64 -> uint32 truncation if needed) of raw indices for a
// single contraction mode.
template <typename index_t>
void build_one_cmode(uint32_t *d_dst, const index_t *h_src, uint64_t nnz,
                     CudaMemoryPool *pool, cudaStream_t stream) {
  if (sizeof(index_t) == sizeof(uint32_t)) {
    common::cuda::h2dcpy_async(d_dst, h_src, nnz * sizeof(uint32_t), stream);
    return;
  }
  // uint64 path: stage on GPU then truncate.
  index_t *d_tmp = pool->allocate<index_t>(nnz);
  common::cuda::h2dcpy_async(d_tmp, h_src, nnz * sizeof(index_t), stream);
  uint64_t grid, block;
  launch_cfg(nnz, grid, block);
  cast_truncate_kernel<index_t><<<grid, block, 0, stream>>>(d_dst, d_tmp, nnz);
  common::cuda::stream_sync(stream);
  pool->deallocate<index_t>(d_tmp, nnz);
}

// Sort (d_coo[0..cnmodes-1][nnz]) lexicographically using a 64-bit composite
// key (mode 0 placed at MSB). Assumes total_bits <= 64.
inline void sort_composite_64(uint32_t **d_coo, int cnmodes, uint64_t nnz,
                              const int *bits, CudaMemoryPool *pool,
                              cudaStream_t stream) {
  // Compute MSB-first bit offsets
  int offsets[8];
  int off = 64;
  for (int c = 0; c < cnmodes; ++c) {
    off -= bits[c];
    offsets[c] = off;
  }
  int key_begin_bit = offsets[cnmodes - 1];
  int key_end_bit = 64;

  uint64_t *d_keys = pool->allocate<uint64_t>(nnz);
  uint64_t *d_keys_out = pool->allocate<uint64_t>(nnz);
  uint64_t *d_perm = pool->allocate<uint64_t>(nnz);
  uint64_t *d_perm_out = pool->allocate<uint64_t>(nnz);

  uint32_t **d_coo_ptrs = nullptr;
  int *d_offsets = nullptr;
  cudaMalloc(&d_coo_ptrs, sizeof(uint32_t *) * cnmodes);
  cudaMalloc(&d_offsets, sizeof(int) * cnmodes);
  cudaMemcpyAsync(d_coo_ptrs, d_coo, sizeof(uint32_t *) * cnmodes,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_offsets, offsets, sizeof(int) * cnmodes,
                  cudaMemcpyHostToDevice, stream);
  common::cuda::stream_sync(stream);

  uint64_t grid, block;
  launch_cfg(nnz, grid, block);

  build_composite_key_kernel<<<grid, block, 0, stream>>>(
      d_keys, d_coo_ptrs, d_offsets, cnmodes, nnz);
  init_perm_kernel<<<grid, block, 0, stream>>>(d_perm, nnz);
  common::cuda::stream_sync(stream);

  // CUB radix sort
  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys, d_keys_out,
                                  d_perm, d_perm_out, nnz, key_begin_bit,
                                  key_end_bit, stream);
  d_temp = pool->allocate(temp_bytes);
  cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys, d_keys_out,
                                  d_perm, d_perm_out, nnz, key_begin_bit,
                                  key_end_bit, stream);
  common::cuda::stream_sync(stream);

  // Apply permutation to each coo array (in-place with 1-buf scratch)
  uint32_t *d_tmp = pool->allocate<uint32_t>(nnz);
  for (int c = 0; c < cnmodes; ++c) {
    gather_u32_kernel<<<grid, block, 0, stream>>>(d_tmp, d_coo[c], d_perm_out,
                                                  nnz);
    common::cuda::stream_sync(stream);
    cudaMemcpyAsync(d_coo[c], d_tmp, nnz * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    common::cuda::stream_sync(stream);
  }
  pool->deallocate<uint32_t>(d_tmp, nnz);

  pool->deallocate(d_temp, temp_bytes);
  cudaFree(d_offsets);
  cudaFree(d_coo_ptrs);
  pool->deallocate<uint64_t>(d_perm_out, nnz);
  pool->deallocate<uint64_t>(d_perm, nnz);
  pool->deallocate<uint64_t>(d_keys_out, nnz);
  pool->deallocate<uint64_t>(d_keys, nnz);
}

// LSD multi-pass radix sort when total_bits > 64. Stable sort mode by mode
// starting from the least-significant contraction mode.
inline void sort_lsd_multipass(uint32_t **d_coo, int cnmodes, uint64_t nnz,
                               const int *bits, CudaMemoryPool *pool,
                               cudaStream_t stream) {
  uint32_t *d_key_a = pool->allocate<uint32_t>(nnz);
  uint32_t *d_key_b = pool->allocate<uint32_t>(nnz);
  uint64_t *d_perm_a = pool->allocate<uint64_t>(nnz);
  uint64_t *d_perm_b = pool->allocate<uint64_t>(nnz);

  uint64_t grid, block;
  launch_cfg(nnz, grid, block);
  init_perm_kernel<<<grid, block, 0, stream>>>(d_perm_a, nnz);
  common::cuda::stream_sync(stream);

  // LSD: sort by cnmodes-1 first, ..., mode 0 last
  for (int c = cnmodes - 1; c >= 0; --c) {
    // key[i] = d_coo[c][d_perm_a[i]]
    gather_u32_kernel<<<grid, block, 0, stream>>>(d_key_a, d_coo[c], d_perm_a,
                                                  nnz);
    common::cuda::stream_sync(stream);

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_key_a, d_key_b,
                                    d_perm_a, d_perm_b, nnz, 0, bits[c],
                                    stream);
    d_temp = pool->allocate(temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_key_a, d_key_b,
                                    d_perm_a, d_perm_b, nnz, 0, bits[c],
                                    stream);
    common::cuda::stream_sync(stream);
    pool->deallocate(d_temp, temp_bytes);

    std::swap(d_perm_a, d_perm_b);
  }

  // Apply final d_perm_a to each coo array
  uint32_t *d_tmp = pool->allocate<uint32_t>(nnz);
  for (int c = 0; c < cnmodes; ++c) {
    gather_u32_kernel<<<grid, block, 0, stream>>>(d_tmp, d_coo[c], d_perm_a,
                                                  nnz);
    common::cuda::stream_sync(stream);
    cudaMemcpyAsync(d_coo[c], d_tmp, nnz * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    common::cuda::stream_sync(stream);
  }
  pool->deallocate<uint32_t>(d_tmp, nnz);

  pool->deallocate<uint64_t>(d_perm_b, nnz);
  pool->deallocate<uint64_t>(d_perm_a, nnz);
  pool->deallocate<uint32_t>(d_key_b, nnz);
  pool->deallocate<uint32_t>(d_key_a, nnz);
}

// CPU sort for COO arrays — lexicographic on contraction modes.
// Sort time is NOT measured (only IndexMatch is timed), so CPU sort is fine.
inline void sort_coo_cpu(uint32_t **h_coo, int cnmodes, uint64_t nnz) {
  printf("[COO Ablation] CPU sorting %llu elements (cnmodes=%d)...\n",
         (unsigned long long)nnz, cnmodes);
  std::vector<uint64_t> perm(nnz);
  std::iota(perm.begin(), perm.end(), 0ULL);

  __gnu_parallel::sort(perm.begin(), perm.end(),
                       [h_coo, cnmodes](uint64_t a, uint64_t b) {
                         for (int c = 0; c < cnmodes; ++c) {
                           if (h_coo[c][a] != h_coo[c][b])
                             return h_coo[c][a] < h_coo[c][b];
                         }
                         return false;
                       });

  // Apply permutation one mode at a time to limit memory
  std::vector<uint32_t> tmp(nnz);
  for (int c = 0; c < cnmodes; ++c) {
    for (uint64_t i = 0; i < nnz; ++i) {
      tmp[i] = h_coo[c][perm[i]];
    }
    std::memcpy(h_coo[c], tmp.data(), nnz * sizeof(uint32_t));
  }
  printf("[COO Ablation] CPU sort done.\n");
}

// Public: build one side (X or Y) of the COO representation on GPU.
template <typename tensor_t>
void BuildCOOGPU(tensor_t *tensor, const int *cpos, int cnmodes,
                 uint32_t **d_coo, CudaMemoryPool *pool, cudaStream_t stream) {
  using index_t = typename tensor_t::index_t;
  uint64_t nnz = tensor->nnz;
  for (int c = 0; c < cnmodes; ++c) {
    d_coo[c] = pool->allocate<uint32_t>(nnz);
    build_one_cmode<index_t>(d_coo[c], tensor->indices[cpos[c]], nnz, pool,
                             stream);
  }
  common::cuda::stream_sync(stream);
}

// Public: sort the COO representation on GPU.
template <typename tensor_t>
void SortCOOGPU(tensor_t *tensor, const int *cpos, int cnmodes,
                uint32_t **d_coo, CudaMemoryPool *pool, cudaStream_t stream) {
  using index_t = typename tensor_t::index_t;
  uint64_t nnz = tensor->nnz;
  int bits[8];
  int total_bits = 0;
  for (int c = 0; c < cnmodes; ++c) {
    index_t dim = tensor->dims[cpos[c]];
    int b = 0;
    index_t v = (dim == 0) ? 0 : dim - 1;
    while (v > 0) {
      ++b;
      v >>= 1;
    }
    if (b == 0) b = 1;
    bits[c] = b;
    total_bits += b;
  }
  if (total_bits <= 64) {
    sort_composite_64(d_coo, cnmodes, nnz, bits, pool, stream);
  } else {
    sort_lsd_multipass(d_coo, cnmodes, nnz, bits, pool, stream);
  }
}

// Estimate peak GPU memory (bytes) for one X-chunk + full Y.
// Sort is done on CPU, so no GPU sort auxiliary memory needed.
inline uint64_t estimate_mem(uint64_t x_nnz, uint64_t y_nnz, int cnmodes) {
  // coo arrays (X+Y): cnmodes * (x+y) * 4
  // IndexMatch aux: mPos + mCnt + mCntPrefix = 3 * x_nnz * 8
  uint64_t coo_bytes = (uint64_t)cnmodes * (x_nnz + y_nnz) * sizeof(uint32_t);
  uint64_t im_bytes = 3ULL * x_nnz * sizeof(uint64_t);
  return coo_bytes + im_bytes;
}

// Process one (X-chunk, full Y) pair: build on CPU, sort on CPU, H2D, IndexMatch.
template <typename tensor_t>
uint64_t process_pair(tensor_t *tensor_x, tensor_t *tensor_y,
                      uint64_t x_off, uint64_t x_nnz, const int *cpos_x,
                      const int *cpos_y, int cnmodes, CudaMemoryPool *pool,
                      cudaStream_t stream, Timer *match_timer) {
  using index_t = typename tensor_t::index_t;
  using lindex_t = uint64_t;
  uint64_t y_nnz = tensor_y->nnz;

  // 1. Build COO arrays on CPU
  uint32_t *h_coo_X[8], *h_coo_Y[8];
  for (int c = 0; c < cnmodes; ++c) {
    h_coo_X[c] = new uint32_t[x_nnz];
    h_coo_Y[c] = new uint32_t[y_nnz];
    const index_t *src_x = tensor_x->indices[cpos_x[c]] + x_off;
    const index_t *src_y = tensor_y->indices[cpos_y[c]];
    for (uint64_t i = 0; i < x_nnz; ++i)
      h_coo_X[c][i] = static_cast<uint32_t>(src_x[i]);
    for (uint64_t i = 0; i < y_nnz; ++i)
      h_coo_Y[c][i] = static_cast<uint32_t>(src_y[i]);
  }

  // 2. Sort on CPU (sort time is NOT included in IndexMatch measurement)
  sort_coo_cpu(h_coo_X, cnmodes, x_nnz);
  sort_coo_cpu(h_coo_Y, cnmodes, y_nnz);

  // 3. H2D copy sorted arrays to GPU
  uint32_t *d_coo_X[8], *d_coo_Y[8];
  for (int c = 0; c < cnmodes; ++c) {
    d_coo_X[c] = pool->allocate<uint32_t>(x_nnz);
    common::cuda::h2dcpy_async(d_coo_X[c], h_coo_X[c],
                               x_nnz * sizeof(uint32_t), stream);
    delete[] h_coo_X[c];
  }
  for (int c = 0; c < cnmodes; ++c) {
    d_coo_Y[c] = pool->allocate<uint32_t>(y_nnz);
    common::cuda::h2dcpy_async(d_coo_Y[c], h_coo_Y[c],
                               y_nnz * sizeof(uint32_t), stream);
    delete[] h_coo_Y[c];
  }
  common::cuda::stream_sync(stream);

  // 4. IndexMatch on GPU
  lindex_t *d_mPos = pool->allocate<lindex_t>(x_nnz);
  lindex_t *d_mCnt = pool->allocate<lindex_t>(x_nnz);
  lindex_t *d_mCntPrefix = pool->allocate<lindex_t>(x_nnz + 1);
  lindex_t *mCntPrefix = new lindex_t[x_nnz + 1];
  cudaMemsetAsync(d_mCnt, 0, x_nnz * sizeof(lindex_t), stream);
  cudaMemsetAsync(d_mPos, 0, x_nnz * sizeof(lindex_t), stream);
  common::cuda::stream_sync(stream);

  lindex_t ir_size = 0;
  match_timer->start();
  IndexMatch_COO<lindex_t>(d_coo_X, d_coo_Y, x_nnz, y_nnz, cnmodes, d_mPos,
                           d_mCnt, d_mCntPrefix, mCntPrefix, &ir_size, stream,
                           pool, 1);
  match_timer->stop();

  delete[] mCntPrefix;
  pool->deallocate<lindex_t>(d_mCntPrefix, x_nnz + 1);
  pool->deallocate<lindex_t>(d_mCnt, x_nnz);
  pool->deallocate<lindex_t>(d_mPos, x_nnz);
  for (int c = cnmodes - 1; c >= 0; --c) {
    pool->deallocate<uint32_t>(d_coo_Y[c], y_nnz);
  }
  for (int c = cnmodes - 1; c >= 0; --c) {
    pool->deallocate<uint32_t>(d_coo_X[c], x_nnz);
  }
  return (uint64_t)ir_size;
}

// Main entry point: runs num_iter iterations of pure COO IndexMatch ablation.
template <typename tensor_t>
void RunCOOAblation(tensor_t *tensor_x, tensor_t *tensor_y,
                    const int *cpos_x, const int *cpos_y, int cnmodes,
                    CudaMemoryPool **pool, int gpu_count, int num_iter) {
  printf("=== Pure COO IndexMatch Ablation (GPU, no SLITOM) ===\n");
  printf("X_nnz=%llu  Y_nnz=%llu  cnmodes=%d\n",
         (unsigned long long)tensor_x->nnz,
         (unsigned long long)tensor_y->nnz, cnmodes);

  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Determine chunking: single partition if it fits, else nnz-equal X chunks.
  uint64_t pool_size = pool[0]->getPoolSize();
  uint64_t need_single =
      estimate_mem(tensor_x->nnz, tensor_y->nnz, cnmodes);
  int num_chunks = 1;
  uint64_t chunk_size = tensor_x->nnz;
  if (need_single >= pool_size) {
    // Find smallest num_chunks such that estimate(nnz/chunks, y_nnz) fits.
    // Use 90% of pool as budget to leave headroom.
    uint64_t budget = (pool_size * 9ULL) / 10ULL;
    num_chunks = 1;
    while (true) {
      uint64_t cs = (tensor_x->nnz + num_chunks - 1) / num_chunks;
      uint64_t need = estimate_mem(cs, tensor_y->nnz, cnmodes);
      if (need <= budget) {
        chunk_size = cs;
        break;
      }
      num_chunks *= 2;
      if (num_chunks > 4096) {
        fprintf(stderr,
                "[COO Ablation] cannot fit even with 4096 chunks; abort\n");
        cudaStreamDestroy(stream);
        return;
      }
    }
    printf("[COO Ablation] nnz-equal chunking: %d chunks (chunk_size=%llu)\n",
           num_chunks, (unsigned long long)chunk_size);
  } else {
    printf("[COO Ablation] single partition\n");
  }

  for (int iter = 0; iter < num_iter; ++iter) {
    printf("===============iteration %d/%d================\n", iter + 1,
           num_iter);
    pool[0]->reset();

    Timer match_timer;
    uint64_t total_ir = 0;
    for (int ck = 0; ck < num_chunks; ++ck) {
      uint64_t x_off = (uint64_t)ck * chunk_size;
      uint64_t x_end = std::min(x_off + chunk_size, tensor_x->nnz);
      uint64_t x_nnz = x_end - x_off;
      if (x_nnz == 0) break;
      uint64_t ir_chunk =
          process_pair(tensor_x, tensor_y, x_off, x_nnz, cpos_x, cpos_y,
                       cnmodes, pool[0], stream, &match_timer);
      total_ir += ir_chunk;
    }
    printf("IndexMatch: %f s\n", match_timer.getTotalTime());
    printf("ir_size: %llu\n", (unsigned long long)total_ir);
    match_timer.reset();
  }

  cudaStreamDestroy(stream);
}

} // namespace coo_ablation
} // namespace gsparc

#endif // COO_ABLATION_CUH
