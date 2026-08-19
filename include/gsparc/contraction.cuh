#ifndef CONTRACTION_CUH
#define CONTRACTION_CUH

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <thrust/iterator/counting_iterator.h>

#include "common/cuda_helper.hpp"
#include "common/size.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/helper.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/timer.hpp"

namespace gsparc {

template <typename ValueType>
__global__ void dense_flag_kernel(ValueType *values, unsigned char *flags,
                                  uint64_t n) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = blockDim.x * gridDim.x;
  for (uint64_t i = tid; i < n; i += stride) {
    flags[i] = (values[i] != static_cast<ValueType>(0)) ? 1 : 0;
  }
}

// find_idx with hint: if hint is valid, narrow search range since idx is monotonically increasing
template <typename IndexType>
__device__ void find_idx(IndexType idx, IndexType *xidx, IndexType *offset,
                         uint64_t X_nnz, IndexType *mCntPrefix,
                         IndexType hint = 0) {
  using index_t = IndexType;

  index_t left = hint;
  index_t right = X_nnz;

  // Quick check: is the hint still valid for this idx?
  if (left < X_nnz && mCntPrefix[left] <= idx && idx < mCntPrefix[left + 1]) {
    *xidx = left;
    *offset = idx - mCntPrefix[left];
    return;
  }

  while (left <= right) {
    index_t mid = left + (right - left) / 2;
    if (mCntPrefix[mid] <= idx && idx < mCntPrefix[mid + 1]) {
      *xidx = mid;
      *offset = idx - mCntPrefix[mid];
      return;
    } else if (idx < mCntPrefix[mid]) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  printf("Error: idx is out of range\n");
}

template <typename IndexType, typename LIndexType, typename ValueType>
__global__ void contraction_dense_kernel(
    LIndexType *X_indices, ValueType *X_values, LIndexType *Y_indices,
    ValueType *Y_values, uint64_t X_nnz, uint64_t Y_nnz, IndexType *mPos,
    IndexType *mCnt, ValueType *result_values, uint64_t result_size,
    uint64_t dense_size, IndexType dense_start, int X_nfbits, int Y_nfbits,
    int cbits) {
  using index_t = IndexType;
  using lindex_t = LIndexType;
  using value_t = ValueType;

  index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;

  for (index_t idx = tid; idx < X_nnz; idx += stride) {
    index_t yidx = mPos[idx];
    index_t cnt = mCnt[idx];
    for (int offset = 0; offset < cnt; ++offset) {
      lindex_t x_fidx = (X_indices[idx] >> cbits);
      lindex_t y_fidx = (Y_indices[yidx + offset] &
                         ((static_cast<lindex_t>(1) << Y_nfbits) - 1));

      lindex_t result_idx = (x_fidx << Y_nfbits) + y_fidx;
      value_t result_val = X_values[idx] * Y_values[yidx + offset];

      atomicAdd(&result_values[result_idx - dense_start], result_val);
    }
  }
}

template <typename IndexType, typename LIndexType, typename ValueType>
__global__ void contraction_esc_kernel(
    LIndexType *X_indices, ValueType *X_values, LIndexType *Y_indices,
    ValueType *Y_values, uint64_t X_nnz, uint64_t Y_nnz, uint64_t ir_nnz,
    IndexType *mPos, IndexType *mCnt, IndexType *mCntPrefix,
    IndexType start_offset, LIndexType *result_indices,
    ValueType *result_values, int X_fbits, int Y_fbits, int cbits) {
  using index_t = IndexType;
  using lindex_t = LIndexType;
  using value_t = ValueType;

  index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;

  index_t cached_xidx = 0;
  for (index_t idx = tid; idx < ir_nnz; idx += stride) {
    index_t xidx, offset;
    find_idx<index_t>(idx + start_offset, &xidx, &offset, X_nnz, mCntPrefix,
                      cached_xidx);
    cached_xidx = xidx;

    index_t yidx = mPos[xidx];

    lindex_t x_fidx = X_indices[xidx] >> cbits;
    lindex_t y_fidx =
        Y_indices[yidx + offset] & ((static_cast<index_t>(1) << Y_fbits) - 1);

    lindex_t result_idx = (x_fidx << Y_fbits) + y_fidx;
    value_t result_val = X_values[xidx] * Y_values[yidx + offset];
    result_indices[idx] = result_idx;
    result_values[idx] = result_val;
  }
}

template <typename IndexType, typename LIndexType, typename ValueType>
void Dynamic_partition(LIndexType *X_indices, uint64_t X_nnz, uint64_t ir_nnz,
                       IndexType *mCntPrefix, int ncbits,
                       CudaMemoryPool *memoryPool,
                       std::vector<IndexType> &prtn_offset,
                       uint64_t *max_prtn_ir_nnz) {
  using index_t = IndexType;
  using lindex_t = LIndexType;
  using value_t = ValueType;

  // Accurate memory cost per intermediate result element in ESC_Contraction:
  //   Double-buffer: 2 * (sizeof(lindex_t) + sizeof(value_t)) = 32 bytes
  //   CUB temp (RadixSort/ReduceByKey): ~2 * sizeof(lindex_t) = 16 bytes
  const size_t bytes_per_ir =
      (sizeof(lindex_t) + sizeof(value_t)) * 2 + sizeof(lindex_t) * 2;
  const size_t fixed_overhead = common::MiB(32);

  size_t available = memoryPool->getRemainingMemory();
  uint64_t budget_ir_nnz = (available > fixed_overhead)
      ? (available - fixed_overhead) / bytes_per_ir
      : 1;

  printf("Dynamic_partition: available=%s, budget_ir=%lu, bytes_per_ir=%lu\n",
         common::byteToString(available), budget_ir_nnz, bytes_per_ir);

  index_t start_pointer = 0;
  while (start_pointer < X_nnz) {
    // All remaining elements share the same contract-mode group
    if ((X_indices[start_pointer] >> ncbits) ==
        (X_indices[X_nnz - 1] >> ncbits)) {
      uint64_t remaining_ir = ir_nnz - mCntPrefix[start_pointer];
      if (remaining_ir > *max_prtn_ir_nnz)
        *max_prtn_ir_nnz = remaining_ir;
      prtn_offset.push_back(X_nnz - 1);
      break;
    }

    // Binary search on mCntPrefix: largest end where ir_nnz fits in budget
    index_t lo = start_pointer, hi = X_nnz - 1;
    while (lo < hi) {
      index_t mid = lo + (hi - lo + 1) / 2;
      uint64_t cost = (mid == X_nnz - 1)
          ? (ir_nnz - mCntPrefix[start_pointer])
          : (mCntPrefix[mid + 1] - mCntPrefix[start_pointer]);
      if (cost <= budget_ir_nnz)
        lo = mid;
      else
        hi = mid - 1;
    }
    index_t candidate = lo;
    lindex_t cand_mode = X_indices[candidate] >> ncbits;

    // Snap forward: find end of this contract-mode group
    index_t forward_end;
    {
      index_t lo2 = candidate, hi2 = X_nnz - 1;
      while (lo2 < hi2) {
        index_t mid = lo2 + (hi2 - lo2 + 1) / 2;
        if ((X_indices[mid] >> ncbits) == cand_mode)
          lo2 = mid;
        else
          hi2 = mid - 1;
      }
      forward_end = lo2;
    }

    // Check if including the full mode group still fits
    uint64_t forward_ir = (forward_end == X_nnz - 1)
        ? (ir_nnz - mCntPrefix[start_pointer])
        : (mCntPrefix[forward_end + 1] - mCntPrefix[start_pointer]);

    if (forward_ir <= budget_ir_nnz) {
      candidate = forward_end;
    } else {
      // Snap backward: find first element of this mode group, take previous
      index_t lo3 = start_pointer, hi3 = candidate;
      while (lo3 < hi3) {
        index_t mid = lo3 + (hi3 - lo3) / 2;
        if ((X_indices[mid] >> ncbits) == cand_mode)
          hi3 = mid;
        else
          lo3 = mid + 1;
      }
      if (lo3 <= start_pointer) {
        // Single mode group exceeds budget — must include it anyway
        candidate = forward_end;
      } else {
        candidate = lo3 - 1;
      }
    }

    uint64_t curr_ir = (candidate == X_nnz - 1)
        ? (ir_nnz - mCntPrefix[start_pointer])
        : (mCntPrefix[candidate + 1] - mCntPrefix[start_pointer]);

    printf("Partition [%lu, %lu] -- ir_nnz: %lu, size: %s\n",
           start_pointer, candidate, curr_ir,
           common::byteToString(curr_ir * bytes_per_ir));

    if (curr_ir > *max_prtn_ir_nnz)
      *max_prtn_ir_nnz = curr_ir;

    prtn_offset.push_back(candidate);
    start_pointer = candidate + 1;
  }

  if (prtn_offset.empty() || prtn_offset.back() != X_nnz - 1) {
    prtn_offset.push_back(X_nnz - 1);
  }
}

template <typename IndexType, typename LIndexType, typename ValueType>
void Dense_Contraction(LIndexType *X_indices, ValueType *X_values,
                       IndexType X_nnz, LIndexType *Y_indices,
                       ValueType *Y_values, IndexType Y_nnz, uint64_t ir_size,
                       IndexType *mPos, IndexType *mCnt, IndexType dense_start,
                       uint64_t dense_size, LIndexType *&Z_indices,
                       ValueType *&Z_values, IndexType *Z_nnz, int X_nfbits,
                       int Y_nfbits, int ncbits, CudaMemoryPool *memoryPool,
                       bool multi, Timer *timer) {
  using index_t = IndexType;
  using lindex_t = LIndexType;
  using value_t = ValueType;

  int num_streams = 3;

  cudaEvent_t start, stop;
  value_t *result_values = static_cast<value_t *>(
      common::cuda::pinned_malloc(sizeof(value_t) * dense_size));

  // checkCudaErrors(cudaEventCreate(&t_start));
  // checkCudaErrors(cudaEventCreate(&t_stop));
  // checkCudaErrors(cudaEventRecord(t_start, 0));
  printf("X_nnz: %lu, Y_nnz: %lu, ir_size: %lu, dense_start: %lu, dense_size: "
         "%lu\n",
         X_nnz, Y_nnz, ir_size, dense_start, dense_size);
  timer->start();

  value_t *d_dense_values = memoryPool->allocate<value_t>(dense_size);
  common::cuda::device_memset(d_dense_values, static_cast<value_t>(0),
                              sizeof(value_t) * dense_size);
  // dim3 dimGrid((result_size + block_size - 1) / block_size, 1, 1);
  index_t block_size = 1024;
  index_t grid_size = (X_nnz + block_size - 1) / block_size;
  dim3 blocks_per_grid(grid_size, 1, 1);
  dim3 threads_per_block(block_size, 1, 1);

  // common::cuda::start_timer(&start, &stop);

  gsparc::contraction_dense_kernel<index_t, lindex_t, value_t>
      <<<blocks_per_grid, threads_per_block>>>(
          X_indices, X_values, Y_indices, Y_values, X_nnz, Y_nnz, mPos, mCnt,
          d_dense_values, ir_size, dense_size, dense_start, X_nfbits, Y_nfbits,
          ncbits);

  // // cub sort
  // common::cuda::end_timer_with_msg(&timer, &stop,
  // "contraction_dense_kernel");

  common::cuda::device_sync();
  printf("finished kernel\n");

  // GPU-side stream compaction: extract non-zero entries on GPU
  // avoiding large D2H transfer of entire dense array
  lindex_t *d_Z_indices = memoryPool->allocate<lindex_t>(dense_size);
  value_t *d_Z_values = memoryPool->allocate<value_t>(dense_size);
  index_t *d_num_selected = memoryPool->allocate<index_t>(1);

  // Generate index sequence + select non-zero entries using a custom kernel
  {
    index_t compact_block = 1024;
    index_t compact_grid = (dense_size + compact_block - 1) / compact_block;

    // Step 1: Create flags array for non-zero entries
    unsigned char *d_flags = memoryPool->allocate<unsigned char>(dense_size);

    // Generate flags
    gsparc::dense_flag_kernel<<<compact_grid, compact_block>>>(
        d_dense_values, d_flags, dense_size);
    common::cuda::device_sync();

    // Use CUB DeviceSelect::Flagged for values
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    thrust::counting_iterator<lindex_t> iota(0);

    // Query temp storage for index selection
    cub::DeviceSelect::Flagged(d_temp, temp_bytes, iota, d_flags,
                               d_Z_indices, d_num_selected, dense_size);
    d_temp = memoryPool->allocate<void>(temp_bytes);

    // Select indices
    cub::DeviceSelect::Flagged(d_temp, temp_bytes, iota, d_flags,
                               d_Z_indices, d_num_selected, dense_size);
    common::cuda::device_sync();

    // Select values
    size_t temp_bytes2 = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_bytes2, d_dense_values, d_flags,
                               d_Z_values, d_num_selected, dense_size);
    if (temp_bytes2 > temp_bytes) {
      memoryPool->deallocate<void>(d_temp, temp_bytes);
      d_temp = memoryPool->allocate<void>(temp_bytes2);
      temp_bytes = temp_bytes2;
    }
    cub::DeviceSelect::Flagged(d_temp, temp_bytes, d_dense_values, d_flags,
                               d_Z_values, d_num_selected, dense_size);
    common::cuda::device_sync();

    memoryPool->deallocate<void>(d_temp, temp_bytes);
    memoryPool->deallocate<unsigned char>(d_flags, dense_size);
  }

  // Copy only the count and compacted results (much smaller than dense_size)
  uint64_t result_nnz = 0;
  index_t h_num_selected = 0;
  common::cuda::d2hcpy(&h_num_selected, d_num_selected, sizeof(index_t));
  result_nnz = h_num_selected;
  printf("result_nnz: %lu\n", result_nnz);

  timer->stop();
  timer->printElapsed("contraction_dense_kernel");

  Z_indices = static_cast<lindex_t *>(
      common::cuda::pinned_malloc(sizeof(lindex_t) * result_nnz));
  Z_values = static_cast<value_t *>(
      common::cuda::pinned_malloc(sizeof(value_t) * result_nnz));

  timer->start();
  // Transfer only compacted results D2H
  if (result_nnz > 0) {
    common::cuda::d2hcpy(Z_indices, d_Z_indices, sizeof(lindex_t) * result_nnz);
    common::cuda::d2hcpy(Z_values, d_Z_values, sizeof(value_t) * result_nnz);
  }
  timer->stop();
  timer->printElapsed("copy result to Z");

  common::cuda::pinned_free(result_values);
  if (multi == false) {
    common::cuda::pinned_free(Z_indices);
    common::cuda::pinned_free(Z_values);
  }
  memoryPool->deallocate<lindex_t>(d_Z_indices, dense_size);
  memoryPool->deallocate<value_t>(d_Z_values, dense_size);
  memoryPool->deallocate<index_t>(d_num_selected, 1);
  memoryPool->deallocate<value_t>(d_dense_values, dense_size);
  *Z_nnz = result_nnz;

  return;
}

template <typename IndexType, typename LIndexType, typename ValueType>
void ESC_Contraction(LIndexType *X_indices, ValueType *X_values, uint64_t X_nnz,
                     LIndexType *Y_indices, ValueType *Y_values, uint64_t Y_nnz,
                     LIndexType *&Z_indices, ValueType *&Z_values,
                     uint64_t *Z_nnz, std::vector<IndexType> prtn_offset,
                     IndexType *mPos, IndexType *mCnt, IndexType *mCntPrefix,
                     IndexType *h_mCntPrefix, uint64_t ir_nnz,
                     uint64_t max_prtn_ir_nnz, int num_prtn, int X_nfbits,
                     int Y_nfbits, int ncbits, CudaMemoryPool *memoryPool,
                     Timer *timer) {

  using index_t = IndexType;
  using lindex_t = LIndexType;
  using value_t = ValueType;

  int num_streams = 5;
  cudaStream_t streams[num_streams];

  for (int i = 0; i < num_streams; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }
  // printf("num_prtn: %d, max_prtn_ir_nnz: %lu\n", num_prtn, max_prtn_ir_nnz);

  // lindex_t **sub_Z_indices = static_cast<lindex_t
  // **>(common::cuda::pinned_malloc(sizeof(lindex_t *) * num_prtn)); value_t
  // **sub_Z_values = static_cast<value_t
  // **>(common::cuda::pinned_malloc(sizeof(value_t *) * num_prtn));
  uint64_t *sub_Z_nnz = static_cast<uint64_t *>(
      common::cuda::pinned_malloc(sizeof(uint64_t) * num_prtn));
  memset(sub_Z_nnz, 0, sizeof(uint64_t) * num_prtn);

  lindex_t **d_Z_indices = static_cast<lindex_t **>(
      common::cuda::pinned_malloc(sizeof(lindex_t *) * 2));
  value_t **d_Z_values = static_cast<value_t **>(
      common::cuda::pinned_malloc(sizeof(value_t *) * 2));
  printf("max_prtn_ir_nnz: %lu\n", max_prtn_ir_nnz);
  for (int i = 0; i < 2; ++i) {
    d_Z_indices[i] = memoryPool->allocate<lindex_t>(max_prtn_ir_nnz);
    d_Z_values[i] = memoryPool->allocate<value_t>(max_prtn_ir_nnz);
  }

  // for (int i = 0; i < num_prtn; ++i)
  // {
  //     uint64_t temp_ir_nnz;
  //     index_t sub_X_start, sub_X_end;

  //     if (num_prtn == 1)
  //     {
  //         temp_ir_nnz = ir_nnz;
  //     }
  //     else
  //     {
  //         sub_X_start = (i == 0) ? 0 : prtn_offset[i - 1] + 1;
  //         sub_X_end = prtn_offset[i] + 1;
  //         temp_ir_nnz = h_mCntPrefix[sub_X_end] - h_mCntPrefix[sub_X_start];
  //     }
  //     sub_Z_indices[i] = static_cast<lindex_t
  //     *>(common::cuda::pinned_malloc(sizeof(lindex_t) * temp_ir_nnz));
  //     sub_Z_values[i] = static_cast<value_t
  //     *>(common::cuda::pinned_malloc(sizeof(value_t) * temp_ir_nnz));
  // }
  lindex_t *sub_Z_indices = static_cast<lindex_t *>(
      common::cuda::pinned_malloc(sizeof(lindex_t) * max_prtn_ir_nnz));
  value_t *sub_Z_values = static_cast<value_t *>(
      common::cuda::pinned_malloc(sizeof(value_t) * max_prtn_ir_nnz));

  // memoryPool->printFree();

  // Pre-query temp_storage size using max_prtn_ir_nnz to avoid per-partition allocation
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  {
    size_t temp_storage_bytes_sort = 0, temp_storage_bytes_reduce = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes_sort, d_Z_indices[0], d_Z_indices[1],
        d_Z_values[0], d_Z_values[1], max_prtn_ir_nnz, 0, X_nfbits + Y_nfbits);
    cub::DeviceReduce::ReduceByKey(nullptr, temp_storage_bytes_reduce,
                                   d_Z_indices[1], d_Z_indices[0],
                                   d_Z_values[1], d_Z_values[0], (index_t *)nullptr,
                                   ::cuda::std::plus<>(), max_prtn_ir_nnz);
    temp_storage_bytes = std::max(temp_storage_bytes_sort, temp_storage_bytes_reduce);
  }
  d_temp_storage = memoryPool->allocate<void>(temp_storage_bytes);
  index_t *d_num_segments = memoryPool->allocate<index_t>(1);

  timer->start();
  for (int i = 0; i < num_prtn; ++i) {
    printf("esc: %d/%d\n", i, num_prtn);
    uint64_t temp_result_nnz = 0;
    index_t sub_X_start, sub_X_end;
    uint64_t prtn_nnz, prtn_ir_nnz;
    if (num_prtn == 1) {
      sub_X_start = 0;
      sub_X_end = X_nnz;
      prtn_nnz = X_nnz;
      prtn_ir_nnz = ir_nnz;
    } else {
      sub_X_start = (i == 0) ? 0 : prtn_offset[i - 1] + 1;
      sub_X_end = prtn_offset[i] + 1;
      prtn_nnz = sub_X_end - sub_X_start;
      prtn_ir_nnz = h_mCntPrefix[sub_X_end] - h_mCntPrefix[sub_X_start];
    }
    printf(
        "sub_X_start: %lu, sub_X_end: %lu, prtn_nnz: %lu, prtn_ir_nnz: %lu\n",
        sub_X_start, sub_X_end, prtn_nnz, prtn_ir_nnz);
    if (prtn_ir_nnz == 0)
      continue;
    cudaStream_t stream = streams[i % num_streams];
    index_t block_size = 1024;
    index_t grid_size = (prtn_ir_nnz + block_size - 1) / block_size;
    dim3 blocks_per_grid(grid_size, 1, 1);
    dim3 threads_per_block(block_size, 1, 1);

    gsparc::contraction_esc_kernel<index_t, lindex_t, value_t>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            X_indices + sub_X_start, X_values + sub_X_start, Y_indices,
            Y_values, prtn_nnz, Y_nnz, prtn_ir_nnz, mPos + sub_X_start,
            mCnt + sub_X_start, mCntPrefix + sub_X_start,
            h_mCntPrefix[sub_X_start], d_Z_indices[0], d_Z_values[0], X_nfbits,
            Y_nfbits, ncbits);

    // Sort: [0] -> [1], ReduceByKey: [1] -> [0] (reuse pre-allocated temp_storage)
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_Z_indices[0], d_Z_indices[1],
        d_Z_values[0], d_Z_values[1], prtn_ir_nnz, 0, X_nfbits + Y_nfbits, stream);

    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                   d_Z_indices[1], d_Z_indices[0],
                                   d_Z_values[1], d_Z_values[0], d_num_segments,
                                   ::cuda::std::plus<>(), prtn_ir_nnz, stream);

    common::cuda::d2hcpy_async(&temp_result_nnz, d_num_segments,
                               sizeof(index_t), stream);
    common::cuda::stream_sync(stream);

    // Result is in d_Z_indices[0] / d_Z_values[0] after ReduceByKey
    common::cuda::d2hcpy_async(sub_Z_indices, d_Z_indices[0],
                               sizeof(lindex_t) * temp_result_nnz, stream);
    common::cuda::d2hcpy_async(sub_Z_values, d_Z_values[0],
                               sizeof(value_t) * temp_result_nnz, stream);
    common::cuda::stream_sync(stream);
    sub_Z_nnz[i] = temp_result_nnz;
    *Z_nnz += temp_result_nnz;
  }

  timer->stop();
  timer->printElapsed("esc contraction kernel");
  common::cuda::device_sync();
  memoryPool->deallocate<void>(d_temp_storage, temp_storage_bytes);
  memoryPool->deallocate<index_t>(d_num_segments, 1);
  for (int i = 0; i < 2; ++i) {
    memoryPool->deallocate<lindex_t>(d_Z_indices[i], max_prtn_ir_nnz);
    memoryPool->deallocate<value_t>(d_Z_values[i], max_prtn_ir_nnz);
  }
  common::cuda::pinned_free(d_Z_indices);
  common::cuda::pinned_free(d_Z_values);

  common::cuda::destory_streams(streams, num_streams);

  common::cuda::pinned_free(sub_Z_indices);
  common::cuda::pinned_free(sub_Z_values);
  return;
}

template <typename IndexType, typename LIndexType, typename ValueType>
void ESC_Contraction_multi(LIndexType *X_indices, ValueType *X_values,
                           uint64_t X_nnz, LIndexType *Y_indices,
                           ValueType *Y_values, uint64_t Y_nnz,
                           LIndexType *&Z_indices, ValueType *&Z_values,
                           uint64_t *Z_nnz, std::vector<IndexType> prtn_offset,
                           IndexType *mPos, IndexType *mCnt,
                           IndexType *mCntPrefix, IndexType *h_mCntPrefix,
                           uint64_t ir_nnz, uint64_t max_prtn_ir_nnz,
                           int num_prtn, int X_nfbits, int Y_nfbits, int ncbits,
                           CudaMemoryPool *memoryPool, Timer *timer) {

  using index_t = IndexType;
  using lindex_t = LIndexType;
  using value_t = ValueType;

  int num_streams = 5;
  cudaStream_t streams[num_streams];

  for (int i = 0; i < num_streams; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }
  // printf("num_prtn: %d, max_prtn_ir_nnz: %lu\n", num_prtn, max_prtn_ir_nnz);

  lindex_t **sub_Z_indices = static_cast<lindex_t **>(
      common::cuda::pinned_malloc(sizeof(lindex_t *) * num_prtn));
  value_t **sub_Z_values = static_cast<value_t **>(
      common::cuda::pinned_malloc(sizeof(value_t *) * num_prtn));
  uint64_t *sub_Z_nnz = static_cast<uint64_t *>(
      common::cuda::pinned_malloc(sizeof(uint64_t) * num_prtn));
  memset(sub_Z_nnz, 0, sizeof(uint64_t) * num_prtn);

  lindex_t **d_Z_indices = static_cast<lindex_t **>(
      common::cuda::pinned_malloc(sizeof(lindex_t *) * 2));
  value_t **d_Z_values = static_cast<value_t **>(
      common::cuda::pinned_malloc(sizeof(value_t *) * 2));
  printf("max_prtn_ir_nnz: %lu\n", max_prtn_ir_nnz);
  for (int i = 0; i < 2; ++i) {
    d_Z_indices[i] = memoryPool->allocate<lindex_t>(max_prtn_ir_nnz);
    d_Z_values[i] = memoryPool->allocate<value_t>(max_prtn_ir_nnz);
  }

  for (int i = 0; i < num_prtn; ++i) {
    uint64_t temp_ir_nnz;
    index_t sub_X_start, sub_X_end;

    if (num_prtn == 1) {
      temp_ir_nnz = ir_nnz;
    } else {
      sub_X_start = (i == 0) ? 0 : prtn_offset[i - 1] + 1;
      sub_X_end = prtn_offset[i] + 1;
      temp_ir_nnz = h_mCntPrefix[sub_X_end] - h_mCntPrefix[sub_X_start];
    }
    sub_Z_indices[i] = static_cast<lindex_t *>(
        common::cuda::pinned_malloc(sizeof(lindex_t) * temp_ir_nnz));
    sub_Z_values[i] = static_cast<value_t *>(
        common::cuda::pinned_malloc(sizeof(value_t) * temp_ir_nnz));
  }
  // lindex_t *sub_Z_indices = static_cast<lindex_t
  // *>(common::cuda::pinned_malloc(sizeof(lindex_t) * max_prtn_ir_nnz));
  // value_t *sub_Z_values = static_cast<value_t
  // *>(common::cuda::pinned_malloc(sizeof(value_t) * max_prtn_ir_nnz));

  // memoryPool->printFree();

  // Pre-query temp_storage size using max_prtn_ir_nnz
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  {
    size_t temp_storage_bytes_sort = 0, temp_storage_bytes_reduce = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes_sort, d_Z_indices[0], d_Z_indices[1],
        d_Z_values[0], d_Z_values[1], max_prtn_ir_nnz, 0, X_nfbits + Y_nfbits);
    cub::DeviceReduce::ReduceByKey(nullptr, temp_storage_bytes_reduce,
                                   d_Z_indices[1], d_Z_indices[0],
                                   d_Z_values[1], d_Z_values[0], (index_t *)nullptr,
                                   ::cuda::std::plus<>(), max_prtn_ir_nnz);
    temp_storage_bytes = std::max(temp_storage_bytes_sort, temp_storage_bytes_reduce);
  }
  d_temp_storage = memoryPool->allocate<void>(temp_storage_bytes);
  index_t *d_num_segments = memoryPool->allocate<index_t>(1);

  timer->start();
  for (int i = 0; i < num_prtn; ++i) {
    printf("esc: %d/%d\n", i, num_prtn);
    uint64_t temp_result_nnz = 0;
    index_t sub_X_start, sub_X_end;
    uint64_t prtn_nnz, prtn_ir_nnz;
    if (num_prtn == 1) {
      sub_X_start = 0;
      sub_X_end = X_nnz;
      prtn_nnz = X_nnz;
      prtn_ir_nnz = ir_nnz;
    } else {
      sub_X_start = (i == 0) ? 0 : prtn_offset[i - 1] + 1;
      sub_X_end = prtn_offset[i] + 1;
      prtn_nnz = sub_X_end - sub_X_start;
      prtn_ir_nnz = h_mCntPrefix[sub_X_end] - h_mCntPrefix[sub_X_start];
    }
    printf(
        "sub_X_start: %lu, sub_X_end: %lu, prtn_nnz: %lu, prtn_ir_nnz: %lu\n",
        sub_X_start, sub_X_end, prtn_nnz, prtn_ir_nnz);
    if (prtn_ir_nnz == 0)
      continue;
    cudaStream_t stream = streams[i % num_streams];
    index_t block_size = 1024;
    index_t grid_size = (prtn_ir_nnz + block_size - 1) / block_size;
    dim3 blocks_per_grid(grid_size, 1, 1);
    dim3 threads_per_block(block_size, 1, 1);

    gsparc::contraction_esc_kernel<index_t, lindex_t, value_t>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            X_indices + sub_X_start, X_values + sub_X_start, Y_indices,
            Y_values, prtn_nnz, Y_nnz, prtn_ir_nnz, mPos + sub_X_start,
            mCnt + sub_X_start, mCntPrefix + sub_X_start,
            h_mCntPrefix[sub_X_start], d_Z_indices[0], d_Z_values[0], X_nfbits,
            Y_nfbits, ncbits);

    // Sort: [0] -> [1], ReduceByKey: [1] -> [0] (reuse pre-allocated temp_storage)
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_Z_indices[0], d_Z_indices[1],
        d_Z_values[0], d_Z_values[1], prtn_ir_nnz, 0, X_nfbits + Y_nfbits, stream);

    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                   d_Z_indices[1], d_Z_indices[0],
                                   d_Z_values[1], d_Z_values[0], d_num_segments,
                                   ::cuda::std::plus<>(), prtn_ir_nnz, stream);

    common::cuda::d2hcpy_async(&temp_result_nnz, d_num_segments,
                               sizeof(index_t), stream);
    common::cuda::stream_sync(stream);

    common::cuda::d2hcpy_async(sub_Z_indices[i], d_Z_indices[0],
                               sizeof(lindex_t) * temp_result_nnz, stream);
    common::cuda::d2hcpy_async(sub_Z_values[i], d_Z_values[0],
                               sizeof(value_t) * temp_result_nnz, stream);
    common::cuda::stream_sync(stream);
    sub_Z_nnz[i] = temp_result_nnz;
    *Z_nnz += temp_result_nnz;
  }

  timer->stop();
  timer->printElapsed("esc contraction kernel");
  common::cuda::device_sync();
  memoryPool->deallocate<void>(d_temp_storage, temp_storage_bytes);
  memoryPool->deallocate<index_t>(d_num_segments, 1);
  for (int i = 0; i < 2; ++i) {
    memoryPool->deallocate<lindex_t>(d_Z_indices[i], max_prtn_ir_nnz);
    memoryPool->deallocate<value_t>(d_Z_values[i], max_prtn_ir_nnz);
  }

  Z_indices = static_cast<lindex_t *>(
      common::cuda::pinned_malloc(sizeof(lindex_t) * *Z_nnz));
  Z_values = static_cast<value_t *>(
      common::cuda::pinned_malloc(sizeof(value_t) * *Z_nnz));

  timer->start();
  index_t offset = 0;
  for (int i = 0; i < num_prtn; i++) {
    common::cuda::h2dcpy_async(Z_indices + offset, sub_Z_indices[i],
                               sizeof(lindex_t) * sub_Z_nnz[i],
                               streams[i % num_streams]);
    common::cuda::h2dcpy_async(Z_values + offset, sub_Z_values[i],
                               sizeof(value_t) * sub_Z_nnz[i],
                               streams[i % num_streams]);

    offset += sub_Z_nnz[i];
  }

  timer->stop();
  common::cuda::pinned_free(d_Z_indices);
  common::cuda::pinned_free(d_Z_values);

  common::cuda::destory_streams(streams, num_streams);

  for (int i = 0; i < num_prtn; ++i) {
    common::cuda::pinned_free(sub_Z_indices[i]);
    common::cuda::pinned_free(sub_Z_values[i]);
  }
  common::cuda::pinned_free(sub_Z_indices);
  common::cuda::pinned_free(sub_Z_values);
  return;
}

} // namespace gsparc
#endif