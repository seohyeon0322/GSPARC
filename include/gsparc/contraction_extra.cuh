#ifndef CONTRACTION_EXTRA_CUH
#define CONTRACTION_EXTRA_CUH

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#include "common/cuda_helper.hpp"
#include "gsparc/helper.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/timer.hpp"

namespace gsparc
{
    namespace extra
    {

        template <typename IndexType>
        __device__ void find_idx(IndexType idx,
                                 IndexType *xidx,
                                 IndexType *offset,
                                 uint64_t X_nnz,
                                 IndexType *mCntPrefix)
        {
            using index_t = IndexType;

            index_t left = 0;
            index_t right = X_nnz;

            while (left <= right)
            {
                index_t mid = left + (right - left) / 2;
                // Check if idx is within the range [mCntPrefix[mid], mCntPrefix[mid + 1])
                if (mCntPrefix[mid] <= idx && idx < mCntPrefix[mid + 1])
                {
                    *xidx = mid;
                    *offset = idx - mCntPrefix[mid];
                    return;
                }
                else if (idx < mCntPrefix[mid])
                {
                    right = mid - 1;
                }
                else
                {
                    left = mid + 1;
                }
            }

            printf("Error: idx is out of range\n");

            // 마지막 인덱스에 대한 처리
            // *xidx = X_nnz - 1;
            // *offset = idx - mCntPrefix[X_nnz - 1];
        }

        template <typename IndexType, typename ULIndexType, typename LIndexType, typename ValueType>
        __global__ void contraction_dense_kernel(ULIndexType *X_uindices,
                                                 LIndexType *X_indices,
                                                 ValueType *X_values,
                                                 ULIndexType *Y_uindices,
                                                 LIndexType *Y_indices,
                                                 ValueType *Y_values,
                                                 uint64_t X_nnz,
                                                 uint64_t Y_nnz,
                                                 IndexType *mPos,
                                                 IndexType *mCnt,
                                                 ValueType *result_values,
                                                 uint64_t result_size,
                                                 uint64_t dense_size,
                                                 IndexType dense_start,
                                                 int X_nfbits,
                                                 int Y_nfbits,
                                                 int Y_cbits_extra,
                                                 int cbits)
        {
            using index_t = IndexType;
            using ulindex_t = ULIndexType;
            using lindex_t = LIndexType;
            using value_t = ValueType;

            index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            index_t stride = blockDim.x * gridDim.x;

            int shift = cbits > 64 ? 0 : 64 - cbits;
            int extra_shift = cbits > 64 ? cbits - 64 : 0;
            for (index_t xidx = tid; xidx < X_nnz; xidx += stride)
            {
                index_t yidx = mPos[xidx];
                index_t cnt = mCnt[xidx];
                for (int offset = 0; offset < cnt; ++offset)
                {

                    __uint128_t x_idx = (static_cast<__uint128_t>(X_uindices[xidx]) << 64) | X_indices[xidx];
                    __uint128_t y_idx = (static_cast<__uint128_t>(Y_uindices[yidx + offset]) << 64) | Y_indices[yidx + offset];

                    lindex_t x_fidx = x_idx >> cbits;
                    lindex_t y_fidx = y_idx & ((static_cast<lindex_t>(1) << Y_nfbits) - 1);
                    lindex_t result_idx = (x_fidx << Y_nfbits) | y_fidx;

                    value_t result_val = X_values[tid] * Y_values[yidx + offset];

                    atomicAdd(&result_values[result_idx - dense_start], result_val);
                }
            }
            __syncthreads();
        }

        template <typename IndexType, typename ULIndexType, typename LIndexType, typename ValueType>
        __global__ void contraction_esc_kernel(ULIndexType *X_uindices,
                                               LIndexType *X_indices,
                                               ValueType *X_values,
                                               ULIndexType *Y_uindices,
                                               LIndexType *Y_indices,
                                               ValueType *Y_values,
                                               uint64_t X_nnz,
                                               uint64_t Y_nnz,
                                               uint64_t ir_nnz,
                                               IndexType *mPos,
                                               IndexType *mCnt,
                                               IndexType *mCntPrefix,
                                               IndexType start_offset,
                                               LIndexType *result_indices,
                                               ValueType *result_values,
                                               int X_fbits,
                                               int Y_fbits,
                                               int cbits)
        {
            using index_t = IndexType;
            using ulindex_t = ULIndexType;
            using lindex_t = LIndexType;
            using value_t = ValueType;

            index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            index_t stride = blockDim.x * gridDim.x;

            for (index_t idx = tid; idx < ir_nnz; idx += stride)
            {
                index_t xidx, offset;
                find_idx<index_t>(idx + start_offset, &xidx, &offset, X_nnz, mCntPrefix);

                index_t yidx = mPos[xidx];

                __uint128_t x_idx = (static_cast<__uint128_t>(X_uindices[xidx]) << 64) | X_indices[xidx];
                __uint128_t y_idx = (static_cast<__uint128_t>(Y_uindices[yidx + offset]) << 64) | Y_indices[yidx + offset];

                lindex_t x_fidx = x_idx >> cbits;
                lindex_t y_fidx = y_idx & ((static_cast<lindex_t>(1) << Y_fbits) - 1);
                lindex_t result_idx = (x_fidx << Y_fbits) | y_fidx;
                value_t result_val = X_values[xidx] * Y_values[yidx + offset];

                result_indices[idx] = result_idx;
                result_values[idx] = result_val;
            }
            __syncthreads();
        }

        template <typename IndexType, typename ULIndexType, typename LIndexType, typename ValueType>
        void Dynamic_partition(ULIndexType *X_uindices,
                               LIndexType *X_indices,
                               uint64_t X_nnz,
                               uint64_t ir_nnz,
                               IndexType *mCntPrefix,
                               int ncbits,
                               CudaMemoryPool *memoryPool,
                               std::vector<IndexType> &prtn_offset,
                               uint64_t *max_prtn_ir_nnz)
        {
            using ulindex_t = ULIndexType;
            using index_t = IndexType;
            using lindex_t = LIndexType;
            using value_t = ValueType;

            index_t start_pointer = 0;
            printf("X_nnz: %llu\n", X_nnz);
            while (start_pointer < X_nnz)
            {
                // 초기 후보 크기 (남은 nnz의 절반 정도에서 시작)
                index_t prtn_nnz = (X_nnz - start_pointer) / 2;
                // 파티션 경계를 저장할 변수
                index_t valid_end = start_pointer;

                while (true)
                {
                    if (X_indices[start_pointer] >> ncbits == X_indices[X_nnz - 1] >> ncbits)
                    {
                        valid_end = X_nnz - 1;

                        if (ir_nnz - mCntPrefix[start_pointer] > *max_prtn_ir_nnz)
                        {
                            *max_prtn_ir_nnz = ir_nnz - mCntPrefix[start_pointer];
                        }
                        break;
                    }
                    if (prtn_nnz < 1)
                    {
                        // prtn_nnz가 1 미만이면 최소 단위조차 만족하지 못하는 것임
                        fprintf(stderr, "Error: Candidate partition size reduced below 1 at start_pointer: %lu, prtn_nnz: %lu\n", start_pointer, prtn_nnz);
                        exit(1);
                    }
                    printf("Try: partition: [%llu, %llu] -- partition size: %llu\n",
                           start_pointer, start_pointer + prtn_nnz, prtn_nnz);

                    // 후보 파티션의 끝 인덱스 계산 (후보: start_pointer + prtn_nnz)
                    index_t candidate = start_pointer + prtn_nnz;
                    if (candidate > X_nnz - 1)
                        candidate = X_nnz - 1;

                    // 만약 인덱스 그룹 (X_indices[i] >> ncbits 값)이 연속된다면 후보를 해당 그룹의 마지막까지 확장
                    auto get_fmode = [&](size_t idx) -> __uint128_t
                    {
                        return ((static_cast<__uint128_t>(X_uindices[idx]) << 64) | X_indices[idx]) >> ncbits;
                    };

                    while (candidate < X_nnz - 1 && get_fmode(candidate) == get_fmode(candidate + 1))
                    {
                        candidate++;
                    }

                    printf("After try: Candidate partition: [%llu, %llu] -- partition size: %llu\n",
                           start_pointer, candidate, candidate - start_pointer);

                    // 현재 파티션의 크기를 mCntPrefix 배열을 통해 계산
                    index_t curr_partition_size = (candidate == X_nnz - 1)
                                                      ? (ir_nnz - mCntPrefix[start_pointer])
                                                      : (mCntPrefix[candidate + 1] - mCntPrefix[start_pointer]);

                    // 예상 메모리 사용량: 인덱스와 값 배열에 필요한 메모리 계산 후, double buffering 고려해 2배로 확장
                    size_t estimated_size = curr_partition_size * (sizeof(lindex_t) + sizeof(value_t)) * 2 + sizeof(index_t);
                    estimated_size *= 2;

                    printf("Candidate partition: [%lu, %lu] -- curr size: %lu, estimated size: %s\n",
                           start_pointer, candidate, curr_partition_size, common::byteToString(estimated_size));

                    // 할당 가능하면 바로 이 후보 범위를 사용 (더 이상 큰 범위를 탐색하지 않음)
                    if (memoryPool->is_available(estimated_size))
                    {
                        printf("Allocation successful for candidate [%lu, %lu] - size: %s\n",
                               start_pointer, candidate, common::byteToString(estimated_size));
                        valid_end = candidate;
                        // 갱신된 최대 파티션 크기가 있으면 업데이트
                        if (curr_partition_size > *max_prtn_ir_nnz)
                        {
                            *max_prtn_ir_nnz = curr_partition_size;
                        }
                        break;
                    }
                    else
                    {
                        // 할당 불가능하면, 후보 크기를 줄여 재시도 (과도한 감소 방지를 위해 최소 prtn_nnz/2 만큼 감소)
                        printf("Allocation failed for candidate [%lu, %lu]. Reducing candidate size.\n", start_pointer, candidate);
                        prtn_nnz -= std::max<index_t>(1, prtn_nnz / 2);
                    }
                } // inner while

                // 결정된 파티션 경계를 기록
                prtn_offset.push_back(valid_end);
                // 다음 파티션의 시작은 현재 파티션의 마지막 인덱스의 다음 인덱스
                start_pointer = valid_end + 1;
            }

            // 마지막 파티션이 X_nnz - 1까지 포함되지 못했다면 추가로 기록
            if (prtn_offset.empty() || prtn_offset.back() != X_nnz - 1)
            {
                prtn_offset.push_back(X_nnz - 1);
            }

            // index_t start_pointer = 0;
            // index_t prtn_nnz = X_nnz / 2;
            // index_t idx = 0;

            // while (start_pointer < X_nnz)
            // {
            //     index_t valid_end = start_pointer; // 해당 파티션의 유효한 끝 인덱스

            //     while (true)
            //     {

            //         if (prtn_nnz < 1)
            //         {
            //             fprintf(stderr, "Error: Candidate partition size reduced below 1. Exiting.\n");
            //             exit(1);
            //         }

            //         index_t candidate = start_pointer + prtn_nnz;
            //         if (candidate > X_nnz - 1)
            //             candidate = X_nnz - 1;
            //         index_t mid = candidate;

            //         index_t last = mid;
            //         index_t same_fmode = 0;
            //         // combine과 fmode를 계산하는 람다
            //         auto get_fmode = [&](size_t idx) -> __uint128_t
            //         {
            //             return ((static_cast<__uint128_t>(X_uindices[idx]) << 64) | X_indices[idx]) >> ncbits;
            //         };

            //         while (last < X_nnz - 1 && get_fmode(last) == get_fmode(last + 1))
            //         {
            //             last++;
            //         }

            //         while (mid > start_pointer && get_fmode(mid) == get_fmode(mid - 1))
            //         {
            //             mid--;
            //             same_fmode++;
            //         }

            //         index_t curr_partition_size = 0;
            //         if (last == X_nnz - 1)
            //         {
            //             curr_partition_size = ir_nnz - mCntPrefix[start_pointer];
            //         }
            //         else
            //         {
            //             curr_partition_size = mCntPrefix[last + 1] - mCntPrefix[start_pointer];
            //         }

            //         size_t estimated_size = curr_partition_size * (sizeof(ulindex_t) + sizeof(lindex_t) + sizeof(value_t)) * 2 + sizeof(index_t);
            //         estimated_size *= 2; // 더블 버퍼링 고려

            //         if (!memoryPool->is_available(estimated_size))
            //         {
            //             if (last == idx)
            //             {
            //                 valid_end = last;
            //                 printf("2curr size: %lu, max size: %lu\n", curr_partition_size, *max_prtn_ir_nnz);

            //                 if (curr_partition_size > *max_prtn_ir_nnz)
            //                 {
            //                     *max_prtn_ir_nnz = curr_partition_size;
            //                 }
            //                 break; // 현재 partition 결정 완료
            //             }
            //             else
            //             {
            //                 index_t reduce = std::max(prtn_nnz / 10, same_fmode);
            //                 if (reduce < 1)
            //                     reduce = 1;
            //                 prtn_nnz -= reduce;
            //                 continue;
            //             }
            //         }
            //         else
            //         {
            //             idx = last;
            //             valid_end = last;

            //             if (curr_partition_size > *max_prtn_ir_nnz)
            //             {
            //                 *max_prtn_ir_nnz = curr_partition_size;
            //             }
            //             break;

            //             printf("estimate sort\n");
            //             index_t *d_result_idx = memoryPool->allocate<lindex_t>(curr_partition_size);
            //             value_t *d_result_vals = memoryPool->allocate<value_t>(curr_partition_size);
            //             index_t *d_num_segments = memoryPool->allocate<index_t>(1);
            //             void *d_temp_storage = nullptr;
            //             size_t temp_storage_bytes_sort = 0;
            //             size_t temp_storage_bytes_reduce = 0;
            //             cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes_sort,
            //                                             d_result_idx, d_result_idx,
            //                                             d_result_vals, d_result_vals,
            //                                             curr_partition_size);
            //             cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes_reduce,
            //                                            d_result_idx, d_result_idx,
            //                                            d_result_vals, d_result_vals,
            //                                            d_num_segments, cub::Sum(),
            //                                            curr_partition_size);
            //             estimated_size += std::max(temp_storage_bytes_sort, temp_storage_bytes_reduce);
            //             memoryPool->deallocate<lindex_t>(d_result_idx, curr_partition_size);
            //             memoryPool->deallocate<value_t>(d_result_vals, curr_partition_size);
            //             memoryPool->deallocate<index_t>(d_num_segments, 1);

            //             if (!memoryPool->is_available(estimated_size))
            //             {
            //                 index_t reduce = std::max(prtn_nnz / 5, same_fmode);
            //                 if (reduce < 1)
            //                     reduce = 1;
            //                 prtn_nnz -= reduce;
            //                 continue;
            //             }
            //             else
            //             {
            //                 valid_end = last;

            //                 if (curr_partition_size > *max_prtn_ir_nnz)
            //                 {
            //                     *max_prtn_ir_nnz = curr_partition_size;
            //                 }
            //                 break;
            //             }
            //         }
            //     } // inner loop 끝

            //     prtn_offset.push_back(valid_end);
            //     start_pointer = valid_end + 1;

            //     prtn_nnz = (X_nnz - start_pointer) / 2;
            // }
        }

        template <typename IndexType, typename ULIndexType, typename LIndexType, typename ValueType>
        void Dense_Contraction(ULIndexType *X_uindices,
                               LIndexType *X_indices,
                               ValueType *X_values,
                               IndexType X_nnz,
                               ULIndexType *Y_uindices,
                               LIndexType *Y_indices,
                               ValueType *Y_values,
                               IndexType Y_nnz,
                               uint64_t ir_size,
                               IndexType *mPos,
                               IndexType *mCnt,
                               IndexType dense_start,
                               uint64_t dense_size,
                               LIndexType *&Z_indices,
                               ValueType *&Z_values,
                               IndexType *Z_nnz,
                               int X_nfbits,
                               int Y_nfbits,
                               int ncbits,
                               CudaMemoryPool *memoryPool,
                               Timer *timer)
        {
            using index_t = IndexType;
            using ulindex_t = ULIndexType;
            using lindex_t = LIndexType;
            using value_t = ValueType;

            int num_streams = 3;

            value_t *result_values = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * dense_size));

            // checkCudaErrors(cudaEventCreate(&t_start));
            // checkCudaErrors(cudaEventCreate(&t_stop));
            // checkCudaErrors(cudaEventRecord(t_start, 0));
            int Y_cbits_extra = ncbits > 64 ? ncbits - 64 : 0;

            value_t *d_dense_values = memoryPool->allocate<value_t>(dense_size);
            common::cuda::device_memset(d_dense_values, static_cast<value_t>(0), sizeof(value_t) * dense_size);
            // dim3 dimGrid((result_size + block_size - 1) / block_size, 1, 1);
            index_t block_size = 1024;
            index_t grid_size = (X_nnz + block_size - 1) / block_size;
            dim3 blocks_per_grid(grid_size, 1, 1);
            dim3 threads_per_block(block_size, 1, 1);

            timer->start();

            gsparc::extra::contraction_dense_kernel<index_t, ulindex_t, lindex_t, value_t><<<blocks_per_grid, threads_per_block>>>(X_uindices,
                                                                                                                                   X_indices,
                                                                                                                                   X_values,
                                                                                                                                   Y_uindices,
                                                                                                                                   Y_indices,
                                                                                                                                   Y_values,
                                                                                                                                   X_nnz,
                                                                                                                                   Y_nnz,
                                                                                                                                   mPos,
                                                                                                                                   mCnt,
                                                                                                                                   d_dense_values,
                                                                                                                                   ir_size,
                                                                                                                                   dense_size,
                                                                                                                                   dense_start,
                                                                                                                                   X_nfbits,
                                                                                                                                   Y_nfbits,
                                                                                                                                   Y_cbits_extra,
                                                                                                                                   ncbits);

            // // cub sort
            common::cuda::device_sync();

            uint64_t result_nnz = 0;

            common::cuda::d2hcpy(result_values, d_dense_values, sizeof(value_t) * dense_size);

#pragma omp parallel for reduction(+ : result_nnz) schedule(static)
            for (index_t i = 0; i < dense_size; i++)
            {
                if (result_values[i] != static_cast<value_t>(0))
                {
                    result_nnz++;
                }
            }
            printf("result_nnz: %lu\n", result_nnz);
            timer->stop();
            timer->printElapsed("Dense contraction kernel");
            Z_indices = static_cast<lindex_t *>(common::cuda::pinned_malloc(sizeof(lindex_t) * result_nnz));
            Z_values = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * result_nnz));

            timer->start();
            index_t offset = 0;
#pragma omp parallel for reduction(+ : offset) schedule(static)
            for (index_t i = 0; i < dense_size; i++)
            {
                if (result_values[i] != static_cast<value_t>(0))
                {
                    Z_indices[offset] = i;
                    Z_values[offset] = result_values[i];
                    offset++;
                }
            }
            timer->stop();
            timer->printElapsed("Copy dense result");
            // checkCudaErrors(cudaStreamSynchronize(stream));

            common::cuda::pinned_free(result_values);
            common::cuda::pinned_free(Z_indices);
            common::cuda::pinned_free(Z_values);
            memoryPool->deallocate<value_t>(d_dense_values, dense_size);
            *Z_nnz = result_nnz;

            return;
        }

        template <typename IndexType, typename ULIndexType, typename LIndexType, typename ValueType>
        void ESC_Contraction(ULIndexType *X_uindicies,
                             LIndexType *X_indices,
                             ValueType *X_values,
                             uint64_t X_nnz,
                             ULIndexType *Y_uindicies,
                             LIndexType *Y_indices,
                             ValueType *Y_values,
                             uint64_t Y_nnz,
                             LIndexType *&Z_indices,
                             ValueType *&Z_values,
                             uint64_t *Z_nnz,
                             std::vector<IndexType> prtn_offset,
                             IndexType *mPos,
                             IndexType *mCnt,
                             IndexType *mCntPrefix,
                             IndexType *h_mCntPrefix,
                             uint64_t ir_nnz,
                             uint64_t max_prtn_ir_nnz,
                             int num_prtn,
                             int X_nfbits,
                             int Y_nfbits,
                             int ncbits,
                             CudaMemoryPool *memoryPool,
                             Timer *timer)
        {

            using index_t = IndexType;
            using ulindex_t = ULIndexType;
            using lindex_t = LIndexType;
            using value_t = ValueType;

            int num_streams = 5;
            cudaStream_t streams[num_streams];

            for (int i = 0; i < num_streams; i++)
            {
                checkCudaErrors(cudaStreamCreate(&streams[i]));
            }
            printf("in esc contraction\n");
            // memoryPool->printFree();

            // lindex_t **sub_Z_indices = static_cast<lindex_t **>(common::cuda::pinned_malloc(sizeof(lindex_t *) * num_prtn));
            // value_t **sub_Z_values = static_cast<value_t **>(common::cuda::pinned_malloc(sizeof(value_t *) * num_prtn));
            uint64_t *sub_Z_nnz = static_cast<uint64_t *>(common::cuda::pinned_malloc(sizeof(uint64_t) * num_prtn));
            memset(sub_Z_nnz, 0, sizeof(uint64_t) * num_prtn);

            lindex_t **d_Z_indices = static_cast<lindex_t **>(common::cuda::pinned_malloc(sizeof(lindex_t *) * 2));
            value_t **d_Z_values = static_cast<value_t **>(common::cuda::pinned_malloc(sizeof(value_t *) * 2));

            for (int i = 0; i < 2; ++i)
            {
                d_Z_indices[i] = memoryPool->allocate<lindex_t>(max_prtn_ir_nnz);
                d_Z_values[i] = memoryPool->allocate<value_t>(max_prtn_ir_nnz);
            }
            lindex_t *sub_Z_indices = static_cast<lindex_t *>(common::cuda::pinned_malloc(sizeof(lindex_t) * max_prtn_ir_nnz));
            value_t *sub_Z_values = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * max_prtn_ir_nnz));

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
            //     sub_Z_indices[i] = static_cast<lindex_t *>(common::cuda::pinned_malloc(sizeof(lindex_t) * temp_ir_nnz));
            //     sub_Z_values[i] = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * temp_ir_nnz));
            // }
            timer->start();

            for (int i = 0; i < num_prtn; ++i)
            {
                uint64_t temp_result_nnz = 0;
                index_t sub_X_start, sub_X_end;
                uint64_t prtn_nnz, prtn_ir_nnz;

                if (num_prtn == 1)
                {
                    sub_X_start = 0;
                    sub_X_end = X_nnz;
                    prtn_nnz = X_nnz;
                    prtn_ir_nnz = ir_nnz;
                }
                else
                {
                    sub_X_start = (i == 0) ? 0 : prtn_offset[i - 1] + 1;
                    sub_X_end = prtn_offset[i] + 1;
                    prtn_nnz = sub_X_end - sub_X_start;
                    prtn_ir_nnz = h_mCntPrefix[sub_X_end] - h_mCntPrefix[sub_X_start];
                }
                if (prtn_nnz == 0)
                    continue;

                // int upper_cbits_X = ncbits - 64 > 0 ? ncbits - 64 : 0;
                // int upper_cbits_Y = (ncbits + Y_nfbits) - 64;
                // int lower_cbits_X = ncbits - upper_cbits_X;
                // int lower_fbits_Y = Y_nfbits;
                // int upper_fbits_X = (ncbits + Y_nfbits) - 64 - upper_cbits_X > 0 ? (ncbits + Y_nfbits) - 64 - upper_cbits_X : 0;
                // int lower_fbits_X = X_nfbits - upper_fbits_X;

                // printf("X_low_fbits: %d\n", X_low_fbits);
                cudaStream_t stream = streams[i % num_streams];
                index_t block_size = 1024;
                index_t grid_size = (prtn_ir_nnz + block_size - 1) / block_size;
                dim3 blocks_per_grid(grid_size, 1, 1);
                dim3 threads_per_block(block_size, 1, 1);

                gsparc::extra::contraction_esc_kernel<index_t, ulindex_t, lindex_t, value_t><<<blocks_per_grid, threads_per_block, 0, stream>>>(
                    X_uindicies + sub_X_start,
                    X_indices + sub_X_start,
                    X_values + sub_X_start,
                    Y_uindicies,
                    Y_indices,
                    Y_values,
                    prtn_nnz,
                    Y_nnz,
                    prtn_ir_nnz,
                    mPos + sub_X_start,
                    mCnt + sub_X_start,
                    mCntPrefix + sub_X_start,
                    h_mCntPrefix[sub_X_start],
                    d_Z_indices[i % 2],
                    d_Z_values[i % 2],
                    X_nfbits,
                    Y_nfbits,
                    ncbits);

                common::cuda::stream_sync(stream);

                void *d_temp_storage = nullptr;
                size_t temp_storage_bytes = 0;
                size_t temp_storage_bytes_sort = 0;
                size_t temp_storage_bytes_reduce = 0;
                // printf("before sort and reduce\n");
                // memoryPool->printFree();

                // lindex_t *d_keys = memoryPool->allocate<lindex_t>(prtn_ir_nnz);
                // value_t *d_vals = memoryPool->allocate<value_t>(prtn_ir_nnz);
                index_t *d_num_segments = memoryPool->allocate<index_t>(1);

                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes_sort,
                                                d_Z_indices[i % 2], d_Z_indices[i % 2],
                                                d_Z_values[i % 2], d_Z_values[i % 2],
                                                prtn_ir_nnz,
                                                0, 64, stream);

                cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes_reduce,
                                               d_Z_indices[i % 2], d_Z_indices[i % 2],
                                               d_Z_values[i % 2], d_Z_values[i % 2],
                                               d_num_segments, cub::Sum(),
                                               prtn_ir_nnz,
                                               stream);
                temp_storage_bytes = std::max(temp_storage_bytes_sort, temp_storage_bytes_reduce);
                d_temp_storage = memoryPool->allocate<void>(temp_storage_bytes);

                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                d_Z_indices[i % 2], d_Z_indices[i % 2],
                                                d_Z_values[i % 2], d_Z_values[i % 2],
                                                prtn_ir_nnz,
                                                0, 64, stream);

                cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                               d_Z_indices[i % 2], d_Z_indices[i % 2],
                                               d_Z_values[i % 2], d_Z_values[i % 2],
                                               d_num_segments, cub::Sum(),
                                               prtn_ir_nnz,
                                               stream);

                common::cuda::d2hcpy_async(&temp_result_nnz, d_num_segments, sizeof(index_t), stream);
                common::cuda::stream_sync(stream);

                // if (i > 0)
                // {
                //     common::cuda::pinned_free(sub_Z_indices[i - 1]);
                //     common::cuda::pinned_free(sub_Z_values[i - 1]);
                // }

                memoryPool->deallocate<void>(d_temp_storage, temp_storage_bytes);
                // memoryPool->deallocate<lindex_t>(d_keys, prtn_ir_nnz);
                // memoryPool->deallocate<value_t>(d_vals, prtn_ir_nnz);
                memoryPool->deallocate<index_t>(d_num_segments, 1);

                common::cuda::d2hcpy_async(sub_Z_indices, d_Z_indices[i % 2], sizeof(lindex_t) * temp_result_nnz, stream);
                common::cuda::d2hcpy_async(sub_Z_values, d_Z_values[i % 2], sizeof(value_t) * temp_result_nnz, stream);

                // common::cuda::d2hcpy_async(sub_Z_indices[i], d_Z_indices[i % 2], sizeof(lindex_t) * temp_result_nnz, stream);
                // common::cuda::d2hcpy_async(sub_Z_values[i], d_Z_values[i % 2], sizeof(value_t) * temp_result_nnz, stream);
                sub_Z_nnz[i] = temp_result_nnz;
                *Z_nnz += temp_result_nnz;
            }
            printf("Z_nnz: %lu\n", *Z_nnz);

            timer->stop();
            timer->printElapsed("ESC contraction kernel");
            common::cuda::device_sync();
            for (int i = 0; i < 2; ++i)
            {
                memoryPool->deallocate<lindex_t>(d_Z_indices[i], max_prtn_ir_nnz);
                memoryPool->deallocate<value_t>(d_Z_values[i], max_prtn_ir_nnz);
            }

            // Z_indices = static_cast<lindex_t *>(common::cuda::pinned_malloc(sizeof(lindex_t) * *Z_nnz));
            // Z_values = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * *Z_nnz));

            // index_t offset = 0;
            // for (int i = 0; i < num_prtn; i++)
            // {
            //     common::cuda::h2dcpy_async(Z_indices + offset, sub_Z_indices[i], sizeof(lindex_t) * sub_Z_nnz[i], streams[i % num_streams]);
            //     common::cuda::h2dcpy_async(Z_values + offset, sub_Z_values[i], sizeof(value_t) * sub_Z_nnz[i], streams[i % num_streams]);
            //     offset += sub_Z_nnz[i];
            // }

            // for (int i = 0; i < *Z_nnz; ++i)
            // {
            //     printf("Z_indices[%d]: %lu, Z_values[%d]: %f\n", i, Z_indices[i], i, Z_values[i]);
            // }
            common::cuda::pinned_free(d_Z_indices);
            common::cuda::pinned_free(d_Z_values);

            common::cuda::destory_streams(streams, num_streams);

            // common::cuda::pinned_free(sub_Z_indices[num_prtn - 1]);
            // common::cuda::pinned_free(sub_Z_values[num_prtn - 1]);
            common::cuda::pinned_free(sub_Z_indices);
            common::cuda::pinned_free(sub_Z_values);

            return;
        }
    }
}
#endif