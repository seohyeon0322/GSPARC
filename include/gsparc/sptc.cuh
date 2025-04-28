#ifndef SPTC_CUH
#define SPTC_CUH

#include "common/cuda_helper.hpp"
#include "gsparc/helper.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/indexMatch.cuh"
#include "gsparc/contraction.cuh"
#include "gsparc/contraction_extra.cuh"
#include "common/size.hpp"
#include "gsparc/timer.hpp"

namespace gsparc
{
    template <typename TensorType, typename SLITOMType>
    void SpTC(SLITOMType *SX,
              SLITOMType *SY,
              SLITOMType *SZ,
              CudaMemoryPool **memory_pools,
              int gpu_count,
              Timer **copy_timers,
              Timer **idxmatch_timers,
              Timer **dprtn_timers,
              Timer **contraction_timers,
              int dense_accumulator = 0)
    {
        using slitom_t = SLITOMType;
        using tensor_t = TensorType;
        using index_t = typename tensor_t::index_t;
        using lindex_t = typename slitom_t::lindex_t;
        using ulindex_t = typename slitom_t::ulindex_t;
        using value_t = typename slitom_t::value_t;

        int X_prtn_num = SX->nprtn;
        int Y_prtn_num = SY->nprtn;
        SZ->nnz = 0;
        printf("before start\n");
        memory_pools[0]->printFree();

// Initialize memory pool
#pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
        for (int i = 0; i < X_prtn_num; ++i)
        {
            int gpu_id = i % gpu_count;
            Timer *copy_timer = copy_timers[gpu_id];
            Timer *idxmatch_timer = idxmatch_timers[gpu_id];
            Timer *dprtn_timer = dprtn_timers[gpu_id];
            Timer *contraction_timer = contraction_timers[gpu_id];
            common::cuda::set_device(gpu_id);

            CudaMemoryPool *memory_pool = memory_pools[gpu_id];

            index_t sub_X_start = SX->prtn_idx[i];
            index_t sub_X_end = SX->prtn_idx[i + 1];
            uint64_t sub_X_nnz = sub_X_end - sub_X_start;
            printf("sub_X_start: %llu, sub_X_end: %llu, sub_X_nnz: %llu\n", sub_X_start, sub_X_end, sub_X_nnz);

            if (sub_X_nnz == 0)
            {
                continue;
            }
            printf("=================X start=================\n");
            memory_pool->printFree();

            lindex_t *d_X_indices = memory_pool->allocate<lindex_t>(sub_X_nnz);
            value_t *d_X_values = memory_pool->allocate<value_t>(sub_X_nnz);
            index_t *d_mPos = memory_pool->allocate<index_t>(sub_X_nnz);
            index_t *d_mCnt = memory_pool->allocate<index_t>(sub_X_nnz);
            index_t *d_mCntPrefix = memory_pool->allocate<index_t>(sub_X_nnz + 1);

            index_t *mCntPrefix = static_cast<index_t *>(common::cuda::pinned_malloc(sizeof(index_t) * (sub_X_nnz + 1)));

            // stream
            cudaStream_t stream1, stream2;
            checkCudaErrors(cudaStreamCreate(&stream1));
            checkCudaErrors(cudaStreamCreate(&stream2));

            // Memcpy

            for (int j = 0; j < Y_prtn_num; j++)
            {
                copy_timer->start();
                if (j == 0)
                {
                    if (SX->sort_gpu == false)
                    {

                        common::cuda::h2dcpy_async(d_X_indices, SX->indices + sub_X_start, sub_X_nnz * sizeof(lindex_t), stream1);
                        common::cuda::h2dcpy_async(d_X_values, SX->values + sub_X_start, sub_X_nnz * sizeof(value_t), stream2);
                    }
                    else
                    {
                        d_X_indices = SX->d_indices[gpu_id] + sub_X_start;
                        d_X_values = SX->d_values[gpu_id] + sub_X_start;
                    }
                }
                printf("===========i: %d, j: %d, gpu_id: %d===========\n", i, j, gpu_id);
                cudaEvent_t start, end;
                index_t sub_Y_start = SY->prtn_idx[j];
                index_t sub_Y_end = SY->prtn_idx[j + 1];
                uint64_t sub_Y_nnz = sub_Y_end - sub_Y_start;

                // printf("indices[%d]: %llu, indices[%d]: %llu\n", sub_X_start, SX->indices[sub_X_start], sub_X_end - 1, SX->indices[sub_X_end - 1]);
                // printf("sub_Y_start: %d, sub_Y_end: %d\n", sub_Y_start, sub_Y_end);
                // printf("sub_Y_nnz: %llu\n", sub_Y_nnz);
                // printf("values[%d]: %f, values[%d]: %f\n", sub_X_start, SX->values[sub_X_start], sub_X_end - 1, SX->values[sub_X_end - 1]);
                // printf("cindex[%d]: %llu, cindex[%d]: %llu\n", sub_X_start, SX->indices[sub_X_start] & ((static_cast<index_t>(1)<<SX->ncbits) - 1), sub_X_end - 1, SX->indices[sub_X_start - 1] & ((static_cast<index_t>(1)<<SX->ncbits) - 1));

                // printf("Y indices[%d]: %llu, indices[%d]: %llu\n", sub_Y_start, SY->indices[sub_Y_start], sub_Y_end - 1, SY->indices[sub_Y_end - 1]);
                // printf("findex [%d]: %llu, findex[%d]: %llu\n", sub_Y_start, SY->indices[sub_Y_start] & ((static_cast<index_t>(1)<<SY->nfbits) - 1), sub_Y_end - 1, SY->indices[sub_Y_end - 1] & ((static_cast<index_t>(1)<<SY->nfbits) - 1));
                // printf("cidxex[%d]: %llu, cindex[%d]: %llu\n", sub_Y_start, SY->indices[sub_Y_start] >> SY->nfbits, sub_Y_end - 1, SY->indices[sub_Y_end - 1] >> SY->nfbits);

                if (sub_Y_nnz == 0)
                {
                    printf("sub_Y_nnz == 0\n");
                    copy_timer->stop();
                    continue;
                }
                lindex_t *Z_indices;
                value_t *Z_values;

                uint64_t Z_nnz = 0;

                lindex_t *d_Y_indices;
                value_t *d_Y_values;

                if (SY->sort_gpu == false)
                {
                    d_Y_indices = memory_pool->allocate<lindex_t>(sub_Y_nnz);
                    d_Y_values = memory_pool->allocate<value_t>(sub_Y_nnz);
                    common::cuda::h2dcpy_async(d_Y_indices, SY->indices + sub_Y_start, sub_Y_nnz * sizeof(lindex_t), stream1);
                    common::cuda::h2dcpy_async(d_Y_values, SY->values + sub_Y_start, sub_Y_nnz * sizeof(value_t), stream2);
                }
                else
                {
                    d_Y_indices = SY->d_indices[gpu_id] + sub_Y_start;
                    d_Y_values = SY->d_values[gpu_id] + sub_Y_start;
                }

                printf("right after allocate\n");
                memory_pool->printFree();

                common::cuda::device_memset(d_mPos, 0, sub_X_nnz * sizeof(index_t));
                common::cuda::device_memset(d_mCnt, 0, sub_X_nnz * sizeof(index_t));
                common::cuda::device_memset(d_mCntPrefix, 0, (sub_X_nnz + 1) * sizeof(index_t));
                common::cuda::stream_sync(stream1);
                copy_timer->stop();
                copy_timer->printElapsed("h2dcpy");

                // common::cuda::start_timer(&start, &end);
                common::cuda::device_sync();

                idxmatch_timer->start();
                uint64_t ir_nnz = 0;
                gsparc::IndexMatch<lindex_t, index_t>(d_X_indices,
                                                      sub_X_nnz,
                                                      d_Y_indices,
                                                      sub_Y_nnz,
                                                      SX->ncbits,
                                                      SY->nfbits,
                                                      d_mPos,
                                                      d_mCnt,
                                                      d_mCntPrefix,
                                                      mCntPrefix,
                                                      &ir_nnz,
                                                      stream1,
                                                      memory_pool,
                                                      idxmatch_timer,
                                                      gpu_count);

                // common::cuda::end_timer_with_msg(&start, &end, "IndexMatch");
                idxmatch_timer->stop();
                idxmatch_timer->printElapsed("IndexMatch");

                if (ir_nnz == 0)
                {
                    printf("ir_nnz == 0\n");

                    memory_pool->deallocate<lindex_t>(d_Y_indices, sub_Y_nnz);
                    memory_pool->deallocate<value_t>(d_Y_values, sub_Y_nnz);

                    continue;
                }

                uint64_t esc_size = ir_nnz * (sizeof(lindex_t) + sizeof(value_t)) * 4; // 2 is for sorting

                lindex_t first_findex = SX->indices[sub_X_start] >> SX->ncbits;
                lindex_t last_findex = SX->indices[sub_X_end - 1] >> SX->ncbits;
                uint64_t dense_range = ((last_findex - first_findex + 1) * static_cast<lindex_t>(1) << (SY->nfbits));
                uint64_t dense_nnz = dense_range * SY->nfbits;
                lindex_t dense_start = first_findex << SY->nfbits;

                uint64_t dense_acc_size = dense_nnz * sizeof(value_t);
                printf("ir_nnz: %llu, dense_nnz: %llu\n", ir_nnz, dense_nnz);
                printf("esc_size : %s\n", common::byteToString(esc_size));
                printf("dense_acc_size : %s\n", common::byteToString(dense_acc_size));
                printf("memory_pool->is_available(esc_size): %d\n", memory_pool->is_available(esc_size));
                memory_pool->printFree();

                bool default_dense = (dense_nnz < ir_nnz
                    && memory_pool->is_available(dense_acc_size)
                    && dense_nnz > sub_X_nnz);

                bool use_dense;
                if (dense_accumulator == 1) {
                use_dense = true;
                if (memory_pool->is_available(dense_acc_size) == false){
                    fprintf(stderr, "Out of memory\n\n\n");
                    exit(1);
                }
                }
                else if (dense_accumulator == -1) {
                use_dense = false;
                }
                else { // == 0
                use_dense = default_dense;
                }
                if (use_dense) /* Dense Accumulator */
                {
                    fprintf(stderr, "Dense Accumulator\n");

                    gsparc::Dense_Contraction<index_t, lindex_t, value_t>(d_X_indices,
                                                                          d_X_values,
                                                                          sub_X_nnz,
                                                                          d_Y_indices,
                                                                          d_Y_values,
                                                                          sub_Y_nnz,
                                                                          ir_nnz,
                                                                          d_mPos,
                                                                          d_mCnt,
                                                                          dense_start,
                                                                          dense_nnz,
                                                                          Z_indices,
                                                                          Z_values,
                                                                          &Z_nnz,
                                                                          SX->nfbits,
                                                                          SY->nfbits,
                                                                          SX->ncbits,
                                                                          memory_pool,
                                                                          0,
                                                                          contraction_timer);
                }
                else /* ESC accumulator */
                {
                    std::vector<index_t> prtn_offset;

                    // Do something
                    if (memory_pool->is_available(esc_size)) /* ESC */
                    {
                        // ESC
                        printf("ESC accumulator available\n");

                        gsparc::ESC_Contraction<index_t, lindex_t, value_t>(d_X_indices,
                                                                            d_X_values,
                                                                            sub_X_nnz,
                                                                            d_Y_indices,
                                                                            d_Y_values,
                                                                            sub_Y_nnz,
                                                                            Z_indices,
                                                                            Z_values,
                                                                            &Z_nnz,
                                                                            prtn_offset,
                                                                            d_mPos,
                                                                            d_mCnt,
                                                                            d_mCntPrefix,
                                                                            mCntPrefix,
                                                                            ir_nnz,
                                                                            ir_nnz,
                                                                            1,
                                                                            SX->nfbits,
                                                                            SY->nfbits,
                                                                            SX->ncbits,
                                                                            memory_pool,
                                                                            contraction_timer);
                    }
                    else /* ESC with dynamic partitioning */
                    {

                        printf("ESC 2\n");
                        uint64_t max_prtn_ir_nnz = 0;
                        dprtn_timer->start();
                        gsparc::Dynamic_partition<index_t, lindex_t, value_t>(SX->indices + sub_X_start, sub_X_nnz, ir_nnz, mCntPrefix, SX->ncbits, memory_pool, prtn_offset, &max_prtn_ir_nnz);
                        dprtn_timer->stop();
                        dprtn_timer->printElapsed("Dynamic_partition");
                        int dynamic_prtn_num = prtn_offset.size();
                        printf("dynamic_prtn_num: %d\n", dynamic_prtn_num);

                        gsparc::ESC_Contraction<index_t, lindex_t, value_t>(d_X_indices,
                                                                            d_X_values,
                                                                            sub_X_nnz,
                                                                            d_Y_indices,
                                                                            d_Y_values,
                                                                            sub_Y_nnz,
                                                                            Z_indices,
                                                                            Z_values,
                                                                            &Z_nnz,
                                                                            prtn_offset,
                                                                            d_mPos,
                                                                            d_mCnt,
                                                                            d_mCntPrefix,
                                                                            mCntPrefix,
                                                                            ir_nnz,
                                                                            max_prtn_ir_nnz,
                                                                            dynamic_prtn_num,
                                                                            SX->nfbits,
                                                                            SY->nfbits,
                                                                            SX->ncbits,
                                                                            memory_pool,
                                                                            contraction_timer);

                        // ESC accumulator
                    }
                }
                if (SY->sort_gpu == false)
                {
                    memory_pool->deallocate<lindex_t>(d_Y_indices, sub_Y_nnz);
                    memory_pool->deallocate<value_t>(d_Y_values, sub_Y_nnz);
                }
#pragma omp atomic
                SZ->nnz += Z_nnz;
                printf("Z_nnz: %llu\n", Z_nnz);

            } // Iterate Y partitions
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);

            if (SX->sort_gpu == false)
            {
                memory_pool->deallocate<lindex_t>(d_X_indices, sub_X_nnz);
                memory_pool->deallocate<value_t>(d_X_values, sub_X_nnz);
            }
            memory_pool->deallocate<index_t>(d_mPos, sub_X_nnz);
            memory_pool->deallocate<index_t>(d_mCnt, sub_X_nnz);
            memory_pool->deallocate<index_t>(d_mCntPrefix, sub_X_nnz + 1);
            common::cuda::pinned_free(mCntPrefix);

        } // Iterate X partitions
        printf("SZ->nnz: %llu\n", SZ->nnz);
        if (SX->sort_gpu == true)
        {
            for (int g = 0; g < gpu_count; ++g)
            {
                memory_pools[g]->deallocate<lindex_t>(SX->d_indices[g], SX->nnz);
                memory_pools[g]->deallocate<value_t>(SX->d_values[g], SX->nnz);
                SX->d_indices[g] = nullptr;
                SX->d_values[g] = nullptr;
            }
        }
        if (SY->sort_gpu == true)
        {
            for (int g = 0; g < gpu_count; ++g)
            {
                memory_pools[g]->deallocate<lindex_t>(SY->d_indices[g], SY->nnz);
                memory_pools[g]->deallocate<value_t>(SY->d_values[g], SY->nnz);
                SY->d_indices[g] = nullptr;
                SY->d_values[g] = nullptr;
            }
        }
    }

    template <typename TensorType, typename SLITOMType>
    void multi_SpTC(SLITOMType *SX,
                    SLITOMType *SY,
                    SLITOMType *SZ,
                    CudaMemoryPool **memory_pools,
                    int gpu_count,
                    Timer **copy_timers,
                    Timer **idxmatch_timers,
                    Timer **dprtn_timers,
                    Timer **contraction_timers)
    {
        using slitom_t = SLITOMType;
        using tensor_t = TensorType;
        using index_t = typename tensor_t::index_t;
        using lindex_t = typename slitom_t::lindex_t;
        using ulindex_t = typename slitom_t::ulindex_t;
        using value_t = typename slitom_t::value_t;

        int X_prtn_num = SX->nprtn;
        int Y_prtn_num = SY->nprtn;
        SZ->nnz = 0;
        memory_pools[0]->printFree();

// Initialize memory pool
#pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
        for (int i = 0; i < X_prtn_num; ++i)
        {
            int gpu_id = i % gpu_count;
            Timer *copy_timer = copy_timers[gpu_id];
            Timer *idxmatch_timer = idxmatch_timers[gpu_id];
            Timer *dprtn_timer = dprtn_timers[gpu_id];
            Timer *contraction_timer = contraction_timers[gpu_id];
            common::cuda::set_device(gpu_id);

            CudaMemoryPool *memory_pool = memory_pools[gpu_id];

            index_t sub_X_start = SX->prtn_idx[i];
            index_t sub_X_end = SX->prtn_idx[i + 1];
            uint64_t sub_X_nnz = sub_X_end - sub_X_start;
            printf("sub_X_start: %llu, sub_X_end: %llu, sub_X_nnz: %llu\n", sub_X_start, sub_X_end, sub_X_nnz);

            if (sub_X_nnz == 0)
            {
                continue;
            }
            memory_pool->printFree();

            lindex_t *d_X_indices = memory_pool->allocate<lindex_t>(sub_X_nnz);
            value_t *d_X_values = memory_pool->allocate<value_t>(sub_X_nnz);
            index_t *d_mPos = memory_pool->allocate<index_t>(sub_X_nnz);
            index_t *d_mCnt = memory_pool->allocate<index_t>(sub_X_nnz);
            index_t *d_mCntPrefix = memory_pool->allocate<index_t>(sub_X_nnz + 1);

            index_t *mCntPrefix = static_cast<index_t *>(common::cuda::pinned_malloc(sizeof(index_t) * (sub_X_nnz + 1)));

            // stream
            cudaStream_t stream1, stream2;
            checkCudaErrors(cudaStreamCreate(&stream1));
            checkCudaErrors(cudaStreamCreate(&stream2));

            // Memcpy

            for (int j = 0; j < Y_prtn_num; j++)
            {
                copy_timer->start();
                if (j == 0)
                {
                    if (SX->sort_gpu == false)
                    {

                        common::cuda::h2dcpy_async(d_X_indices, SX->indices + sub_X_start, sub_X_nnz * sizeof(lindex_t), stream1);
                        common::cuda::h2dcpy_async(d_X_values, SX->values + sub_X_start, sub_X_nnz * sizeof(value_t), stream2);
                    }
                    else
                    {
                        d_X_indices = SX->d_indices[gpu_id] + sub_X_start;
                        d_X_values = SX->d_values[gpu_id] + sub_X_start;
                    }
                }
                printf("===========i: %d, j: %d, gpu_id: %d===========\n", i, j, gpu_id);
                cudaEvent_t start, end;
                index_t sub_Y_start = SY->prtn_idx[j];
                index_t sub_Y_end = SY->prtn_idx[j + 1];
                uint64_t sub_Y_nnz = sub_Y_end - sub_Y_start;

                if (sub_Y_nnz == 0)
                {
                    printf("sub_Y_nnz == 0\n");
                    copy_timer->stop();
                    continue;
                }
                lindex_t *Z_indices;
                value_t *Z_values;

                uint64_t Z_nnz = 0;

                lindex_t *d_Y_indices;
                value_t *d_Y_values;

                if (SY->sort_gpu == false)
                {
                    d_Y_indices = memory_pool->allocate<lindex_t>(sub_Y_nnz);
                    d_Y_values = memory_pool->allocate<value_t>(sub_Y_nnz);
                    common::cuda::h2dcpy_async(d_Y_indices, SY->indices + sub_Y_start, sub_Y_nnz * sizeof(lindex_t), stream1);
                    common::cuda::h2dcpy_async(d_Y_values, SY->values + sub_Y_start, sub_Y_nnz * sizeof(value_t), stream2);
                }
                else
                {
                    d_Y_indices = SY->d_indices[gpu_id] + sub_Y_start;
                    d_Y_values = SY->d_values[gpu_id] + sub_Y_start;
                }

                printf("right after allocate\n");
                memory_pool->printFree();

                common::cuda::device_memset(d_mPos, 0, sub_X_nnz * sizeof(index_t));
                common::cuda::device_memset(d_mCnt, 0, sub_X_nnz * sizeof(index_t));
                common::cuda::device_memset(d_mCntPrefix, 0, (sub_X_nnz + 1) * sizeof(index_t));
                common::cuda::stream_sync(stream1);
                copy_timer->stop();
                copy_timer->printElapsed("h2dcpy");

                // common::cuda::start_timer(&start, &end);
                common::cuda::device_sync();

                idxmatch_timer->start();
                uint64_t ir_nnz = 0;
                gsparc::IndexMatch<lindex_t, index_t>(d_X_indices,
                                                      sub_X_nnz,
                                                      d_Y_indices,
                                                      sub_Y_nnz,
                                                      SX->ncbits,
                                                      SY->nfbits,
                                                      d_mPos,
                                                      d_mCnt,
                                                      d_mCntPrefix,
                                                      mCntPrefix,
                                                      &ir_nnz,
                                                      stream1,
                                                      memory_pool,
                                                      idxmatch_timer,
                                                      gpu_count);

                // common::cuda::end_timer_with_msg(&start, &end, "IndexMatch");
                idxmatch_timer->stop();
                idxmatch_timer->printElapsed("IndexMatch");

                if (ir_nnz == 0)
                {
                    printf("ir_nnz == 0\n");

                    memory_pool->deallocate<lindex_t>(d_Y_indices, sub_Y_nnz);
                    memory_pool->deallocate<value_t>(d_Y_values, sub_Y_nnz);

                    continue;
                }

                uint64_t esc_size = ir_nnz * (sizeof(lindex_t) + sizeof(value_t)) * 4; // 2 is for sorting

                lindex_t first_findex = SX->indices[sub_X_start] >> SX->ncbits;
                lindex_t last_findex = SX->indices[sub_X_end - 1] >> SX->ncbits;
                uint64_t dense_range = ((last_findex - first_findex + 1) * static_cast<lindex_t>(1) << (SY->nfbits));
                uint64_t dense_nnz = dense_range * SY->nfbits;
                lindex_t dense_start = first_findex << SY->nfbits;

                uint64_t dense_acc_size = dense_nnz * sizeof(value_t);
                printf("ir_nnz: %llu, dense_nnz: %llu\n", ir_nnz, dense_nnz);
                printf("esc_size : %s\n", common::byteToString(esc_size));
                printf("dense_acc_size : %s\n", common::byteToString(dense_acc_size));
                printf("memory_pool->is_available(esc_size): %d\n", memory_pool->is_available(esc_size));
                memory_pool->printFree();
                if (dense_nnz < ir_nnz && memory_pool->is_available(dense_acc_size) && dense_nnz > sub_X_nnz) /* Dense Accumulator */
                // if(1 == 0 )
                {
                    // Dense
                    fprintf(stderr, "Dense Accumulator\n");

                    gsparc::Dense_Contraction<index_t, lindex_t, value_t>(d_X_indices,
                                                                          d_X_values,
                                                                          sub_X_nnz,
                                                                          d_Y_indices,
                                                                          d_Y_values,
                                                                          sub_Y_nnz,
                                                                          ir_nnz,
                                                                          d_mPos,
                                                                          d_mCnt,
                                                                          dense_start,
                                                                          dense_nnz,
                                                                          Z_indices,
                                                                          Z_values,
                                                                          &Z_nnz,
                                                                          SX->nfbits,
                                                                          SY->nfbits,
                                                                          SX->ncbits,
                                                                          memory_pool,
                                                                          1,
                                                                          contraction_timer);
                }
                else /* ESC accumulator */
                {
                    std::vector<index_t> prtn_offset;

                    // Do something
                    if (memory_pool->is_available(esc_size)) /* ESC */
                    {
                        // ESC
                        printf("ESC accumulator available\n");

                        gsparc::ESC_Contraction_multi<index_t, lindex_t, value_t>(d_X_indices,
                                                                                  d_X_values,
                                                                                  sub_X_nnz,
                                                                                  d_Y_indices,
                                                                                  d_Y_values,
                                                                                  sub_Y_nnz,
                                                                                  Z_indices,
                                                                                  Z_values,
                                                                                  &Z_nnz,
                                                                                  prtn_offset,
                                                                                  d_mPos,
                                                                                  d_mCnt,
                                                                                  d_mCntPrefix,
                                                                                  mCntPrefix,
                                                                                  ir_nnz,
                                                                                  ir_nnz,
                                                                                  1,
                                                                                  SX->nfbits,
                                                                                  SY->nfbits,
                                                                                  SX->ncbits,
                                                                                  memory_pool,
                                                                                  contraction_timer);
                    }
                    else /* ESC with dynamic partitioning */
                    {

                        printf("ESC 2\n");
                        uint64_t max_prtn_ir_nnz = 0;
                        dprtn_timer->start();
                        gsparc::Dynamic_partition<index_t, lindex_t, value_t>(SX->indices + sub_X_start, sub_X_nnz, ir_nnz, mCntPrefix, SX->ncbits, memory_pool, prtn_offset, &max_prtn_ir_nnz);
                        dprtn_timer->stop();
                        dprtn_timer->printElapsed("Dynamic_partition");
                        int dynamic_prtn_num = prtn_offset.size();
                        printf("dynamic_prtn_num: %d\n", dynamic_prtn_num);

                        gsparc::ESC_Contraction_multi<index_t, lindex_t, value_t>(d_X_indices,
                                                                                  d_X_values,
                                                                                  sub_X_nnz,
                                                                                  d_Y_indices,
                                                                                  d_Y_values,
                                                                                  sub_Y_nnz,
                                                                                  Z_indices,
                                                                                  Z_values,
                                                                                  &Z_nnz,
                                                                                  prtn_offset,
                                                                                  d_mPos,
                                                                                  d_mCnt,
                                                                                  d_mCntPrefix,
                                                                                  mCntPrefix,
                                                                                  ir_nnz,
                                                                                  max_prtn_ir_nnz,
                                                                                  dynamic_prtn_num,
                                                                                  SX->nfbits,
                                                                                  SY->nfbits,
                                                                                  SX->ncbits,
                                                                                  memory_pool,
                                                                                  contraction_timer);

                        // ESC accumulator
                    }
                }
                if (SY->sort_gpu == false)
                {
                    memory_pool->deallocate<lindex_t>(d_Y_indices, sub_Y_nnz);
                    memory_pool->deallocate<value_t>(d_Y_values, sub_Y_nnz);
                }
#pragma omp atomic
                SZ->nnz += Z_nnz;

                printf("Z_nnz: %llu\n", Z_nnz);
                SZ->indices = Z_indices;
                SZ->values = Z_values;

            } // Iterate Y partitions
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);

            if (SX->sort_gpu == false)
            {
                memory_pool->deallocate<lindex_t>(d_X_indices, sub_X_nnz);
                memory_pool->deallocate<value_t>(d_X_values, sub_X_nnz);
            }
            memory_pool->deallocate<index_t>(d_mPos, sub_X_nnz);
            memory_pool->deallocate<index_t>(d_mCnt, sub_X_nnz);
            memory_pool->deallocate<index_t>(d_mCntPrefix, sub_X_nnz + 1);
            common::cuda::pinned_free(mCntPrefix);

            SZ->nprtn = 1;
            SZ->prtn_idx = gsparc::allocate<index_t>(2);
            SZ->prtn_idx[0] = 0;
            SZ->prtn_idx[1] = SZ->nnz;
            SZ->dims = gsparc::allocate<index_t>(SX->fnmodes + SY->fnmodes);
            for (int k = 0; k < SX->fnmodes; ++k)
            {
                SZ->dims[k] = SX->dims[SX->fpos[k]];
            }
            for (int k = 0; k < SY->fnmodes; ++k)
            {
                SZ->dims[k + SX->fnmodes] = SY->dims[SY->fpos[k]];
            }

            SZ->nbits = SX->nfbits + SY->nfbits;
            SZ->ncbits = SX->nfbits;
            SZ->nfbits = SY->nfbits;
            SZ->input_order = 1;

        } // Iterate X partitions
        printf("SZ->nnz: %llu\n", SZ->nnz);
        if (SX->sort_gpu == true)
        {
            for (int g = 0; g < gpu_count; ++g)
            {
                memory_pools[g]->deallocate<lindex_t>(SX->d_indices[g], SX->nnz);
                memory_pools[g]->deallocate<value_t>(SX->d_values[g], SX->nnz);
                SX->d_indices[g] = nullptr;
                SX->d_values[g] = nullptr;
            }
        }
        // if (SY->sort_gpu == true)
        // {
        //     for (int g = 0; g < gpu_count; ++g)
        //     {
        //         memory_pools[g]->deallocate<lindex_t>(SY->d_indices[g], SY->nnz);
        //         memory_pools[g]->deallocate<value_t>(SY->d_values[g], SY->nnz);
        //         SY->d_indices[g] = nullptr;
        //         SY->d_values[g] = nullptr;
        //     }
        // }
    }

    namespace extra
    {
        template <typename TensorType, typename SLITOMType>
        void SpTC(SLITOMType *SX,
                  SLITOMType *SY,
                  SLITOMType *SZ,
                  CudaMemoryPool **memory_pools,
                  int gpu_count,
                  Timer **copy_timers,
                  Timer **idxmatch_timers,
                  Timer **dprtn_timers,
                  Timer **contraction_timers,
                  int dense_accumulator = 0)


        {
            using slitom_t = SLITOMType;
            using tensor_t = TensorType;
            using index_t = typename tensor_t::index_t;
            using lindex_t = typename slitom_t::lindex_t;
            using ulindex_t = typename slitom_t::ulindex_t;
            using value_t = typename slitom_t::value_t;

            int X_prtn_num = SX->nprtn;
            int Y_prtn_num = SY->nprtn;
            SZ->nnz = 0;
            printf("SX gpu: %d, SY gpu: %d\n", SX->sort_gpu, SY->sort_gpu);

#pragma omp parallel for num_threads(gpu_count) schedule(static, 1)
            for (int i = 0; i < X_prtn_num; ++i)
            {
                int gpu_id = i % gpu_count;
                Timer *copy_timer = copy_timers[gpu_id];
                Timer *idxmatch_timer = idxmatch_timers[gpu_id];
                Timer *dprtn_timer = dprtn_timers[gpu_id];
                Timer *contraction_timer = contraction_timers[gpu_id];

                common::cuda::set_device(gpu_id);

                CudaMemoryPool *memory_pool = memory_pools[gpu_id];

                index_t sub_X_start = SX->prtn_idx[i];
                index_t sub_X_end = SX->prtn_idx[i + 1];
                uint64_t sub_X_nnz = sub_X_end - sub_X_start;

                if (sub_X_nnz == 0)
                {
                    continue;
                }

                ulindex_t *d_X_uindices = memory_pool->allocate<ulindex_t>(sub_X_nnz);
                lindex_t *d_X_indices = memory_pool->allocate<lindex_t>(sub_X_nnz);
                value_t *d_X_values = memory_pool->allocate<value_t>(sub_X_nnz);
                index_t *d_mPos = memory_pool->allocate<index_t>(sub_X_nnz);
                index_t *d_mCnt = memory_pool->allocate<index_t>(sub_X_nnz);
                index_t *d_mCntPrefix = memory_pool->allocate<index_t>(sub_X_nnz + 1);

                index_t *mCntPrefix = static_cast<index_t *>(common::cuda::pinned_malloc(sizeof(index_t) * (sub_X_nnz + 1)));

                // stream
                cudaStream_t stream1, stream2;
                checkCudaErrors(cudaStreamCreate(&stream1));
                checkCudaErrors(cudaStreamCreate(&stream2));

                for (int j = 0; j < Y_prtn_num; j++)
                {
                    printf("===========i: %d, j: %d, gpu_id: %d===========\n", i, j, gpu_id);

                    copy_timer->start();

                    if (j == 0)
                    {
                        // Memcpy
                        if (SX->sort_gpu == false)
                        {
                            common::cuda::h2dcpy_async(d_X_uindices, SX->uindices + sub_X_start, sub_X_nnz * sizeof(ulindex_t), stream1);
                            common::cuda::h2dcpy_async(d_X_indices, SX->indices + sub_X_start, sub_X_nnz * sizeof(lindex_t), stream1);
                            common::cuda::h2dcpy_async(d_X_values, SX->values + sub_X_start, sub_X_nnz * sizeof(value_t), stream2);
                        }
                        else
                        {
                            d_X_uindices = SX->d_uindices[gpu_id] + sub_X_start;
                            d_X_indices = SX->d_indices[gpu_id] + sub_X_start;
                            d_X_values = SX->d_values[gpu_id] + sub_X_start;
                        }
                    }
                    cudaEvent_t start, end;
                    index_t sub_Y_start = SY->prtn_idx[j];
                    index_t sub_Y_end = SY->prtn_idx[j + 1];
                    uint64_t sub_Y_nnz = sub_Y_end - sub_Y_start;

                    if (sub_Y_nnz == 0)
                    {
                        copy_timer->stop();
                        continue;
                    }
                    lindex_t *Z_indices;
                    value_t *Z_values;

                    uint64_t Z_nnz = 0;

                    ulindex_t *d_Y_uindices;
                    lindex_t *d_Y_indices;
                    value_t *d_Y_values;

                    if (SY->sort_gpu == false)
                    {
                        d_Y_uindices = memory_pool->allocate<ulindex_t>(sub_Y_nnz);
                        d_Y_indices = memory_pool->allocate<lindex_t>(sub_Y_nnz);
                        d_Y_values = memory_pool->allocate<value_t>(sub_Y_nnz);
                        common::cuda::h2dcpy_async(d_Y_uindices, SY->uindices + sub_Y_start, sub_Y_nnz * sizeof(ulindex_t), stream1);
                        common::cuda::h2dcpy_async(d_Y_indices, SY->indices + sub_Y_start, sub_Y_nnz * sizeof(lindex_t), stream1);
                        common::cuda::h2dcpy_async(d_Y_values, SY->values + sub_Y_start, sub_Y_nnz * sizeof(value_t), stream2);
                    }
                    else
                    {
                        d_Y_uindices = SY->d_uindices[gpu_id] + sub_Y_start;
                        d_Y_indices = SY->d_indices[gpu_id] + sub_Y_start;
                        d_Y_values = SY->d_values[gpu_id] + sub_Y_start;
                    }

                    common::cuda::device_memset(d_mPos, 0, sub_X_nnz * sizeof(index_t));
                    common::cuda::device_memset(d_mCnt, 0, sub_X_nnz * sizeof(index_t));
                    common::cuda::device_memset(d_mCntPrefix, 0, (sub_X_nnz + 1) * sizeof(index_t));
                    common::cuda::stream_sync(stream1);
                    copy_timer->stop();
                    copy_timer->printElapsed("h2dcpy");
                    common::cuda::device_sync();
                    // common::cuda::start_timer(&start, &end);
                    idxmatch_timer->start();
                    uint64_t ir_nnz = 0;
                    gsparc::IndexMatch_extra<ulindex_t, lindex_t, index_t>(d_X_uindices,
                                                                           d_X_indices,
                                                                           sub_X_nnz,
                                                                           d_Y_uindices,
                                                                           d_Y_indices,
                                                                           sub_Y_nnz,
                                                                           SX->ncbits,
                                                                           SY->nfbits,
                                                                           d_mPos,
                                                                           d_mCnt,
                                                                           d_mCntPrefix,
                                                                           mCntPrefix,
                                                                           &ir_nnz,
                                                                           stream1,
                                                                           memory_pool,
                                                                           gpu_count);

                    uint64_t esc_size = ir_nnz * (sizeof(lindex_t) + sizeof(value_t)) * 3; // 3 is for sorting
                    idxmatch_timer->stop();
                    idxmatch_timer->printElapsed("IndexMatch");
                    printf("ir_nnz: %llu\n", ir_nnz);
                    if (ir_nnz == 0)
                    {
                        memory_pool->deallocate<ulindex_t>(d_Y_uindices, sub_Y_nnz);
                        memory_pool->deallocate<lindex_t>(d_Y_indices, sub_Y_nnz);
                        memory_pool->deallocate<value_t>(d_Y_values, sub_Y_nnz);

                        continue;
                    }
                    __uint128_t first = (static_cast<__uint128_t>(SX->uindices[sub_X_start]) << 64) | SX->indices[sub_X_start];
                    lindex_t first_findex = first >> SX->ncbits;
                    __uint128_t last = (static_cast<__uint128_t>(SX->uindices[sub_X_end - 1]) << 64) | SX->indices[sub_X_end - 1];
                    lindex_t last_findex = last >> SX->ncbits;
                    uint64_t dense_range = ((last_findex - first_findex + 1) * static_cast<lindex_t>(1) << (SY->nfbits));
                    uint64_t dense_nnz = dense_range * SY->nfbits;
                    lindex_t dense_start = first_findex << SY->nfbits;

                    uint64_t dense_acc_size = dense_nnz * sizeof(value_t);
                    printf("ir_nnz: %llu, dense_nnz: %llu\n", ir_nnz, dense_nnz);
                    printf("esc_size : %s\n", common::byteToString(esc_size));
                    printf("dense_acc_size : %s\n", common::byteToString(dense_acc_size));
                    printf("memory_pool->is_available(esc_size): %d\n", memory_pool->is_available(esc_size));
                    bool default_dense = (gpu_count == 1 
                        && dense_nnz < ir_nnz
                        && memory_pool->is_available(dense_acc_size)
                        && dense_nnz > sub_X_nnz);
    
                    bool use_dense;
                    if (dense_accumulator == 1) {
                    use_dense = true;
                    if (memory_pool->is_available(dense_acc_size) == false){
                        fprintf(stderr, "Out of memory\n\n\n");
                        exit(1);
                    }
                    }
                    else if (dense_accumulator == -1) {
                    use_dense = false;
                    }
                    else { // == 0
                    use_dense = default_dense;
                    }
                    if (use_dense) /* Dense Accumulator */
                        {
                        // Dense
                        fprintf(stderr, "Dense Accumulator\n");
                        gsparc::extra::Dense_Contraction<index_t, ulindex_t, lindex_t, value_t>(d_X_uindices,
                                                                                                d_X_indices,
                                                                                                d_X_values,
                                                                                                sub_X_nnz,
                                                                                                d_Y_uindices,
                                                                                                d_Y_indices,
                                                                                                d_Y_values,
                                                                                                sub_Y_nnz,
                                                                                                ir_nnz,
                                                                                                d_mPos,
                                                                                                d_mCnt,
                                                                                                dense_start,
                                                                                                dense_nnz,
                                                                                                Z_indices,
                                                                                                Z_values,
                                                                                                &Z_nnz,
                                                                                                SX->nfbits,
                                                                                                SY->nfbits,
                                                                                                SX->ncbits,
                                                                                                memory_pool,
                                                                                                contraction_timer);
                    }
                    else /* ESC accumulator */
                    {
                        std::vector<index_t> prtn_offset;
                        // Do something
                        if (memory_pool->is_available(esc_size)) /* ESC */
                        {
                            // ESC
                            // common::cuda::start_timer(&start, &end);
                            printf("ESC 1\n");
                            gsparc::extra::ESC_Contraction<index_t, ulindex_t, lindex_t, value_t>(d_X_uindices,
                                                                                                  d_X_indices,
                                                                                                  d_X_values,
                                                                                                  sub_X_nnz,
                                                                                                  d_Y_uindices,
                                                                                                  d_Y_indices,
                                                                                                  d_Y_values,
                                                                                                  sub_Y_nnz,
                                                                                                  Z_indices,
                                                                                                  Z_values,
                                                                                                  &Z_nnz,
                                                                                                  prtn_offset,
                                                                                                  d_mPos,
                                                                                                  d_mCnt,
                                                                                                  d_mCntPrefix,
                                                                                                  mCntPrefix,
                                                                                                  ir_nnz,
                                                                                                  ir_nnz,
                                                                                                  1,
                                                                                                  SX->nfbits,
                                                                                                  SY->nfbits,
                                                                                                  SX->ncbits,
                                                                                                  memory_pool,
                                                                                                  contraction_timer);

                            // common::cuda::end_timer_with_msg(&start, &end, "ESC_Contraction");
                        }
                        else /* ESC with dynamic partitioning */
                        {
                            printf("ESC 2\n");
                            uint64_t max_prtn_ir_nnz = 0;
                            dprtn_timer->start();
                            gsparc::extra::Dynamic_partition<index_t, ulindex_t, lindex_t, value_t>(SX->uindices + sub_X_start, SX->indices + sub_X_start, sub_X_nnz, ir_nnz, mCntPrefix, SX->ncbits, memory_pool, prtn_offset, &max_prtn_ir_nnz);
                            dprtn_timer->stop();
                            dprtn_timer->printElapsed("Dynamic_partition");
                            // common::cuda::end_timer_with_msg(&start, &end, "Dynamic_partition");
                            int dynamic_prtn_num = prtn_offset.size();
                            printf("max_prtn_ir_nnz: %llu\n", max_prtn_ir_nnz);
                            printf("dynamic_prtn_num: %d\n", dynamic_prtn_num);

                            gsparc::extra::ESC_Contraction<index_t, ulindex_t, lindex_t, value_t>(d_X_uindices,
                                                                                                  d_X_indices,
                                                                                                  d_X_values,
                                                                                                  sub_X_nnz,
                                                                                                  d_Y_uindices,
                                                                                                  d_Y_indices,
                                                                                                  d_Y_values,
                                                                                                  sub_Y_nnz,
                                                                                                  Z_indices,
                                                                                                  Z_values,
                                                                                                  &Z_nnz,
                                                                                                  prtn_offset,
                                                                                                  d_mPos,
                                                                                                  d_mCnt,
                                                                                                  d_mCntPrefix,
                                                                                                  mCntPrefix,
                                                                                                  ir_nnz,
                                                                                                  max_prtn_ir_nnz,
                                                                                                  dynamic_prtn_num,
                                                                                                  SX->nfbits,
                                                                                                  SY->nfbits,
                                                                                                  SX->ncbits,
                                                                                                  memory_pool,
                                                                                                  contraction_timer);

                            // ESC accumulator
                        }
                    }
                    if (SY->sort_gpu == false)
                    {
                        memory_pool->deallocate<ulindex_t>(d_Y_uindices, sub_Y_nnz);
                        memory_pool->deallocate<lindex_t>(d_Y_indices, sub_Y_nnz);
                        memory_pool->deallocate<value_t>(d_Y_values, sub_Y_nnz);
                    }
#pragma omp atomic
                    SZ->nnz += Z_nnz;
                    printf("Z_nnz: %llu\n", Z_nnz);
                } // Iterate Y partitions

                cudaStreamDestroy(stream1);
                cudaStreamDestroy(stream2);

                if (SX->sort_gpu == false)
                {
                    memory_pool->deallocate<ulindex_t>(d_X_uindices, sub_X_nnz);
                    memory_pool->deallocate<lindex_t>(d_X_indices, sub_X_nnz);
                    memory_pool->deallocate<value_t>(d_X_values, sub_X_nnz);
                }
                memory_pool->deallocate<index_t>(d_mPos, sub_X_nnz);
                memory_pool->deallocate<index_t>(d_mCnt, sub_X_nnz);
                memory_pool->deallocate<index_t>(d_mCntPrefix, sub_X_nnz + 1);
                common::cuda::pinned_free(mCntPrefix);

            } // Iterate X partitions
            printf("SZ->nnz: %llu\n", SZ->nnz);

            if (SX->sort_gpu == true)
            {
                for (int g = 0; g < gpu_count; ++g)
                {
                    memory_pools[g]->deallocate<ulindex_t>(SX->d_uindices[g], SX->nnz);
                    memory_pools[g]->deallocate<lindex_t>(SX->d_indices[g], SX->nnz);
                    memory_pools[g]->deallocate<value_t>(SX->d_values[g], SX->nnz);
                    SX->d_uindices[g] = nullptr;
                    SX->d_indices[g] = nullptr;
                    SX->d_values[g] = nullptr;
                }
            }

            if (SY->sort_gpu == true)
            {
                for (int g = 0; g < gpu_count; ++g)
                {
                    memory_pools[g]->deallocate<ulindex_t>(SY->d_uindices[g], SY->nnz);
                    memory_pools[g]->deallocate<lindex_t>(SY->d_indices[g], SY->nnz);
                    memory_pools[g]->deallocate<value_t>(SY->d_values[g], SY->nnz);
                    SY->d_uindices[g] = nullptr;
                    SY->d_indices[g] = nullptr;
                    SY->d_values[g] = nullptr;
                }
            }
        }

    } // namespace extra

} // namespace gsparc

#endif