
#include <omp.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <execution>
#include <immintrin.h>

#include "gsparc/tensor_manager.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/sparse_tensor.hpp"
#include "gsparc/cuda_memory.hpp"
#include "common/bitops.hpp"
#include "common/cuda_helper.hpp"
#include "gsparc/sort.cuh"
#include "gsparc/timer.hpp"
#include "common/size.hpp"

namespace gsparc
{

    TENSOR_MANAGER_TEMPLATE
    TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::TensorManager(int gpu_count)
        : prtn_num(1), gpu_count(gpu_count) {}

    TENSOR_MANAGER_TEMPLATE
    TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::~TensorManager()
    {
    }

    TENSOR_MANAGER_TEMPLATE
    bool TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::FindPartitionNum(uint64_t nnz_count_x, uint64_t nnz_count_y, size_t poolSize, int nbits)
    {
        uint64_t x_storage = (sizeof(lindex_t) + sizeof(value_t)) * nnz_count_x;
        uint64_t y_storage = (sizeof(lindex_t) + sizeof(value_t)) * nnz_count_y;
        uint64_t mPosIdx = sizeof(index_t) * nnz_count_x * 4;
        uint64_t total_storage = (x_storage + y_storage)*2 + mPosIdx;
        if (nbits > 64)
        {
            total_storage += sizeof(ulindex_t) * (nnz_count_x + nnz_count_y);
        }

        
        common::cuda::set_device(0);
        printf("Available memory: %s\n", common::byteToString(poolSize));
        size_t half_poolSize = poolSize / 2;

        prtn_num = ((total_storage) + half_poolSize - 1) / half_poolSize;
        if (gpu_count > prtn_num)
        {
            prtn_num = gpu_count;
        }
        bool sort_gpu = prtn_num == 1 ? true : false;

        printf("total_storage: %s\n", common::byteToString(total_storage));
        printf("Partition number: %d\n", prtn_num);
        max_block_size = ((x_storage + y_storage) / prtn_num) / (sizeof(lindex_t) + sizeof(value_t));
        printf("sort_gpu: %d\n", sort_gpu);
        return sort_gpu;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::get_fmode(slitom_t *slitom, const int *cpos)
    {
        for (int i = 0; i < slitom->cnmodes; i++)
        {
            slitom->cpos[i] = cpos[i];
        }

        int f = 0;
        for (int n = 0; n < slitom->nmodes; ++n)
        {
            bool in_cmode = 0;
            for (int i = 0; i < slitom->cnmodes; ++i)
            {
                if (n == cpos[i])
                {
                    in_cmode = 1;
                }
            }
            if (!in_cmode)
            {
                slitom->fpos[f] = n;
                f++;
            }
        }
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::compute_nbtis(slitom_t *slitom)
    {
        int nmode = slitom->nmodes;
        int fmode = slitom->fnmodes;
        int cmode = slitom->cnmodes;

        int *cpos = slitom->cpos;
        int *fpos = slitom->fpos;

        MPair *mode_bits = gsparc::allocate<MPair>(nmode);

        int nbits = 0;
        int ncbits = 0;
        int nfbits = 0;

        // Initial mode values.
        for (int n = 0; n < nmode; ++n)
        {
            int mbits = (sizeof(uint64_t) * 8) - common::clz(slitom->dims[n] - 1);
            mode_bits[n].mode = n;
            mode_bits[n].bits = mbits;
            nbits += mbits;
        }

        for (int c = 0; c < cmode; ++c)
        {
            int cbits = (sizeof(uint64_t) * 8) - common::clz(slitom->dims[slitom->cpos[c]] - 1);
            ncbits += cbits;
        }

        for (int f = 0; f < fmode; ++f)
        {
            int fbits = (sizeof(uint64_t) * 8) - common::clz(slitom->dims[slitom->fpos[f]] - 1);
            nfbits += fbits;
        }

        slitom->mode_bits = mode_bits;
        slitom->nbits = nbits;
        slitom->ncbits = ncbits;
        slitom->nfbits = nfbits;

        return;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::setup_slitom(slitom_t *slitom)
    {

        mask_t SLITOM_MASK[slitom->nmodes] = {}; // initialized to zeros by default

        int nmode = slitom->nmodes;
        int nbits = slitom->nbits;
        mask_t slitom_mask = 0;

        MPair *mode_bits = slitom->mode_bits;

        int *cpos = slitom->cpos;
        int *fpos = slitom->fpos;

        // lindex_t slitom_fmask = 0;
        // lindex_t slitom_cmask = 0;

        int f_shift = 0;
        // for (int n = 0; n < slitom->fnmodes; ++n)
        for (int n = slitom->fnmodes - 1; n >= 0; --n)
        {
            int fn = fpos[n];
            mask_t mask = (static_cast<mask_t>(1) << mode_bits[fn].bits) - 1;
            SLITOM_MASK[mode_bits[fn].mode] = mask << f_shift;
            f_shift += mode_bits[fn].bits;
        }

        int c_shift = 0;
        // for (int n = 0; n < slitom->cnmodes; ++n)
        for (int n = slitom->cnmodes - 1; n >= 0; --n)

        {
            int cn = cpos[n];

            mask_t mask = (static_cast<mask_t>(1) << mode_bits[cn].bits) - 1;
            SLITOM_MASK[mode_bits[cn].mode] = mask << c_shift;
            c_shift += mode_bits[cn].bits;
            // printf("modes: %d, bits: %d, mask: %llu\n", mode_bits[cn].mode, mode_bits[cn].bits, mask);
        }

        slitom->mode_masks = gsparc::allocate<mask_t>(nmode);
        for (int n = 0; n < nmode; ++n)
        {
            slitom->mode_masks[n] = SLITOM_MASK[n];
            slitom_mask |= SLITOM_MASK[n];
        }

        slitom->slitom_mask = slitom_mask;

        return;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::create_mask(slitom_t *slitom)
    {
        slitom->cmask = 0;
        slitom->cmode_masks = gsparc::allocate<mask_t>(slitom->cnmodes);
        for (int n = 0; n < slitom->cnmodes; ++n)
        {
            slitom->cmode_masks[n] = slitom->mode_masks[slitom->cpos[n]];
            slitom->cmask |= slitom->cmode_masks[n];
            // printf("cmode_masks[%d]: %llu\n", n, slitom->cmode_masks[n]);
        }

        slitom->fmask = 0;
        slitom->fmode_masks = gsparc::allocate<mask_t>(slitom->fnmodes);
        for (int n = 0; n < slitom->fnmodes; ++n)
        {
            slitom->fmode_masks[n] = slitom->mode_masks[slitom->fpos[n]];
            slitom->fmask |= slitom->fmode_masks[n];
            // printf("fmode_masks[%d]: %llu\n", n, slitom->fmode_masks[n]);
        }

        return;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::ConvertTensor(sptensor_t *sptensor, slitom_t *slitom, int cnmodes, const int *cpos, Timer *timer)
    {

        slitom->nmodes = sptensor->order;
        slitom->cnmodes = cnmodes;
        slitom->fnmodes = slitom->nmodes - slitom->cnmodes;
        slitom->nprtn = prtn_num;
        slitom->nnz = sptensor->nnz;

        slitom->dims = gsparc::allocate<lindex_t>(sptensor->order);
        slitom->indices = static_cast<lindex_t *>(common::cuda::pinned_malloc(sizeof(lindex_t) * slitom->nnz));
        slitom->values = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * slitom->nnz));

        slitom->cpos = gsparc::allocate<int>(slitom->cnmodes);
        slitom->fpos = gsparc::allocate<int>(slitom->fnmodes);

        for (int i = 0; i < sptensor->order; i++)
        {
            slitom->dims[i] = sptensor->dims[i];
        }
        printf("flag1\n");
        timer->start();
        get_fmode(slitom, cpos);

        compute_nbtis(slitom);

        setup_slitom(slitom);

        create_mask(slitom);

        mask_t *SLITOM_CMASKS = slitom->cmode_masks;
        mask_t *SLITOM_FMASKS = slitom->fmode_masks;

        if (slitom->input_order == 1 || (slitom->input_order == 2 && slitom->nprtn > 1))
        {
            printf("slitom->input_order: %d\n", slitom->input_order);
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < slitom->nnz; ++i)
            {
                lindex_t s_index = 0;
                lindex_t f_index = 0;
                lindex_t c_index = 0;

                slitom->values[i] = sptensor->values[i];

                for (int c = 0; c < slitom->cnmodes; ++c)
                {
                    c_index |= common::pdep(sptensor->indices[slitom->cpos[c]][i], slitom->cmode_masks[c]);
                }
                for (int f = 0; f < slitom->fnmodes; ++f)
                {
                    f_index |= common::pdep(sptensor->indices[slitom->fpos[f]][i], slitom->fmode_masks[f]);
                }
                s_index = (f_index << (slitom->ncbits)) | c_index;
                slitom->indices[i] = s_index;
            }
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < slitom->nnz; ++i)
            {
                lindex_t s_index = 0;
                lindex_t f_index = 0;
                lindex_t c_index = 0;

                slitom->values[i] = sptensor->values[i];

                for (int c = 0; c < slitom->cnmodes; ++c)
                {
                    c_index |= common::pdep(sptensor->indices[slitom->cpos[c]][i], SLITOM_CMASKS[c]);
                }
                for (int f = 0; f < slitom->fnmodes; ++f)
                {

                    f_index |= common::pdep(sptensor->indices[slitom->fpos[f]][i], SLITOM_FMASKS[f]);
                }
                s_index = (c_index << (slitom->nfbits)) | f_index;

                slitom->indices[i] = s_index;
            }
        }

        timer->stop();
        timer->printElapsed("ConvertTensor");
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::ConvertTensor_extra(sptensor_t *sptensor, slitom_t *slitom, int cnmodes, const int *cpos, Timer *timer)
    {
        slitom->nmodes = sptensor->order;
        slitom->cnmodes = cnmodes;
        slitom->fnmodes = slitom->nmodes - slitom->cnmodes;
        slitom->nprtn = prtn_num;
        slitom->nnz = sptensor->nnz;
        slitom->dims = gsparc::allocate<lindex_t>(sptensor->order);

        for (int i = 0; i < sptensor->order; i++)
        {
            slitom->dims[i] = sptensor->dims[i];
        }

        slitom->cpos = gsparc::allocate<int>(slitom->cnmodes);
        slitom->fpos = gsparc::allocate<int>(slitom->fnmodes);

        get_fmode(slitom, cpos);

        slitom->uindices = static_cast<ulindex_t *>(common::cuda::pinned_malloc(sizeof(ulindex_t) * slitom->nnz));
        slitom->indices = static_cast<lindex_t *>(common::cuda::pinned_malloc(sizeof(lindex_t) * slitom->nnz));
        slitom->values = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * slitom->nnz));

        timer->start();
        compute_nbtis(slitom);

        setup_slitom(slitom);

        create_mask(slitom);

        mask_t *SLITOM_CMASKS = slitom->cmode_masks;
        mask_t *SLITOM_FMASKS = slitom->fmode_masks;
        printf("input_order: %d\n", slitom->input_order);
        if (slitom->input_order == 1 || (slitom->input_order == 2 && slitom->nprtn > 1))
        {
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < slitom->nnz; ++i)
            {
                mask_t s_index = 0;
                mask_t f_index = 0;
                mask_t c_index = 0;

                slitom->values[i] = sptensor->values[i];

                for (int c = 0; c < slitom->cnmodes; ++c)
                {

                    c_index |= common::pdep(static_cast<mask_t>(sptensor->indices[slitom->cpos[c]][i]), slitom->cmode_masks[c]);
                    // printf("c upper mask: %llu\n", slitom->cmode_masks[c] >> 64);
                    // printf("c lower mask: %llu\n", slitom->cmode_masks[c] & 0xffffffffffffffff);
                    // printf("c %d: %llu\n", i, common::pdep(static_cast<mask_t>(sptensor->indices[slitom->cpos[c]][i]), slitom->cmode_masks[c]));
                    // printf("c_index: %llu\n", c_index);
                }
                for (int f = 0; f < slitom->fnmodes; ++f)
                {
                    f_index |= common::pdep(static_cast<mask_t>(sptensor->indices[slitom->fpos[f]][i]), slitom->fmode_masks[f]);
                    // printf("f upper mask: %llu\n", slitom->fmode_masks[f] >> 64);
                    // printf("f lower mask: %llu\n", slitom->fmode_masks[f] & 0xffffffffffffffff);
                }

                s_index = (c_index << (slitom->nfbits)) | f_index;

                s_index = (f_index << (slitom->ncbits)) | c_index;
                slitom->uindices[i] = common::uhalf(s_index);
                slitom->indices[i] = common::lhalf(s_index);
            }
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < slitom->nnz; ++i)
            {
                mask_t s_index = 0;
                mask_t f_index = 0;
                mask_t c_index = 0;

                slitom->values[i] = sptensor->values[i];

                for (int c = 0; c < slitom->cnmodes; ++c)
                {
                    c_index |= common::pdep(sptensor->indices[slitom->cpos[c]][i], SLITOM_CMASKS[c]);
                }
                for (int f = 0; f < slitom->fnmodes; ++f)
                {

                    f_index |= common::pdep(sptensor->indices[slitom->fpos[f]][i], SLITOM_FMASKS[f]);
                }

                unsigned long long ylow = slitom->cmask & 0xffffffffffffffff;
                unsigned long long xlow = c_index & 0xffffffffffffffff;
                int shift = __builtin_popcountll(ylow);

                c_index = common::pext(c_index, slitom->cmask);
                // c_index = ((LIT)(pext((unsigned long long)(c_index >> 64), (unsigned long long)(slitom->cmask >> 64))) << shift) | pext(xlow, ylow); // TODO: change to function

                f_index = common::pext(f_index, slitom->fmask);

                s_index = (c_index << (slitom->nfbits)) | f_index;

                slitom->uindices[i] = s_index >> 64;
                slitom->indices[i] = common::lhalf(s_index);
            }
        }
        timer->stop();
        timer->printElapsed("ConvertTensor_extra");
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::sort_tensor_cpu(slitom_t *slitom, Timer *timer)
    {
        uint64_t nnz = slitom->nnz;
        timer->start();

        if (slitom->nbits <= 64)
        {
            typename TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::SPair_64 *st_pair;

            st_pair = gsparc::allocate<SPair_64>(nnz);

#pragma omp parallel for
            for (uint64_t i = 0; i < nnz; i++)
            {
                st_pair[i].idx = slitom->indices[i];
                st_pair[i].val = slitom->values[i];
            }
            std::sort(std::execution::par, st_pair, st_pair + nnz, [](auto &a, auto &b)
                      { return a.idx < b.idx; });

#pragma omp parallel for
            for (uint64_t i = 0; i < nnz; i++)
            {
                slitom->indices[i] = st_pair[i].idx;
                slitom->values[i] = st_pair[i].val;
            }
            gsparc::deallocate(st_pair);
        }
        else
        {
            typename TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::SPair_128 *st_pair;
            st_pair = gsparc::allocate<SPair_128>(nnz);

#pragma omp parallel for
            for (uint64_t i = 0; i < nnz; i++)
            {
                st_pair[i].uidx = slitom->uindices[i];
                st_pair[i].idx = slitom->indices[i];
                st_pair[i].val = slitom->values[i];
            }

            std::sort(std::execution::par, st_pair, st_pair + nnz, [](auto &a, auto &b)
                      { if (a.uidx != b.uidx)
                            return a.uidx < b.uidx;
                        return a.idx < b.idx; });

#pragma omp parallel for
            for (uint64_t i = 0; i < nnz; i++)
            {
                slitom->uindices[i] = st_pair[i].uidx;
                slitom->indices[i] = st_pair[i].idx;
                slitom->values[i] = st_pair[i].val;
            }
            gsparc::deallocate(st_pair);
            // printf("slitom->uindices[0]: %llu\n", slitom->uindices[0]);
            // printf("slitom->indices[0]: %llu\n", slitom->indices[0]);
            // printf("slitom->values[0]: %f\n", slitom->values[0]);
            // printf("slitom->uindices[nnz-1]: %llu\n", slitom->uindices[nnz - 1]);
            // printf("slitom->indices[nnz-1]: %llu\n", slitom->indices[nnz - 1]);
            // printf("slitom->values[nnz-1]: %f\n", slitom->values[nnz - 1]);
        }
        timer->stop();
        timer->printElapsed("SORT SLITOM (CPU)");
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::sort_tensors_gpu_64(slitom_t *SX, slitom_t *SY, memorypool_t **pools, Timer *timer)
    {
        sort_64(SX, SY, pools, gpu_count, timer);
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::sort_tensors_gpu_128(slitom_t *SX, slitom_t *SY, memorypool_t **pools, Timer *timer)
    {

        gsparc::sort_128<slitom_t, index_t>(SX, SY, pools, gpu_count, timer);
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::SortSlitomXY(slitom_t *SX, slitom_t *SY, memorypool_t **pools, Timer *timer)
    {
        uint64_t nnzX = SX->nnz;
        uint64_t nnzY = SY->nnz;

        SX->d_uindices = gsparc::allocate<ulindex_t *>(gpu_count);
        SX->d_indices = gsparc::allocate<lindex_t *>(gpu_count);
        SX->d_values = gsparc::allocate<value_t *>(gpu_count);
        SY->d_uindices = gsparc::allocate<ulindex_t *>(gpu_count);
        SY->d_indices = gsparc::allocate<lindex_t *>(gpu_count);
        SY->d_values = gsparc::allocate<value_t *>(gpu_count);
        if (SX->sort_gpu == 1)
        { /* GPU SORT */
            printf("GPU SORT\n");

            printf("SX->sort_gpu: %d\n", SX->sort_gpu);
            if (SX->nbits <= 64)
            {
                sort_tensors_gpu_64(SX, SY, pools, timer);
            }
            else
            {
                sort_tensors_gpu_128(SX, SY, pools, timer);
            }
        }
        else
        {
            printf(("CPU SORT\n"));
            /* CPU SORT */
            sort_tensor_cpu(SX, timer);
            sort_tensor_cpu(SY, timer);
        }

        return;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::Partition_128(slitom_t *slitom)
    {
        uint64_t prtn_num = slitom->nprtn;
        uint64_t nnz = slitom->nnz;
        int nshiftbit = 0;
        if (slitom->input_order == 1 || (slitom->input_order == 2 && slitom->nprtn > 1))
            nshiftbit = slitom->ncbits;
        else
            nshiftbit = slitom->nfbits;

        uint64_t prtn_size = nnz / prtn_num;
        uint64_t remainder = nnz % prtn_num;
        std::vector<uint64_t> prtn_idx;
        prtn_idx.push_back(0);

        uint64_t start = 0;
        uint64_t end = 0;
        for (uint64_t i = 0; i < prtn_num; i++)
        {
            start = end;
            uint64_t initial_end = std::min(start + prtn_size, nnz); // 최초 end 저장

            if (i < remainder)
                initial_end++;

            end = initial_end;

            // end를 확장하면서 slitom->idx[i] >> nshiftbit 값이 동일한지 확인
            while (end < nnz)
            {
                __uint128_t current_index = static_cast<__uint128_t>(slitom->uindices[end]) << 64 | slitom->indices[end];
                __uint128_t previous_index = static_cast<__uint128_t>(slitom->uindices[end - 1]) << 64 | slitom->indices[end - 1];
                if ((current_index >> nshiftbit) == (previous_index >> nshiftbit))
                    end++;
                else
                    break;
            }

            // // 최대 블록 크기 초과 시, 초기 end에서 분할 위치를 다시 찾음
            if (end - start > this->max_block_size)
            {
                uint64_t temp_end = end;
                end = initial_end; // 초기 end로 되돌림
                __uint128_t current_index = static_cast<__uint128_t>(slitom->uindices[end]) << 64 | slitom->indices[end];
                __uint128_t previous_index = static_cast<__uint128_t>(slitom->uindices[end - 1]) << 64 | slitom->indices[end - 1];

                while (end > start && (current_index >> nshiftbit) == (previous_index >> nshiftbit))
                {

                    end--;
                }
                if (end == start)
                    end = temp_end;

                prtn_idx.push_back(end);
                start = end; // 다음 블록의 시작점 조정
            }
            else
            {
                prtn_idx.push_back(end);
                if (end == nnz)
                {
                    break;
                }
            }
        }
        // 마지막 값이 nnz가 아닐 경우 추가
        if (!prtn_idx.empty() && prtn_idx.back() != nnz)
        {
            prtn_idx.push_back(nnz);
        }

        // prtn_num 업데이트
        prtn_num = prtn_idx.size() - 1;
        this->prtn_num = prtn_num;

        slitom->prtn_idx = gsparc::allocate<uint64_t>(prtn_num + 1);
        for (uint64_t i = 0; i < prtn_num + 1; i++)
        {
            slitom->prtn_idx[i] = prtn_idx[i];
        }

        slitom->prtn_coord = gsparc::allocate<lindex_t>(prtn_num);

        slitom->nprtn = prtn_num;

        uint64_t max_nnz = 0;

        for (uint64_t i = 0; i < (this->prtn_num); i++)
        {
            uint64_t start = slitom->prtn_idx[i];
            uint64_t end = slitom->prtn_idx[i + 1];
            printf("slitom->prtn_idx[%llu]: %llu, slitom->prtn_idx[%llu]: %llu, nnz: %llu\n", i, start, i + 1, end, end - start);
        }

#pragma omp parallel for schedule(static) reduction(max : max_nnz)
        for (uint64_t i = 0; i < (this->prtn_num); i++)
        {
            uint64_t start = slitom->prtn_idx[i];
            uint64_t end = slitom->prtn_idx[i + 1];
            if (end - start > max_nnz)
            {
                max_nnz = end - start;
            }
        }
        slitom->max_prtn_size = max_nnz;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::Partition(slitom_t *slitom)
    {
        uint64_t prtn_num = slitom->nprtn;
        uint64_t nnz = slitom->nnz;
        int nshiftbit = 0;
        if (slitom->input_order == 1 || (slitom->input_order == 2 && slitom->nprtn > 1))
            nshiftbit = slitom->ncbits;
        else
            nshiftbit = slitom->nfbits;

        uint64_t prtn_size = nnz / prtn_num;
        uint64_t remainder = nnz % prtn_num;
        std::vector<uint64_t> prtn_idx;
        prtn_idx.push_back(0);
        if (prtn_num == 1)
        {
            prtn_idx.push_back(nnz);
            slitom->prtn_idx = gsparc::allocate<uint64_t>(2);
            slitom->prtn_idx[0] = 0;
            slitom->prtn_idx[1] = nnz;
            slitom->nprtn = 1;
            return;
        }

        uint64_t start = 0;
        uint64_t end = 0;
        for (uint64_t i = 0; i < prtn_num; i++)
        {
            start = end;
            uint64_t initial_end = std::min(start + prtn_size, nnz); // 최초 end 저장

            if (i < remainder)
                initial_end++;

            end = initial_end;
            if (end > nnz)
                end = nnz;

            // end를 확장하면서 slitom->idx[i] >> nshiftbit 값이 동일한지 확인
            while (end < nnz)
            {
                if ((slitom->indices[end] >> nshiftbit) == (slitom->indices[end - 1] >> nshiftbit))
                    end++;
                else
                    break;
            }

            // // 최대 블록 크기 초과 시, 초기 end에서 분할 위치를 다시 찾음
            // if (end - start > this->max_block_size)
            // {
            //     end = initial_end; // 초기 end로 되돌림

            //     while (end > start && slitom->indices[end] >> nshiftbit == slitom->indices[end - 1] >> nshiftbit)
            //     {

            //         end--;
            //     }

            //     prtn_idx.push_back(end);
            //     start = end; // 다음 블록의 시작점 조정
            // }
            // else
            // {
            prtn_idx.push_back(end);
            printf("end: %llu\n", end);
            if (end == nnz)
            {
                break;
            }
            // }
        }

        // 마지막 값이 nnz가 아닐 경우 추가
        if (!prtn_idx.empty() && prtn_idx.back() != nnz)
        {
            printf("last end: %llu\n", nnz);
            prtn_idx.push_back(nnz);
        }

        // prtn_num 업데이트
        prtn_num = prtn_idx.size() - 1;
        this->prtn_num = prtn_num;
        printf("prtn_num: %llu\n", prtn_num);

        slitom->prtn_idx = gsparc::allocate<uint64_t>(prtn_num + 1);
        for (uint64_t i = 0; i < prtn_num + 1; i++)
        {
            slitom->prtn_idx[i] = prtn_idx[i];
        }

        slitom->prtn_coord = gsparc::allocate<lindex_t>(prtn_num);

        slitom->nprtn = prtn_num;

        uint64_t max_nnz = 0;
        printf("partitioned\n");

        for (uint64_t i = 0; i < (this->prtn_num); i++)
        {
            uint64_t start = slitom->prtn_idx[i];
            uint64_t end = slitom->prtn_idx[i + 1];
            printf("start: %llu, end: %llu, nnz: %llu\n", start, end, end - start);
        }

#pragma omp parallel for schedule(static) reduction(max : max_nnz)
        for (uint64_t i = 0; i < (this->prtn_num); i++)
        {
            uint64_t start = slitom->prtn_idx[i];
            uint64_t end = slitom->prtn_idx[i + 1];
            if (end - start > max_nnz)
            {
                max_nnz = end - start;
            }
        }
        slitom->max_prtn_size = max_nnz;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::convert_and_sort_64(slitom_t *slitom, CudaMemoryPool **pools)
    {
        CudaMemoryPool *pool = pools[0];
        printf("in convert_and_sort_64\n");
        pools[0]->printFree();
#pragma omp parallel for
        for (uint64_t i = 0; i < slitom->nnz; i++)
        {
            lindex_t free_index = slitom->indices[i] >> slitom->ncbits;
            lindex_t contract_index = slitom->indices[i] & ((static_cast<lindex_t>(1) << slitom->ncbits) - 1);
            slitom->indices[i] = (contract_index << slitom->nfbits) | free_index;
        }
        printf("convert finished\n");
        for (int p = 0; p < slitom->nprtn; ++p)
        {
            index_t start = slitom->prtn_idx[p];
            index_t end = slitom->prtn_idx[p + 1];
            uint64_t nnz = end - start;
            if (nnz == 0)
                continue;
            printf("start: %llu, end: %llu, nnz: %llu\n", start, end, nnz);
            lindex_t *d_indices = pool->allocate<lindex_t>(nnz);
            value_t *d_values = pool->allocate<value_t>(nnz);

            common::cuda::h2dcpy(d_indices, slitom->indices + start, sizeof(lindex_t) * nnz);
            common::cuda::h2dcpy(d_values, slitom->values + start, sizeof(value_t) * nnz);
            // Sort the indices and values
            char *d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_indices, d_indices, d_values, d_values, nnz);
            d_temp_storage = pool->allocate<char>(temp_storage_bytes);

            // printf("%d/ %d, d_temp_storage bytes: %s\n", p + 1, slitom->nprtn, common::byteToString(temp_storage_bytes));
            // pool->printFree();
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_indices, d_indices, d_values, d_values, nnz);

            common::cuda::d2hcpy(slitom->indices + start, d_indices, sizeof(lindex_t) * nnz);
            common::cuda::d2hcpy(slitom->values + start, d_values, sizeof(value_t) * nnz);

            pool->deallocate<lindex_t>(d_indices, nnz);
            pool->deallocate<value_t>(d_values, nnz);
            pool->deallocate<char>(d_temp_storage, temp_storage_bytes);
        }
        printf("finished convert_and_sort_64\n");

        return;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::convert_and_sort_128(slitom_t *slitom, CudaMemoryPool **pools)
    {

        CudaMemoryPool *pool = pools[0];

        pools[0]->printFree();

#pragma omp parallel for
        for (uint64_t i = 0; i < slitom->nnz; i++)
        {
            __uint128_t index = (static_cast<__uint128_t>(slitom->uindices[i]) << 64) | slitom->indices[i];
            __uint128_t free_index = index >> slitom->ncbits;
            __uint128_t contract_index = index & ((static_cast<__uint128_t>(1) << slitom->ncbits) - 1);
            index = (contract_index << slitom->nfbits) | free_index;
            slitom->uindices[i] = common::uhalf(index);
            slitom->indices[i] = common::lhalf(index);
        }

        for (int p = 0; p < slitom->nprtn; ++p)
        {
            index_t start = slitom->prtn_idx[p];
            index_t end = slitom->prtn_idx[p + 1];
            uint64_t nnz = end - start;
            if (nnz == 0)
                continue;

                ulindex_t *d_uindices_X = pool->allocate<ulindex_t>(nnz);
            lindex_t *d_indices_X = pool->allocate<lindex_t>(nnz);
            value_t *d_values_X = pool->allocate<value_t>(nnz);

            common::cuda::h2dcpy(d_uindices_X, slitom->uindices + start, nnz * sizeof(ulindex_t));
            common::cuda::h2dcpy(d_indices_X, slitom->indices + start, nnz * sizeof(lindex_t));
            common::cuda::h2dcpy(d_values_X, slitom->values + start, nnz * sizeof(value_t));

            gsparc::CudaMemoryPoolAllocator<index_t> alloc(pool);
            auto exec_policy = thrust::cuda::par(alloc);

            // ==== X 배열 처리 ====
            index_t *sorted_idx_X = pool->allocate<index_t>(nnz);
            auto idx_X_begin = thrust::device_pointer_cast(sorted_idx_X);
            auto idx_X_end = idx_X_begin + nnz;
            thrust::sequence(exec_policy, idx_X_begin, idx_X_end);

            // X 배열에 대해 비교 연산자 생성 후 인덱스 정렬
            IndirectComparator<index_t, ulindex_t, lindex_t> comp_X{d_uindices_X, d_indices_X};
            thrust::sort(exec_policy, idx_X_begin, idx_X_end, comp_X);

            // gather를 위한 임시 버퍼 할당
            ulindex_t *d_uindices_temp_X = pool->allocate<ulindex_t>(nnz);
            lindex_t *d_indices_temp_X = pool->allocate<lindex_t>(nnz);
            value_t *d_values_temp_X = pool->allocate<value_t>(nnz);

            auto d_uindices_ptr_X = thrust::device_pointer_cast(d_uindices_X);
            auto d_indices_ptr_X = thrust::device_pointer_cast(d_indices_X);
            auto d_values_ptr_X = thrust::device_pointer_cast(d_values_X);

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
            common::cuda::d2hcpy(slitom->uindices, d_uindices_temp_X, nnz * sizeof(ulindex_t));
            common::cuda::d2hcpy(slitom->indices, d_indices_temp_X, nnz * sizeof(lindex_t));
            common::cuda::d2hcpy(slitom->values, d_values_temp_X, nnz * sizeof(value_t));

            // 메모리 해제
            pool->deallocate<ulindex_t>(d_uindices_X, nnz);
            pool->deallocate<lindex_t>(d_indices_X, nnz);
            pool->deallocate<value_t>(d_values_X, nnz);

            pool->deallocate<index_t>(sorted_idx_X, nnz);

            pool->deallocate<ulindex_t>(d_uindices_temp_X, nnz);
            pool->deallocate<lindex_t>(d_indices_temp_X, nnz);
            pool->deallocate<value_t>(d_values_temp_X, nnz);
        }
        printf("finished convert_and_sort_128\n");

        return;
    }

} // namespace gsparc
