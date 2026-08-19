
#include <omp.h>

#include <iostream>
#include <algorithm>
#include <numeric>
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
        uint64_t total_storage = (x_storage + y_storage) * 2 + mPosIdx;
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
        // GPU sort processes X and Y sequentially, so only one tensor needs to fit at a time
        // Need: in(indices+values) + out(indices+values) + temp for one tensor
        uint64_t max_single = std::max(x_storage, y_storage);
        uint64_t max_nnz = std::max(nnz_count_x, nnz_count_y);
        uint64_t sort_storage = max_single * 2; // in + out buffers for larger tensor
        sort_storage += sizeof(lindex_t) * max_nnz; // CUB temp storage (approx)
        if (nbits > 64)
        {
            sort_storage += sizeof(ulindex_t) * max_nnz * 2; // uindices in+out
            sort_storage += sizeof(index_t) * max_nnz * 2;   // perm arrays
            sort_storage += sizeof(lindex_t) * max_nnz;      // gather workspace
        }
        bool sort_gpu = (sort_storage < poolSize) ? true : false;

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
            // Parallel LSD radix sort on 128-bit keys (uidx:idx)
            const int RADIX_BITS = 8;
            const int RADIX = 1 << RADIX_BITS;
            // Sort lower 64 bits first, then upper bits
            int lower_passes = 8;  // full 64 bits of idx
            int upper_bits = slitom->nbits - 64;
            int upper_passes = (upper_bits + RADIX_BITS - 1) / RADIX_BITS;
            int total_passes = lower_passes + upper_passes;

            lindex_t *idx_in = slitom->indices;
            ulindex_t *uidx_in = slitom->uindices;
            value_t *vals_in = slitom->values;
            lindex_t *idx_out = gsparc::allocate<lindex_t>(nnz);
            ulindex_t *uidx_out = gsparc::allocate<ulindex_t>(nnz);
            value_t *vals_out = gsparc::allocate<value_t>(nnz);

            int nthreads = omp_get_max_threads();
            std::vector<std::vector<uint64_t>> thread_hist(nthreads, std::vector<uint64_t>(RADIX, 0));

            for (int pass = 0; pass < total_passes; ++pass)
            {
                // Determine which key and shift to use
                bool use_upper = (pass >= lower_passes);
                int shift = use_upper ? (pass - lower_passes) * RADIX_BITS : pass * RADIX_BITS;

                // 1. Histogram
                for (auto &h : thread_hist)
                    std::fill(h.begin(), h.end(), 0);

#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    auto &hist = thread_hist[tid];
#pragma omp for schedule(static)
                    for (uint64_t i = 0; i < nnz; i++)
                    {
                        uint8_t digit = use_upper
                            ? ((uidx_in[i] >> shift) & (RADIX - 1))
                            : ((idx_in[i] >> shift) & (RADIX - 1));
                        hist[digit]++;
                    }
                }

                // 2. Global prefix sum
                std::vector<uint64_t> global_offset(RADIX, 0);
                for (int d = 0; d < RADIX; d++)
                {
                    for (int t = 0; t < nthreads; t++)
                    {
                        uint64_t cnt = thread_hist[t][d];
                        thread_hist[t][d] = global_offset[d];
                        global_offset[d] += cnt;
                    }
                }
                uint64_t sum = 0;
                for (int d = 0; d < RADIX; d++)
                {
                    uint64_t tmp = global_offset[d];
                    global_offset[d] = sum;
                    sum += tmp;
                }
                for (int t = 0; t < nthreads; t++)
                    for (int d = 0; d < RADIX; d++)
                        thread_hist[t][d] += global_offset[d];

                // 3. Scatter
#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    auto &offsets = thread_hist[tid];
#pragma omp for schedule(static)
                    for (uint64_t i = 0; i < nnz; i++)
                    {
                        uint8_t digit = use_upper
                            ? ((uidx_in[i] >> shift) & (RADIX - 1))
                            : ((idx_in[i] >> shift) & (RADIX - 1));
                        uint64_t pos = offsets[digit]++;
                        idx_out[pos] = idx_in[i];
                        uidx_out[pos] = uidx_in[i];
                        vals_out[pos] = vals_in[i];
                    }
                }

                // Swap buffers
                std::swap(idx_in, idx_out);
                std::swap(uidx_in, uidx_out);
                std::swap(vals_in, vals_out);
            }

            // Ensure result is in slitom arrays
            if (idx_in != slitom->indices)
            {
#pragma omp parallel for schedule(static)
                for (uint64_t i = 0; i < nnz; i++)
                {
                    slitom->indices[i] = idx_in[i];
                    slitom->uindices[i] = uidx_in[i];
                    slitom->values[i] = vals_in[i];
                }
                gsparc::deallocate(idx_in);
                gsparc::deallocate(uidx_in);
                gsparc::deallocate(vals_in);
            }
            else
            {
                gsparc::deallocate(idx_out);
                gsparc::deallocate(uidx_out);
                gsparc::deallocate(vals_out);
            }
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

            // Binary search for last element with same mode group at boundary
            auto get_mode_128 = [&](uint64_t idx) -> __uint128_t {
                return (static_cast<__uint128_t>(slitom->uindices[idx]) << 64 | slitom->indices[idx]) >> nshiftbit;
            };
            if (end < nnz)
            {
                __uint128_t current_mode = get_mode_128(end - 1);
                uint64_t lo = end, hi = nnz;
                while (lo < hi) {
                    uint64_t mid = lo + (hi - lo) / 2;
                    if (get_mode_128(mid) == current_mode)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                end = lo;
            }

            if (end - start > this->max_block_size)
            {
                uint64_t temp_end = end;
                end = initial_end;

                // Binary search backward: find first element of end's mode group
                {
                    __uint128_t end_mode = get_mode_128(end);
                    uint64_t lo = start, hi = end;
                    while (lo < hi) {
                        uint64_t mid = lo + (hi - lo) / 2;
                        if (get_mode_128(mid) < end_mode)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    end = lo;
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

            // Binary search for last element with same mode group at boundary
            if (end < nnz)
            {
                lindex_t current_mode = slitom->indices[end - 1] >> nshiftbit;
                uint64_t lo = end, hi = nnz;
                // Find first index where mode changes (upper bound)
                while (lo < hi) {
                    uint64_t mid = lo + (hi - lo) / 2;
                    if ((slitom->indices[mid] >> nshiftbit) == current_mode)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                end = lo;
            }

            // 최대 블록 크기 초과 시, 초기 end에서 분할 위치를 다시 찾음
            if (end - start > this->max_block_size)
            {
                uint64_t temp_end = end;
                end = initial_end; // 초기 end로 되돌림

                // Binary search backward: find first index where mode differs from end
                {
                    lindex_t end_mode = slitom->indices[end] >> nshiftbit;
                    uint64_t lo = start, hi = end;
                    while (lo < hi) {
                        uint64_t mid = lo + (hi - lo) / 2;
                        if ((slitom->indices[mid] >> nshiftbit) < end_mode)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    end = lo; // first element of end's mode group
                }
                if (end == start)
                    end = temp_end;

                prtn_idx.push_back(end);
                printf("end: %llu\n", end);
                start = end; // 다음 블록의 시작점 조정
            }
            else
            {
                prtn_idx.push_back(end);
                printf("end: %llu\n", end);
                if (end == nnz)
                {
                    break;
                }
            }
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
        // Convert [f|c] -> [c|f] bit layout
#pragma omp parallel for
        for (uint64_t i = 0; i < slitom->nnz; i++)
        {
            lindex_t free_index = slitom->indices[i] >> slitom->ncbits;
            lindex_t contract_index = slitom->indices[i] & ((static_cast<lindex_t>(1) << slitom->ncbits) - 1);
            slitom->indices[i] = (contract_index << slitom->nfbits) | free_index;
        }

        CudaMemoryPool *pool = pools[0];

        for (int p = 0; p < slitom->nprtn; ++p)
        {
            index_t start = slitom->prtn_idx[p];
            index_t end = slitom->prtn_idx[p + 1];
            uint64_t nnz = end - start;
            if (nnz == 0)
                continue;

            // Check if GPU sort fits: need 4*nnz*8 + temp (~nnz*4) bytes
            size_t gpu_need = nnz * (4 * sizeof(lindex_t) + 4);
            if (pool->is_available(gpu_need))
            {
                // GPU CUB RadixSort per partition (pool-allocated)
                lindex_t *d_keys_in = pool->allocate<lindex_t>(nnz);
                lindex_t *d_keys_out = pool->allocate<lindex_t>(nnz);
                value_t *d_vals_in = pool->allocate<value_t>(nnz);
                value_t *d_vals_out = pool->allocate<value_t>(nnz);

                common::cuda::h2dcpy(d_keys_in, slitom->indices + start, nnz * sizeof(lindex_t));
                common::cuda::h2dcpy(d_vals_in, slitom->values + start, nnz * sizeof(value_t));

                cudaStream_t sort_stream;
                checkCudaErrors(cudaStreamCreate(&sort_stream));

                void *d_temp = nullptr;
                size_t temp_bytes = 0;
                cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                    d_keys_in, d_keys_out, d_vals_in, d_vals_out,
                    nnz, 0, slitom->nbits, sort_stream);
                d_temp = pool->allocate<void>(temp_bytes);
                cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                    d_keys_in, d_keys_out, d_vals_in, d_vals_out,
                    nnz, 0, slitom->nbits, sort_stream);

                common::cuda::stream_sync(sort_stream);
                checkCudaErrors(cudaStreamDestroy(sort_stream));

                common::cuda::d2hcpy(slitom->indices + start, d_keys_out, nnz * sizeof(lindex_t));
                common::cuda::d2hcpy(slitom->values + start, d_vals_out, nnz * sizeof(value_t));

                pool->deallocate(d_temp, temp_bytes);
                pool->deallocate<value_t>(d_vals_out, nnz);
                pool->deallocate<value_t>(d_vals_in, nnz);
                pool->deallocate<lindex_t>(d_keys_out, nnz);
                pool->deallocate<lindex_t>(d_keys_in, nnz);
            }
            else
            {
                // CPU fallback when partition doesn't fit in GPU memory
                fprintf(stderr, "[convert_and_sort_64] partition %d: GPU OOM (need %s), using CPU sort\n",
                        p, common::byteToString(gpu_need));
                SPair_64 *st_pair = gsparc::allocate<SPair_64>(nnz);
#pragma omp parallel for
                for (uint64_t i = 0; i < nnz; i++)
                {
                    st_pair[i].idx = slitom->indices[start + i];
                    st_pair[i].val = slitom->values[start + i];
                }
                std::sort(std::execution::par, st_pair, st_pair + nnz, [](auto &a, auto &b)
                          { return a.idx < b.idx; });
#pragma omp parallel for
                for (uint64_t i = 0; i < nnz; i++)
                {
                    slitom->indices[start + i] = st_pair[i].idx;
                    slitom->values[start + i] = st_pair[i].val;
                }
                gsparc::deallocate(st_pair);
            }
        }
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

            // 2-pass CUB RadixSort (replaces Thrust indirect sort)
            cudaStream_t sort_stream;
            checkCudaErrors(cudaStreamCreate(&sort_stream));
            gsparc::radix_sort_128_single<index_t>(d_uindices_X, d_indices_X, d_values_X,
                                                    nnz, slitom->nbits, pool, sort_stream);
            common::cuda::stream_sync(sort_stream);
            checkCudaErrors(cudaStreamDestroy(sort_stream));

            // 결과를 host 메모리로 복사
            common::cuda::d2hcpy(slitom->uindices + start, d_uindices_X, nnz * sizeof(ulindex_t));
            common::cuda::d2hcpy(slitom->indices + start, d_indices_X, nnz * sizeof(lindex_t));
            common::cuda::d2hcpy(slitom->values + start, d_values_X, nnz * sizeof(value_t));

            // 메모리 해제
            pool->deallocate<ulindex_t>(d_uindices_X, nnz);
            pool->deallocate<lindex_t>(d_indices_X, nnz);
            pool->deallocate<value_t>(d_values_X, nnz);
        }
        printf("finished convert_and_sort_128\n");

        return;
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::ExtractCOOCmodes(slitom_t *slitom)
    {
        uint64_t nnz = slitom->nnz;
        int cnmodes = slitom->cnmodes;

        slitom->coo_cmodes = new uint32_t*[cnmodes];
        for (int c = 0; c < cnmodes; ++c) {
            slitom->coo_cmodes[c] = static_cast<uint32_t*>(
                common::cuda::pinned_malloc(sizeof(uint32_t) * nnz));
        }

        // Extract per-mode contraction indices from packed SLITOM indices
        // input_order==1: packed = [f_index(MSB) | c_index(LSB)] → c_index = packed & cmask
        // input_order==2: packed = [c_index(MSB) | f_index(LSB)] → c_index = packed >> nfbits
        // Then pext(c_index, cmode_masks[c]) extracts each mode value

        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < nnz; ++i) {
            mask_t packed = static_cast<mask_t>(slitom->indices[i]);
            if (slitom->nbits > 64 && slitom->uindices) {
                packed = (static_cast<mask_t>(slitom->uindices[i]) << 64) | slitom->indices[i];
            }
            // Extract c_index portion based on bit layout
            mask_t c_index;
            if (slitom->input_order == 1 || (slitom->input_order == 2 && slitom->nprtn > 1)) {
                c_index = packed & ((static_cast<mask_t>(1) << slitom->ncbits) - 1);
            } else {
                c_index = packed >> slitom->nfbits;
            }
            for (int c = 0; c < cnmodes; ++c) {
                slitom->coo_cmodes[c][i] = static_cast<uint32_t>(
                    common::pext(c_index, slitom->cmode_masks[c]));
            }
        }
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::ExtractBLCOKeys(slitom_t *slitom)
    {
        uint64_t nnz = slitom->nnz;
        int nmodes = slitom->nmodes;
        int cnmodes = slitom->cnmodes;
        int fnmodes = slitom->fnmodes;

        slitom->blco_ckeys = static_cast<uint64_t*>(
            common::cuda::pinned_malloc(sizeof(uint64_t) * nnz));

        // Compute per-mode bit widths (ALL modes, in natural order)
        int mode_bits[nmodes];
        for (int m = 0; m < nmodes; ++m) {
            mode_bits[m] = (sizeof(uint64_t) * 8) - common::clz(slitom->dims[m] - 1);
        }

        // Build mapping: for each original mode m, find which c/f index it corresponds to
        // mode_is_c[m] = c index if contraction mode, -1 otherwise
        // mode_is_f[m] = f index if free mode, -1 otherwise
        int mode_is_c[nmodes], mode_is_f[nmodes];
        for (int m = 0; m < nmodes; ++m) { mode_is_c[m] = -1; mode_is_f[m] = -1; }
        for (int c = 0; c < cnmodes; ++c) mode_is_c[slitom->cpos[c]] = c;
        for (int f = 0; f < fnmodes; ++f) mode_is_f[slitom->fpos[f]] = f;

        // Compute contraction mode bit offsets within the BLCO key
        // BLCO key = m_{N-1}(MSB) | ... | m1 | m0(LSB)
        // mode m starts at bit offset = sum(mode_bits[0..m-1])
        slitom->blco_cmode_offsets = new int[cnmodes];
        slitom->blco_cmode_widths = new int[cnmodes];
        for (int c = 0; c < cnmodes; ++c) {
            int m = slitom->cpos[c];
            int offset = 0;
            for (int k = 0; k < m; ++k) offset += mode_bits[k];
            slitom->blco_cmode_offsets[c] = offset;
            slitom->blco_cmode_widths[c] = mode_bits[m];
        }

        // Extract ALL mode values and re-pack in natural mode order (no pdep interleaving)
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < nnz; ++i) {
            mask_t packed = static_cast<mask_t>(slitom->indices[i]);
            if (slitom->nbits > 64 && slitom->uindices) {
                packed = (static_cast<mask_t>(slitom->uindices[i]) << 64) | slitom->indices[i];
            }
            // Extract c_index and f_index portions based on bit layout
            mask_t c_index, f_index;
            if (slitom->input_order == 1 || (slitom->input_order == 2 && slitom->nprtn > 1)) {
                // packed = [f_index(MSB) | c_index(LSB)]
                c_index = packed & ((static_cast<mask_t>(1) << slitom->ncbits) - 1);
                f_index = packed >> slitom->ncbits;
            } else {
                // packed = [c_index(MSB) | f_index(LSB)]
                c_index = packed >> slitom->nfbits;
                f_index = packed & ((static_cast<mask_t>(1) << slitom->nfbits) - 1);
            }

            // Pack all modes in natural order: m0(LSB), m1, ..., m_{N-1}(MSB)
            uint64_t blco_key = 0;
            int shift = 0;
            for (int m = 0; m < nmodes; ++m) {
                uint64_t mode_val;
                if (mode_is_c[m] >= 0) {
                    mode_val = static_cast<uint64_t>(
                        common::pext(c_index, slitom->cmode_masks[mode_is_c[m]]));
                } else {
                    mode_val = static_cast<uint64_t>(
                        common::pext(f_index, slitom->fmode_masks[mode_is_f[m]]));
                }
                blco_key |= (mode_val << shift);
                shift += mode_bits[m];
            }
            slitom->blco_ckeys[i] = blco_key;
        }
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::SortCOOPerPartition(slitom_t *slitom)
    {
        int cnmodes = slitom->cnmodes;
        uint32_t **coo = slitom->coo_cmodes;

        for (int p = 0; p < slitom->nprtn; ++p) {
            uint64_t start = slitom->prtn_idx[p];
            uint64_t end = slitom->prtn_idx[p + 1];
            uint64_t len = end - start;
            if (len <= 1) continue;

            // Create permutation
            std::vector<uint64_t> perm(len);
            std::iota(perm.begin(), perm.end(), 0);

            std::sort(perm.begin(), perm.end(), [&](uint64_t a, uint64_t b) {
                for (int c = 0; c < cnmodes; ++c) {
                    if (coo[c][start + a] != coo[c][start + b])
                        return coo[c][start + a] < coo[c][start + b];
                }
                return false;
            });

            // Apply permutation to all mode arrays
            for (int c = 0; c < cnmodes; ++c) {
                std::vector<uint32_t> tmp(len);
                for (uint64_t i = 0; i < len; ++i)
                    tmp[i] = coo[c][start + perm[i]];
                memcpy(&coo[c][start], tmp.data(), len * sizeof(uint32_t));
            }
        }
    }

    TENSOR_MANAGER_TEMPLATE
    void TensorManager<TENSOR_MANAGER_TEMPLATE_ARGS>::SortBLCOPerPartition(slitom_t *slitom)
    {
        // Sort by full BLCO key (all modes in natural order)
        // Contraction values are NOT contiguous after this sort → binary search on contraction won't work
        for (int p = 0; p < slitom->nprtn; ++p) {
            uint64_t start = slitom->prtn_idx[p];
            uint64_t end = slitom->prtn_idx[p + 1];
            if (end - start <= 1) continue;
            std::sort(slitom->blco_ckeys + start, slitom->blco_ckeys + end);
        }
    }

} // namespace gsparc
