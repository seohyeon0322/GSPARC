

#include "gsparc/slitom.hpp"
#include "gsparc/helper.hpp"
#include "common/cuda_helper.hpp"

namespace gsparc
{
    SLITOM_TEMPLATE
    SLITOM<SLITOM_TEMPLATE_ARGS>::SLITOM()
        : sort_gpu(false), coo_cmodes(nullptr), blco_ckeys(nullptr),
          blco_cmode_offsets(nullptr), blco_cmode_widths(nullptr) {}

    SLITOM_TEMPLATE
    SLITOM<SLITOM_TEMPLATE_ARGS>::SLITOM(int input_order)
        : input_order(input_order), sort_gpu(false), coo_cmodes(nullptr), blco_ckeys(nullptr),
          blco_cmode_offsets(nullptr), blco_cmode_widths(nullptr) {}

    SLITOM_TEMPLATE
    SLITOM<SLITOM_TEMPLATE_ARGS>::SLITOM(int input_order, bool sort_gpu)
        : input_order(input_order), sort_gpu(sort_gpu), coo_cmodes(nullptr), blco_ckeys(nullptr),
          blco_cmode_offsets(nullptr), blco_cmode_widths(nullptr) {}

    // SLITOM_TEMPLATE
    // SLITOM<SLITOM_TEMPLATE_ARGS>::SLITOM(unsigned short nmodes, lindex_t *dims, uint64_t nnz, int input_number)
    //     : nmodes(nmodes), nnz(nnz), input_order(input_number)
    // {
    //     indices = gsparc::allocate<lindex_t *>(nmodes);
    //     values = gsparc::allocate<value_t>(nnz);
    //     if (nbtis > 64)
    //         uindices = gsparc::allocate<ulindex_t *>(nmodes);
    // }

    SLITOM_TEMPLATE
    SLITOM<SLITOM_TEMPLATE_ARGS>::~SLITOM()
    {
        common::cuda::pinned_free(indices);
        common::cuda::pinned_free(values);
        if (nbits > 64)
            common::cuda::pinned_free(uindices);
        gsparc::deallocate(mode_bits);
        gsparc::deallocate(cpos);
        gsparc::deallocate(fpos);
        gsparc::deallocate(cmode_masks);
        gsparc::deallocate(fmode_masks);
        gsparc::deallocate(mode_masks);
        gsparc::deallocate(d_indices);
        gsparc::deallocate(d_values);
        gsparc::deallocate(d_uindices);
        if (coo_cmodes) {
            for (int c = 0; c < cnmodes; ++c)
                common::cuda::pinned_free(coo_cmodes[c]);
            delete[] coo_cmodes;
        }
        if (blco_ckeys)
            common::cuda::pinned_free(blco_ckeys);
        delete[] blco_cmode_offsets;
        delete[] blco_cmode_widths;
    }
}