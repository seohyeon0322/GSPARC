#ifndef SLITOM_HPP_
#define SLITOM_HPP_

#include <iostream>

namespace gsparc
{
#define SLITOM_TEMPLATE template <typename MaskType, typename ULIndexType, typename LIndexType, typename ValueType>
#define SLITOM_TEMPLATE_ARGS MaskType, ULIndexType, LIndexType, ValueType
    typedef enum PackOrder_
    {
        LSB_FIRST,
        MSB_FIRST
    } PackOrder;
    typedef enum ModeOrder_
    {
        SHORT_FIRST,
        LONG_FIRST,
        NATURAL
    } ModeOrder;
    struct MPair
    {
        int mode;
        int bits;
    };

    SLITOM_TEMPLATE
    class SLITOM
    {
    public:
        using mask_t = MaskType;
        using ulindex_t = ULIndexType;
        using lindex_t = LIndexType;
        using value_t = ValueType;

    public:
        SLITOM();
        SLITOM(int input_order);
        SLITOM(int input_order, bool sort_gpu);

        // SLITOM(unsigned short nmodes, lindex_t *dims, uint64_t nnz, int input_number);
        ~SLITOM();

    public:
        int input_order;
        MPair *mode_bits;

        /* Metadata */
        lindex_t *dims;
        uint64_t nnz;

        unsigned short nmodes;
        unsigned short fnmodes;
        unsigned short cnmodes;

        /* Bits */
        int nfbits;
        int ncbits;
        int nbits;

        /* mode */
        int *cpos;
        int *fpos;

        /* Mode mask */
        mask_t *mode_masks;
        mask_t *cmode_masks;
        mask_t *fmode_masks;

        /*Mask per modes*/
        mask_t fmask;
        mask_t cmask;
        mask_t slitom_mask;
        

        /* Data */
        ulindex_t *uindices; /* For large-scale tensor */
        lindex_t *indices;
        value_t *values;

        /* Partition */
        int nprtn;
        bool sort_gpu;
        uint64_t *prtn_idx;
        lindex_t *prtn_coord;
        uint64_t max_prtn_size;

        /* Device pointer */
        ulindex_t **d_uindices;
        lindex_t **d_indices;
        value_t **d_values;
    };
} // namespace gsparc

#include "gsparc/slitom.tpp"
#endif