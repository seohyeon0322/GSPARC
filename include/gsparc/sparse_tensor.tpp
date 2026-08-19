#include <iostream>

#include "gsparc/sparse_tensor.hpp"
#include "common/bitops.hpp"

namespace gsparc
{

    TENSOR_TEMPLATE
    SparseTensor<TENSOR_TEMPLATE_ARGS>::SparseTensor(int new_input_number)
        : input_number(new_input_number){ }
        
    TENSOR_TEMPLATE
    SparseTensor<TENSOR_TEMPLATE_ARGS>::SparseTensor(unsigned short new_order, index_t *new_dims, uint64_t new_nnz, int new_input_number)
        : order(new_order), nnz(new_nnz), input_number(new_input_number)
    {
        dims = gsparc::allocate<index_t>(order);
        nbits = 0;
        for (int i = 0; i < order; i++)
        {
            dims[i] = new_dims[i];
            int num_bits = (sizeof(uint64_t) * 8) - common::clz(dims[i] - 1);
            nbits += num_bits;
        }

        // indices = gsparc::allocate<index_t *>(order);
        // values = gsparc::allocate<value_t>(nnz);
    }

    TENSOR_TEMPLATE
    SparseTensor<TENSOR_TEMPLATE_ARGS>::~SparseTensor()
    {
        gsparc::deallocate(dims);
        gsparc::deallocate(indices);
        gsparc::deallocate(values);
    }

    TENSOR_TEMPLATE
    void SparseTensor<TENSOR_TEMPLATE_ARGS>::ComputeNBits()
    {
        nbits = 0;
        for (int i = 0; i < order; i++)
        {
            int num_bits = (sizeof(uint64_t) * 8) - common::clz(dims[i] - 1);
            nbits += num_bits;
        }
    }
} // namespace gsparc
