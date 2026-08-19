#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP
#include <iostream>

#include "gsparc/tensor.hpp"

namespace gsparc
{
#define TENSOR_TEMPLATE template <typename IndexType, typename ValueType>
#define TENSOR_TEMPLATE_ARGS IndexType, ValueType

    TENSOR_TEMPLATE
    class SparseTensor : public Tensor<TENSOR_TEMPLATE_ARGS>
    {
    public:
        using tensor_t = Tensor<TENSOR_TEMPLATE_ARGS>;
        using index_t = typename tensor_t::index_t;
        using value_t = typename tensor_t::value_t;

    public:
        SparseTensor(int input_number);
        SparseTensor(unsigned short order, index_t *dims, uint64_t nnz, int input_number);
        ~SparseTensor();
        void ComputeNBits();

    private:
    public:
        unsigned short order;
        // unsigned short nmodes;

        index_t *dims;
        uint64_t nnz;
        int nbits;

        // /* Data */
        value_t *values;
        index_t **indices;

        int input_number;

        // /* Partition */
        // index_t *prtn_idx;
        // index_t *prtn_coord;
    };
} // namespace gsparc

#include "gsparc/sparse_tensor.tpp"
#endif