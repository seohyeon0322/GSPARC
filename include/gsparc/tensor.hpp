#ifndef TENSOR_HPP_
#define TENSOR_HPP_

namespace gsparc
{
#define TENSOR_TEMPLATE template <typename IndexType, typename ValueType>
#define TENSOR_TEMPLATE_ARGS IndexType, ValueType

    TENSOR_TEMPLATE
    class Tensor
    {
    public:
        // using this_t = Tensor<TENSOR_TEMPLATE_ARGS>;
        using index_t = IndexType;
        using value_t = ValueType;

    public:
        Tensor();
        Tensor(unsigned short order);
        ~Tensor();
        void ToString();
        const int get_nnz() const { return nnz; }

    private:
    public:
        unsigned short order;
        index_t *dims;
        uint64_t nnz;

        // // /* Data */
        // value_t *values;
        // index_t **indices;

        // /* Partition */
        // index_t *prtn_idx;
        // index_t *prtn_coord;
    };
}

#include "gsparc/tensor.tpp"
#endif // TENSOR_HPP_