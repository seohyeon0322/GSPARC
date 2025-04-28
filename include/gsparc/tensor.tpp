#include <omp.h>

#include "gsparc/tensor.hpp"
#include "gsparc/helper.hpp"

namespace gsparc
{

    TENSOR_TEMPLATE
    Tensor<TENSOR_TEMPLATE_ARGS>::Tensor()
        : nnz(0) {}

    TENSOR_TEMPLATE
    Tensor<TENSOR_TEMPLATE_ARGS>::Tensor(unsigned short new_order)
        : order(new_order)
    {
        dims = gsparc::allocate<index_t>(order);
        nnz = 0;
    }

    TENSOR_TEMPLATE
    void Tensor<TENSOR_TEMPLATE_ARGS>::ToString()
    {
        uint64_t tmp = 1;
        for (int i = 0; i < order; i++)
        {
            tmp *= dims[i];
        }

        double sparsity = ((double)nnz) / tmp;
        fprintf(stderr, "# Modes         = %u\n", order);
        fprintf(stderr, "Sparsity        = %f\n", sparsity);
        fprintf(stderr, "Dimensions      = [%llu", dims[0]);
        for (int i = 1; i < order; i++)
        {
            fprintf(stderr, " X %llu", dims[i]);
        }
        fprintf(stderr, "]\n");
        fprintf(stderr, "NNZ             = %llu\n", nnz);
    }
}