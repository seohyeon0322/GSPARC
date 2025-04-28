#ifndef TENSOR_MANAGER_HPP_
#define TENSOR_MANAGER_HPP_
#include <thrust/tuple.h>
#include <immintrin.h>

#include "gsparc/sparse_tensor.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/cuda_memory.hpp"
#include "gsparc/timer.hpp"

namespace gsparc
{

#define TENSOR_MANAGER_TEMPLATE template <typename TensorType, typename SLITOMType>
#define TENSOR_MANAGER_TEMPLATE_ARGS TensorType, SLITOMType
    // #define TENSOR_TEMPLATE template <typename TensorType, typename SLITOMType>
    // #define TENSOR_TEMPLATE_ARGS IndexType, ValueType

    // #define SLITOM_TEMPLATE template <typename LIndexType, typename ValueType>
    // #define SLITOM_TEMPLATE_ARGS LIndexType, ValueType

    TENSOR_MANAGER_TEMPLATE
    class TensorManager
    {
    public:
        using sptensor_t = TensorType;
        using slitom_t = SLITOMType;
        using index_t = typename sptensor_t::index_t;
        using ulindex_t = typename slitom_t::ulindex_t;
        using lindex_t = typename slitom_t::lindex_t;
        using value_t = typename sptensor_t::value_t;
        using memorypool_t = CudaMemoryPool;
        using mask_t = typename slitom_t::mask_t;
        TensorManager(int gpu_count);
        ~TensorManager();

        struct SPair_64
        {
            lindex_t idx;
            value_t val;
        };

        struct SPair_128
        {
            ulindex_t uidx;
            lindex_t idx;
            value_t val;
        };

        struct CompositeComparator
        {
            __host__ __device__ bool operator()(const thrust::tuple<ulindex_t, lindex_t> &a,
                                                const thrust::tuple<ulindex_t, lindex_t> &b) const
            {
                ulindex_t a_u = thrust::get<0>(a);
                ulindex_t b_u = thrust::get<0>(b);
                if (a_u < b_u)
                    return true;
                if (a_u > b_u)
                    return false;
                return thrust::get<1>(a) < thrust::get<1>(b);
            }
        };

        bool FindPartitionNum(uint64_t nnz_count_x, uint64_t nnz_count_y, size_t poolSize, int nbits);
        void ConvertTensor(sptensor_t *sptensor, slitom_t *slitom, int cnmodes, const int *cpos, Timer *timer);
        void ConvertTensor_extra(sptensor_t *sptensor, slitom_t *slitom, int cnmodes, const int *cpos, Timer *timer);
        void Partition(slitom_t *slitom);
        void Partition_128(slitom_t *slitom);
        void SortSlitomXY(slitom_t *SX, slitom_t *SY, memorypool_t **pools, Timer *timer);
        void get_fmode(slitom_t *slitom, const int *cpos);
        void compute_nbtis(slitom_t *slitom);
        void setup_slitom(slitom_t *slitom);
        void create_mask(slitom_t *slitom);
        void sort_tensor_cpu(slitom_t *slitom, Timer *timer);
        void sort_tensors_gpu_64(slitom_t *SX, slitom_t *SY, memorypool_t **pools, Timer *timer);
        void sort_tensors_gpu_128(slitom_t *SX, slitom_t *SY, memorypool_t **pools, Timer *timer);
        void convert_and_sort_64(slitom_t *slitom, CudaMemoryPool **pool);
        void convert_and_sort_128(slitom_t *slitom, CudaMemoryPool **pools);


    private:
    public:
        int prtn_num;
        int gpu_count;
        int max_block_size;
    };
} // namespace gsparc

#include "gsparc/tensor_manager.tpp"
#endif // TENSOR_MANAGER_HPP_
