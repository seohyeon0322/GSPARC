#ifndef CUDA_MEMORY_ALLOCATOR_HPP
#define CUDA_MEMORY_ALLOCATOR_HPP

#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/execution_policy.h>

#include "cuda_memory.hpp" // 너의 CudaMemoryPool 헤더

namespace gsparc
{
    // thrust의 custom allocator 형태에 맞추기 위해 allocator 템플릿 사용
    template <typename T>
    class CudaMemoryPoolAllocator
    {
    public:
        using value_type = T;

        CudaMemoryPool* pool;

        CudaMemoryPoolAllocator(CudaMemoryPool* p = nullptr) : pool(p) {}

        template <typename U>
        CudaMemoryPoolAllocator(const CudaMemoryPoolAllocator<U>& other) noexcept
            : pool(other.pool) {}

        T* allocate(std::size_t n)
        {
            return pool->allocate<T>(n);
        }

        void deallocate(T* ptr, std::size_t n)
        {
            pool->deallocate(ptr, n);
        }

        // Allocator comparison (required for thrust)
        bool operator==(const CudaMemoryPoolAllocator& other) const { return pool == other.pool; }
        bool operator!=(const CudaMemoryPoolAllocator& other) const { return pool != other.pool; }
    };
} // namespace gsparc

#endif // CUDA_MEMORY_ALLOCATOR_HPP
