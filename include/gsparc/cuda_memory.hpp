#ifndef CUDA_MEMORY_HPP
#define CUDA_MEMORY_HPP

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <unordered_set>
#include <type_traits> // optional, for static_assert

namespace gsparc
{
    class CudaMemoryPool
    {
    public:
        CudaMemoryPool(unsigned id);
        ~CudaMemoryPool();

        bool is_available(size_t size);
        void *allocate(size_t size);
        void deallocate(void *ptr, size_t size);
        void reset();
        void printFree();
        size_t getRemainingMemory();

        const size_t getPoolSize() const { return poolSize; }

        template <typename T>
        T *allocate(size_t count)
        {
            // static_assert(std::is_trivially_copyable<T>::value, "CudaMemoryPool only supports trivially copyable types.");
            size_t size = sizeof(T) * count;
            return static_cast<T *>(allocate(size));
        }

        template <typename T>
        void deallocate(T *ptr, size_t count)
        {
            // static_assert(std::is_trivially_copyable<T>::value, "CudaMemoryPool only supports trivially copyable types.");
            size_t size = sizeof(T) * count;
            deallocate(static_cast<void *>(ptr), size);
        }

    private:
        struct Block
        {
            size_t offset;
            size_t size;
            Block(size_t o, size_t s) : offset(o), size(s) {}
        };

        void *pool;
        size_t poolSize;
        std::list<Block> freeBlocks;
        std::unordered_set<void *> allocatedBlocks;
        size_t currentOffset;
        int alignment;
        unsigned _device_id;

        void mergeFreeBlocks();
    };
} // namespace gsparc

#endif // CUDA_MEMORY_HPP
