#ifndef PINNED_MEMORY_CUH
#define PINNED_MEMORY_CUH

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <mutex>
#include <unordered_set>
#include <list>

#include "helper.hpp"
#include "cuda_memory.cuh"

class PinnedMemoryPool
{
public:
    PinnedMemoryPool(size_t size) : poolSize(size), currentOffset(0), pinnedOffset(0)
    {
        checkCudaErrors(cudaHostAlloc(&pool, poolSize, cudaHostAllocPortable));
    }

    ~PinnedMemoryPool()
    {
        checkCudaErrors(cudaFreeHost(pool));
    }

    void *allocate(size_t size)
    {

        size_t alignment = 8; // 8-byte alignment for 64-bit
        size_t padding = (alignment - (reinterpret_cast<size_t>(static_cast<char *>(pool) + currentOffset) % alignment)) % alignment;
        if (currentOffset + padding + size > poolSize)
        {
            std::cout << "currentOffset: " << currentOffset << ", size: " << size << ", poolSize: " << poolSize << std::endl;
            std::cout << "pinned memory pool is full" << std::endl;
            exit(1);
            return nullptr; // 메모리 풀이 가득 찬 경우 nullptr 반환
        }
        void *ptr = static_cast<char *>(pool) + currentOffset + padding;
        currentOffset += padding + size;
        return ptr;    }

    void pre_reset()
    {

        currentOffset = 0;
    }

    void pin()
    {
        pinnedOffset = currentOffset;
    }

    void set_offset()
    {
        currentOffset = pinnedOffset;
    }

    void reset()
    {
        currentOffset = 0;
    }

    size_t getPoolSize() const
    {
        return poolSize;
    }

private:
    void *pool;
    size_t poolSize;
    size_t currentOffset;
    size_t pinnedOffset;
};

#endif // PINNED_MEMORY_CUH
