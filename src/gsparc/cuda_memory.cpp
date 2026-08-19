#ifndef CUDA_MEMORY_CUH
#define CUDA_MEMORY_CUH

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <set> // make sure set is included for allocatedBlocks

#include "gsparc/cuda_memory.hpp"
#include "gsparc/helper.hpp"
#include "common/size.hpp"

namespace gsparc
{

    CudaMemoryPool::CudaMemoryPool(unsigned id, size_t memory) : _device_id(id), currentOffset(0), alignment(8)
    {
        size_t avail, total;
        checkCudaErrors(cudaSetDevice(id));
        checkCudaErrors(cudaMemGetInfo(&avail, &total));
        // poolSize = avail - common::MiB(256);
        if (memory == 0)
        {
            poolSize = avail - common::GiB(1);
        }
        else
        {
            poolSize = common::GiB(memory);
        }

        printf("poolSize: %s\n", common::byteToString(poolSize));

        checkCudaErrors(cudaMalloc(&pool, poolSize));
        freeBlocks.push_back(Block(0, poolSize));
        // poolSize = poolSize;
    }

    CudaMemoryPool::~CudaMemoryPool()
    {
        checkCudaErrors(cudaFree(pool));
    }

    bool CudaMemoryPool::is_available(size_t size)
    {
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            size_t offset = it->offset;
            double sz = it->size;
            size_t padding = (alignment - (offset % alignment)) % alignment;

            if (it->size >= size + padding)
            {
                return true;
            }
        }
        return false;
    }

    void *CudaMemoryPool::allocate(size_t size)
    {
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            size_t offset = it->offset;
            double sz = it->size;
            if (it->size >= size)
            {
                void *ptr = static_cast<char *>(pool) + offset;

                if (it->size > size)
                {
                    size_t free_offset = offset + size;
                    size_t padding = (alignment - (free_offset % alignment)) % alignment;
                    freeBlocks.push_back(Block(free_offset + padding, it->size - size - padding));
                }
                allocatedBlocks.insert(ptr); 
                freeBlocks.erase(it);       
                return ptr;
            }
        }

        char *byte_str = common::byteToString(size);

        std::cerr << "Failed to allocate memory of size " << byte_str << std::endl;
        exit(1);
        return nullptr; 
    }

    void CudaMemoryPool::deallocate(void *ptr, size_t size)
    {
        if (allocatedBlocks.find(ptr) == allocatedBlocks.end())
        {
            fprintf(stderr, "Attempt to free unallocated or already freed block\n");

            exit(1);
            return;
        }
        // get size of allocated blocks
        size_t offset = static_cast<char *>(ptr) - static_cast<char *>(pool);
        size_t padding = (alignment - ((offset + size) % alignment)) % alignment;
        freeBlocks.push_back(Block(offset, size + padding));
        mergeFreeBlocks();
        // iterate allocatedBlocks
        allocatedBlocks.erase(ptr); 
        ptr = NULL;
    }

    void CudaMemoryPool::printFree()
    {
        printf("============printFree============\n");
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            double sz = it->size;
            char *sz_str = common::byteToString(sz);
            printf("free block size: %s\n", sz_str);
        }

        printf("=================================\n");
    }

    void CudaMemoryPool::reset()
    {
        freeBlocks.clear();
        freeBlocks.push_back(Block(currentOffset, poolSize));
        allocatedBlocks.clear();
        currentOffset = 0;
    }

    void CudaMemoryPool::mergeFreeBlocks()
    {
        freeBlocks.sort([](const Block &a, const Block &b)
                        { return a.offset < b.offset; });

        for (auto it = freeBlocks.begin(); it != freeBlocks.end();)
        {
            if (it->size == 0)
            {
                it = freeBlocks.erase(it); 
                continue;                 
            }

            auto next = std::next(it);
            if (next != freeBlocks.end() && it->offset + it->size == next->offset)
            {
                it->size += next->size;
                freeBlocks.erase(next);
            }
            else
            {
                ++it;
            }
        }
    }

    size_t CudaMemoryPool::getRemainingMemory()
    {
        size_t remaining = 0;
        for (const auto &block : freeBlocks)
        {
            remaining += block.size;
        }
        return remaining;
    }

} // namespace gsparc

#endif // CUDA_MEMORY_CUH
