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

    CudaMemoryPool::CudaMemoryPool(unsigned id) : _device_id(id), currentOffset(0), alignment(8)
    {
        size_t avail, total;
        checkCudaErrors(cudaSetDevice(id));
        checkCudaErrors(cudaMemGetInfo(&avail, &total));
        poolSize = avail - common::MiB(256);
        // poolSize = common::MiB(256);

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
        return false; // 적절한 크기의 블록을 찾을 수 없으면 false 반환
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
                allocatedBlocks.insert(ptr); // 할당된 블록 추적
                freeBlocks.erase(it);        // 사용된 블록 제거
                return ptr;
            }
        }

        char *byte_str = common::byteToString(size);

        std::cerr << "Failed to allocate memory of size " << byte_str << std::endl;
        exit(1);
        return nullptr; // 적절한 크기의 블록을 찾을 수 없으면 nullptr 반환
    }

    void CudaMemoryPool::deallocate(void *ptr, size_t size)
    {
        // 중복 해제 방지
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
        allocatedBlocks.erase(ptr); // 할당된 블록에서 제거
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
        // freeBlocks를 offset 기준으로 정렬
        freeBlocks.sort([](const Block &a, const Block &b)
                        { return a.offset < b.offset; });

        for (auto it = freeBlocks.begin(); it != freeBlocks.end();)
        {
            // 현재 블록의 크기가 0이면 제거
            if (it->size == 0)
            {
                it = freeBlocks.erase(it); // erase는 유효한 다음 반복자를 반환
                continue;                  // 제거 후 다시 루프 시작
            }

            auto next = std::next(it);
            // 현재 블록과 다음 블록이 병합 가능한 경우 병합
            if (next != freeBlocks.end() && it->offset + it->size == next->offset)
            {
                it->size += next->size;
                freeBlocks.erase(next); // next가 삭제되었으므로 반복자 이동 생략
            }
            else
            {
                ++it; // 병합하지 않은 경우 다음 반복자로 이동
            }
        }
    }

    // 새로 추가된 함수: 남은 메모리의 총 크기를 반환한다.
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
