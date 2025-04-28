#ifndef CONTRACTION_CUH_
#define CONTRACTION_CUH_

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <omp.h>

#include "common.hpp"
#include "sptensor.hpp"
#include "cuda_memory.cuh"
#include "pinned_memory.cuh"
// #include "helper.hpp"
const int block_size = 1024;

__device__ int compare_cmodes_device(IType cmode_len, IType **tmp_ind, IType **X_ind, IType tmp_idx, IType X_idx)
{
    for (int i = 0; i < cmode_len; i++)
    {

        if (tmp_ind[i][tmp_idx] > X_ind[i][X_idx])
        {
            return 1;
        }
        else if (tmp_ind[i][tmp_idx] < X_ind[i][X_idx])
        {
            return -1;
        }
    }

    return 0;
}

__device__ IType pretend(IType X_idx, IType Y_idx, IType **X_f_ind, IType **Y_f_ind, IType *dims_acc, IType X_fmode_num, IType Y_fmode_num)
{
    IType f = 0;

    for (IType i = 0; i < X_fmode_num; i++)
    {
        IType temp = X_f_ind[i][X_idx];
        temp *= dims_acc[i + 1];

        f += temp;
    }
    for (IType j = 0; j < Y_fmode_num; j++)
    {
        IType temp = Y_f_ind[j][Y_idx];

        if (j == Y_fmode_num - 1)
            f += temp;
        else
        {
            temp *= dims_acc[j + X_fmode_num + 1];
            f += temp;
        }
    }
    return f;
}

__global__ void contraction_kernel(IType **X_f_ind,
                                   IType **Y_f_ind,
                                   FType *X_vals,
                                   FType *Y_vals,
                                   IType *dims_acc,
                                   IType *P_start,
                                   IType *Q_start,
                                   IType *P_end,
                                   IType *Q_end,
                                   IType *XO_start,
                                   IType *XO_end,
                                   FType *ZT_vals,
                                   IType **Y_cmode_ind,
                                   IType **XO_cmode_ind,
                                   IType *Q_mc,
                                   IType cnmodes,
                                   IType X_fmode_num,
                                   IType Y_fmode_num,
                                   IType Q_size)
{
    IType k = blockIdx.x * blockDim.x + threadIdx.x; // thread id
    IType stride = blockDim.x * gridDim.x;

    while (k < Q_size)
    {
        IType flag = Q_mc[k];
        IType i = P_start[flag];
        IType j = Q_start[k];
        IType Pend = P_end[flag];
        IType Qend = Q_end[k];
        while (i < Pend && j < Qend)
        {

            int com = compare_cmodes_device(cnmodes, XO_cmode_ind, Y_cmode_ind, i, j); // i, j: nnz index
            if (com > 0)
                j++;
            else if (com < 0)
                i++;
            else
            {
                IType start = XO_start[i];
                IType end = XO_end[i];
                for (IType l = start; l < end; l++)
                {
                    IType f = pretend(l, j, X_f_ind, Y_f_ind, dims_acc, X_fmode_num, Y_fmode_num);
                    FType value = X_vals[l] * Y_vals[j];
                    atomicAdd(&ZT_vals[f], value);
                }
                j++;
            }
        }
        k += stride;
    }
    return;
}

void contraction(SparseTensor *X, SparseTensor *Y, tmp_SparseTensor *P, tmp_SparseTensor *Q, tmp_SparseTensor *XO, IType *dims_acc, FType *ZT_vals, CudaMemoryPool *pool, PinnedMemoryPool *pinned_pool)
{

    IType **d_X_f_ind, **d_Y_f_ind;
    FType *d_X_vals, *d_Y_vals, *d_ZT_vals;
    IType *d_dims_acc, *P_start, *Q_start, *P_end, *Q_end, *Q_mc, *d_Q_mc, *XO_start, *XO_end;

    pool->pin();
    d_Y_f_ind = static_cast<IType **>(pinned_pool->allocate(Y->fnmodes * sizeof(IType *)));
    d_X_f_ind = static_cast<IType **>(pinned_pool->allocate(X->fnmodes * sizeof(IType *)));
    Q_mc = static_cast<IType *>(pinned_pool->allocate(Q->size * sizeof(IType)));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("flag00: %s\n", cudaGetErrorString(err));
    }

    printf("storage size for vals: %s\n", sptBytesString(X->nnz * sizeof(FType)));
    d_X_vals = static_cast<FType *>(pool->allocate(X->nnz * sizeof(FType)));
    d_Y_vals = static_cast<FType *>(pool->allocate(Y->nnz * sizeof(FType)));
    d_ZT_vals = static_cast<FType *>(pool->allocate(dims_acc[0] * sizeof(FType)));
    cudaMemcpy(d_ZT_vals, ZT_vals, dims_acc[0] * sizeof(FType), cudaMemcpyHostToDevice);
    // cudaMemset(d_ZT_vals, 0.0f, dims_acc[0] * sizeof(FType));
    printf("dms_acc[0]: %d\n", dims_acc[0]);
    printf("after vals\n");
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("flag0: %s\n", cudaGetErrorString(err));
    }
    d_dims_acc = static_cast<IType *>(pool->allocate((X->fnmodes + Y->fnmodes) * sizeof(IType)));
    // P_start = static_cast<IType *>(pool->allocate(P->size * sizeof(IType)));
    // Q_start = static_cast<IType *>(pool->allocate(Q->size * sizeof(IType)));
    // P_end = static_cast<IType *>(pool->allocate(P->size * sizeof(IType)));
    // Q_end = static_cast<IType *>(pool->allocate(Q->size * sizeof(IType)));
    d_Q_mc = static_cast<IType *>(pool->allocate(Q->size * sizeof(IType)));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("flag1: %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemcpy(d_X_vals, X->vals, X->nnz * sizeof(FType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Y_vals, Y->vals, Y->nnz * sizeof(FType), cudaMemcpyHostToDevice));

    P_start = static_cast<IType *>(pinned_pool->allocate(P->size * sizeof(IType)));
    Q_start = static_cast<IType *>(pinned_pool->allocate(Q->size * sizeof(IType)));
    P_end = static_cast<IType *>(pinned_pool->allocate(P->size * sizeof(IType)));
    Q_end = static_cast<IType *>(pinned_pool->allocate(Q->size * sizeof(IType)));
    XO_start = static_cast<IType *>(pinned_pool->allocate(XO->size * sizeof(IType)));
    XO_end = static_cast<IType *>(pinned_pool->allocate(XO->size * sizeof(IType)));
    printf("P->size: %llu\n", P->size);
    printf("Q->size: %llu\n", Q->size);
    printf("XO->size: %llu\n", XO->size);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("flag2: %s\n", cudaGetErrorString(err));
    }
    IType **d_Y_cmode_ind = static_cast<IType **>(pinned_pool->allocate(Y->cnmodes * sizeof(IType **)));
    IType **d_XO_cmode_ind = static_cast<IType **>(pinned_pool->allocate(XO->cmode_len * sizeof(IType **)));

    IType *d_P_start, *d_Q_start, *d_P_end, *d_Q_end, *d_XO_start, *d_XO_end;
    printf("P->size: %llu\n", P->size);
    d_P_start = static_cast<IType *>(pool->allocate(P->size * sizeof(IType)));
    d_Q_start = static_cast<IType *>(pool->allocate(Q->size * sizeof(IType)));
    d_P_end = static_cast<IType *>(pool->allocate(P->size * sizeof(IType)));
    d_Q_end = static_cast<IType *>(pool->allocate(Q->size * sizeof(IType)));
    d_XO_start = static_cast<IType *>(pool->allocate(XO->size * sizeof(IType)));
    d_XO_end = static_cast<IType *>(pool->allocate(XO->size * sizeof(IType)));
    if (d_P_start == nullptr || d_Q_start == nullptr || d_P_end == nullptr || d_Q_end == nullptr || d_XO_start == nullptr || d_XO_end == nullptr)
        printf("null pointer\n");

#pragma omp parallel for
    for (IType i = 0; i < XO->size; i++)
    {
        XO_start[i] = XO->XO[i]->start;
        XO_end[i] = XO->XO[i]->end;
    }

#pragma omp parallel for
    for (IType i = 0; i < P->size; i++)
    {
        P_start[i] = P->XO[i]->start;
        P_end[i] = P->XO[i]->end;
    }

#pragma omp parallel for
    for (IType i = 0; i < Q->size; i++)
    {
        Q_start[i] = Q->XO[i]->start;
        Q_end[i] = Q->XO[i]->end;
        Q_mc[i] = Q->XO[i]->mc;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("before memcpy Error: %s\n", cudaGetErrorString(err));
    }

    IType *tmp_cmode = static_cast<IType *>(pinned_pool->allocate(XO->size * sizeof(IType)));
    for (int i = 0; i < XO->cmode_len; i++)
    {
        d_XO_cmode_ind[i] = static_cast<IType *>(pool->allocate(XO->size * sizeof(IType)));

#pragma omp parallel for
        for (int j = 0; j < XO->size; j++)
        {
            tmp_cmode[j] = XO->XO[j]->cmode_ind[i];
        }
        checkCudaErrors(cudaMemcpy(d_XO_cmode_ind[i], tmp_cmode, XO->size * sizeof(IType), cudaMemcpyHostToDevice));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cmode ind Error: %s\n", cudaGetErrorString(err));
    }

    IType *tmp_cmode2 = static_cast<IType *>(pinned_pool->allocate(Y->nnz * sizeof(IType)));
    for (int i = 0; i < Y->cnmodes; i++)
    {
        d_Y_cmode_ind[i] = static_cast<IType *>(pool->allocate(Y->nnz * sizeof(IType)));

#pragma omp parallel for
        for (IType j = 0; j < Y->nnz; j++)
        {
            tmp_cmode2[j] = Y->cidx[i][j];
        }
        checkCudaErrors(cudaMemcpy(d_Y_cmode_ind[i], tmp_cmode2, Y->nnz * sizeof(IType), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < X->fnmodes; i++)
    {
        // checkCudaErrors(((void **)&(d_X_f_ind[i]), X->nnz * sizeof(IType)));
        // printf("X->fmode[i]: %d\n", X->fmode[i]);
        d_X_f_ind[i] = static_cast<IType *>(pool->allocate(X->nnz * sizeof(IType)));
        checkCudaErrors(cudaMemcpy(d_X_f_ind[i], X->cidx[X->fmode[i]] , X->nnz * sizeof(IType), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < Y->fnmodes; i++)
    {
        // checkCudaErrors(cudaMalloc((void **)&d_Y_f_ind[i], Y->nnz * sizeof(IType)));
        // printf("Y->fmode[i]: %d\n", Y->fmode[i]);
        d_Y_f_ind[i] = static_cast<IType *>(pool->allocate(Y->nnz * sizeof(IType)));
        checkCudaErrors(cudaMemcpy(d_Y_f_ind[i], Y->cidx[Y->fmode[i]] , Y->nnz * sizeof(IType), cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMemcpy(d_P_start, P_start, P->size * sizeof(IType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_P_end, P_end, P->size * sizeof(IType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_XO_start, XO_start, XO->size * sizeof(IType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_XO_end, XO_end, XO->size * sizeof(IType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dims_acc, dims_acc, (X->fnmodes + Y->fnmodes) * sizeof(IType), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_Q_start, Q_start, Q->size * sizeof(IType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Q_end, Q_end, Q->size * sizeof(IType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Q_mc, Q_mc, Q->size * sizeof(IType), cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("before contraction Error: %s\n", cudaGetErrorString(err));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    printf("before kernel\n");
    printf("Q->size: %llu\n", Q->size);
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid((Q->size + block_size - 1) / block_size, 1, 1);
    contraction_kernel<<<dimGrid, dimBlock>>>(d_X_f_ind, d_Y_f_ind, d_X_vals, d_Y_vals, d_dims_acc,
                                              d_P_start, d_Q_start, d_P_end, d_Q_end,
                                              d_XO_start, d_XO_end, d_ZT_vals, d_Y_cmode_ind, d_XO_cmode_ind, d_Q_mc,
                                              X->cnmodes, X->fnmodes, Y->fnmodes, Q->size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float miliseconds = 0;
    cudaEventElapsedTime(&miliseconds, start, stop);
    printf("contraction time: %fs\n", miliseconds / 1000);

    // int num_streams = 3;
    // int parallel_factor = 6;
    // cudaStream_t streams[num_streams];
    // for (int i = 0; i < num_streams; i++)
    // {
    //     checkCudaErrors(cudaStreamCreate(&streams[i]));
    // }

    // for (int i = 0; i < parallel_factor; i++)
    // {
    //     cudaStream_t stream = streams[i % num_streams];

    //     int start = i * Q->size / parallel_factor;
    //     int end = i == parallel_factor - 1 ? Q->size : ((i + 1) * Q->size / parallel_factor);
    //     int size = end - start;

    //     checkCudaErrors(cudaMemcpyAsync(d_Q_start + start, Q_start + start, size * sizeof(IType), cudaMemcpyHostToDevice, stream));
    //     checkCudaErrors(cudaMemcpyAsync(d_Q_end + start, Q_end + start, size * sizeof(IType), cudaMemcpyHostToDevice, stream));
    //     checkCudaErrors(cudaMemcpyAsync(d_Q_mc + start, Q_mc + start, size * sizeof(IType), cudaMemcpyHostToDevice, stream));

    //     for (int j = 0; j < Y->fnmodes; j++)
    //     {
    //         checkCudaErrors(cudaMemcpyAsync(d_Y_f_ind[j] + Q_start[start], Y->cidx[Y->fmode[j]] + Q_start[start], (Q_end[end - 1] - Q_start[end]) * sizeof(IType), cudaMemcpyHostToDevice, stream));
    //     }

    //     dim3 dimBlock(block_size, 1, 1);
    //     dim3 dimGrid((size + block_size - 1) / block_size, 1, 1);
    //     contraction_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_X_f_ind, d_Y_f_ind, d_X_vals, d_Y_vals, d_dims_acc,
    //                                                          d_P_start, d_Q_start + start, d_P_end, d_Q_end + start,
    //                                                          d_XO_start, d_XO_end, d_ZT_vals, d_Y_cmode_ind, d_XO_cmode_ind, d_Q_mc + start,
    //                                                          X->cnmodes, X->fnmodes, Y->fnmodes, size);
    // }

    // for (int i = 0; i < num_streams; i++)
    // {
    //     checkCudaErrors(cudaStreamSynchronize(streams[i]));
    // }

    // for (int i = 0; i < num_streams; i++)
    // {
    //     checkCudaErrors(cudaStreamDestroy(streams[i]));
    // }

    // printf("Q->size: %d\n", Q->size);
    // printf("X->cnmodes: %d, X->fnmodes: %d, Y->fnmodes: %d\n", X->cnmodes, X->fnmodes, Y->fnmodes);
    // dim3 dimBlock(block_size, 1, 1);
    // dim3 dimGrid((Q->size + block_size - 1) / block_size, 1, 1);
    // contraction_kernel<<<dimGrid, dimBlock>>>(d_X_f_ind, d_Y_f_ind, d_X_vals, d_Y_vals, d_dims_acc, d_P_start, d_Q_start, d_P_end, d_Q_end, d_XO_start, d_XO_end, d_ZT_vals, d_Y_cmode_ind, d_XO_cmode_ind, d_Q_mc, X->cnmodes, X->fnmodes, Y->fnmodes, Q->size);
    // checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(ZT_vals, d_ZT_vals, dims_acc[0] * sizeof(FType), cudaMemcpyDeviceToHost));
    pool->set_offset();


}
#endif