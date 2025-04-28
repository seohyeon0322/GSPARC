#include <cuda_runtime_api.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "cmdline_parser.hpp"
#include "sptensor.hpp"
#include "partition.hpp"
#include "contraction.cuh"
#include "common/size.hpp"
#include "cuda_memory.cuh"
#include "pinned_memory.cuh"
#include "common/timer.hpp"
// #include "helper.hpp"

int main(int argc, char **argv)
{
    omp_set_num_threads(32);
    omp_set_nested(1);

    InitTSC();
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    size_t poolSize = avail - common::MiB(1024);
    unsigned long long pinnedpoolSize = common::GiB(100);
    // size_t poolSize = common::MiB(2);
    CudaMemoryPool *pool = new CudaMemoryPool(poolSize);

    PinnedMemoryPool *pinned_pool = new PinnedMemoryPool(pinnedpoolSize);

    CommandLineParser *cmd_parser = new CommandLineParser();
    cmd_parser->Parse(argc, argv);
    printf("finish parsing\n");
    int num_gpus = cmd_parser->get_g();
    printf("get_p: %d\n", cmd_parser->get_p());
    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaFree(0);
    }

    SparseTensor *_X;
    SparseTensor *_Y;

    printf("cmd_parser->get_q(): %d\n", cmd_parser->get_q());

    double wtime_s = omp_get_wtime();
    if (cmd_parser->get_q())
    {
        ImportQuantumTensor(cmd_parser->get_tensor_X().c_str(), &_X, pinned_pool);
        ImportQuantumTensor(cmd_parser->get_tensor_X().c_str(), &_Y, pinned_pool);
    }
    else
    {
        read_sparse_tensor(cmd_parser->get_tensor_X().c_str(), &_X, 1, pinned_pool);
        read_sparse_tensor(cmd_parser->get_tensor_Y().c_str(), &_Y, 1, pinned_pool);
    }
    printf("read tensor time: %f\n", omp_get_wtime() - wtime_s);

    // _Y = _X;
    pinned_pool->pin();

    fprintf(stderr, "Tensor X        = %s\n", cmd_parser->get_tensor_X().c_str());
    fprintf(stderr, "Tensor Y      = %s\n", cmd_parser->get_tensor_Y().c_str());
    // ImportSparseTensor(cmd_parser->get_tensor_X().c_str(),TEXT_FORMAT, &X);
    // ImportSparseTensor(cmd_parser->get_tensor_Y().c_str(),TEXT_FORMAT, &Y);
    printf("X tensor NNZ: %d\n", _X->nnz);

    _X->cnmodes = cmd_parser->get_num_cmodes();
    _Y->cnmodes = cmd_parser->get_num_cmodes();
    _X->fnmodes = _X->nmodes - _X->cnmodes;
    _Y->fnmodes = _Y->nmodes - _X->cnmodes;

    printf("X nmodes: %d, cnmodes: %d, fnmodes: %d\n", _X->nmodes, _X->cnmodes, _X->fnmodes);
    int partition_dim = cmd_parser->get_p();
    if (partition_dim == 0)
        partition_dim = cmd_parser->get_num_cmodes();
    printf("PARTITION DIM: %d\n", partition_dim);

    _X->cmode = cmd_parser->get_cmodes_X();
    _Y->cmode = cmd_parser->get_cmodes_Y();
    int num_iter = 1;
    IType final_result;
    double preprocess_time = 0, partition_time = 0, computation_time = 0, write_back_time = 0;
    for (int iter = 0; iter < (num_iter); iter++)
    {
        printf("==========ITERATION %d==========\n", iter);
        pool->pre_reset();
        pinned_pool->set_offset();

        IType *dims_acc = static_cast<IType *>(pinned_pool->allocate((_X->fnmodes + _Y->fnmodes) * sizeof(IType)));
        tmp_SparseTensor *tmp_X = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
        tmp_SparseTensor *P1 = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
        tmp_SparseTensor *Q1 = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
        tmp_SparseTensor *P = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
        tmp_SparseTensor *Q = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
        SparseTensor *Z = (SparseTensor *)pinned_pool->allocate(sizeof(SparseTensor));

        if (iter == 1)
        {
            preprocess_time = 0, partition_time = 0, computation_time = 0, write_back_time = 0;
        }

        double wtime_s = omp_get_wtime();
        preprocess_tensor(_X, _Y, dims_acc, &tmp_X, pinned_pool, pool);
        preprocess_time += omp_get_wtime() - wtime_s;
        printf("preprocess time: %f\n", preprocess_time);

        IType Z_memory = dims_acc[0] * sizeof(FType);
        FType *ZT_vals = (FType *)pinned_pool->allocate(dims_acc[0] * sizeof(FType));
        memset(ZT_vals, 0.0f, dims_acc[0] * sizeof(FType));

        if (Z_memory > poolSize)
        {
            fprintf(stderr, "Not enough memory for dense accumulator\n");
            return 0;
        }

        pinned_pool->pin();

        SparseTensor *X = (SparseTensor *)pinned_pool->allocate(sizeof(SparseTensor));
        SparseTensor *Y = (SparseTensor *)pinned_pool->allocate(sizeof(SparseTensor));
        X = _X;
        Y = _Y;



        X->cidx = static_cast<IType **>(pinned_pool->allocate(X->nmodes * sizeof(IType *)));
        X->vals = static_cast<FType *>(pinned_pool->allocate(X->nnz * sizeof(FType)));
        for (int i = 0; i < X->nmodes; i++)
        {
            X->cidx[i] = static_cast<IType *>(pinned_pool->allocate(X->nnz * sizeof(IType)));
#pragma omp parallel for
            for (int j = 0; j < X->nnz; j++)
            {
                X->cidx[i][j] = _X->orig_cidx[i][j];
            }
        }

#pragma omp parallel for
        for (int i = 0; i < X->nnz; i++)
        {
            X->vals[i] = _X->orig_vals[i];
        }
        Y->cidx = static_cast<IType **>(pinned_pool->allocate(Y->nmodes * sizeof(IType *)));
        Y->vals = static_cast<FType *>(pinned_pool->allocate(Y->nnz * sizeof(FType)));
        for (int i = 0; i < Y->nmodes; i++)
        {
            Y->cidx[i] = static_cast<IType *>(pinned_pool->allocate(Y->nnz * sizeof(IType)));
#pragma omp parallel for
            for (int j = 0; j < Y->nnz; j++)
            {
                Y->cidx[i][j] = _Y->orig_cidx[i][j];
            }
        }
#pragma omp parallel for
        for (int i = 0; i < Y->nnz; i++)
        {
            Y->vals[i] = _Y->orig_vals[i ];
        }

        wtime_s = omp_get_wtime();
        double temp_time = omp_get_wtime();
        partition1D(tmp_X, X, X->cnmodes, pinned_pool);
        printf("partition1D time: %f\n", omp_get_wtime() - temp_time);
        IType Xnnz = 0;
        IType Ynnz = 0;

        for (IType i = 0; i < tmp_X->size; i++)
        {
            Xnnz += tmp_X->XO[i]->nnz;
        }
        printf("before partition 2d, Xnnz: %llu\n", Xnnz);
        partition2D(tmp_X, Y, &P1, &Q1, pinned_pool, partition_dim);
        printf("partition2D time: %f\n", omp_get_wtime() - temp_time);
        temp_time = omp_get_wtime();
        Xnnz = 0;
        Ynnz = 0;

        for (IType i = 0; i < P1->size; i++)
        {
            Xnnz += P1->XO[i]->nnz;
        }
        for (IType i = 0; i < Q1->size; i++)
        {
            Ynnz += Q1->XO[i]->nnz;
        }
        printf("Xnnz: %llu, Ynnz: %llu\n", Xnnz, Ynnz);
        partition3D(tmp_X, Y, P1, &P, Q1, &Q, pinned_pool, partition_dim);
        printf("partition3D time: %f\n", omp_get_wtime() - temp_time);
        partition_time += omp_get_wtime() - wtime_s;
        printf("Partition time: %f\n", partition_time);

        Xnnz = 0;
        Ynnz = 0;
        for (IType i = 0; i < P->size; i++)
        {
            Xnnz += P->XO[i]->nnz;
        }
        for (IType i = 0; i < Q->size; i++)
        {
            Ynnz += Q->XO[i]->nnz;
        }
        printf("Xnnz: %llu, Ynnz: %llu\n", Xnnz, Ynnz);

        wtime_s = omp_get_wtime();
        contraction(X, Y, P, Q, tmp_X, dims_acc, ZT_vals, pool, pinned_pool);
        // wtime_s = omp_get_wtime();

        pinned_pool->set_offset();
        Z->nmodes = _X->fnmodes + _Y->fnmodes;
        Z->dims = static_cast<IType *>(pinned_pool->allocate(Z->nmodes * sizeof(IType)));
        int i;
        for (i = 0; i < _X->fnmodes; i++)
        {
            Z->dims[i] = _X->dims[_X->fmode[i]];
        }
        for (i = 0; i < _Y->fnmodes; i++)
        {
            Z->dims[i + _X->fnmodes] = _Y->dims[_Y->fmode[i]];
        }

        computation_time += omp_get_wtime() - wtime_s;
        // Z->vals = static_cast<FType *>(malloc(dims_acc[0] * sizeof(FType)));
        write_back(ZT_vals, Z, dims_acc, pinned_pool);
        write_back_time += omp_get_wtime() - wtime_s;
        printf("preprocess time\t partition time\t computation time\t write back time\n");
        printf("total time: %f\n", preprocess_time + partition_time + computation_time + write_back_time);
        std::cout << "Z tensor NNZ: " << Z->nnz << std::endl;
        final_result = Z->nnz;
        printf("%f\t%f\t%f\t%f\t\n", preprocess_time, partition_time, computation_time, write_back_time);
        // for (IType i = 0; i < Z->nnz; i++){
        //     printf("Z->indices[%d]: %d, %d    ", i, Z->cidx[0][i], Z->cidx[1][i]);
        //     printf("Z->vals[%d]: %f\n", i, Z->vals[i]);
        // }
    }
    fprintf(stderr, "preprocess time\t partition time\t computation time\t write back time\n");
    fprintf(stderr, "%f\t%f\t%f\t%f\t\n", preprocess_time / num_iter, partition_time / num_iter, computation_time / num_iter, write_back_time / num_iter);
    fprintf(stderr, "total time: %f\n", (preprocess_time + partition_time + computation_time + write_back_time) / num_iter);
    fprintf(stderr, "\n\n\n");

    return 0;
}