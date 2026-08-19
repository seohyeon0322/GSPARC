// [local-build] nvcc workaround: stub _mm_pause + block immintrin.h for nvcc,
//               host compiler (g++) gets real intrinsics via bitops.hpp __CUDA_ARCH__ guard.
#ifdef __CUDACC__
static __host__ __device__ inline void _mm_pause(void) {}
#define _IMMINTRIN_H_INCLUDED  // prevent GCC immintrin.h from being parsed by nvcc
#endif

#include "cuda_runtime.h"

#include "common/cuda_helper.hpp"
#include "gsparc/cmdline_opts.hpp"
#include "gsparc/coo_ablation.cuh"
#include "gsparc/coo_natural_ablation.cuh"
#include "gsparc/helper.hpp"
#include "gsparc/io_manager.hpp"
#include "gsparc/slitom.hpp"
#include "gsparc/sparse_tensor.hpp"
#include "gsparc/sptc.cuh"
#include "gsparc/tensor.hpp"
#include "gsparc/tensor_manager.hpp"
#include "gsparc/timer.hpp"
#include <algorithm>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char *argv[]) {
  using namespace gsparc;

  omp_set_num_threads(64);
  // Normal benchmark runs keep the verbose per-stage trace quiet.  The
  // variability suite enables it so every in-process iteration is recorded.
  if (getenv("GSPARC_LOG_ITERATIONS") == nullptr) {
    freopen("/dev/null", "w", stdout);
  }
  CommandLineOptions *options = new CommandLineOptions;
  options->Parse(argc, argv);
  using index_t = uint64_t;

  using value_t = float;
  using tensor_t = SparseTensor<index_t, value_t>;
  using io_manager_t = IOManager<tensor_t>;
  using cuda_memory_t = CudaMemoryPool;

  // Read tensor
  io_manager_t *io_manager = new io_manager_t;
  tensor_t *tensor_x = new tensor_t(1);
  tensor_t *tensor_y = new tensor_t(2);
  tensor_t *tensor_z = new tensor_t(2);
  double reorder_bit_multi_sptc = 0.0;

  io_manager->ParseFromFile(options->get_first_input_path(), &tensor_x, 1,
                            options->get_quantum());
  io_manager->ParseFromFile(options->get_second_input_path(), &tensor_y, 1,
                            options->get_quantum());
  if (options->is_multi_sptc())
    io_manager->ParseFromFile(options->get_second_input_path(), &tensor_z, 1,
                              options->get_quantum());
  /*Read dimension*/
  for (int i = 0; i < tensor_x->order; i++) {
    fprintf(stderr, "%llu ", tensor_x->dims[i]);
  }
  fprintf(stderr, " nnz: %llu\n", tensor_x->nnz);

  for (int i = 0; i < tensor_y->order; i++) {
    fprintf(stderr, "%llu ", tensor_y->dims[i]);
  }
  fprintf(stderr, " nnz: %llu\n", tensor_y->nnz);
  printf("tensor_x->nnz %llu\n", tensor_x->nnz);
  printf("tensor_y->nnz %llu\n", tensor_y->nnz);

  tensor_x->ComputeNBits();
  tensor_y->ComputeNBits();
  tensor_z->ComputeNBits();
  printf("tensor_x nbits %d\n", tensor_x->nbits);
  int memory = options->get_memory();

  cuda_memory_t **cuda_memory = new cuda_memory_t *[options->get_gpu_count()];
  for (int i = 0; i < options->get_gpu_count(); i++) {
    cuda_memory[i] = new cuda_memory_t(i, memory);
  }
  for (int i = 0; i < options->get_gpu_count(); i++) {
    cudaSetDevice(i);
    cudaFree(0);
  }
  int num_iter = options->get_num_iter();
  uint64_t result_nnz = 0;

  /*Timer*/
  Timer *slitom_timer = new Timer;
  Timer *partition_timer = new Timer;
  Timer **idxmatch_timers = new Timer *[options->get_gpu_count()];
  Timer **dynamicprtn_timers = new Timer *[options->get_gpu_count()];
  Timer **contraction_timers = new Timer *[options->get_gpu_count()];
  Timer **copy_timers = new Timer *[options->get_gpu_count()];
  for (int i = 0; i < options->get_gpu_count(); i++) {
    idxmatch_timers[i] = new Timer;
    dynamicprtn_timers[i] = new Timer;
    contraction_timers[i] = new Timer;
    copy_timers[i] = new Timer;
  }

  // Pure COO IndexMatch ablation path (-i 1).
  // Skips SLITOM entirely: builds/sorts COO directly on GPU from raw
  // tensor->indices and runs only the IndexMatch kernel.
  if (options->get_idxmatch_mode() == 1) {
    coo_ablation::RunCOOAblation<tensor_t>(
        tensor_x, tensor_y, options->get_first_input_cmodes(),
        options->get_second_input_cmodes(), options->get_num_cmodes(),
        cuda_memory, options->get_gpu_count(), num_iter);
    printf("\n\n");
    fprintf(stderr, "\n\n");
    return 0;
  }

  // Natural-sorted COO IndexMatch ablation path (-i 3).
  // c-tuples are NOT contiguous → O(X*Y) linear scan.
  if (options->get_idxmatch_mode() == 3) {
    coo_natural_ablation::RunCOONaturalAblation<tensor_t>(
        tensor_x, tensor_y, options->get_first_input_cmodes(),
        options->get_second_input_cmodes(), options->get_num_cmodes(),
        cuda_memory, options->get_gpu_count(), num_iter);
    printf("\n\n");
    fprintf(stderr, "\n\n");
    return 0;
  }

  for (int iter = 0; iter < num_iter; iter++) {
    printf("===============iteration %d/%d================\n", iter + 1,
           num_iter);
    for (int i = 0; i < options->get_gpu_count(); i++) {
      cuda_memory[i]->reset();
      printf("pool size: %s\n",
             common::byteToString(cuda_memory[i]->getPoolSize()));
    }

    printf("tensor_x->nbits %d\n", tensor_x->nbits);

    if (tensor_x->nbits > 64) {

      int extra_bits = tensor_x->nbits - 64;
      int aligned_extra_bits = (extra_bits <= 8)    ? 8
                               : (extra_bits <= 16) ? 16
                               : (extra_bits <= 32) ? 32
                                                    : 64;

      int extra_bytes = (aligned_extra_bits + 7) / 8;
      printf("extra_bytes %d\n", extra_bytes);
      if (extra_bytes == 1) {
        using lindex_t = uint64_t;
        using ulindex_t = uint8_t;
        using mask_t = __uint128_t;
        using slitom_t = SLITOM<mask_t, ulindex_t, lindex_t, value_t>;

        using tensor_manager_t = TensorManager<tensor_t, slitom_t>;

        tensor_manager_t *tensor_manager =
            new tensor_manager_t(options->get_gpu_count());
        bool sort_gpu = tensor_manager->FindPartitionNum(
            tensor_x->nnz, tensor_y->nnz, cuda_memory[0]->getPoolSize(),
            tensor_x->nbits);

        slitom_t *SX = new slitom_t(1, sort_gpu);
        slitom_t *SY = new slitom_t(2, sort_gpu);
        slitom_t *SZ = new slitom_t();

        tensor_manager->ConvertTensor_extra(
            tensor_x, SX, options->get_num_cmodes(),
            options->get_first_input_cmodes(), slitom_timer);
        tensor_manager->ConvertTensor_extra(
            tensor_y, SY, options->get_num_cmodes(),
            options->get_second_input_cmodes(), slitom_timer);
        // slitom_timer->printElapsed("Construct SLITOM");

        tensor_manager->SortSlitomXY(SX, SY, cuda_memory, slitom_timer);

        partition_timer->start();
        tensor_manager->Partition_128(SX);
        tensor_manager->Partition_128(SY);
        partition_timer->stop();
        partition_timer->printElapsed("Partition");

        if (SY->nprtn > 1) {
          printf("convert and sort\n");
          partition_timer->start();
          tensor_manager->convert_and_sort_128(SY, cuda_memory);
          partition_timer->stop();
          partition_timer->printElapsed("Convert and Sort SLITOM");
        }

        {
          int idxmatch_mode = options->get_idxmatch_mode();
          if (idxmatch_mode == 1) {
            tensor_manager->ExtractCOOCmodes(SX);
            tensor_manager->ExtractCOOCmodes(SY);
            tensor_manager->SortCOOPerPartition(SX);
            tensor_manager->SortCOOPerPartition(SY);
          } else if (idxmatch_mode == 2) {
            tensor_manager->ExtractBLCOKeys(SX);
            tensor_manager->ExtractBLCOKeys(SY);
            tensor_manager->SortBLCOPerPartition(SX);
            tensor_manager->SortBLCOPerPartition(SY);
          }
        }

        gsparc::extra::SpTC<tensor_t, slitom_t>(
            SX, SY, SZ, cuda_memory, options->get_gpu_count(), copy_timers,
            idxmatch_timers, dynamicprtn_timers, contraction_timers,
            options->is_dense_accumulator(), options->get_idxmatch_mode());
        result_nnz += SZ->nnz;

        delete SX;
        delete SY;
      } else if (extra_bytes == 2) {
        using lindex_t = uint64_t;
        using ulindex_t = uint16_t;
        using mask_t = __uint128_t;
        using slitom_t = SLITOM<mask_t, ulindex_t, lindex_t, value_t>;

        using tensor_manager_t = TensorManager<tensor_t, slitom_t>;

        tensor_manager_t *tensor_manager =
            new tensor_manager_t(options->get_gpu_count());
        /* Assume all GPU device has the same size of device memory*/
        bool sort_gpu = tensor_manager->FindPartitionNum(
            tensor_x->nnz, tensor_y->nnz, cuda_memory[0]->getPoolSize(),
            tensor_x->nbits);

        slitom_t *SX = new slitom_t(1, sort_gpu);
        slitom_t *SY = new slitom_t(2, sort_gpu);
        slitom_t *SZ = new slitom_t();

        tensor_manager->ConvertTensor_extra(
            tensor_x, SX, options->get_num_cmodes(),
            options->get_first_input_cmodes(), slitom_timer);
        tensor_manager->ConvertTensor_extra(
            tensor_y, SY, options->get_num_cmodes(),
            options->get_second_input_cmodes(), slitom_timer);
        // slitom_timer->printElapsed("Construct SLITOM");

        tensor_manager->SortSlitomXY(SX, SY, cuda_memory, slitom_timer);

        partition_timer->start();
        tensor_manager->Partition_128(SX);
        tensor_manager->Partition_128(SY);
        partition_timer->stop();
        partition_timer->printElapsed("Partition");

        if (SY->nprtn > 1) {
          printf("convert and sort\n");
          partition_timer->start();
          tensor_manager->convert_and_sort_128(SY, cuda_memory);
          partition_timer->stop();
          partition_timer->printElapsed("Convert and Sort SLITOM");
        }

        {
          int idxmatch_mode = options->get_idxmatch_mode();
          if (idxmatch_mode == 1) {
            tensor_manager->ExtractCOOCmodes(SX);
            tensor_manager->ExtractCOOCmodes(SY);
            tensor_manager->SortCOOPerPartition(SX);
            tensor_manager->SortCOOPerPartition(SY);
          } else if (idxmatch_mode == 2) {
            tensor_manager->ExtractBLCOKeys(SX);
            tensor_manager->ExtractBLCOKeys(SY);
            tensor_manager->SortBLCOPerPartition(SX);
            tensor_manager->SortBLCOPerPartition(SY);
          }
        }

        gsparc::extra::SpTC<tensor_t, slitom_t>(
            SX, SY, SZ, cuda_memory, options->get_gpu_count(), copy_timers,
            idxmatch_timers, dynamicprtn_timers, contraction_timers,
            options->is_dense_accumulator(), options->get_idxmatch_mode());
        result_nnz += SZ->nnz;

        delete SX;
        delete SY;
      } else if (extra_bytes == 4) {
        using lindex_t = uint64_t;
        using ulindex_t = uint32_t;
        using mask_t = __uint128_t;
        using slitom_t = SLITOM<mask_t, ulindex_t, lindex_t, value_t>;

        using tensor_manager_t = TensorManager<tensor_t, slitom_t>;

        tensor_manager_t *tensor_manager =
            new tensor_manager_t(options->get_gpu_count());
        bool sort_gpu = tensor_manager->FindPartitionNum(
            tensor_x->nnz, tensor_y->nnz, cuda_memory[0]->getPoolSize(),
            tensor_x->nbits);

        slitom_t *SX = new slitom_t(1, sort_gpu);
        slitom_t *SY = new slitom_t(2, sort_gpu);
        slitom_t *SZ = new slitom_t();

        tensor_manager->ConvertTensor_extra(
            tensor_x, SX, options->get_num_cmodes(),
            options->get_first_input_cmodes(), slitom_timer);
        tensor_manager->ConvertTensor_extra(
            tensor_y, SY, options->get_num_cmodes(),
            options->get_second_input_cmodes(), slitom_timer);
        // slitom_timer->printElapsed("Construct SLITOM");

        tensor_manager->SortSlitomXY(SX, SY, cuda_memory, slitom_timer);

        partition_timer->start();
        tensor_manager->Partition_128(SX);
        tensor_manager->Partition_128(SY);
        partition_timer->stop();
        partition_timer->printElapsed("Partition");

        if (SY->nprtn > 1) {
          printf("convert and sort\n");
          partition_timer->start();
          tensor_manager->convert_and_sort_128(SY, cuda_memory);
          partition_timer->stop();
          partition_timer->printElapsed("Convert and Sort SLITOM");
        }

        {
          int idxmatch_mode = options->get_idxmatch_mode();
          if (idxmatch_mode == 1) {
            tensor_manager->ExtractCOOCmodes(SX);
            tensor_manager->ExtractCOOCmodes(SY);
            tensor_manager->SortCOOPerPartition(SX);
            tensor_manager->SortCOOPerPartition(SY);
          } else if (idxmatch_mode == 2) {
            tensor_manager->ExtractBLCOKeys(SX);
            tensor_manager->ExtractBLCOKeys(SY);
            tensor_manager->SortBLCOPerPartition(SX);
            tensor_manager->SortBLCOPerPartition(SY);
          }
        }

        gsparc::extra::SpTC<tensor_t, slitom_t>(
            SX, SY, SZ, cuda_memory, options->get_gpu_count(), copy_timers,
            idxmatch_timers, dynamicprtn_timers, contraction_timers,
            options->is_dense_accumulator(), options->get_idxmatch_mode());
        result_nnz += SZ->nnz;

        delete SX;
        delete SY;
      } else if (extra_bytes == 8) {
        using lindex_t = uint64_t;
        using ulindex_t = uint64_t;
        using mask_t = __uint128_t;
        using slitom_t = SLITOM<mask_t, ulindex_t, lindex_t, value_t>;

        using tensor_manager_t = TensorManager<tensor_t, slitom_t>;

        tensor_manager_t *tensor_manager =
            new tensor_manager_t(options->get_gpu_count());
        bool sort_gpu = tensor_manager->FindPartitionNum(
            tensor_x->nnz, tensor_y->nnz, cuda_memory[0]->getPoolSize(),
            tensor_x->nbits);

        slitom_t *SX = new slitom_t(1, sort_gpu);
        slitom_t *SY = new slitom_t(2, sort_gpu);
        slitom_t *SZ = new slitom_t();

        tensor_manager->ConvertTensor_extra(
            tensor_x, SX, options->get_num_cmodes(),
            options->get_first_input_cmodes(), slitom_timer);
        tensor_manager->ConvertTensor_extra(
            tensor_y, SY, options->get_num_cmodes(),
            options->get_second_input_cmodes(), slitom_timer);
        // slitom_timer->printElapsed("Construct SLITOM");

        tensor_manager->SortSlitomXY(SX, SY, cuda_memory, slitom_timer);

        partition_timer->start();
        tensor_manager->Partition_128(SX);
        tensor_manager->Partition_128(SY);
        partition_timer->stop();
        partition_timer->printElapsed("Partition");

        if (SY->nprtn > 1) {
          printf("convert and sort\n");
          partition_timer->start();
          tensor_manager->convert_and_sort_128(SY, cuda_memory);
          partition_timer->stop();
          partition_timer->printElapsed("Convert and Sort SLITOM");
        }

        {
          int idxmatch_mode = options->get_idxmatch_mode();
          if (idxmatch_mode == 1) {
            tensor_manager->ExtractCOOCmodes(SX);
            tensor_manager->ExtractCOOCmodes(SY);
            tensor_manager->SortCOOPerPartition(SX);
            tensor_manager->SortCOOPerPartition(SY);
          } else if (idxmatch_mode == 2) {
            tensor_manager->ExtractBLCOKeys(SX);
            tensor_manager->ExtractBLCOKeys(SY);
            tensor_manager->SortBLCOPerPartition(SX);
            tensor_manager->SortBLCOPerPartition(SY);
          }
        }

        gsparc::extra::SpTC<tensor_t, slitom_t>(
            SX, SY, SZ, cuda_memory, options->get_gpu_count(), copy_timers,
            idxmatch_timers, dynamicprtn_timers, contraction_timers,
            options->is_dense_accumulator(), options->get_idxmatch_mode());
        result_nnz += SZ->nnz;

        delete SX;
        delete SY;
      } else {
        printf("extra bytes %d\n", extra_bytes);
        exit(0);
      }
    } else {
      using lindex_t = uint64_t;
      using ulindex_t = uint8_t; // not used
      using mask_t = uint64_t;
      using slitom_t = SLITOM<mask_t, ulindex_t, lindex_t, value_t>;

      using tensor_manager_t = TensorManager<tensor_t, slitom_t>;

      tensor_manager_t *tensor_manager =
          new tensor_manager_t(options->get_gpu_count());
      bool sort_gpu = tensor_manager->FindPartitionNum(
          tensor_x->nnz, tensor_y->nnz, cuda_memory[0]->getPoolSize(),
          tensor_x->nbits);

      slitom_t *SX = new slitom_t(1, sort_gpu);
      slitom_t *SY = new slitom_t(2, sort_gpu);
      slitom_t *SZ = new slitom_t();

      tensor_manager->ConvertTensor(tensor_x, SX, options->get_num_cmodes(),
                                    options->get_first_input_cmodes(),
                                    slitom_timer);
      tensor_manager->ConvertTensor(tensor_y, SY, options->get_num_cmodes(),
                                    options->get_second_input_cmodes(),
                                    slitom_timer);
      // slitom_timer->printElapsed("Construct SLITOM");

      // common::end_timer_with_msg(&timer, "Construct SLITOM");

      tensor_manager->SortSlitomXY(SX, SY, cuda_memory, slitom_timer);

      partition_timer->start();
      tensor_manager->Partition(SX);
      tensor_manager->Partition(SY);
      partition_timer->stop();
      partition_timer->printElapsed("Partition");

      if (SY->nprtn > 1) {
        printf("convert and sort\n");
        partition_timer->start();
        tensor_manager->convert_and_sort_64(SY, cuda_memory);
        partition_timer->stop();
        partition_timer->printElapsed("Convert and Sort SLITOM");
      }

      // Extract COO/BLCO data after partition+sort, then sort per partition
      int idxmatch_mode = options->get_idxmatch_mode();
      if (idxmatch_mode == 1) {
        tensor_manager->ExtractCOOCmodes(SX);
        tensor_manager->ExtractCOOCmodes(SY);
        tensor_manager->SortCOOPerPartition(SX);
        tensor_manager->SortCOOPerPartition(SY);
      } else if (idxmatch_mode == 2) {
        tensor_manager->ExtractBLCOKeys(SX);
        tensor_manager->ExtractBLCOKeys(SY);
        tensor_manager->SortBLCOPerPartition(SX);
        tensor_manager->SortBLCOPerPartition(SY);
      }

      if (options->is_multi_sptc()) {
        printf("in multi sptc\n");
        gsparc::multi_SpTC<tensor_t, slitom_t>(
            SX, SY, SZ, cuda_memory, options->get_gpu_count(), copy_timers,
            idxmatch_timers, dynamicprtn_timers, contraction_timers);

        printf("after multi sptc\n");
        /* Total time until here */
        double first_sptc_total_time =
            slitom_timer->getTotalTime() + partition_timer->getTotalTime();
        std::vector<double> copy_timers_vec_temp;
        std::vector<double> idxmatch_timers_vec_temp;
        std::vector<double> contraction_timers_vec_temp;
        std::vector<double> dynamicprtn_timers_vec_temp;
        for (int i = 0; i < options->get_gpu_count(); i++) {
          copy_timers_vec_temp.push_back(copy_timers[i]->getTotalTime());
          idxmatch_timers_vec_temp.push_back(
              idxmatch_timers[i]->getTotalTime());
          contraction_timers_vec_temp.push_back(
              contraction_timers[i]->getTotalTime());
          dynamicprtn_timers_vec_temp.push_back(
              dynamicprtn_timers[i]->getTotalTime());
        }
        first_sptc_total_time += (std::accumulate(
            copy_timers_vec_temp.begin(), copy_timers_vec_temp.end(), 0.0));
        first_sptc_total_time +=
            (std::accumulate(idxmatch_timers_vec_temp.begin(),
                             idxmatch_timers_vec_temp.end(), 0.0));
        first_sptc_total_time +=
            (std::accumulate(contraction_timers_vec_temp.begin(),
                             contraction_timers_vec_temp.end(), 0.0));
        first_sptc_total_time +=
            (std::accumulate(dynamicprtn_timers_vec_temp.begin(),
                             dynamicprtn_timers_vec_temp.end(), 0.0));

        fprintf(stderr, "First SpTC time: %f\n", first_sptc_total_time);
        reorder_bit_multi_sptc = slitom_timer->getTotalTime();
        slitom_timer->start();
#pragma omp parallel for
        for (index_t i = 0; i < SZ->nnz; ++i) {

          lindex_t first = SZ->indices[i] >> SZ->nfbits;
          lindex_t second = SZ->indices[i] & ((1 << SZ->nfbits) - 1);
          SZ->indices[i] = (first << SZ->nfbits) | second;
        }
        SZ->sort_gpu = true;
        slitom_timer->stop();
        slitom_timer->printElapsed("Bit reordering for multi-SpTC");
        reorder_bit_multi_sptc =
            slitom_timer->getTotalTime() - reorder_bit_multi_sptc;

        slitom_t *Z = new slitom_t();
        tensor_manager->ConvertTensor(
            tensor_z, Z, options->get_add_num_cmodes(),
            options->get_third_cmodes(), slitom_timer);
        tensor_manager->SortSlitomXY(SZ, Z, cuda_memory, slitom_timer);

        gsparc::multi_SpTC<tensor_t, slitom_t>(
            SZ, SY, Z, cuda_memory, options->get_gpu_count(), copy_timers,
            idxmatch_timers, dynamicprtn_timers, contraction_timers);
        fprintf(stderr, "\n\n\n");
        result_nnz += Z->nnz;
      } else {
        gsparc::SpTC<tensor_t, slitom_t>(
            SX, SY, SZ, cuda_memory, options->get_gpu_count(), copy_timers,
            idxmatch_timers, dynamicprtn_timers, contraction_timers,
            options->is_dense_accumulator(), idxmatch_mode);
        result_nnz += SZ->nnz;
      }
      delete SX;
      delete SY;
    }
    // delete SZ;

    // Reset timers after first iteration (warmup) so that only
    // iterations 2..N are counted in the final average.
    if (iter == 0 && num_iter > 1) {
      slitom_timer->reset();
      partition_timer->reset();
      for (int i = 0; i < options->get_gpu_count(); i++) {
        copy_timers[i]->reset();
        idxmatch_timers[i]->reset();
        contraction_timers[i]->reset();
        dynamicprtn_timers[i]->reset();
      }
      result_nnz = 0;
    }
  }

  // When warmup is used, effective iteration count excludes iter 0.
  int effective_iter = (num_iter > 1) ? (num_iter - 1) : num_iter;

  /* Final Print*/
  double slitom_t = slitom_timer->getTotalTime();
  double partition_t = partition_timer->getTotalTime();

  std::vector<double> copy_timers_vec;
  std::vector<double> idxmatch_timers_vec;
  std::vector<double> contraction_timers_vec;
  std::vector<double> dynamicprtn_timers_vec;
  for (int i = 0; i < options->get_gpu_count(); i++) {
    copy_timers_vec.push_back(copy_timers[i]->getTotalTime());
    idxmatch_timers_vec.push_back(idxmatch_timers[i]->getTotalTime());
    contraction_timers_vec.push_back(contraction_timers[i]->getTotalTime());
    dynamicprtn_timers_vec.push_back(dynamicprtn_timers[i]->getTotalTime());
  }

  double copy_t =
      (std::accumulate(copy_timers_vec.begin(), copy_timers_vec.end(), 0.0));
  double idxmatch_t = (std::accumulate(idxmatch_timers_vec.begin(),
                                       idxmatch_timers_vec.end(), 0.0));
  double contraction_t = (std::accumulate(contraction_timers_vec.begin(),
                                          contraction_timers_vec.end(), 0.0));
  double dynamicprtn_t = (std::accumulate(dynamicprtn_timers_vec.begin(),
                                          dynamicprtn_timers_vec.end(), 0.0));

  copy_t = copy_t / options->get_gpu_count();
  idxmatch_t = idxmatch_t / options->get_gpu_count();
  contraction_t = contraction_t / options->get_gpu_count();
  dynamicprtn_t = dynamicprtn_t / options->get_gpu_count();

  double total = slitom_t + partition_t + copy_t + idxmatch_t + contraction_t +
                 dynamicprtn_t;
  double total_per_iter = total / effective_iter;
  fprintf(stderr, "--------Break Down--------\n");
  fprintf(stderr, "SLITOM time: %f\n", slitom_t / effective_iter);
  fprintf(stderr, "Partition time: %f\n", partition_t / effective_iter);
  fprintf(stderr, "Copy time: %f\n", copy_t / effective_iter);
  fprintf(stderr, "Idxmatch time: %f\n", idxmatch_t / effective_iter);
  fprintf(stderr, "Contraction time: %f\n", contraction_t / effective_iter);
  fprintf(stderr, "Dynamic Partition time: %f\n", dynamicprtn_t / effective_iter);
  if (options->is_multi_sptc()) {
    fprintf(stderr, "Bit reordering time: %f\n", reorder_bit_multi_sptc);
    fprintf(stderr, "Bit reordering ratio: %f\%\n",
            (reorder_bit_multi_sptc / total_per_iter) * 100);
  }
  fprintf(stderr, "--------Elapsed Time--------\n");
  fprintf(stderr, "Result NNZ: %llu\n", result_nnz / effective_iter);
  fprintf(stderr, "SLITOM time: %f\n", slitom_t / effective_iter);
  fprintf(stderr, "Partition time: %f\n", partition_t / effective_iter);
  fprintf(stderr, "SpTC(IdxMatch + Contraction + Data Copy) time: %f\n",
          (copy_t + idxmatch_t + contraction_t + dynamicprtn_t) / effective_iter);
  fprintf(stderr, "Total time: %f\n", total_per_iter);

  printf("\n\n");
  fprintf(stderr, "\n\n");
  return 0;
}
