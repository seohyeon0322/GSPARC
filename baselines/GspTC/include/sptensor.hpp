#ifndef SPTENSOR_HPP_
#define SPTENSOR_HPP_



#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <getopt.h>
#include <omp.h>
#include <time.h>
#include "common.hpp"
#include "helper.hpp" 
#include "pinned_memory.cuh"


struct sptensor_struct {
  // number of modes/dimensions in the tensor
  int nmodes;
  int fnmodes;
  int cnmodes;
  // array that stores the length of each mode 
  IType* dims;
  IType* orig_dims;

  IType* mode;

  // array that stores free mode index
  IType * cmode;
  IType* fmode;
  // number of non-zeros
  IType nnz;

  // a 'num_modes' x 'nnz' matrix containing the indices of the non-zeros
  // index[0][1...nnz] stores the 1st mode indices
  // index[nmodes - 1][1...nnz] stores the last mode indices
  IType** cidx;
  IType** orig_cidx;

  // IType ** free_idx;

  // stores the non-zeros
  FType* vals;
  FType * orig_vals;
};



struct tmp_sptensor_struct_element {

  // array that stores the length of each mode 
  IType* cmode_ind;

  // number of non-zeros
  IType start;
  IType end;
  IType nnz;
  IType mc;

};

typedef struct sptensor_struct SparseTensor;
typedef struct tmp_sptensor_struct_element tmp_SparseTensor_element;


struct tmp_sptensor_struct{

  // array that stores the length of each mode 
  tmp_SparseTensor_element** XO;

  // number of non-zeros
  IType size;
  IType cmode_len;
  int prtn_dim;
};
typedef struct tmp_sptensor_struct tmp_SparseTensor;


typedef enum FileFormat_
{
    TEXT_FORMAT = 0,
    BINARY_FORMAT = 1
} FileFormat;

void PrintTensorInfo(SparseTensor* X);

void ExportSparseTensor(const char *file_path, FileFormat f, SparseTensor *X);

void read_sparse_tensor(const char *file_path, SparseTensor **sptensor, int based, PinnedMemoryPool *pool);

void parse_metadata_and_data(const char *buffer, const size_t buffer_length, SparseTensor *sptensor, int based, PinnedMemoryPool *pool);

int get_nmodes(const char *buffer);

void ImportSparseTensor(const char *file_path, FileFormat f, SparseTensor **X_);

void ImportQuantumTensor(const char* filepath, SparseTensor** X_, PinnedMemoryPool *pool);

void DestroySparseTensor(SparseTensor *X);


void permute_tensor(SparseTensor *X, IType *mode_order_X);

void preprocess_tensor(SparseTensor *X, SparseTensor *Y, IType *dims_acc, tmp_sptensor_struct **tmp_X, PinnedMemoryPool *pinned_pool, CudaMemoryPool *cuda_pool);
int compare_cmodes(IType cmode_len, IType *tmp_ind, IType **X_ind, IType idx);
int compare_cmodes_tmp(IType cmode_len, IType *tmp_ind, IType *X_ind);

void write_back(FType *ZT_vals, SparseTensor *Z, IType *acc_dims, PinnedMemoryPool *pinned_pool);
#endif // SPTENSOR_HPP_