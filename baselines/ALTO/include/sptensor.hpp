#ifndef SPTENSOR_HPP_
#define SPTENSOR_HPP_


#include "common.hpp"


struct sptensor_struct {
  // number of modes/dimensions in the tensor
  int nmodes;

  // array that stores the length of each mode 
  IType* dims;

  // number of non-zeros
  IType nnz;

  // a 'num_modes' x 'nnz' matrix containing the indices of the non-zeros
  // index[0][1...nnz] stores the 1st mode indices
  // index[nmodes - 1][1...nnz] stores the last mode indices
  IType** cidx;

  // stores the non-zeros
  FType* vals;
};
typedef struct sptensor_struct SparseTensor;

typedef enum FileFormat_
{
    TEXT_FORMAT = 0,
    BINARY_FORMAT = 1
} FileFormat;

void ExportSparseTensor(const char *file_path, FileFormat f, SparseTensor *X);

void ImportSparseTensor(const char *file_path, FileFormat f, SparseTensor **X_);

void ImportQuantumTensor(const char* filepath, SparseTensor** X_);


void read_sparse_tensor(const char *file_path, SparseTensor **sptensor, int based);

void parse_metadata_and_data(const char *buffer, const size_t buffer_length, SparseTensor *sptensor, int based);

int get_nmodes(const char *buffer);

void DestroySparseTensor(SparseTensor *X);

void CreateSparseTensor(
  IType mode, 
  IType* dims, 
  IType nnz,
  IType* cidx, 
  FType* vals, 
  SparseTensor** X_
);


#endif // SPTENSOR_HPP_
