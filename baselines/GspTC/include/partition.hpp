#ifndef PARTITION_HPP
#define PARTITION_HPP

#include "sptensor.hpp" 

// #include "helper.hpp"

void basic_partition(tmp_SparseTensor *tmp_XO, tmp_SparseTensor *P, PinnedMemoryPool *pinned_pool, int partition_dim);
void partition1D(tmp_SparseTensor *tmp_X, SparseTensor *X, int cmode_len, PinnedMemoryPool *pinned_pool);
void partition2D(tmp_SparseTensor *tmp_XO, SparseTensor *Y, tmp_SparseTensor** P1, tmp_SparseTensor** Q1, PinnedMemoryPool *pinned_pool, int partition_dim);
void partition3D(tmp_SparseTensor *tmpXO, SparseTensor *Y, tmp_SparseTensor *P1, tmp_SparseTensor **P, tmp_SparseTensor *Q1, tmp_SparseTensor **Q, PinnedMemoryPool *pinned_pool, int partition_dim);


#endif