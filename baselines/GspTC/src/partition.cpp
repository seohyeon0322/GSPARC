#include "partition.hpp"
#include "sptensor.hpp"
#include "pinned_memory.cuh"

void partition1D(tmp_SparseTensor *tmp_X, SparseTensor *X, int cmode_len, PinnedMemoryPool *pinned_pool){
  IType sum = 0;
  tmp_X->XO[sum] = (tmp_SparseTensor_element *)pinned_pool->allocate(sizeof(tmp_SparseTensor_element));
  tmp_X->XO[sum]->cmode_ind = (IType *)pinned_pool->allocate(sizeof(IType) * cmode_len);

  for (IType j = 0; j < cmode_len; j++)
  {
    tmp_X->XO[sum]->cmode_ind[j] = X->cidx[j][0];
  }
  tmp_X->XO[sum]->start = 0;
  tmp_X->XO[sum]->end = 1;
  for (IType i = 1 ; i < X->nnz; i++)
  {
    int m = compare_cmodes(cmode_len, tmp_X->XO[sum]->cmode_ind, X->cidx, i);
    if (m == 0)
    {
      tmp_X->XO[sum]->end++;
    }
    else
    {
      tmp_X->XO[sum]->nnz = tmp_X->XO[sum]->end - tmp_X->XO[sum]->start;
      sum++;
      tmp_X->XO[sum] = (tmp_SparseTensor_element *)pinned_pool->allocate(sizeof(tmp_SparseTensor_element));
      tmp_X->XO[sum]->cmode_ind = (IType *)pinned_pool->allocate(sizeof(IType) * cmode_len);

      for (int j = 0; j < cmode_len; j++)
      {
        tmp_X->XO[sum]->cmode_ind[j] = X->cidx[j][i];
      }
      tmp_X->XO[sum]->start = tmp_X->XO[sum - 1]->end;
      tmp_X->XO[sum]->end = tmp_X->XO[sum]->start + 1;
    }
  }
  printf("flag2\n");
  tmp_X->XO[sum]->end = X->nnz;
  tmp_X->XO[sum]->nnz = tmp_X->XO[sum]->end - tmp_X->XO[sum]->start;
  tmp_X->size = ++sum;
  // printf("sum: %d\n", sum);
  // printf("tmp_X->end: %d\n", tmp_X->XO[sum - 1]->end);
  // printf("cmode len: %d\n", cmode_len);
  tmp_X->cmode_len = cmode_len;
}


void basic_partition(tmp_SparseTensor *tmp_XO, tmp_SparseTensor *P, PinnedMemoryPool *pinned_pool, int partition_dim)
{
  IType sumP = 0;
  int K = partition_dim;
  P->XO[sumP] = (tmp_SparseTensor_element *)pinned_pool->allocate(sizeof(tmp_SparseTensor_element));
  P->XO[sumP]->cmode_ind = (IType *)pinned_pool->allocate(sizeof(IType) * K);
  for (IType j = 0; j < K; j++)
  {
    P->XO[sumP]->cmode_ind[j] = tmp_XO->XO[0]->cmode_ind[j];
  }
  P->XO[sumP]->start = 0;
  P->XO[sumP]->end = P->XO[sumP]->start + 1;
  for (IType i = 1; i < tmp_XO->size; i++)
  {
    int m = compare_cmodes_tmp(K, P->XO[sumP]->cmode_ind, tmp_XO->XO[i]->cmode_ind);

    if (m == 0)
    {
      P->XO[sumP]->end++;
    }
    else
    {
      P->XO[sumP]->nnz = P->XO[sumP]->end - P->XO[sumP]->start;
      sumP++;
      P->XO[sumP] = (tmp_SparseTensor_element *)pinned_pool->allocate(sizeof(tmp_SparseTensor_element));
      P->XO[sumP]->cmode_ind = (IType *)pinned_pool->allocate(sizeof(IType) * K);

      for (IType j = 0; j < K; j++)
      {
        P->XO[sumP]->cmode_ind[j] = tmp_XO->XO[i]->cmode_ind[j];
      }
      P->XO[sumP]->start = P->XO[sumP - 1]->end;
      P->XO[sumP]->end = P->XO[sumP]->start + 1;
    }
  }
  P->XO[sumP]->end = tmp_XO->XO[tmp_XO->size - 1]->end;
  P->XO[sumP]->nnz = P->XO[sumP]->end - P->XO[sumP]->start;
  P->size = ++sumP;
  P->cmode_len = tmp_XO->cmode_len;
  P->prtn_dim = K;
}
void partition2D(tmp_SparseTensor *tmp_XO, SparseTensor *Y, tmp_SparseTensor **P1, tmp_SparseTensor **Q1, PinnedMemoryPool *pinned_pool, int partition_dim)
{

  // P->XO = (tmp_SparseTensor_element**)pinned_pool->allocate(sizeof(tmp_SparseTensor_element*)*tmp_XO->size);
  // P2->XO = (tmp_SparseTensor_element**)pinned_pool->allocate(sizeof(tmp_SparseTensor_element*)*tmp_XO->size);
  tmp_SparseTensor *_P1 = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
  tmp_SparseTensor *_Q1 = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
  _P1->XO = (tmp_SparseTensor_element **)pinned_pool->allocate(sizeof(tmp_SparseTensor_element *) * Y->nnz); //TODO: change to X->nnz?
  _Q1->XO = (tmp_SparseTensor_element **)pinned_pool->allocate(sizeof(tmp_SparseTensor_element *) * Y->nnz);

  int K = partition_dim;
  // int K = 1;
  basic_partition(tmp_XO, _P1, pinned_pool, K);
  IType cmode_len = tmp_XO->cmode_len;
  IType sumQ1 = 0;
  _Q1->XO[sumQ1] = (tmp_SparseTensor_element *)pinned_pool->allocate(sizeof(tmp_SparseTensor_element));
  _Q1->XO[sumQ1]->cmode_ind = (IType *)pinned_pool->allocate(sizeof(IType) * K);

  for (IType j = 0; j < K; j++)
  {
    _Q1->XO[sumQ1]->cmode_ind[j] = Y->cidx[j][0];
  }
  _Q1->XO[sumQ1]->start = 0;
  _Q1->XO[sumQ1]->end = 1;

  for (IType i = 1 ; i <  Y->nnz; i++)
  {
    bool m = compare_cmodes(K, _Q1->XO[sumQ1]->cmode_ind, Y->cidx, i);
        
    if (m == 0)
    {
      _Q1->XO[sumQ1]->end++;
    }
    else
    {

      _Q1->XO[sumQ1]->nnz = _Q1->XO[sumQ1]->end - _Q1->XO[sumQ1]->start;
      sumQ1++;
      _Q1->XO[sumQ1] = (tmp_SparseTensor_element *)pinned_pool->allocate(sizeof(tmp_SparseTensor_element));
      _Q1->XO[sumQ1]->cmode_ind = (IType *)pinned_pool->allocate(sizeof(IType) * K);

      for (int j = 0; j < K; j++)
      {
        _Q1->XO[sumQ1]->cmode_ind[j] = Y->cidx[j][i];
      }
      _Q1->XO[sumQ1]->start = _Q1->XO[sumQ1 - 1]->end;
      _Q1->XO[sumQ1]->end = _Q1->XO[sumQ1]->start + 1;
    }
  }
  _Q1->XO[sumQ1]->end = Y->nnz;
  _Q1->XO[sumQ1]->nnz = _Q1->XO[sumQ1]->end - _Q1->XO[sumQ1]->start;
  _Q1->size = ++sumQ1;
  _Q1->cmode_len = cmode_len;
  _Q1->prtn_dim = K;

  *P1 = _P1;
  *Q1 = _Q1;

  return;
}

void partition3D(tmp_SparseTensor *tmpXO, SparseTensor *Y, tmp_SparseTensor *P1, tmp_SparseTensor **P, tmp_SparseTensor *Q1, tmp_SparseTensor **Q, PinnedMemoryPool *pinned_pool, int partition_dim)
{
  IType sumP2 = 0;
  IType sumQ2 = 0;
  int K = partition_dim;
  // int K = 1;

  IType i = 0;
  IType j = 0;
  IType thrval = 1000000000000; // TODO: modify
  tmp_SparseTensor *P2 = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
  tmp_SparseTensor *Q2 = (tmp_SparseTensor *)pinned_pool->allocate(sizeof(tmp_SparseTensor));
  P2->XO = (tmp_SparseTensor_element **)pinned_pool->allocate(sizeof(tmp_SparseTensor_element *) * P1->size);
  Q2->XO = (tmp_SparseTensor_element **)pinned_pool->allocate(sizeof(tmp_SparseTensor_element *) * Y->nnz);
  // printf("P1 size: %d\n", P1->size);
  // printf("Q1 size: %d\n", Q1->size);
  // printf("tmpXO size: %d\n", tmpXO->size);

  while (i < P1->size && j < Q1->size)
  {

    int m = compare_cmodes_tmp(K, P1->XO[i]->cmode_ind, Q1->XO[j]->cmode_ind);
    if (m > 0)
      j++;
    else if (m < 0)
      i++;
    else
    {
      P2->XO[sumP2] = P1->XO[i];
      IType t1 = P1->XO[i]->nnz; // P1의 sum 크기
      IType q1 = Q1->XO[j]->nnz; // Q1의 sum 크기
      if (t1 * q1 > thrval)
      {
        printf("t1: %d, q1: %d\n", t1, q1);
        // Calculate how many parts to divide into
        IType num_parts = (t1 * q1) / thrval + ((t1 * q1) % thrval != 0);
        IType nnz_per_part = q1 / num_parts; // Base size of nnz for each part

        for (IType part = 0; part < num_parts; part++)
        {
          // Determine the start index for this part
          IType part_start_index = part * nnz_per_part + std::min(part, q1 % num_parts);
          IType part_nnz = nnz_per_part + (part < (q1 % num_parts) ? 1 : 0); // Adjust the nnz count for uneven splits

          // Allocate and set properties for each part
          Q2->XO[sumQ2] = (tmp_SparseTensor_element *)pinned_pool->allocate(sizeof(tmp_SparseTensor_element));
          Q2->XO[sumQ2]->start = Q1->XO[j]->start + part_start_index;
          Q2->XO[sumQ2]->end = Q2->XO[sumQ2]->start + part_nnz;
          Q2->XO[sumQ2]->nnz = part_nnz;
          Q2->XO[sumQ2]->cmode_ind = Q1->XO[j]->cmode_ind;
          Q2->XO[sumQ2]->mc = sumP2;
          sumQ2++;
        }
      }
      else
      {
        Q2->XO[sumQ2] = Q1->XO[j];
        Q2->XO[sumQ2]->mc = sumP2;
        sumQ2++;
      }
      sumP2++;
      i++;
      j++;
    }
  }
  printf("P1 size: %d, Q1 size: %d\n", P1->size, Q1->size);
  printf("sumP2: %d, sumQ2: %d\n", sumP2, sumQ2);
  P2->size = sumP2;
  Q2->size = sumQ2;
  // printf("P size: %d, Q size: %d\n", P2->size, Q2->size);
  // printf("P end: %d, Q end: %d\n", P2->XO[sumP2 - 1]->end, Q2->XO[sumQ2 - 1]->end);
  P2->cmode_len = tmpXO->cmode_len;
  Q2->cmode_len = tmpXO->cmode_len;
  P2->prtn_dim = K;
  Q2->prtn_dim = K;
  *P = P2;
  *Q = Q2;
  return;
}