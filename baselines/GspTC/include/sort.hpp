#ifndef SORT_HPP_
#define SORT_HPP_

#include <iostream>
#include "sptensor.hpp"


int compare_indices(SparseTensor * const X1, IType loc1,  SparseTensor * const X2, IType loc2);
static void quicksort_index(SparseTensor *X, IType l, IType r);
void swap_values(SparseTensor *X, IType ind1, IType ind2);
void sort_index(SparseTensor *X, int force, int tk);
bool is_sorted(SparseTensor *X, IType l, IType r);

#endif // SORT_HPP_