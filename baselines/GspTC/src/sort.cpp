
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <iostream>
#include <omp.h>


#include "sptensor.hpp"
#include "sort.hpp"

#include <cstdlib>  // for std::rand, std::srand
#include <ctime>    // for std::time


int compare_indices(SparseTensor * const X1, IType loc1,  SparseTensor * const X2, IType loc2)
{
    IType i;
    assert(X1->nmodes == X2->nmodes);
    for(i = 0; i < X1->nmodes; ++i) {
        IType eleind1 = X1->orig_cidx[i][loc1];
        IType eleind2 = X2->orig_cidx[i][loc2];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
}


bool is_sorted(SparseTensor *X, IType l, IType r) {
    for (IType i = l; i < r; ++i) {
        if (compare_indices(X, i, X, i + 1) > 0) {
            return false;  // 다음 원소가 이전 원소보다 작으면 정렬되지 않은 것입니다.
        }
    }
    return true;  // 모든 원소가 올바른 순서로 정렬된 경우
}

void insertion_sort(SparseTensor *X, IType l, IType r) {
    for (IType i = l + 1; i < r; ++i) {
        for (IType j = i; j > l && compare_indices(X, j - 1, X, j) > 0; --j) {
            swap_values(X, j, j - 1);
        }
    }
}

// Median-of-three 피벗 선택
IType median_of_three(SparseTensor* X, IType l, IType r) {
    IType mid = l + (r - l) / 2;
    if (compare_indices(X, l, X, mid) > 0) {
        swap_values(X, l, mid);
    }
    if (compare_indices(X, l, X, r) > 0) {
        swap_values(X, l, r);
    }
    if (compare_indices(X, mid, X, r) > 0) {
        swap_values(X, mid, r);
    }
    swap_values(X, mid, r - 1); // 피벗을 끝에서 두 번째로 이동
    return r - 1; // 피벗 인덱스 반환
}


static void quicksort_index(SparseTensor *X, IType l, IType r) {
    IType i, j, p;

    if(r-l < 2) {
        return;
    }
    if (is_sorted(X, l, r - 1)) {
        return;  // 배열이 이미 정렬된 상태면 정렬 작업을 수행하지 않습니다.
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(compare_indices(X, i, X, p) < 0) {
            ++i;
        }
        while(compare_indices(X, p, X, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        swap_values(X, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }

    #pragma omp task 	
	{ quicksort_index(X, l, i); }
	#pragma omp task 	
	{ quicksort_index(X, i, r); }	
}


void swap_values(SparseTensor *X, IType ind1, IType ind2) {

    for(IType i = 0; i < X->nmodes; ++i) {
        IType eleind1 = X->orig_cidx[i][ind1];
        X->orig_cidx[i][ind1] = X->orig_cidx[i][ind2];
        X->orig_cidx[i][ind2] = eleind1;
    }
    FType val1 = X->orig_vals[ind1];
    X->orig_vals[ind1] = X->orig_vals[ind2];
    X->orig_vals[ind2] = val1;
}




/**
 * Reorder the elements in a sparse tensor lexicographically
 * @param X  the sparse tensor to operate on
 */
void sort_index(SparseTensor *X, int force, int tk)
{
        #pragma omp parallel num_threads(tk)
        {    
            #pragma omp single nowait
            {
                quicksort_index(X, 0, X->nnz);
            }
        }
}
