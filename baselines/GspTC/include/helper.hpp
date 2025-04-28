#ifndef HELPER_HPP
#define HELPER_HPP

#include <iostream>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


template <typename T>
inline void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(99);
    }
}

inline void *pinned_malloc(size_t size){
    void *p;
    checkCudaErrors(cudaHostAlloc(&p, size, cudaHostAllocPortable));
    if (p == NULL) {
        printf("p is NULL\n");
        exit(1);
    }
    
    return p;
}


inline void pinned_free(void *p){
    checkCudaErrors(cudaFreeHost(p));
    p = NULL;
}

inline char * sptBytesString(uint64_t const bytes)
{
  double size = (double)bytes;
  int suff = 0;
  const char *suffix[5] = {"B", "KiB", "MiB", "GiB", "TiB"};
  while(size > 1024 && suff < 4) {
    size /= 1024.;
    ++suff;
  }
  char * ret = NULL;
  if(asprintf(&ret, "%0.2f %s", size, suffix[suff]) == -1) {
    fprintf(stderr, "SPT: asprintf failed with %" PRIu64 " bytes.\n", bytes);
    ret = NULL;
  }
  return ret;
}

#endif