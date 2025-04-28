cmake_minimum_required(VERSION 3.10)
project(GSPARC LANGUAGES CXX CUDA)


set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS_RELEASE   "-O3 -march=native -fopenmp -D_GLIBCXX_PARALLEL -mbmi2")
set(CMAKE_CXX_FLAGS_DEBUG     "-g -O0")
set(CMAKE_CUDA_FLAGS          "-O3 -arch=sm_75 -Xcompiler=-fopenmp -mbmi2 -rdc=true")
# set(CMAKE_CUDA_FLAGS          "-O3 -arch=sm_80 -Xcompiler=-fopenmp -mbmi2 -rdc=true")



set(CMAKE_CUDA_SEPARABLE_COMPILATION   ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


set(CMAKE_CUDA_ARCHITECTURES 75)


find_package(OpenMP REQUIRED)
find_package(CUDAToolkit 12.2 REQUIRED)
find_package(TBB       REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/include)


file(GLOB_RECURSE COMMON_SRC src/common/*.cpp)
file(GLOB_RECURSE GSPARC_SRC  src/gsparc/*.cpp)
set(CUDA_MAIN main.cu)


add_executable(gsparc
  ${COMMON_SRC}
  ${GSPARC_SRC}
  ${CUDA_MAIN}
)


target_link_libraries(gsparc PUBLIC
  OpenMP::OpenMP_CXX
  TBB::tbb                     
  pthread
  dl
  m
  openblas
  CUDA::cudart
)

