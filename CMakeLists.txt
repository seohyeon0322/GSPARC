cmake_minimum_required(VERSION 3.18)
project(GSPARC LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Tested with: CUDA 13.0, g++ 13.3, Ubuntu 24.04, NVIDIA RTX PRO 6000 (sm_120)
# Requires   : OpenMP, oneTBB (std::execution::par), a CPU with BMI2 (pdep/pext)
# Note       : -march=native ties the binary to the machine it is built on
# ---------------------------------------------------------------------------

# GPU Compute Capability: 75 = T4/RTX 20xx, 86 = RTX 30xx/A6000, 90 = H100,
# 120 = RTX PRO 6000 (Blackwell, the GPU used in the paper)
set(CMAKE_CUDA_ARCHITECTURES 120)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Host (.cpp) flags — same as the Makefile used for the paper binary
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mbmi2 -D_GLIBCXX_PARALLEL")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall")

# Device (main.cu) flags — same as the Makefile: nvcc -O3 -rdc=true -Xcompiler -fopenmp,-mbmi2
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG   "-O0 -g -G")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp,-mbmi2")
set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)   # -rdc=true
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Dependencies
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 12.8)
  message(WARNING "GSPARC was tested with CUDA 13.0; sm_120 (Blackwell) needs CUDA >= 12.8. Found ${CUDAToolkit_VERSION}.")
endif()

# Sources
file(GLOB_RECURSE COMMON_SRC ${PROJECT_SOURCE_DIR}/src/common/*.cpp)
file(GLOB_RECURSE GSPARC_SRC ${PROJECT_SOURCE_DIR}/src/gsparc/*.cpp)

add_executable(gsparc
  ${COMMON_SRC}
  ${GSPARC_SRC}
  ${PROJECT_SOURCE_DIR}/main.cu
)
target_include_directories(gsparc PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link: cudart (host runtime), OpenMP, TBB (parallel std::sort), pthread/dl/m
target_link_libraries(gsparc PRIVATE
  CUDA::cudart
  OpenMP::OpenMP_CXX
  TBB::tbb
  Threads::Threads
  ${CMAKE_DL_LIBS}
  m
)

# Binary at <build>/gsparc (scripts use ./build/gsparc)
set_target_properties(gsparc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "GSPARC: CUDA ${CUDAToolkit_VERSION}, arch=${CMAKE_CUDA_ARCHITECTURES}, host compiler=${CMAKE_CUDA_HOST_COMPILER}")
