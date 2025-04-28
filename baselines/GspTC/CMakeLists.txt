cmake_minimum_required(VERSION 3.22)
project(GspTC LANGUAGES CXX CUDA)


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


find_package(CUDAToolkit REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(TBB REQUIRED)


include_directories(${OpenBLAS_INCLUDE_DIRS})
link_libraries(${OpenBLAS_LIBRARIES})


file(GLOB SRC_COMMON src/common/*.cpp)
file(GLOB SRC_GSPTC src/*.cpp)

add_executable(GspTC
    main.cu
    ${SRC_COMMON}
    ${SRC_GSPTC}
)


set_target_properties(GspTC PROPERTIES
    CXX_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINK_FLAGS "-Wl,--no-as-needed"
)


target_compile_options(GspTC PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -arch=sm_75 -Xcompiler=-fopenmp -Xcompiler=-mbmi2 -rdc=true>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -D_GLIBCXX_PARALLEL -mbmi2>
)


target_include_directories(GspTC PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(GspTC PRIVATE
    TBB::tbb
    CUDA::cudart
    CUDA::cublas
    gfortran
    pthread
    dl
    openblas
)
