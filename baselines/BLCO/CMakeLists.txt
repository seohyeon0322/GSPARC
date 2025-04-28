cmake_minimum_required(VERSION 3.10)
project(BLCO LANGUAGES CXX CUDA)

# 1) CUDA 호스트 컴파일러 지정
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS_RELEASE   "-O3 -march=native -fopenmp -D_GLIBCXX_PARALLEL -mbmi2")
set(CMAKE_CXX_FLAGS_DEBUG     "-g -O0")
set(CMAKE_CUDA_FLAGS          "-O3 -arch=sm_75 -Xcompiler=-fopenmp -mbmi2 -rdc=true")
# set(CMAKE_CUDA_FLAGS          "-O3 -arch=sm_80 -Xcompiler=-fopenmp -mbmi2 -rdc=true")


set(CMAKE_CUDA_SEPARABLE_COMPILATION   ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Options
option(ALTO_ALTERNATIVE_PEXT "Use alternative PEXT implementation" OFF)
option(ALTO_THP_PRE_ALLOCATION "Enable THP pre-allocation" OFF)
option(ALTO_MEM_TRACE "Enable memory tracing" OFF)
option(ALTO_DEBUG "Enable debug mode" OFF)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(CUDAToolkit 12.2 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenBLAS_INCLUDE_DIRS})

# Add common definitions
add_definitions(-DALTO_MODES_SPECIALIZED=0)
add_definitions(-DALTO_RANKS_SPECIALIZED=0)
add_definitions(-DMAX_NUM_MODES=10)
add_definitions(-D__BMI2__)

if(ALTO_ALTERNATIVE_PEXT)
    add_definitions(-DALT_PEXT)
endif()
if(ALTO_THP_PRE_ALLOCATION)
    add_definitions(-DALTO_PRE_ALLOC)
endif()
if(ALTO_MEM_TRACE)
    add_definitions(-DALTO_MEM_TRACE)
endif()
if(ALTO_DEBUG)
    add_definitions(-DALTO_DEBUG)
endif()

# Source files
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CU_SOURCES "src/*.cu")
set(SOURCES ${CPP_SOURCES} ${CU_SOURCES})

# Function to create executable with specific mask length
function(add_alto_executable MASK_LENGTH)
    set(TARGET_NAME "cpd${MASK_LENGTH}")
    
    # Create executable
    add_executable(${TARGET_NAME} ${SOURCES})
    
    # Add mask length definition
    target_compile_definitions(${TARGET_NAME} PRIVATE ALTO_MASK_LENGTH=${MASK_LENGTH})
    
    # Link libraries
    target_link_libraries(${TARGET_NAME}
        PRIVATE
        OpenMP::OpenMP_CXX
        ${OpenBLAS_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::curand
        openblas
    )
    
    # Set output directory
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # Install target
    install(TARGETS ${TARGET_NAME}
        DESTINATION bin
    )
endfunction()

# Create both 64-bit and 128-bit versions
add_alto_executable(64)
add_alto_executable(128) 