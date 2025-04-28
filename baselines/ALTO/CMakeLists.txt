cmake_minimum_required(VERSION 3.10)
project(ALTO VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -mbmi2")

# Options
option(ALTO_USE_MKL "Use MKL instead of OpenBLAS" OFF)
option(ALTO_ALTERNATIVE_PEXT "Use alternative PEXT implementation" OFF)
option(ALTO_THP_PRE_ALLOCATION "Enable THP pre-allocation" OFF)
option(ALTO_MEM_TRACE "Enable memory tracing" OFF)
option(ALTO_DEBUG "Enable debug mode" OFF)

# Find required packages
find_package(OpenMP REQUIRED)
if(ALTO_USE_MKL)
    find_package(MKL REQUIRED)
else()
    find_package(OpenBLAS REQUIRED)
endif()

include_directories(${OpenBLAS_INCLUDE_DIRS})
link_libraries(${OpenBLAS_LIBRARIES})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add common definitions
add_definitions(-DALTO_MODES_SPECIALIZED=0,3,4,5)
add_definitions(-DALTO_RANKS_SPECIALIZED=0,8,16,100)
add_definitions(-DMAX_NUM_MODES=10)
add_definitions(-D__BMI2__)

if(ALTO_USE_MKL)
    add_definitions(-DMKL)
endif()
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
file(GLOB_RECURSE SOURCES "src/*.cpp")

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
    )
    
    if(ALTO_USE_MKL)
        target_link_libraries(${TARGET_NAME} PRIVATE ${MKL_LIBRARIES})
    endif()
    
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