# Makefile for GSPARC project

# CUDA 설정
CUDA_ROOT_DIR   = /usr/local/cuda
NVCC            = nvcc
CUDA_LIB_DIR    = -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR    = -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS  = -lcudart

# 컴파일러 및 공통 옵션
CXX       = g++
OPENMP    = -fopenmp
BLASLIBS  =

# NVCC 플래그 (TBB, BMI2 등 기존 옵션 포함)
NVCC_FLAGS = -Xcompiler -fopenmp -ltbb -Xcompiler -mbmi2 -O3 -arch=sm_120 -rdc=true

# 빌드 설정
DEBUG    = false
MEMTRACE = false

ifeq ($(DEBUG),true)
    CXXFLAGS = $(OPENMP) -O0 -g -Wall -std=c++17 -D_GLIBCXX_PARALLEL
    LIBS     = -lpthread -lm -ldl $(BLASLIBS)
else
    CXXFLAGS = $(OPENMP) -O3 -march=native -static -g -std=c++17 -D_GLIBCXX_PARALLEL
    LIBS     = -lpthread -lm -ldl $(BLASLIBS)
endif

# 인클루드 디렉토리
INCLUDES = -Iinclude
CXXFLAGS = $(OPENMP) -O3 -march=native -static -g -std=c++17 -D_GLIBCXX_PARALLEL
CPPFLAGS = $(INCLUDES) $(CUDA_INC_DIR)

# 빌드 디렉토리d
BUILD_DIR       = build-gsparc
BUILD_COMMON    = $(BUILD_DIR)/common
BUILD_GSPARC    = $(BUILD_DIR)/gsparc

# 소스 파일 (src 디렉토리의 .cpp 파일과 최상위의 main.cu)
SRC_COMMON  = $(wildcard src/common/*.cpp)
SRC_GSPARC  = $(wildcard src/gsparc/*.cpp)
CUDA_SRC    = main.cu

# 오브젝트 파일 (빌드 디렉토리 내 동일한 구조로 생성)
OBJ_COMMON  = $(patsubst src/common/%.cpp, $(BUILD_COMMON)/%.o, $(SRC_COMMON))
OBJ_GSPARC  = $(patsubst src/gsparc/%.cpp, $(BUILD_GSPARC)/%.o, $(SRC_GSPARC))
OBJ_MAIN    = $(BUILD_DIR)/main.o

# 최종 실행 파일 이름
TARGET = gsparc

# 기본 타겟
all: $(TARGET)

# 링크 단계 (CUDA를 사용하기 위해 nvcc로 링크)
$(TARGET): $(OBJ_COMMON) $(OBJ_GSPARC) $(OBJ_MAIN) | $(BUILD_DIR)
	@echo "===>  LINKING $(TARGET)"
	$(NVCC) $(NVCC_FLAGS) $(CPPFLAGS) $(OBJ_COMMON) $(OBJ_GSPARC) $(OBJ_MAIN) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(LIBS) -o $(TARGET)

# src/common 의 .cpp 파일 컴파일
$(BUILD_COMMON)/%.o: src/common/%.cpp | $(BUILD_COMMON)
	@echo "===>  COMPILE $@"
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# src/gsparc 의 .cpp 파일 컴파일
$(BUILD_GSPARC)/%.o: src/gsparc/%.cpp | $(BUILD_GSPARC)
	@echo "===>  COMPILE $@"
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# 최상위의 main.cu (CUDA 파일) 컴파일
$(OBJ_MAIN): main.cu | $(BUILD_DIR)
	@echo "===>  COMPILE $@"
	$(NVCC) -c $(NVCC_FLAGS) $(CPPFLAGS) $(CUDA_INC_DIR) $< -o $@

# 빌드 디렉토리 생성
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_COMMON):
	@mkdir -p $(BUILD_COMMON)

$(BUILD_GSPARC):
	@mkdir -p $(BUILD_GSPARC)

# 클린 타겟
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
