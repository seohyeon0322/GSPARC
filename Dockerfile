FROM nvidia/cuda:12.2.0-devel-ubuntu22.04


RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc-11 \
    g++-11 \
    libomp-dev \
    libtbb-dev \
    tar \
    libnuma-dev \
    libblas-dev \
    liblapack-dev \
    git \
    cmake \
    pkg-config \
    gfortran \
    python3 \
    python3-pip \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*


  RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib

# 2. CMake 
WORKDIR /root/Util
RUN wget https://cmake.org/files/v3.22/cmake-3.22.1-linux-x86_64.tar.gz
RUN tar xvfz cmake-3.22.1-linux-x86_64.tar.gz
ENV CMAKE_HOME=/root/Util/cmake-3.22.1-linux-x86_64
ENV PATH=$CMAKE_HOME/bin:$PATH

# 3. OpenBLAS 
WORKDIR /root/OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git . && \
    git checkout v0.3.23 && \
    make -j$(nproc) && \
    make PREFIX=/usr install && \
    ldconfig

WORKDIR /root/GSPARC
COPY ./ ./

# GSPARC 
RUN rm -rf build && mkdir build
WORKDIR /root/GSPARC/build
RUN cmake ../ \
    -DCMAKE_CXX_COMPILER=g++-11 \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include \
    -DCUDA_CUBLAS_LIBRARIES=/usr/local/cuda/lib64/libcublas.so \
    -DCUDA_CUSOLVER_LIBRARIES=/usr/local/cuda/lib64/libcusolver.so
RUN make -j$(nproc)

# Sparta 
WORKDIR /root/GSPARC/baselines/Sparta
RUN rm -rf build && mkdir -p build
WORKDIR /root/GSPARC/baselines/Sparta/build
RUN cmake .. \
    -DCMAKE_CXX_COMPILER=g++-11 \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_OPENMP=ON \
    -DUSE_OpenBLAS=ON \
    -DBUILD_SHARED=ON \
    -DOpenBLAS_DIR=/usr/lib/cmake/openblas \
    -DOpenBLAS_INCLUDE_DIRS=/usr/include \
    -DOpenBLAS_LIBRARIES=/usr/lib/libopenblas.so
RUN make -j$(nproc)

# GspTC 
WORKDIR /root/GSPARC/baselines/GspTC
RUN rm -rf build && mkdir -p build
WORKDIR /root/GSPARC/baselines/GspTC/build
RUN cmake .. \
    -DCMAKE_CXX_COMPILER=g++-11 \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include \
    -DCUDA_CUBLAS_LIBRARIES=/usr/local/cuda/lib64/libcublas.so \
    -DCUDA_CUSOLVER_LIBRARIES=/usr/local/cuda/lib64/libcusolver.so \
    -DOpenBLAS_DIR=/usr/lib/cmake/openblas \
    -DOpenBLAS_INCLUDE_DIRS=/usr/include \
    -DOpenBLAS_LIBRARIES=/usr/lib/libopenblas.so
RUN make -j$(nproc)

# ALTO 
WORKDIR /root/GSPARC/baselines/ALTO
RUN rm -rf build && mkdir -p build
WORKDIR /root/GSPARC/baselines/ALTO/build
RUN cmake .. \
    -DCMAKE_CXX_COMPILER=g++-11 \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_BUILD_TYPE=Release \
    -DALTO_MASK_LENGTH=64 \
    -DCMAKE_CXX_FLAGS="-march=native -mbmi2" \
    -DALTERNATIVE_PEXT=OFF \
    -DOpenBLAS_DIR=/usr/lib/cmake/openblas \
    -DOpenBLAS_INCLUDE_DIRS=/usr/include \
    -DOpenBLAS_LIBRARIES=/usr/lib/libopenblas.so
RUN make -j$(nproc)

# BLCO 
WORKDIR /root/GSPARC/baselines/BLCO
RUN rm -rf build && mkdir -p build
WORKDIR /root/GSPARC/baselines/BLCO/build
RUN cmake .. \
    -DCMAKE_CXX_COMPILER=g++-11 \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLCO_MASK_LENGTH=64 \
    -DCMAKE_CXX_FLAGS="-march=native -mbmi2" \
    -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
    -DALTO_MASK_LENGTH=64 \
    -DBLAS_LIBRARY=OPENBLAS \
    -DTHP_PRE_ALLOCATION=OFF \
    -DALTERNATIVE_PEXT=OFF \
    -DMEMTRACE=OFF \
    -DDEBUG=OFF \
    -DOpenBLAS_DIR=/usr/lib/cmake/openblas \
    -DOpenBLAS_INCLUDE_DIRS=/usr/include \
    -DOpenBLAS_LIBRARIES=/usr/lib/libopenblas.so
  RUN make -j$(nproc)

WORKDIR /root/GSPARC
