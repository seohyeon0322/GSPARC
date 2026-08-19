FROM nvidia/cuda:13.0.0-devel-ubuntu24.04


RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    tar \
    build-essential \
    gcc-13 \
    g++-13 \
    libomp-dev \
    libtbb-dev \
    libnuma-dev \
    git \
    cmake \
    pkg-config \
    python3 \
    python3-numpy \
    python3-pandas \
    python3-matplotlib \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root/GSPARC
COPY ./ ./

# The GPU architecture is set inside CMakeLists.txt (see step 3 of README.md).
RUN rm -rf build && mkdir build
WORKDIR /root/GSPARC/build
RUN cmake ../ \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER=g++-13
RUN make -j$(nproc)

# Dataset mount point used by scripts/*.sh
RUN mkdir -p /var/GSPARC/dataset
ENV OMP_NUM_THREADS=64

WORKDIR /root/GSPARC
