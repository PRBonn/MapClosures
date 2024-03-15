#FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04
From ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y

RUN apt-get install -y \
    build-essential \
    cmake \
    ccache \
    clang-format \
    git \
    cmake \
    pkg-config \
    python3 \
    python3-pip \
    python3-numpy \
    pybind11-dev \
    libgl1-mesa-dev \
    libglvnd-dev \
    libglew-dev \
    libeigen3-dev \
    ca-certificates \
    software-properties-common

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade numpy
RUN pip3 install kiss_icp
RUN pip3 install --ignore-installed open3d

RUN git clone --depth 1 https://github.com/opencv/opencv.git -b 4.x &&\
    cd opencv && mkdir build && cd build &&\
    cmake .. && make -j4 && make install

RUN git clone https://github.com/PRBonn/MapClosures.git &&\
    cd MapClosures &&\
    make install
    
WORKDIR MapClosures
