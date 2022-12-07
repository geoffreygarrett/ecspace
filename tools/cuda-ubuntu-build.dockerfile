ARG CUDA_VERSION
ARG UBUNTU_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CUDA_VERSION
ARG UBUNTU_VERSION
ARG ENTT_VERSION
ARG EIGEN_VERSION
ENV CUDA_VERSION=${CUDA_VERSION}
ENV UBUNTU_VERSION=${UBUNTU_VERSION}
ENV ENTT_VERSION=${ENTT_VERSION}
ENV EIGEN_VERSION=${EIGEN_VERSION}

LABEL Description="CUDA Ubuntu Build Environment"

ENV HOME /root
SHELL ["/bin/bash", "-c"]

# choose eigen install directory
ENV EIGEN_INSTALL_DIR /usr/local

# DEPENDENCIES
RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    cmake \
    gdb \
    wget \
    g++-11  \
    pciutils \
    tar \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libtbb-dev \
    && wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz \
    && tar -xvf eigen-${EIGEN_VERSION}.tar.gz \
    && cd eigen-${EIGEN_VERSION}  \
    && mkdir "build"  \
    && cd "build"  \
    && cmake -DCMAKE_INSTALL_PREFIX=/root/eigen ..  \
    && make install
#    libeigen3-dev
#    && wget https://github.com/skypjack/entt/archive/refs/tags/v$ENTT_VERSION.tar.gz \
#    && tar -xvf v$ENTT_VERSION.tar.gz  \
#    && wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz \
#    && tar -xvf eigen-${EIGEN_VERSION}.tar.gz

# Eigen
#RUN wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz \
#    && tar -xvf eigen-${EIGEN_VERSION}.tar.gz \
#    && cd eigen-${EIGEN_VERSION}  \
#    && mkdir "build"  \
#    && cd "build"  \
#    && cmake ..  \
#    && make install

#RUN #while true; do echo Ready to build!; sleep 10; done

#SET ENTT_INCLUDE_DIR
#ENV ENTT_INCLUDE_DIR /root/entt/src

#ARG EIGEN_VERSION
#ENV EIGEN_VERSION=${EIGEN_VERSION}
#ENV EIGEN3_INCLUDE_DIR /root/eigen-${EIGEN_VERSION}
#ENV Eigen3_DIR /root/eigen-${EIGEN_VERSION}/cmake


