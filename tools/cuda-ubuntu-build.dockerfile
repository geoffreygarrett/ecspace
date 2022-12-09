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
ENV WORKING /work
ENV USR /usr
ENV PREFIX /tmp_prefix
SHELL ["/bin/bash", "-c"]

RUN mkdir -p ${WORKING} && \
    mkdir -p ${PREFIX}


# DEPENDENCIES
RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    cmake \
    gdb \
    wget \
    g++-11  \
    pciutils \
    tar \
    git \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libtbb-dev

# EIGEN
RUN cd $WORKING \
    && wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz \
    && tar -xvf eigen-${EIGEN_VERSION}.tar.gz \
    && cd eigen-${EIGEN_VERSION}  \
    && mkdir -p "build"  \
    && cd "build"  \
    && cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..  \
    && make install

# ENTT
RUN cd $WORKING \
    && wget https://github.com/skypjack/entt/archive/refs/tags/v${ENTT_VERSION}.tar.gz \
    && tar -xvf v${ENTT_VERSION}.tar.gz  \
    && mv entt-${ENTT_VERSION} $PREFIX/entt
#    && cd entt-${ENTT_VERSION}  \
#    && mkdir -p "build"  \
#    && cd "build"  \
#    && cmake -DCMAKE_INSTALL_PREFIX=$USR ..  \
#    && make install

# MATPLOTLIBCPP
RUN cd $WORKING \
    && git clone https://github.com/lava/matplotlib-cpp.git \
    && cd matplotlib-cpp \
    && mkdir -p "build"  \
    && cd "build"  \
    && cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..  \
    && make install





