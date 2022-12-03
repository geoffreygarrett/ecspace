ARG CUDA_VERSION
ARG UBUNTU_VERSION
ARG ENTT_VERSION

ENV CUDA_VERSION=${CUDA_VERSION}
ENV UBUNTU_VERSION=${UBUNTU_VERSION}
ENV ENTT_VERSION=${ENTT_VERSION}

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
LABEL Description="CUDA Ubuntu Build Environment"

ENV HOME /root
SHELL ["/bin/bash", "-c"]

# DEPENDENCIES
RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    cmake \
    gdb \
    wget \
    g++-11  \
    pciutils \
    tar \
    && wget https://github.com/skypjack/entt/archive/refs/tags/v${ENTT_VERSION}.tar.gz \
    && tar -xvf v${ENTT_VERSION}.tar.gz && while true; do echo Ready to build!; sleep 10; done

# SET ENTT_INCLUDE_DIR
ENV ENTT_INCLUDE_DIR /root/entt/src


