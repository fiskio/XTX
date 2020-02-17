FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

MAINTAINER Marco Fiscato <marco.fiscato@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        locales \
        swig \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8

RUN curl -LO --silent https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
    bash Miniconda3-4.7.12.1-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-4.7.12.1-Linux-x86_64.sh

# Set the default conda env
ENV PATH=/miniconda/bin:${PATH}
ENV PYTHONPATH=.:$PYTHONPATH

# we need pytorch built with 10.0 for apex
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# try apex
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

RUN conda install -y -c anaconda jupyter seaborn pandas
RUN conda install -y -c anaconda h5py
RUN conda install -y -c anaconda pytest

RUN pip install reformer_pytorch