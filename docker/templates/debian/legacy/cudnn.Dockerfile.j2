{{ metadata.header }}

{% set cuda = metadata.cuda %}
{% set cudnn = cuda.cudnn %}
{% set cudnn_version = cudnn.version %}
{% set cudnn_major_version = cudnn.major_version %}
ARG PYTHON_VERSION

FROM {{ metadata.package }}:{{ build_tag }} as build_image

RUN curl -fsSL {{ cuda.base_repo }}/7fa2af80.pub | apt-key add - && \
    echo "deb {{ cuda.base_repo }} /" >/etc/apt/sources.list.d/cuda.list && \
    apt-get purge --auto-remove -y curl && \
    rm -rf /var/lib/apt/lists/*

{%- set cudart_version = cuda.version.major+"-"+cuda.version.minor+"="+ cuda.components['cudart'] -%}
{%- set cublas_version = cuda.version.major+"-"+cuda.version.minor+"="+ cuda.components['libcublas'] -%}
{%- set curand_version = cuda.version.major+"-"+cuda.version.minor+"="+ cuda.components['libcurand'] -%}
{%- set cusparse_version = cuda.version.major+"-"+cuda.version.minor+"="+ cuda.components['libcusparse'] -%}
{%- set cufft_version = cuda.version.major+"-"+cuda.version.minor+"="+ cuda.components['libcufft'] -%}
{% set cusolver_version = cuda.version.major+"-"+cuda.version.minor+"="+ cuda.components['libcusolver'] %}

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-{{ cudart_version }} \
        libcublas-{{ cublas_version }} \
        libcurand-{{ curand_version }} \
        libcusparse-{{ cusparse_version }} \
        libcufft-{{ cufft_version }} \
        libcusolver-{{ cusolver_version }} \
        libcudnn{{ cudnn_major_version }}={{ cudnn_version }}+cuda{{ cuda.version.major }}.{{ cuda.version.minor }} && \
    ln -s cuda-{{ cuda.version.major }}.{{ cuda.version.minor }} /usr/local/cuda && \
    apt-mark hold libcudnn{{ cudnn_major_version }} libcublas-{{ cublas_version }} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# nvidia-container-runtime, which is needed for nvidia-docker
# https://github.com/NVIDIA/nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "{{ cuda.requires }}"

FROM build_image as release_image

ENV BENTOML_VERSION={{ metadata.envars['BENTOML_VERSION'] }}

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir && \
    rm -rf /var/lib/apt/lists/*
