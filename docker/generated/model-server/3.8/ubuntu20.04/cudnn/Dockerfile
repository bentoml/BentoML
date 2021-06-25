# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


FROM model-server:python3.8-ubuntu20.04-base as build_image

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" >/etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" >/etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-cudart-11-3=11.3.109-1 \
        libcublas-11-3=11.5.1.109-1 \
        libcurand-11-3=10.2.4.109-1 \
        libcusparse-11-3=11.6.0.109-1 \
        libcufft-11-3=10.4.2.109-1 \
        libcusolver-11-3=11.1.2.109-1 \
        libcudnn8=8.2.0.53-1+cuda11.3 && \
    ln -s cuda-11.3 /usr/local/cuda && \
    apt-mark hold libcudnn8 libcublas-11-3=11.5.1.109-1 && \
    apt-get clean && \
    apt-get purge --autoremove -y build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# nvidia-container-runtime, which is needed for nvidia-docker
# https://github.com/NVIDIA/nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

FROM build_image as release_image

RUN pip install bentoml[model_server]==$BENTOML_VERSION --no-cache-dir && \
    apt-get purge --autoremove -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]
