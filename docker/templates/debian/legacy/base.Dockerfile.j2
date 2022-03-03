{{ metadata.header }}

FROM {{ metadata.base_image }}

# setup ENV and ARG
{% for key, value in metadata.envars.items() if key != "BENTOML_VERSION"%}
{% if key in ["PYTHON_VERSION"] %}
ARG {{ key }}
{% else %}
ENV {{ key }}={{ value }}
{% endif %}
{% endfor %}

ENV PATH /opt/conda/bin:$PATH

ENV DEBIAN_FRONTEND noninteractive

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    ca-certificates curl wget git gnupg build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    apt-get clean && apt-get autoremove -y && \
    apt-get purge -y wget

# Install python via conda
RUN /opt/conda/bin/conda install -y python=$PYTHON_VERSION pip && \
    /opt/conda/bin/conda clean -afy

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc