{{ metadata.header }}

FROM {{ metadata.base_image }}

# setup ENV and ARG
{% for key, value in metadata.envars.items() if key not in ["BENTOML_VERSION","LC_ALL","LANG"] %}
{% if key in ["PYTHON_VERSION"] %}
ARG {{ key }}
{% else %}
ENV {{ key }}={{ value }}
{% endif %}
{% endfor %}

ENV PATH /opt/conda/bin:$PATH

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

RUN yum upgrade -y \
    && yum install -y wget git ca-certificates curl gcc gcc-c++ make \
    && yum clean all \
    && rm -rf /var/cache/yum

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
    yum clean all && \
    yum autoremove -y wget && \
    rm -rf /var/cache/yum

RUN /opt/conda/bin/conda install -y python=$PYTHON_VERSION pip && \
    /opt/conda/bin/conda clean -afy

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc