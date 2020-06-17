AZURE_API_FUNCTION_JSON = """\
{{
  "scriptFile": "__init__.py",
  "bindings": [
    {{
      "authLevel": "{function_auth_level}",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": [
        "get",
        "post",
        "put",
        "delete",
        "patch"
      ],
      "route": "{{*route}}"
    }},
    {{
      "type": "http",
      "direction": "out",
      "name": "$return"
    }}
  ]
}}
"""

# TODO change the version
BENTO_SERVICE_AZURE_FUNCTION_DOCKERFILE = """\
FROM bentoml/azure-function:{bentoml_version}

COPY . /home/site/wwwroot

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /home/site/wwwroot/bentoml-init.sh ]; then /bin/bash -c /home/site/wwwroot/bentoml-init.sh; fi
"""  # noqa: E501

BENTO_SERVICE_AZURE_FUNCTION_DOCKERFILE_OLD = """\
FROM mcr.microsoft.com/azure-functions/python:3.0.13901-python3.7
# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:3.0-python3.7-appservice

# Install miniconda3
# https://hub.docker.com/r/continuumio/miniconda3/dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \\
    apt-get install -y wget bzip2 ca-certificates curl git && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \\
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \\
    rm ~/miniconda.sh && \\
    /opt/conda/bin/conda clean -tipsy && \\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \\
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${{TINI_VERSION}}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
# Finish install miniconda3

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \\
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

COPY . /home/site/wwwroot

# Install BentoML related
RUN set -x && \\
    apt-get update && \\
    apt-get install --no-install-recommends --no-install-suggests -y libpq-dev build-essential && \\
    rm -rf /var/lib/apt/lists/*

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /home/site/wwwroot/bentoml-init.sh ]; then /bin/bash -c /home/site/wwwroot/bentoml-init.sh; fi
"""  # noqa: E501
