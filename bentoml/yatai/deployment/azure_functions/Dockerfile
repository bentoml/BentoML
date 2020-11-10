ARG BENTOML_VERSION=latest
ARG PYTHON_VERSION=3.7
FROM bentoml/azure-functions:$BENTOML_VERSION-py$PYTHON_VERSION

# https://github.com/MicrosoftDocs/azure-docs/issues/26761
# Copying BentoService to Azure function specific directory
COPY . /home/site/wwwroot

# Install pip dependencies
RUN if [ -f /home/site/wwwroot/bentoml-init.sh ]; then /bin/bash -c /home/site/wwwroot/bentoml-init.sh; fi
