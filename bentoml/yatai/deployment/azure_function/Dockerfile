ARG BENTOML_VERSION=latest
FROM bentoml/azure-function:$BENTOML_VERSION

COPY . /home/site/wwwroot

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /home/site/wwwroot/bentoml-init.sh ]; then /bin/bash -c /home/site/wwwroot/bentoml-init.sh; fi
