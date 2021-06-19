WORKDIR /
RUN git clone https://github.com/bentoml/BentoML.git \
    && cd BentoML \
    && pip install --editable .

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]