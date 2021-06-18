ARG UID=1034
ARG GID=1034
RUN groupadd -g $GID -o bentoml && useradd -m -u $UID -g $GID -o -r bentoml

USER bentoml
WORKDIR $HOME
RUN git clone https://github.com/bentoml/BentoML.git && \
    cd BentoML && \
    make install-local

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]