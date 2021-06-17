RUN pip install bentoml[model-server]==${BENTOML_VERSION} --no-cache-dir

COPY tools/${BENTOML_PACKAGE}/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]