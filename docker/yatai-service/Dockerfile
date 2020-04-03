FROM python:3.7-buster

# YataiService GRPC port
EXPOSE 50051

# YataiService Web UI port
EXPOSE 3000

# Build time argument - the BentoML version to be used in this image
ARG version
ENV BENTOML_VERSION=$version
RUN pip install bentoml==$version

ENTRYPOINT ["bentoml", "yatai-service-start"]
CMD []