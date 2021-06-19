FROM ubuntu:${OS_VERSION} as build-image

# Redefine BENTOML_VERSION because of multistage
ARG BENTOML_VERSION

COPY --from=base-image /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH