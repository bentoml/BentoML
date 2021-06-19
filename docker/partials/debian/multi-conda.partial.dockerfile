FROM debian:${OS_VERSION} as build-image

COPY --from=base-image /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH