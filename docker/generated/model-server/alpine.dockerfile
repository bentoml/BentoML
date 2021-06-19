# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG OS_VERSION

FROM alpine:${OS_VERSION} as base-image

ENV PATH /usr/local/bin:$PATH

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

SHELL ["/bin/ash", "-eo", "pipefail", "-c"]

# Direct port from https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/alpine/Dockerfile
ARG ALPINE_GLIBC_BASE_URL="https://github.com/sgerrand/alpine-pkg-glibc/releases/download"
ARG ALPINE_GLIBC_PACKAGE_VERSION="2.32-r0"
ARG ALPINE_GLIBC_BASE_PACKAGE_FILENAME="glibc-${ALPINE_GLIBC_PACKAGE_VERSION}.apk"
ARG ALPINE_GLIBC_BIN_PACKAGE_FILENAME="glibc-bin-${ALPINE_GLIBC_PACKAGE_VERSION}.apk"
ARG ALPINE_GLIBC_I18N_PACKAGE_FILENAME="glibc-i18n-${ALPINE_GLIBC_PACKAGE_VERSION}.apk"

RUN apk add -q --no-cache --virtual=.build-dependencies wget ca-certificates && \
    echo \
        "-----BEGIN PUBLIC KEY-----\
        MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEApZ2u1KJKUu/fW4A25y9m\
        y70AGEa/J3Wi5ibNVGNn1gT1r0VfgeWd0pUybS4UmcHdiNzxJPgoWQhV2SSW1JYu\
        tOqKZF5QSN6X937PTUpNBjUvLtTQ1ve1fp39uf/lEXPpFpOPL88LKnDBgbh7wkCp\
        m2KzLVGChf83MS0ShL6G9EQIAUxLm99VpgRjwqTQ/KfzGtpke1wqws4au0Ab4qPY\
        KXvMLSPLUp7cfulWvhmZSegr5AdhNw5KNizPqCJT8ZrGvgHypXyiFvvAH5YRtSsc\
        Zvo9GI2e2MaZyo9/lvb+LbLEJZKEQckqRj4P26gmASrZEPStwc+yqy1ShHLA0j6m\
        1QIDAQAB\
        -----END PUBLIC KEY-----" | sed 's/   */\n/g' > "/etc/apk/keys/sgerrand.rsa.pub" && \
    wget -q \
        "${ALPINE_GLIBC_BASE_URL}/${ALPINE_GLIBC_PACKAGE_VERSION}/${ALPINE_GLIBC_BASE_PACKAGE_FILENAME}" \
        "${ALPINE_GLIBC_BASE_URL}/${ALPINE_GLIBC_PACKAGE_VERSION}/${ALPINE_GLIBC_BIN_PACKAGE_FILENAME}" \
        "${ALPINE_GLIBC_BASE_URL}/${ALPINE_GLIBC_PACKAGE_VERSION}/${ALPINE_GLIBC_I18N_PACKAGE_FILENAME}" && \
    apk add -q --no-cache \
        "${ALPINE_GLIBC_BASE_PACKAGE_FILENAME}" \
        "${ALPINE_GLIBC_BIN_PACKAGE_FILENAME}" \
        "${ALPINE_GLIBC_I18N_PACKAGE_FILENAME}" && \
    \
    rm "/etc/apk/keys/sgerrand.rsa.pub" && \
    /usr/glibc-compat/bin/localedef --force --inputfile POSIX --charmap UTF-8 "$LANG" || true && \
    echo "export LANG=$LANG" > /etc/profile.d/locale.sh && \
    \
    apk del -q glibc-i18n && \
    \
    rm "/root/.wget-hsts" && \
    apk del -q .build-dependencies && \
    rm \
        "$ALPINE_GLIBC_BASE_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_BIN_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_I18N_PACKAGE_FILENAME"

FROM base-image

ENV ENV /root/.shinit   # set path to `ash` equivalent of ~/.bashrc
CMD [ "/bin/sh" ]

ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

# we will install python from conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip \
    && /opt/conda/bin/conda clean -ya

RUN pip install bentoml[model-server]==${BENTOML_VERSION} --no-cache-dir

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]