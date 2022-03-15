# syntax = docker/dockerfile:1.4
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================



ARG XX_VERSION=1.1.0


FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx


FROM alpine:3.14 as base

COPY --from=xx / /

ARG TARGETPLATFORM

ARG TARGETARCH

FROM base as base-amd64

ARG GLIBC_PACKAGE_VERSION="2.33-r0"

# slight modification to improve build time
ARG GLIBC_BASE_URL="https://github.com/sgerrand/alpine-pkg-glibc/releases/download"
ARG GLIBC_BASE_PACKAGE_FILENAME="glibc-${GLIBC_PACKAGE_VERSION}.apk"
ARG GLIBC_BIN_PACKAGE_FILENAME="glibc-bin-${GLIBC_PACKAGE_VERSION}.apk"
ARG GLIBC_I18N_PACKAGE_FILENAME="glibc-i18n-${GLIBC_PACKAGE_VERSION}.apk"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

SHELL ["/bin/ash", "-eo", "pipefail", "-c"]

RUN xx-apk add -q --no-cache --virtual=.build-dependencies wget ca-certificates && \
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
        "$GLIBC_BASE_URL/$GLIBC_PACKAGE_VERSION/$GLIBC_BASE_PACKAGE_FILENAME" \
        "$GLIBC_BASE_URL/$GLIBC_PACKAGE_VERSION/$GLIBC_BIN_PACKAGE_FILENAME" \
        "$GLIBC_BASE_URL/$GLIBC_PACKAGE_VERSION/$GLIBC_I18N_PACKAGE_FILENAME" && \
    xx-apk add -q --no-cache \
        "$GLIBC_BASE_PACKAGE_FILENAME" \
        "$GLIBC_BIN_PACKAGE_FILENAME" \
        "$GLIBC_I18N_PACKAGE_FILENAME" && \
    \
    rm "/etc/apk/keys/sgerrand.rsa.pub" && \
    /usr/glibc-compat/bin/localedef --force --inputfile POSIX --charmap UTF-8 "$LANG" || true && \
    echo "export LANG=$LANG" > /etc/profile.d/locale.sh && \
    \
    xx-apk del -q glibc-i18n && \
    \
    rm "/root/.wget-hsts" && \
    xx-apk del -q .build-dependencies && \
    rm \
        "$GLIBC_BASE_PACKAGE_FILENAME" \
        "$GLIBC_BIN_PACKAGE_FILENAME" \
        "$GLIBC_I18N_PACKAGE_FILENAME"

FROM base as base-arm64

ARG GLIBC_TARGET_ARCH=aarch64

RUN set -xe \
    && xx-apk add --no-cache --purge -uU curl \
    && mkdir -p /glibc \
    && GLIBC_PACKAGE_VERSION="$(curl -SL https://api.github.com/repos/SatoshiPortal/alpine-pkg-glibc/releases/latest | awk '/tag_name/{print $4;exit}' FS='[""]' | sed -e 's_v__')" \
    && echo "Using GLIBC Version: ${GLIBC_PACKAGE_VERSION}" \
    && GLIBC_KEY="https://github.com/SatoshiPortal/alpine-pkg-glibc/releases/download/${GLIBC_PACKAGE_VERSION}/cyphernode@satoshiportal.com.rsa.pub" \
    && GLIBC_BASE_URL='https://github.com/SatoshiPortal/alpine-pkg-glibc/releases/download' \
    && GLIBC_BASE_PACKAGE_FILENAME="glibc-${GLIBC_PACKAGE_VERSION}.apk" \
    && GLIBC_BIN_PACKAGE_FILENAME="glibc-bin-${GLIBC_PACKAGE_VERSION}.apk" \
    && GLIBC_I18N_PACKAGE_FILENAME="glibc-i18n-${GLIBC_PACKAGE_VERSION}.apk" \
    && curl \
        -jkSL ${GLIBC_KEY} \
        -o /etc/apk/keys/cyphernode@satoshiportal.com.rsa.pub \
    && curl \
        -jkSL ${GLIBC_BASE_URL}/${GLIBC_PACKAGE_VERSION}/glibc-${GLIBC_PACKAGE_VERSION}-${GLIBC_TARGET_ARCH}.apk \
        -o /glibc/${GLIBC_BASE_PACKAGE_FILENAME} \
    && curl \
        -jkSL ${GLIBC_BASE_URL}/${GLIBC_PACKAGE_VERSION}/glibc-bin-${GLIBC_PACKAGE_VERSION}-${GLIBC_TARGET_ARCH}.apk \
        -o /glibc/${GLIBC_BIN_PACKAGE_FILENAME} \
    && curl \
        -jkSL ${GLIBC_BASE_URL}/${GLIBC_PACKAGE_VERSION}/glibc-i18n-${GLIBC_PACKAGE_VERSION}-${GLIBC_TARGET_ARCH}.apk \
        -o /glibc/${GLIBC_I18N_PACKAGE_FILENAME} \
    && xx-apk add --update --no-cache /glibc/*.apk; \
    /usr/glibc-compat/bin/localedef --force --inputfile POSIX --charmap UTF-8 C.UTF-8 || true \
    && echo "export LANG=C.UTF-8" > /etc/profile.d/locale.sh \
    && /usr/glibc-compat/sbin/ldconfig /lib /usr/glibc-compat/lib \
    && echo 'hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4' >> /etc/nsswitch.conf \
    && xx-apk del --purge curl glibc-i18n \
    && rm -rf /var/cache/apk/* /tmp/* /glibc /etc/apk/keys/cyphernode@satoshiportal.com.rsa.pub



FROM base-${TARGETARCH}

ARG TARGETPLATFORM

ARG TARGETARCH

ENV PATH /opt/conda/bin:/usr/local/bin:$PATH

LABEL maintainer "BentoML Team <contact@bentoml.com>"

# setup ENV and ARG
ARG PYTHON_VERSION
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1

ARG CONDA_VERSION=py39_4.10.3

# Installing miniconda
RUN set -x \
    && UNAME_M="$(uname -m)" \
    && if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh"; \
    elif [ "${UNAME_M}" = "ppc64le" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-ppc64le.sh"; \
    fi \
    && xx-apk add --no-cache --virtual .build-dependencies ca-certificates bash procps \
    && wget --quiet ${MINICONDA_URL} -O ~/miniconda.sh \
	&& mkdir -p /opt \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    \
    && xx-apk del --purge .build-dependencies \
    && rm -f ~/miniconda.sh \
    && find /opt/conda/ -follow -type f \( -iname '*.a' -o -iname '*.pyc' -o -iname '*.js.map' \) -delete \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.shinit \
    && echo "conda activate base" >> ~/.shinit \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && conda clean -afy

RUN xx-apk add --no-cache bash

RUN conda create -n bentoml-python${PYTHON_VERSION} python=$PYTHON_VERSION

RUN conda install -y pip \
    && conda init bash

CMD [ "/bin/bash" ]