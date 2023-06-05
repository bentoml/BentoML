# syntax=docker/dockerfile:1.4-labs

FROM --platform=linux/amd64 python:3-bullseye

# [Option] Install zsh
ARG INSTALL_ZSH="true"
# [Option] Upgrade OS packages to their latest versions
ARG UPGRADE_PACKAGES="false"
# [Option] Enable non-root Docker access in container
ARG ENABLE_NONROOT_DOCKER="true"
# [Option] Use the OSS Moby CLI instead of the licensed Docker CLI
ARG USE_MOBY="true"
# [Option] Select CLI version
ARG CLI_VERSION="latest"
# Enable new "BUILDKIT" mode for Docker CLI
ENV DOCKER_BUILDKIT=1
ENV DEBIAN_FRONTEND=noninteractive

# Install needed packages and setup non-root user. Use a separate RUN statement to add your
# own dependencies. A user of "automatic" attempts to reuse an user ID if one already exists.
ARG USERNAME=automatic
ARG USER_UID=1000
ARG USER_GID=$USER_UID
COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/

RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    && apt-get purge -y imagemagick imagemagick-6-common

# install common packages
RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    apt-get install -y build-essential software-properties-common vim \
    && /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    # Use Docker script from script library to set things up
    && /bin/bash /tmp/library-scripts/docker-debian.sh "${ENABLE_NONROOT_DOCKER}" "/var/run/docker-host.sock" "/var/run/docker.sock" "${USERNAME}" "${USE_MOBY}" "${CLI_VERSION}" \
    # Clean up
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

COPY requirements/*.txt /tmp/pip-tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-warn-script-location -r /tmp/pip-tmp/dev-requirements.txt -r /tmp/pip-tmp/docs-requirements.txt \
    && rm -rf /tmp/pip-tmp && pre-commit install

# Setting the ENTRYPOINT to docker-init.sh will configure non-root access to
# the Docker socket if "overrideCommand": false is set in devcontainer.json.
# The script will also execute CMD if you need to alter startup behaviors.
ENTRYPOINT [ "/usr/local/share/docker-init.sh" ]
CMD [ "sleep", "infinity" ]
