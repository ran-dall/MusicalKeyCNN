# syntax=docker/dockerfile:1

FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Some cached versions of the upstream image contain an obsolete Yarn APT
# source whose rotated signing key can make every apt-get update fail. This
# project does not use Yarn, so disable only that optional source before
# installing TorchCodec's system libraries.
RUN set -eux; \
    for source_file in /etc/apt/sources.list.d/*; do \
        [ -f "${source_file}" ] || continue; \
        if grep -q 'dl.yarnpkg.com/debian' "${source_file}"; then \
            rm -f "${source_file}"; \
        fi; \
    done; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i '\|dl.yarnpkg.com/debian|d' /etc/apt/sources.list; \
    fi; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

USER user
