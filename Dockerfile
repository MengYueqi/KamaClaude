# Build the Docker sandbox test image with:
# docker build --target sandbox-runtime -t kama-sandbox:py312 .
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS sandbox-base

RUN groupadd --gid 10001 kama \
    && useradd --uid 10001 --gid kama --create-home --shell /bin/sh kama

# This target deliberately contains no application source, API keys, or user configuration.
FROM sandbox-base AS sandbox-runtime

USER kama
WORKDIR /workspace
