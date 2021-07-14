# syntax=docker/dockerfile:1.2

ARG DEBIAN_FRONTEND=noninteractive

FROM python:3.9.6-slim-buster AS python-base

RUN groupadd --gid 1000 python && useradd --uid 1000 --gid python --shell /bin/bash --create-home python

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app


FROM python-base AS python-requirements

RUN set -x; \
        if ! command -v curl > /dev/null; then \
            apt-get update; \
            apt-get install --no-install-recommends -y curl; \
            rm -rf /var/lib/apt/lists/*; \
        fi

ENV POETRY_VERSION=1.1.7 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    PIP_DEFAULT_TIMEOUT=100

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app/target

RUN set -ex && \
        pip3 install --no-cache-dir -U pip && \
        pip3 install --no-cache-dir -U setuptools wheel && \
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py >get-poetry.py && \
        python3.9 get-poetry.py

COPY pyproject.toml poetry.lock ./

RUN poetry export --format=requirements.txt >requirements-prod.txt


FROM python-base AS python-deps-ml

RUN --mount=from=python-requirements,src=/app/target,target=/app/target set -x && \
        pip3 install --no-cache-dir -U pip && \
        pip3 install --no-cache-dir -U setuptools wheel && \
        pip3 install --no-cache-dir -r target/requirements-prod.txt


FROM python-base AS python-deps-server

# size of docker image is too large due to precompiled numpy, scipy, matplotlib, ...
# https://towardsdatascience.com/how-to-shrink-numpy-scipy-pandas-and-matplotlib-for-your-data-product-4ec8d7e86ee4
RUN --mount=from=python-requirements,src=/app/target,target=/app/target set -x && \
        pip3 install --no-cache-dir -U pip && \
        pip3 install --no-cache-dir -U setuptools wheel && \
        pip3 install --no-cache-dir -r target/requirements-prod.txt && \
        pip3 uninstall --no-cache-dir -y numpy pandas scikit-learn


FROM python-deps-server AS hades-server

COPY hades/server /app/hades/server

COPY hades/store /app/hades/store

EXPOSE 3002

CMD ["python",  "-m",  "hades.server.main"]


FROM python-deps-ml AS hades-mlpipeline

COPY hades/mlpipeline /app/hades/mlpipeline

COPY hades/store /app/hades/store

CMD ["python",  "-m",  "hades.mlpipeline.main"]


FROM python-deps-ml AS hades-mlmodels

COPY hades/mlmodels /app/hades/mlmodels

COPY hades/store /app/hades/store

CMD ["python",  "-m",  "hades.mlmodels.main"]
