FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Python 패키지

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

RUN useradd -ms /bin/bash appuser
USER appuser

WORKDIR /work
