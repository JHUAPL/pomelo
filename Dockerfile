FROM ubuntu:focal
FROM python:3.9.12

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION=3.9.12

RUN curl -fksSL https://apllinuxdepot.jhuapl.edu/linux/APL-root-cert/JHUAPL-MS-Root-CA-05-21-2038-B64-text.cer >  /usr/local/share/ca-certificates/JHUAPL-MS-Root-CA-05-21-2038-B64-text.crt

RUN update-ca-certificates

RUN apt-get update && apt-get install zip unzip

ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs

RUN alias curl='curl --cacert /etc/ssl/certs/ca-certificates.crt' \
&& alias wget='wget --ca-certificate /etc/ssl/certs/ca-certificates.crt'

RUN curl --location --output /usr/local/bin/release-cli "https://release-cli-downloads.s3.amazonaws.com/latest/release-cli-linux-amd64" \
&& chmod +x /usr/local/bin/release-cli \
&& release-cli -v

COPY ./requirements.txt /tmp/

COPY ./dev-requirements.txt /tmp/

RUN groupadd jovyan; \
    useradd -m -g jovyan jovyan; \
    chown -R jovyan:jovyan /home/jovyan

USER jovyan

RUN python --version; \
    pip install --default-timeout=100 --upgrade pip

RUN pip install --default-timeout=100 -r /tmp/requirements.txt