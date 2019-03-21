FROM python:3.6.5 as mlworkbench
MAINTAINER "S M Y ALTAMASH <smy.altamash@gmail.com>"
WORKDIR /home/ML-Workbench/
ADD . /home/ML-Workbench/
RUN apt update \
    && apt install libhdf5-dev -y
RUN bash -x build.sh

FROM python:3.6.5
WORKDIR /home/ML-Workbench/
RUN apt update && apt install libhdf5-dev -y && rm -rf /var/cache/apt/archives/*
COPY --from=mlworkbench /home/ML-Workbench/bin/ .
RUN pip install numpy
RUN pip install bin/daggit-0.5.0.tar.gz
