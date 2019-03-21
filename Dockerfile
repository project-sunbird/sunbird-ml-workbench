FROM python:3.6.5 as mlworkbench
MAINTAINER "S M Y ALTAMASH <smy.altamash@gmail.com>"
WORKDIR /home/ml-workbench/
RUN apt update \
    && apt install libhdf5-dev -y
ADD . /home/ml-workbench/
RUN bash -x build.sh

FROM python:3.6.5
WORKDIR /home/ml-workbench/
RUN apt update && apt install libhdf5-dev -y 
ADD . /home/ml-workbench/
COPY --from=mlworkbench /home/ml-workbench/bin/ .
RUN pip install numpy
RUN pip install bin/daggit-0.5.0.tar.gz
