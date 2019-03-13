FROM python:3.6.5 as buildingml
MAINTAINER "S M Y ALTAMASH <smy.altamash@gmail.com>"
WORKDIR /home/ml-workbench/
RUN apt update && git clone https://github.com/SMYALTAMASH/sunbird-ml-workbench -b release-1.15 .
RUN bash -x build.sh

FROM python:3.6.5-alpine
WORKDIR /home/daggit/bin
COPY --from=buildingml /home/ml-workbench/bin/* .
WORKDIR /home/daggit/
RUN pip install bin/daggit-0.5.0.tar.gz \
        && daggit init dag_examples/iris_classification/iris_classification_graph.yaml \
        && daggit run Example1_Iris_Classification
