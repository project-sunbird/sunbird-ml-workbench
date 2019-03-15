FROM python:3.6.5 as mlworkbench                                                                                                        
MAINTAINER "S M Y ALTAMASH <smy.altamash@gmail.com>"                                                                                    
WORKDIR /home/ml-workbench/                                                                                                             
RUN apt update \
    && apt install libhdf5-dev -y \
    && git clone https://github.com/SMYALTAMASH/sunbird-ml-workbench -b release-1.15 .         
RUN bash -x build.sh                                                                                                                   
                                                                                                                                        
FROM python:3.6.5                                                                                                                       
WORKDIR /home/ml-workbench/
RUN apt update && apt install libhdf5-dev -y && git clone https://github.com/SMYALTAMASH/sunbird-ml-workbench -b release-1.15 .         
COPY --from=mlworkbench /home/ml-workbench/bin .
RUN pip install numpy
RUN pip install bin/daggit-0.5.0.tar.gz
RUN daggit init dag_examples/iris_classification/iris_classification_graph.yaml
RUN airflow list_dags
RUN daggit run Example1_Iris_Classification
