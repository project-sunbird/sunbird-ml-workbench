#
# ML-Workbench Dockerfile
#

FROM aleenaraj/raj_ubuntu
MAINTAINER Aleena Raj "aleenar@ilimi.in"

# Setting up DS_DATA_HOME
RUN mkdir /home/DS_DATA_HOME
RUN mkdir /home/ML-Workbench

# Setting the working directory
WORKDIR /home

ADD . /home/ML-Workbench

ADD google_cred.json /home
ADD credentials.ini /home

# Setting the environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS /home/google_cred.json

RUN pwd

#Running MLWB
RUN pip3 install -r /home/ML-Workbench/requirements.txt

