FROM python:3.6
RUN mkdir /app
WORKDIR /app
COPY . /app
RUN pip install bin/daggit-0.5.0.tar.gz
RUN mkdir /data
ENV APP_HOME=/app
ENV DS_DATA_HOME=/data
ENTRYPOINT python apps/vectorise_service/startService.py
