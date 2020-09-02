FROM python:3.6
MAINTAINER Harsha <harshavardhanc95@gmail.com>
RUN mkdir /app
WORKDIR /app
COPY --from=sunbird/ml-build /app/bin/daggit-0.5.0.tar.gz /app
RUN pip install daggit-0.5.0.tar.gz
COPY daggit_api.py /app
COPY start.sh /app
RUN mkdir /app/examples -p
COPY examples /app/examples
COPY expt_name_map.json /app
ENV APP_HOME=/app
CMD sh /app/start.sh
