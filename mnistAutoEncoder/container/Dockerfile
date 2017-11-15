# https://hub.docker.com/r/tensorflow/tensorflow/
FROM tensorflow/tensorflow:1.3.0
#-gpu
LABEL maintainer="Troy Toggweiler <ttoggweiler@fit.edu>"

ENV LOG_DIR=/home/log
ENV MODEL_ARGS=""

#RUN pip install setuptools


#RUN apt-get install python-matplotlib

COPY scripts/start.sh target /home
#CMD chmod +x /home/start.sh
ENTRYPOINT bash /home/start.sh