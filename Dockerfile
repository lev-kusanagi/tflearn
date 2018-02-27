FROM tensorflow/tensorflow:1.6.0-rc1-devel
RUN pip install tflearn h5py six
RUN apt-get install python3-pip
RUN pip3 install tflearn h5py six
COPY / /