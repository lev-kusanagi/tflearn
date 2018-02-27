FROM tensorflow/tensorflow:1.6.0-rc1-devel
RUN pip install tflearn h5py six
RUN apt-get install pip3
RUN pip3 install tflearn h5py six
COPY / /