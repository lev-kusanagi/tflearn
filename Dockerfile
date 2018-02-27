FROM tensorflow/tensorflow:1.6.0-rc1-devel-py3
RUN pip3 install tflearn h5py six
COPY / /