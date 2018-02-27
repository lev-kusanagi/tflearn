FROM tensorflow/tensorflow:1.6.0-rc1-devel
RUN pip install tflearn h5py six
COPY / /