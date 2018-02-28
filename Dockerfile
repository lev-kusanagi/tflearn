FROM javiermares/miguel-model
COPY / /
RUN pip3 install tflearn h5py six
