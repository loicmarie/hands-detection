FROM gcr.io/tensorflow/tensorflow

# Google Cloud SDK installation
ENV CLOUD_SDK_REPO=cloud-sdk-xenial
RUN echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# APT dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    git \
    wget \
    python-tk \
    google-cloud-sdk

# Hand tracker files
ADD requirements.txt /tensorflow/requirements.txt
ADD hands_label_map.pbtxt /tensorflow/hands_label_map.pbtxt
ADD create_inputs_from_dataset_nb.ipynb /tensorflow/create_inputs_from_dataset_nb.ipynb
ADD create_inputs_from_dataset.py /tensorflow/create_inputs_from_dataset.py
COPY models /tensorflow/models

WORKDIR /tensorflow

# PIP dependencies
RUN pip install -r requirements.txt

# Tensorflow Object Detection API installation
RUN cd /tensorflow/models \
    && protoc object_detection/protos/*.proto --python_out=. \
    && python setup.py sdist \
    && (cd slim && python setup.py sdist)

ENV PYTHONPATH=$PYTHONPATH:/tensorflow/models:/tensorflow/models/slim
