FROM gcr.io/tensorflow/tensorflow

RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    git \
    wget \
    python-tk

ADD requirements.txt /tensorflow/requirements.txt
ADD hands_label_map.pbtxt /tensorflow/hands_label_map.pbtxt
ADD create_inputs_from_dataset_nb.ipynb /tensorflow/create_inputs_from_dataset_nb.ipynb
ADD create_inputs_from_dataset.py /tensorflow/create_inputs_from_dataset.py
COPY models /tensorflow/models

WORKDIR /tensorflow

RUN pip install -r requirements.txt

RUN cd /tensorflow/models \
    && protoc object_detection/protos/*.proto --python_out=. \
    && python setup.py sdist \
    && (cd slim && python setup.py sdist)

ENV PYTHONPATH=$PYTHONPATH:/tensorflow/models:/tensorflow/models/slim
