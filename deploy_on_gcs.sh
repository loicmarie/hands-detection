#!/bin/bash

wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
sed -i "s|PATH_TO_BE_CONFIGURED|"gs://$1"/data|g" faster_rcnn_resnet101_hands.config

gsutil cp faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* gs://$1/data/
gsutil cp faster_rcnn_resnet101_hands.config gs://$1/data/faster_rcnn_resnet101_hands.config
gsutil cp hands_train.record gs://$1/data/hands_train.record
gsutil cp hands_val.record gs://$1/data/hands_val.record
gsutil cp hands_label_map.pbtxt gs://$1/data/hands_label_map.pbtxt
