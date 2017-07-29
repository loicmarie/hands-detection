# Hands Detection
Hands video tracker using the **Tensorflow Object Detection API** and **Faster RCNN model**. The data used is the **"Hand Dataset" from University of Oxford**. The dataset [can be found here](http://www.robots.ox.ac.uk/~vgg/data/hands/index.html). More informations: _"Hand detection using multiple proposals"_, A. Mittal, A. Zisserman, P. H. S. Torr, British Machine Vision Conference, 2011.

You can find [demo here](https://youtu.be/-klQ_bEPwfs).

[![Demo](http://img.youtube.com/vi/-klQ_bEPwfs/0.jpg)](http://www.youtube.com/watch?v=-klQ_bEPwfs)

## Installation
First we need to install the Tensorflow Object Detection API. You can either install dependencies or run the provided docker image.
### Installing dependencies
Please follow [Tensorflow Object Detection API installation](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) tutorial in models/ directory
### Using docker
We use the gcr.io/tensorflow/tensorflow image, so we have already Jupyter and Tensorboard services. The TOD API is already installed, the next step is to pull data from Hands Dataset.
```
docker build -t hands-tracker .
docker run -it -p 8888:8888 -p 6006:6006 hands-tracker bash
```
## Training on Google Cloud ML
### Pulling data from the Oxford University Hands Dataset
To pull data in dataset/ directory, use the following python script:
```
python create_inputs_from_dataset.py
```
If you need more informations about pulling data from University of Oxford, or MAT files to TFRecord files conversion, see the [IPython notebook for generating inputs](create_inputs_from_dataset_nb.ipynb)
The dataset folder should be structured as following:
```
dataset/
|---  test_dataset/
|------  test_data/
|----------  images/
|----------  annotations/
|---  training_dataset/
|------  training_data/
|----------  images/
|----------  annotations/
|---  validation_dataset/
...
```
### Deploy model to Google Cloud Storage
To make following steps easier, use the following variables:
```
export GC_PROJECT_ID=<your_project_id>
export GCS_BUCKET=<your_gcs_bucket>
```
First, we have to log in with our Google Cloud account and setup config
```
gcloud auth login
gcloud config set project $GC_PROJECT_ID
gcloud auth application-default login
```
Next, we can deploy our project files to Google Cloud Storage. You can use the following script:
```
./deploy_on_gcs.sh $GCS_BUCKET
```
### Create training and eval jobs
Our project is ready for training. We can create our job on Google Cloud ML
```
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
    --job-dir=gs://${GCS_BUCKET}/train \
    --packages models/dist/object_detection-0.1.tar.gz,models/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-central1 \
    --scale-tier BASIC \
    -- \
    --train_dir=gs://${GCS_BUCKET}/train \
    --pipeline_config_path=gs://${GCS_BUCKET}/data/faster_rcnn_resnet101_hands.config
```
The scale tier used here is 'BASIC' and training takes absolutely forever, but with 'BASIC_GPU' config training takes approximatively two hours. Be aware that after job began **you'll be charged** on your credit card.

Once the job has started, you can run an evaluation job as following:
```
gcloud ml-engine jobs submit training `whoami`_object_detection_eval_`date +%s` \
    --job-dir=gs://${GCS_BUCKET}/train \
    --packages models/dist/object_detection-0.1.tar.gz,models/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=gs://${GCS_BUCKET}/train \
    --eval_dir=gs://${GCS_BUCKET}/eval \
    --pipeline_config_path=gs://${GCS_BUCKET}/data/faster_rcnn_resnet101_hands.config
```

### Monitoring
Finally, if you are using the provided docker image, you can monitor your training job with Tensorboard:
```
tensorboard --logdir=gs://${GCS_BUCKET}
```
