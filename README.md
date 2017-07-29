# Hands Detection
Hands video tracker using the **Tensorflow Object Detection API** and **Faster RCNN model**. The data used is the **"Hand Dataset" from University of Oxford**. The dataset [can be found here](http://www.robots.ox.ac.uk/~vgg/data/hands/index.html). More informations: _"Hand detection using multiple proposals"_, A. Mittal, A. Zisserman, P. H. S. Torr, British Machine Vision Conference, 2011.

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
## Training
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
