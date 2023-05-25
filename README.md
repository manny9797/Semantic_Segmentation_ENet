# Binary and instance segmentation using CNN ENet
This repository contains an implementation of ENet to do semantic segmentation on ResortIt dataset containing waste images with different classes of objects. 
The goal is to train a network with a small size being able to return high pixel accuracy and Miou (mean intersection over unit).

## Usage

### Data Preparation
* Download the [ResortIT dataset.](https://drive.google.com/file/d/14ThGc53okYC61AnTXFAofiYYY8PTZYtl/view?usp=share_link).
* Unzip the ```dataset.zip``` into the project folder.
* Modify the root path of the dataset by changing ```__C.DATA.DATA_PATH``` in ```config.py```.
* Set the BATCH_SIZES, TRAIN_MAXEPOCH and NUM_CLASSES (binary_segmentation -> 1 ; instance_segmentation -> 5) in config.py
* Comment the line 49 on resortit.py to do instance segmentation, otherwise no

### Training

* Insert segmentation_type in the main ("binary" or "instance")
* Use ```python train.py``` command to train the model.
* ```train.py``` also provides the flexibility of either training the entire model (encoder + decoder) or just the encoder which can be performed by changing ```__C.TRAIN.STAGE``` in ```config.py```.

