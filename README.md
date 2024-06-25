This repository contains scripts for model training and data post processing.
`model_training` contains code for modified UNet, model training, data loader setup, and batch-testing a model checkpoint on a set of images.
`helper` contains helper scripts for data post-processing.

Data post processing:
- RGB pipeline: cropping (`crop_img.py`) -> downsampling (`downsample.py`) -> greyscale conversion (`to_greyscale.py`) -> flip (`flip_image.py`) -> rotate (`rotate_img.py`)
- Depth pipeline: cropping (`crop_img.py`) -> normalization (`renormalization.py`) -> downsampling (`downsample.py`)
Each script can be called independently (remember to change the input & output directory). Pipeline scripts are WIP to achieve integrate these steps.

The model training script, `train_greyscale.py`, takes post-processed data and creates a data loader object, `RGBDDataset`, which is defined in `rgbd_dataset_greyscale`. Refer to this script for the specifics of model training.
`batch_test.py` takes in a model weight and outputs its predictions on a directory of images. 

The model trainign script with updated loss function is 'train_greyscale_updated_loss.py' which does the same process as above mentioned 'train_greyscale.py' but with the updated custom loss function which consists of both the modified mse loss for distance and angle difference. This script takes configs2.json as the input file in which all the necessary paths are required to be changed in order for it to function.
