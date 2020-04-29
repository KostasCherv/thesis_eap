# YOLO3 (Detection, Training, and Evaluation)

## Installing

To install the dependencies, run
```bash
pip install -r requirements.txt
```
And for the GPU to work, make sure you've got the drivers installed beforehand (CUDA).

It has been tested to work with Python 2.7.13 and 3.5.3.

## Training

### 1. Data preparation 
Organize the dataset into 4 folders:

+ train_image_folder <= the folder that contains the train images.

+ train_annot_folder <= the folder that contains the train annotations in VOC format.

+ valid_image_folder <= the folder that contains the validation images.

+ valid_annot_folder <= the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.

### 2. Edit the configuration file


### 3. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file).

### 4. Perform detection using trained weights on image, set of images, video, or webcam
`python detect.py -c config.json -i /path/to/image/or/video`

Config file must contain anchors, traffic light labels and model, traffic sign labels and model.

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

If you wish to change the object threshold or IOU threshold, you can do it by altering `obj_thresh` and `nms_thresh` variables. By default, they are set to `0.5` and `0.45` respectively.

## Evaluation

`python evaluate.py -c config.json`

Compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.
