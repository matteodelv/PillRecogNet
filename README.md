# PillRecogNet
A convolutional neural network based on ```VGG16```, specialized in recognizing pills from images. This is Part 1 of my Undergraduate Thesis Project @ UniBo, where the Part 2 is an iOS application based on Metal Performance Shaders to run that ConvNet on Apple devices.  
  
This network hasn't been trained from scratch but it uses ```VGG16``` pre-trained weights on ImageNet, together with *bottleneck feature*, *image augmentation*, *transfer learning* and *fine tuning* on a custom dataset made on purpose.  
  
This repository contains all the scripts used to train the net and export its weights so that they can then be used inside the iOS app. Custom pill dataset and intermediate training results won't be provided since these scripts are fairly general, allowing anyone to use them on any dataset. Furthermore, example final trained model and weights can be downloaded from the [Releases](https://github.com/matteodelv/PillRecogNet/releases) page.  
A ```p2.xlarge``` instance on Amazon AWS has been used to perform training and fine tuning, lasting about few minutes for feature extraction and about an hour and half for fine tuning on a dataset of almost 1000 images of 1120x1120 pixels.

### Usage on a custom dataset
Before starting to use these scripts, it is mandatory to have a dataset ready and split in ```train/validation/test``` groups, using the following path structure (image names are not important):

    data
    ├── test
    │   ├── Class1
    │   │   ├── test-image1.jpg
    │   │   └── ...
    │   ├── Class2
    │   │   ├── test-image1.jpg
    │   │   └── ...
    │   └── Class3
    │       ├── test-image1.jpg
    │       └── ...
    ├── train
    │   ├── Class1
    │   │   ├── train-image1.jpg
    │   │   └── ...
    │   ├── Class2
    │   │   ├── train-image1.jpg
    │   │   └── ...
    │   └── Class3
    │       ├── train-image1.jpg
    │       └── ...
    └── validation
        ├── Class1
        │   ├── validation-image1.jpg
        │   └── ...
        ├── Class2
        │   ├── validation-image1.jpg
        │   └── ...
        └── Class3
            ├── validation-image1.jpg
            └── ...
**NOTE:** Each class MUST have the same amount of samples in every split  
**NOTE:** Class1, Class2, Class3 will be the labels the network will recognize, so name your folder accordingly  
  
After doing so, you can use the script and start training the network.  

1. Edit ```settings.py``` accordingly to your dataset, specifying the number of train, validation and test samples, the batch size, the number of classes to recognize and the epochs for feature extraction and fine tuning  
2. Launch a terminal, change directory to the scripts folder and start the extraction part issuing these commands: ```cd path/to/downloaded/scripts``` and then ```python bottlenecking.py```
3. After, custom top layers of the network are initialized and they have to be fine tuned, together the last convolutional block of ```VGG16```. Use the following command to do that: ```python fine-tuning.py```
4. When fine tuning is over, you can evaluate the model by predicting on every test image. This is possible by issuing the command ```python model-evaluator.py```. A log file will be generated to check the results
5. Finally, do ```python weights-converter.py``` to load the fine tuned model and extract its weights to use later with Metal Performance Shaders on iOS

**NOTE:** ```params``` will contain the weights in binary format for MPS, ```plots``` will contain the graphs for accuracy and loss, regarding the extraction and the fine tuning part, ```results``` will contain all the data regarding the trained network (mainly ```.npy``` and ```.h5``` files; the latter can be inspected using [HDFView](https://support.hdfgroup.org/products/java/release/download.html))
  
Feel free to edit the scripts to change optimizers, learning rates and every parameter that can better fit your dataset.

### Requirements
* Python 2.7.14
* Keras 2.0.8
* TensorFlow 1.3.0
* NumPy 1.13
* MatPlotLib 2.1.0