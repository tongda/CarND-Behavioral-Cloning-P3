# **Behavioral Cloning** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[dirt road]: ./writeup/dirt_road.png "Dirt Road Place"
[model architecture]: ./writeup/model_architecture.png "Model Architecture"
[image1]: ./writeup/image1.jpg "Image 1"
[left out of track]: ./writeup/left_out_of_track.jpg "left out of track"
[right out of track]: ./writeup/right_out_of_track.jpg "right out of track"
[wrong way]: ./writeup/wrong_way.jpg "wrong way"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* data.py containing the functions that reads data from specified dir
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4 containing the recording of trained model driving car for 1 loop in track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 1 1x1 convolution layer for transform of the RGB to another color space (model.py code line 18), and several convolution layers with 5x5 or 3x3 filters, sizing from 64 to 128 (code line 19-36).

The model includes `ELU` layers to introduce nonlinearity (code line 13), and the data is normalized in the model using a Keras lambda layer (code line 16).

The architecture borrowed ideas from Nvidia Neural network mentioned in the lecture video.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 29, 32, 34). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (data.py code line 33). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 38).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, clockwise and anti-clockwise driving. The total number of images for training and validation are arount 43k.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a model and see what it will do in the simulator, then tuning the model by adding regularization and capability.

My first step was to use a convolution neural network model similar to the VGGNet. I thought this model might be appropriate because VGGNet has been proved to be good in image classification, and additionally, it is very simple.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Since I added Dropout layer in the beginning, the overfitting problem did not seems very serious. But it did overfitting when I increase the number of epochs to 15. Finally, I decide to decrease the number of epoch to 4, which is similar to what students discussed in the forum.

Because my Dropout rate was set to be 0.5 in the beginning, the training loss was alwasy higher than validation loss (train loss is about 0.02, whereas validation loss is about 0.005). I think this is because that the Dropout rate was a little too high. So I tried different Dropout rate from 0.5 to 0.1. At last, I found 0.2 worked better.

The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicle keep going the wrong way. The car keep run into the dirt road every time it encountered the place in the screenshot as below.

![The place the car keep going to the wrong way][dirt road]

I think the problem may be because the training data is not enough. So I added more recovery lane data from the specific place that force the car to turn left. Finally, it worked and the car would turn left after a little "hesitation" at the same place.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 15-36) consists of 1 1x1 convolution layer for transform of the RGB to another color space (model.py code line 18), and several convolution layers with 5x5 or 3x3 filters, sizing from 64 to 128 (code line 19-36). At last, there were 4 fully connected layers to do the regression.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model Architecture][model architecture]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![demo image][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it get out of the track. These images show what a recovery looks like starting from out of track from left side, out of track from right track, and nearly drive to the wrong way:

![left out of track][left out of track]
![right out of track][right out of track]
![wrong way][wrong way]

I didnot collect data from Track 2.

After the collection process, I had X number of data points. I then preprocessed this data by normalize the data to -0.5 to 0.5 (model.py code line 16)and cropping the top and bottom noise data (code line 17).


When generating generators, I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by 4 I used an adam optimizer so that manually training the learning rate wasn't necessary.
