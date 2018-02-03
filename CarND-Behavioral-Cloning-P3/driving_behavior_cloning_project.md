# Use Deep Learning to Clone Driving Behavior

Overview
---
In this project, deep neural networks and convolutional neural networks will be utilized to clone driving behavior. 

With Udacity simulator (the interface is very similar to a video game!), data is collected from mannually steering a car around tracks. Image data and steering angles collected from the simulator are used to train a neural network as input and output respectively. Keras with TensorFlow backend is used for the training, validation and testing on the model. The model will then output a steering angle to an autonomous vehicle to drive the car around the track.

Goals
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

Files included
---

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Model Architecture and Training Strategy
---

#### 1. Model Architecture

The network architecture used in this project is the similar to the Navidia architecture(https://arxiv.org/pdf/1704.07911.pdf) shown in 
Figure 1 shows the network  used in this project, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Train the model

The weights of the network is trained to minimize the mean-squared error between the steering angle output by the network, and the ground truth human driver steering angle.


The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model initially contains dropout layers in the effort to reduce overfitting, but it does not seem to improve the result. In the final implementation, the dropout layer is not implemented.

The model used an adam optimizer, in which the learning rate parameter is adaptively changed based on the average moments, no mannual tunning is necessary. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Efforts were make to make sure the car was driving on the center lane. To training the model to learn recovering from the left and right sides of the road to the center, additionally efforts were made to drive the car the either side of the road, and recover to the center. 


My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
