# Use Deep Learning to Clone Driving Behavior

Overview
---
In this project, deep convolutional neural networks will be utilized to clone driving behavior. 

With Udacity simulator (the interface is very similar to a video game!), data is collected from mannually driving a car around tracks to mimic good human driving hehavior. Image data and steering angles collected from the simulator are used to train a neural network as input and output respectively. Keras with TensorFlow backend is used as framework for the building, training, validation and testing on the model. The model will then output a steering angle to an autonomous vehicle to drive the car around the track alone the center line without leaving the raod.

![Car Animation][image1] 

[//]: # (Image References)

[image1]: ./examples/car.gif "Car Animation"
[image2]: ./examples/Navidia-cnn-architecture.png "Navidia architecture"
[image3]: ./examples/driving_center.png "Center driving image"
[image4]: ./examples/driving_right_side.png "Recovery image from right"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/angle1.png ""
[image7]: ./examples/angle2.png "Flipped Image"

Files included
---

The project includes the following files in the same GitHub folder as this document:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* driving_behavior_cloning_project.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Model Architecture
---

The network architecture used in this project is similar to the Navidia architecture in their End-to-End Learning Deep Neural Network (https://arxiv.org/pdf/1704.07911.pdf), which consists of 9 layers, including input (from camera image), a Cropping2D layer, a normalization layer, 5 convolutional layers, 1 flatten layer, and 3 fully connected layers, and 1 output (steering angle). 

A visualization of the architecture is displayed below.

<!-- ![Navidia architecture][image1] -->
<img src="./examples/Navidia-cnn-architecture.png" align="middle" height="500">

Let's look at a summary of the model by excuting the following command: 
```
model.summary()
```
 
|Layer (type)                    |   Output Shape     |  Param #  |   Connected to             |
|:-------------------------------|:------------------:|:---------:|:--------------------------:|
|cropping2d_1 (Cropping2D)       | (None, 75, 320, 3) |   0       |   cropping2d_input_2[0][0] |
|lambda_1 (Lambda)               | (None, 75, 320, 3) |   0       |    cropping2d_1[0][0]      |         
|convolution2d_1 (Convolution2D) | (None, 36, 158, 24)|   1824    |    lambda_1[0][0]          |         
|convolution2d_2 (Convolution2D) | (None, 16, 77, 36) |   21636   |    convolution2d_1[0][0]   |         
|convolution2d_3 (Convolution2D) | (None, 6, 37, 48)  |   43248   |    convolution2d_2[0][0]   |         
|convolution2d_4 (Convolution2D) | (None, 4, 35, 64)  |   27712   |    convolution2d_3[0][0]   |         
|convolution2d_5 (Convolution2D) | (None, 2, 33, 64)  |   36928   |    convolution2d_4[0][0]   |         
|flatten_1 (Flatten)             | (None, 4224)       |   0       |    convolution2d_5[0][0]   |         
|dense_1 (Dense)                 | (None, 100)        |   422500  |    flatten_1[0][0]         |         
|dense_2 (Dense)                 | (None, 50)         |   5050    |    dense_1[0][0]           |         
|dense_3 (Dense)                 | (None, 10)         |   510     |    dense_2[0][0]           |         
|dense_4 (Dense)                 | (None, 1)          |   11      |    dense_3[0][0]           |         

* Total params: 559,419
* Trainable params: 559,419
* Non-trainable params: 0


Creation of the Training Set
---

To capture good driving behavior, a few laps human driving in the simulator was recorded as training data. These images include both vehicle center lane  driving and off center-lane driving on both left and right sides of the road to the center. The intention of off-center lane driving is for the model to learn how to recover from the left side or right side of the road back to center. 

An example image of center lane driving:
![Center driving][image3]

Another image shows what a recovery driving looks like:
![Recovery from right side of the road][image4]

Since track 1 is a counter-clock wise loop, to augment the data sat, I also flipped images and angles. These will not only increase the size of the data by a factor of 2, but also teach the model for both left and right turning behavior. I also intentionally drove the car in the opporsite direction to let the network learn somewhat different environment, and create more data with clock-wise direction.

After the collection process, I had 113,562 images (inlcuding Udacity provided training data). Before images fed into the model, the data set was splitt into 80% training samples and 20% validation samples to ensure that the model was not overfitting. 

Training Process and Model Parameter Tuning
---

The images captured in the car simulator are too large to store in the memory all at once. Generators are used to pull pieces of the data and process them on the fly. 

The weights of the network is trained to minimize the mean-squared error between the steering angle output by the network and the ground truth human driver steering angle.

The model initially contains dropout layers in the effort to reduce overfitting, but it does not seem to improve the result. In the final implementation, the dropout layer is not implemented.

The model used an adam optimizer, in which the learning rate parameter is adaptively changed based on the average moments, no mannual tunning is necessary. 


Final Result
---

To test the final trained model, the model was runn through the simulator and to see how well the car was driving around track and to ensure the vehicle stay on the track. The recored video named ["demo.mp4"](./demo.mp4) is located in the same folder as this document.


![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]


