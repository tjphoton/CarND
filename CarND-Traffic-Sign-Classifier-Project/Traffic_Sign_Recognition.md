# **Traffic Sign Recognition Project** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figs/1_all_traffic_signs.png "All signs"
[image2]: ./figs/6_label_distribution.png "Distribution"
[image3]: ./figs/7_argumentation.png "Argumentation"
[image4]: ./figs/6_label_distribution_2.png "New Distribution"
[image5]: ./figs/LeNet-5.png "LeNet"
[image6]: ./figs/2_LeNet_model_accuracy.png "LeNet Accuracy"
[image7]: ./figs/4_new_images.png "New images"


<!-- 
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
 -->

---
### Writeup and Code

* Link to the [project code](https://github.com/tjphoton/CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)
* Link to the [writeup](https://github.com/tjphoton/CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Recognition.md)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

Pandas library is used to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

The type and the image labels of German traffic signs included in the training dataset are listed below:

| ClassId         		 |     Sign Name and Description	      			| 
|:----------------------:|:------------------------------------------------:| 
| 	0	         		 |Speed limit (20km/h)	      						| 
| 	1	         		 |Speed limit (30km/h)	      						| 
| 	2	         		 |Speed limit (50km/h)	      						| 
| 	3	         		 |Speed limit (60km/h)	      						| 
| 	4	         		 |Speed limit (70km/h)	      						| 
| 	5	         		 |Speed limit (80km/h)	      						| 
| 	6	         		 |End of speed limit (80km/h)	      						| 
| 	7	         		 |Speed limit (100km/h)	      						| 
| 	8	         		 |Speed limit (120km/h)	      						| 
| 	9	         		 |No passing	      								| 
| 	10	         		 |No passing for vehicles over 3.5 metric tons	    | 
| 	11	         		 |Right-of-way at the next intersection	      		| 
| 	12	         		 |Priority road	      								| 
| 	13	         		 |Yield	      										| 
| 	14	         		 |Stop	      										| 
| 	15	         		 |No vehicles	      								| 
| 	16	         		 |Vehicles over 3.5 metric tons prohibited			|	
| 	17	         		 |No entry	      									| 
| 	18	         		 |General caution	      							| 
| 	19	         		 |Dangerous curve to the left	      				| 
| 	20	         		 |Dangerous curve to the right	      				| 
| 	21	         		 |Double curve	      								| 
| 	22	         		 |Bumpy road	      								| 
| 	23	         		 |Slippery road	      								| 
| 	24	         		 |Road narrows on the right	      					| 
| 	25	         		 |Road work	      									| 
| 	26	         		 |Traffic signals	      							| 
| 	27	         		 |Pedestrians	      								| 
| 	28	         		 |Children crossing	      							| 
| 	29	         		 |Bicycles crossing	      							| 
| 	30	         		 |Beware of ice/snow	      						| 
| 	31	         		 |Wild animals crossing	      						| 
| 	32	         		 |End of all speed and passing limits	      		| 
| 	33	         		 |Turn right ahead	   								| 
| 	34	         		 |Turn left ahead	   								| 
| 	35	         		 |Ahead only	      								| 
| 	36	         		 |Go straight or right								| 
| 	37	         		 |Go straight or left								| 
| 	38	         		 |Keep right	     								| 
| 	39	         		 |Keep left	      									| 
| 	40	         		 |Roundabout mandatory	      						| 
| 	41	         		 |End of no passing	      							| 
| 	42	         		 |End of no passing by vehicles over 3.5 metric ...	| 

The figure below listed one image for each traffic sign type in the training data set.
![All traffic signs][image1]

Below is a histogram chart showing how the number of training data set labels are distributed. 
The x axis shows the sign label, while the height of each bin shows the number of images for each label. 
It's clear the data sampling distribution is uneven among diffrent traffic signs, with the largest number 
is roughly 10 times larger than the smallest one. This non-uniform distribution will affect the accuracy 
for these under-sampled signs.

![distribution of traffic sign labels][image2]




### Design and Test a Model Architecture

#### 1. Image data preprocess

Since the traffic sign in the training data distribution is highly non-uniform, under-sampled image 
data need to be added to avoid loss in accuracy. One way to add more data to the under-sampled data 
is to collect more. Another way to do is use the argumentation technique.

Python package [Augmentor](http://augmentor.readthedocs.io/en/master/) is used to artificially generate image data
with operations like rotate, flip, zoom. Here is a few examples of augmented Stop Sign images 
(in the actual data preprocessing, the flip/mirror operation is not used, since the stop sign does not possess 
mirror symmetry). 

![Data argumentation][image2]

With image data argumented, data sample size is increased from 34,799 for the original data set 
to 91,736 for the augmented data set. Roughly a factor of 3 increase in the number of images. 
The numbers of different traffic signs are more evenly distributed, as seen below:

![distribution of argumented traffic sign labels][image3]


In a last step, image data is normalized with the code below to rescale pixel value  from [0, 256] to [-1, 1]. 
The intention of this image intensity rescaling is to make sure in the neural network training stage, 
the gradient descent converges much faster than the data that is not normalized.

```
(np.array(image).astype(float)-128)/128.
```    

#### 2. Final model architecture

<!-- 
looks like including model type, layers, layer sizes, connectivity, etc.) 
Consider including a diagram and/or table describing the final model.
 -->
 
LeNet is a great deep network architecture for image recognition.
It's a good starting point to build the model on.

![LeNet-5 Architecture][image4]

Final model consisted of the following layers:

| Layer         		 |     Description	        					| 
|:----------------------:|:--------------------------------------------:| 
| Input         		 | 32x32x3 RGB image   							| 
| Convolution  3x3 	     | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					 |												|
| Max pooling	      	 | 2x2 stride, outputs 14x14x6  				|
| Convolution 3x3	     | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					 |												|
| Max pooling	      	 | 2x2 stride, outputs 5x5x16   				|
| Flatten layer      	 | outputs 400 									|
| Fully connected layer 1| outputs 120 									|
| RELU					 |												|
| Fully connected layer 2| outputs 84  									|
| RELU					 |												|
| Fully connected layer 3| outputs 10  									|
| Softmax				 | probability distribution of traffic signs   	|

#### 3. Train the model

To train the model, I used AdamOptimizer. The following hyperparameters are used: batch size 128, learning rate 0.001. 
Training was repeated with 50 epochs. 

#### 4. Approach to find a solution
With the choice of LeNet, the initial accuracy on the validation data is about 84%. With image data normalization,
the validation accuracy improves to ~90%. 
![LeNet Accuracy][image5]

Even though the accuracy on the training set is high (surpasses 99% after just 10 epochs),  but the validation 
data set accuracy is low, indicating overfitting.

Overfitting can be solved in a few ways, one is to add more data to make the model recognized more images in the 
hope to make it more make more generalized decision. We have already used the argumentation technique mentioned above to achieve this.
Another way is to apply regularization. After apply dropout regularization on each convolution and fully connected layers 
(not on the final output layer) with keep_prob = 0.6, the validation accuracy improves to more than 95%.

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 95.6%
* test set accuracy of 94.2% 

### Test a Model on New Images

#### 1. German traffic signs found on the web 

<!-- 
For each image, discuss what quality or qualities might be difficult to classify.
 -->

Here are 20 German traffic signs that I found on the web:

![New images found on the web][image5]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

### Improve Model
* Add regularization features
  * drop out regularization
  * L2 regularization
* Pre-process Data
  * normalization and setting zero mean
  * histogram equalization
    * improve the value range and detail of many of the images 
    * locally adaptive equalization works better than global
    * Histogram equalization https://en.wikipedia.org/wiki/Histogram_equalization
    * scikit-image Local Histogram Equalization http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html
* Argument training data
  * Augmentor is an image augmentation library  https://github.com/mdbloice/Augmentor
  * rotate or shift image
  * change color

* Batch normalisation
  *  https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82
* Change activation functions, play around with Leaky ReLUs, PReLUs
* Change optimizer
* Tune hyperparameters
  * 128 batch size with 50 epochs
  * increasing batch size also helps
* Learning rate with decay and a large momentum -
  * increase your learning rate by a factor of 10 to 100
  * use a high momentum value of 0.9 or 0.99
* Experiment different network architectures
  * try deeper and more recent networks than lenet
* Change dimentions of LeNet layers
* i get 99.3 on validation using 
  * more deeper network + 
  * batch normalization +  
  * using YUV (Y channel) instead of grayscale image + 
  * cv2.Histogram for contrast problem + 
  * data augment

* Found a video on youTube with title "driving Berlin" and made some screenshots
  * The signs are real, the signs are German and they were filmed on the car's camera - everything is legal
  * *Driving through Berlin* https://www.youtube.com/watch?v=3SQe2xlHEiU
  * *Driving in Berlin Streets, Germany* https://www.youtube.com/watch?v=FllWycSZKpk
  * *Driving through... Berlin!* https://www.youtube.com/watch?v=JlASX8L04hI
  * *Driving Through (München) Munich Germany* https://www.youtube.com/watch?v=2LXwr2bRNic
  * https://www.howtogermany.com/images/roadsigns1.jpg

