# **Traffic Sign Recognition Project** 

[//]: # (Image References)
[image1]: ./figs/1_all_traffic_signs.png "All signs"
[image2]: ./figs/6_label_distribution.png "Distribution"
[image3]: ./figs/7_argumentation.png "Argumentation"
[image4]: ./figs/6_label_distribution_2.png "New Distribution"
[image5]: ./figs/LeNet-5.png "LeNet"
[image6]: ./figs/2_LeNet_model_accuracy.png "LeNet Accuracy"
[image7]: ./figs/3_improved_model_accuracy.png "Improved Accuracy"
[image8]: ./figs/4_new_images.png "New images"
[image9]: ./figs/5_top5_prob.png "Top 5"
[image10]: ./figs/predict1.png "Prediction 1"
[image11]: ./figs/predict2.png "Prediction 2"
[image12]: ./figs/predict3.png "Prediction 3"


### Writeup and Code

* Link to the [project code](https://github.com/tjphoton/CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)
* Link to the [writeup](https://github.com/tjphoton/CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Recognition.md)

---
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

Displayed below is a histogram chart showing how the number of training data set labels are distributed. 
The x axis shows the sign label, while the height of each bin shows the number of images for each label. 
It's clear the data sampling distribution is uneven among diffrent traffic signs, with the largest number 
is roughly 10 times larger than the smallest one. This non-uniform distribution will affect the accuracy 
for these under-sampled signs.

![distribution of traffic sign labels][image2]

---
### Design and Test a Model Architecture

#### 1. Image data preprocess

Each type of the traffic sign in the training data distribution is highly non-uniform, under-sampled image 
data need to be added to avoid loss of accuracy. One way to add more data to the under-sampled data 
is to collect more. Since this is not feasible, data argumentation technique is used to increase the size of data sample.

Python package [Augmentor](http://augmentor.readthedocs.io/en/master/) is used to artificially generate image data
with operations like rotate, flip, zoom. Here is a few examples of augmented Stop Sign images 
(in the actual data preprocessing, the flip/mirror operation is not used, since the stop sign does not possess 
mirror symmetry). 

![Data argumentation][image3]

With image data argumented with more data generated towards under-sampled images, data sample size increased 
from 34,799 for the original data set to 91,736 for the augmented data set. Roughly a factor of 3 increase.
The numbers of different traffic signs are more evenly distributed, as seen below:

![distribution of argumented traffic sign labels][image4]


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

![LeNet-5 Architecture][image5]

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

![LeNet Accuracy][image6]

Even though the accuracy on the training set is high (surpasses 99.8% after 20 epochs),  but the validation 
data set accuracy is relatively low (91.8%), indicating overfitting.

Overfitting can be solved in a few ways, one is to add more data to make the model recognized more images in the 
hope to make it more make more generalized decision. We have already used the argumentation technique mentioned above to achieve this.
Another way is to apply regularization. After apply dropout regularization on each convolution and fully connected layers 
(not on the final output layer) with keep_prob = 0.6, the validation accuracy improves to more than 97.0%.

The other technique that were tried but did not immediately seem to improve the accuracy includes: 
histogram equalization, batch normalization, L2 regularization.

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 95.9%
* test set accuracy of 94.0% 

![Improved Accuracy][image7]

---
### Test a Model on New Images

#### 1. German traffic signs found on the web 

To find additional German traffic signs on the web, I looked up a few video on youTube with title 
similar to "driving Berlin" and made hundreds of screenshots. Since these videos were filmed in 
German with car's camera, they provide as close image as possible to the training data sets.

The link to these videos are listed below:
* [Driving through Berlin](https://www.youtube.com/watch?v=3SQe2xlHEiU)
* [Driving in Berlin Streets, Germany](https://www.youtube.com/watch?v=FllWycSZKpk)
* [Driving through... Berlin!](https://www.youtube.com/watch?v=JlASX8L04hI)
* [Driving Through (München) Munich Germany](https://www.youtube.com/watch?v=2LXwr2bRNic)

After the screenshots were made, final process has to be made to make sure the traffic signs are 
roughly centered, and image size are resized or cropped to be 32x32 in pixels.

Here are 20 German traffic signs that I captured with this method:
![New images found on the web][image8]

Efforts were made to make sure these traffic signs are as close to the German traffic sign training data sets.
But in reality, there are a few differences in the new images. 
1. Some images are captured from a titled angel, resulting in different perspective angle. 
2. Some images are zoomed in too much with some parts of the sign cropped off from the images.
3. Lighting condition and/or camera quality may slightly differ between the training and newly captured images.
All these contribute to the difference beween "out of sample" data differ from the training sample data, 
may result in reduced prediction accuracy.

#### 2. Model's predictions on these new traffic signs

Here are the results of the prediction:

| 	 Prediction       	|     Image	True Label			|    Correct ?   |
|:----------------------|:------------------------------|:--------------:|
|  Keep right 			| Keep right                  	|     ✓ 		 |
|  Road work 			| Road work                  	|     ✓ 		 |
|  Priority road 		| Priority road                 |     ✓ 		 |
|  Keep right 			| Go straight or left           |     ✘ 		 |
|  Speed limit (30km/h) | Speed limit (30km/h)          |     ✓ 		 |
|  Children crossing 	| Children crossing             |     ✓ 		 |
|  Speed limit (30km/h) | Speed limit (30km/h)          |     ✓ 		 |
|  Speed limit (80km/h) | Speed limit (50km/h)          |     ✘ 		 |
|  Yield 				| Yield                  		|     ✓ 		 |
|  Go straight or right | Go straight or right          |     ✓ 		 |
|  No vehicles 			| No vehicles                  	|     ✓ 		 |
|  Yield 				| Yield                  		|     ✓ 		 |
|  Roundabout mandatory | Turn right ahead            	|     ✘ 		 |
|  Turn left ahead 		| Turn left ahead               |     ✓ 		 |
|  Go straight or right | Go straight or right          |     ✓ 		 |
|  Ahead only 			| Ahead only                  	|     ✓ 		 |
|  Stop 				| Stop                  		|     ✓ 		 |
|  Right-of-way at the next intersection | Right-of-way at the next intersection  | ✓ |
|  No entry 			| No entry                  	|     ✓ 		 |
|  Bicycles crossing 	| Bicycles crossing             |     ✓ 		 |

The model was able to correctly guess 17 of the 20 traffic signs, which gives an accuracy of 85%. 
This compares worse to the accuracy on the test set of 94.2%. 

#### 3. Model prediction certainty 

By looking at the softmax probabilities for each prediction, we may know how certain the model is when predicting on 
each of the new images 
(OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th and 13th cell of the Ipython notebook.

The figure below displays the top 5 softmax probabilities for each newly captured signs image along with the sign type of each probability
The first column are the input images, the 2nd column are the sample image with highest probability (probability labeled above images),
the 3rd column are the sample image with 2nd highest probability, etc.

![Top 5 Prediction][image9]


Let's look at these three mis-identified images.

![Prediction #1][image10]

The 1st mis-identified image "Go straight or left".
Model predicted traffic sign name with probability:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .81         			| Keep right									| 
| .18     				| Go straight or right							|
| .002					| Turn left ahead								|
| .002	      			| End of all speed and passing limits			|
| .002				    | Roundabout mandatory 							|

The model is relatively sure that this is a Keep right sign (probability of 0.81), 
and may be a "Go straight or right" sign with probability 0.18. It's actually a "Go straight or left" sign. 
It's a bit surprise that the correct sign is not even in the top 5 candidates. 
It's probably due to the red traffic light in the image that confused the model, or the sign is off the center to the top.

![Prediction #2][image11]

For the 2nd mis-identified image "Speed limit (50km/h)".
Model predicted traffic sign name with probability:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| Speed limit (80km/h)							| 
| .20     				| Speed limit (60km/h)							|
| .14					| Speed limit (50km/h)							|
| .07	      			| Speed limit (30km/h)							|
| .04				    | Speed limit (100km/h) 						|

The model seem very confident the image is a speed limit sign (top 5 prediction are all speed limit signs), 
since they all have the same round shape, red circle color, and black number inside. 
But it is not very confident on the actually number in the speed limit sign. 
The probability of the top candidates are not differ much with the correct one the 3rd on the list.

![Prediction #3][image12]

For the 3rd mis-identified image "Turn right ahead"

Model predicted traffic sign name with probability:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Roundabout mandatory							| 
| .005     				| Speed limit (100km/h)							|
| .003					| End of no passing by vehicles over 3.5 metric ...	|
| .002	      			| Speed limit (30km/h)							|
| .001				    | Turn right ahead 								|

The model is very confident the image is a "Roundabout" sign (probability 98%), 
but it is actually a "Turn right ahead" sign. 
The are some similarity in the "Roundabout" and "Turn right ahead" signs, so it is possible to have this wrongly predicted. 
The correct sign is listed on the 5th top candidate, though with very low probability.

---
### Conclusion

This document describes how to use LeNet architecture to train a model to make satisfactory prediction 
on the German traffic sign images, both for the validation, test data, and on the newly captured images. 

There are more rooms to improve the prediction accuracy with the following techniques:
* More augmented image on most incorrectly classified images
* Change activation functions, play around with Leaky ReLUs, PReLUs
* Change optimizer
* Tune hyperparameters: 
  * batch size, 
  * epochs,
  * learning rate with decay and a large momentum
* Change dimentions of LeNet layers, or 
* Different deeper network architectures

