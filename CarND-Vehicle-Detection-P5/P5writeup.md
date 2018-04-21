## Project 5: Vehicle Detection and Tracking

---

****

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/car_image.png
[image3]: ./output_images/HOG_vis.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/vechile_identification.png
[video1]: ./project_video_output.mp4

### Files
* P5writeup.md : this file
* object detection video.ipynb : main code for this project
* output_images : images used to illustrate the steps taken to complete this project
* videos : processed output video with vehicle detected
* README.md : readme file

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.

Some examples the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an car example image, and HOG Visilization using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on my final choice of HOG parameters based upon the performance of the SVM classifier produced using them.  I tried various combinations of parameters and final parameters chosen were YUV colorspace, 9 orientations, 8 pixels per cell, 2 cells per block, and ALL channels of the colorspace. 

#### 3. Describe how you trained a classifier using your selected HOG features and color features.

I trained a linear SVM using...
In the section titled "## Train linear SVC to classify car and no car" I trained a linear SVM with the default classifier parameters and using HOG features along with spatial intensity and channel intensity histogram features.The achieved test accuracy is 98.8%.

### Sliding Window Search

#### 1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled "Hog Sub-sampling Window Search" I adapted the method find_cars from the lesson materials. The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted once per image for the selected portion of the image and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

The image below shows one of the test images, using a variable window size:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

* scale: 1.0, ystart: 380, ystop: 450
* scale: 1.5, ystart: 400, ystop: 530
* scale: 2.0, ystart: 400, ystop: 560
* scale: 3.5, ystart: 400, ystop: 660

Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Link to your final video output. 
Here's a [video1](./project_video_output.mp4)


#### 2. Describe how you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were 
1. Make sure reading image file consistently. OpenCV loads images in the *BGR* format. mpimg to read images and pipeline will receive images in the *RGB* format.
2. A lot of parameters tunning is needed, for HOG feature extraction, SVM parameters, sliding windows scale, etc.
3. Although most of the time, the car detection is accurate, and false positives are miminum. There are still some false positive even with heatmap filter implemented. That could be possitivley improved by using more reliable car detections technique, such as choice of better feature vector, thresholding the decision function, hard negative mining etc.
4. Although traditional Computer Vision pipeline for object detection works great in this project, more recently, Deep Neural Network designed for object detection such as [YOLO, Redmon et al., 2015. You Only Look Once: Unified, Real-Time Object Detection.] (https://arxiv.org/abs/1506.02640) is trending. I would like to try it out in the future to compare the performance of the computer vision method.


