## Advanced Lane Finding Project

The goal of the project is to find in images taken by a dashboard camera the lane lines the car is driving in between.

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The code for the project is in the [IPython notebook][https://github.com/tjphoton/CarND/blob/master/CarND-Advanced-Lane-Lines-P4/Advanced_Lane_Lines_Finding.ipynb] "Advanced_Lane_Lines_Finding.ipynb". 

[//]: # (Image References)

[image1]: ./output_images/undist_calibration.png "Undistorted"
[image2]: ./output_images/road_undistort1.png "Road Image Distortion Corrected #1"
[image3]: ./output_images/road_undistort2.png "Road Image Distortion Corrected #2"
[image4]: ./output_images/road_transformed1.png "Road Image Perspective Transformed #1"
[image5]: ./output_images/road_transformed1.png "Road Image Perspective Transformed #2"
[image6]: ./output_images/binary1.png "Binary Example #1"
[image7]: ./output_images/binary4.png "Binary Example #2"
[image8]: ./output_images/color_fit_sliding.png "Sliding Window Fit"
[image9]: ./output_images/color_fit_roi.png "ROI Fit"
[image10]: ./output_images/lane_overlay.png "Lane Overlay Output"
[video1]: ./project_video_output.mp4 "Video"

---

### Camera Calibration

Started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Assume the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners were successfully detected in the calibration chessboard images provided in the repository.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection with `cv2.findChessboardCorners()` function in code cell #2.

The output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in code cell #3. The camera calibration result (camera matrix and distortion coefficients) is also saved in pickle file for later access.

This distortion correction is applied to the test image using the `cv2.undistort()` function in code cell #4 to obtaine the following result: 

![Undistored Image][image1]

### Pipeline (single images)

#### 1. Distortion correction on road image

With the calibration matrix and distortion coefficients calculated above from the chessboard calibration images, `cv2.undistort()` function was used again on the real world road image taken from the same camera mounted in the center of the car. Two straight lanes test images can be distortion corrected in the same manner, as shown below. The correction is successfully applied by observing the tree in the top right corner is more vertical than the un-corrected on. The car dashboard is also more (correctly) downward curvered on two sides in the corrected images.

![Road Image Distortion Corrected #1][image2]
![Road Image Distortion Corrected #2][image3]

#### 2. Perspective transform

The code for the perspective transform includes a function called `bird_view()`, which appears in the #8 code cell of the IPython notebook.  The `bird_view()` function takes an road image as input, outputs a perspective transformed bird view image. 
the source (`src`) and destination (`dst`) points are hardcoded in the following manner:

```python
x_len, y_len = (image.shape[1], image.shape[0])
offset = 300
src = np.float32([[x_len/16*7.3,  y_len/16*10.2], 
                  [x_len/16*8.7,  y_len/16*10.2], 
                  [x_len/16*13.2, y_len],
                  [x_len/16*2.8,  y_len]])
dst = np.float32([[offset,       0], 
                  [x_len-offset, 0], 
                  [x_len-offset, y_len], 
                  [offset,       y_len]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
|  584,  459    |   300,    0   | 
|  696,  459    |   980,    0   |
| 1056,  720    |   980,  720   |
|  224,  720    |   300,  720   |

The perspective transform is verified to work as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart. The lines appear parallel in the warped image.

![Road Image Perspective Transformed #1][image4]
![Road Image Perspective Transformed #2][image5]

#### 3. Thresholded binary image from color and gradient threshold methods

A combination of color and gradient thresholds is used to generate a binary image containing likely lane pixels (thresholding steps in code cell #11 and #12). 

Saturation (S) channel in the HSL Color Space is thresholded between 170 and 255. The reason to choose S channel is because it does a fairly robust job of picking up the lines under very different color and contrast conditions.

By exploring different combination of color threshold and gradient threshold, it is found the gradient threshold (specifically, Sobel operator in the x direction) can complement color threshold by picking up some portion of the lane lines not chosen by S channel color threshold method. After some experiment, the gradient threshold is chosen between 20 and 100. 

A combined threhold is implemented in the final code (code cell #11) to take advantage of both methods.

Two examples of output are shown below. Most of the white pixels identified by color and gradients threshold methods are visual verified to be part of the lane lines, with some noise pixels which will be rejected later with some other techniques, such as sliding window search or region of interest search methods.

![Binary Example #1][image6]
![Binary Example #2][image7]

#### 4. Identified lane-line pixels and fit their positions with a polynomial

After applying calibration, perspective transform, thresholding to a road image, we now have a binary image where the lane line pixels stand out clearly. Next, we need to decide explicitly which pixels are part of the lines and separate them to the left line and the right line.

One way to do that is to take a histogram along all the columns in the lower half of the image, and find peaks in such histogram. The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. Use that as a starting point to do a sliding window search for the lines, then find and follow the lines up to the top of the frame. The python code for the sliding window search method is located in code cell #14.

Once we know where the pixels for left and right lane lines are, we may fit all these identified pixels in each side of the lane line with two separate second order polynomial curves.

No blind search is needed in the next frame of video, since we already know where the lines are in the previous frame, and we assume these lines will not change their positions significantly from frame to frame at 24Hz sampling rate. We may just search in a margin around the previous line position. The python code for the Region of Interest (ROI) search is located in code cell #15. This smart search should help us track the lanes through sharp curves, shadow, missing marks, or other kind tricky conditions. If for some reason we lose track of the lines, just go back to start from scratch with the sliding windows search to rediscover the lane lines again. 

Below are the lane line search results with these two different methods.

Sliding window search method result:

![Sliding Window Fit][image8]

Region of Interest (ROI) search method result:

![ROI Fit][image9]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center

With lane lines idetified, we may take measurements of lane lines and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. Eventually, we would like to use these information to steer and drive the car autonomously.

As we have already done in the last step, a second order polynomial curve f(y) = Ay^2 + By + C are fited to the x and y pixel positions of the lane line pixels.  The radius of curvature can be calculated with formula mentioned in the lecture (GitHub is not currently supporting LaTex yet as far as I know, so I don't bother to put formula here). One note on the sign of the value not mentioned in the lecture is: the negative sign for the curvature means the road lane is curving to the left, while the positive sign means it's curving to the right.

Assuming the camera is mounted at the center of the car, the deviation of the midpoint of the lane from the center of the image is the vehicle position offset with respect to the certer of the lane. A negative value means the car is left to the center, while positive means it's right to the center.

The curvature and offset values are measured in pixels. To convert the unit from pixels to meters, conversion factors 740 pixels = 3.7 meters in x dimension, 720 pixel = 20 meters in y dimension were used in the calculation.

These steps are implemented in code cell #17 through #18 in functions named `calc_curvature()` and `calc_shift()` respectively.

For the test image above, the left lane curverture is 290 meters curving to the left, the right lane curverture is 162 meters also curvigng to the left, the car is driving at 0.30 meters distance left off center of the lanes.

#### 6. Plot lane area back down onto the road

The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This demonstrates the lane boundaries were correctly identified. This step is implemented in code cell #25 in the `draw_road()` function. Information of curvature and position from center are to be included later on the video frame. Below is an example of my result on a test image.

![Lane Overlay Output][image10]

---

### Pipeline (video)

The image processing pipeline (in code cell #26) that was established to find the lane lines in images successfully processes the video. The [output video][video1] identified the lanes in every frame with the radius of curvature of the lane and vehicle position within the lane displayed on the top right corner. The are two picture in picture images displayed on the top as well, one for the color lane bird-view image, the other for the lane binary image with identified lane line pixels and lane pixel fit lines. The 2nd image is helpful, since it tells us whether the line pixels identified in the current frame are from sliding window search (with green search window boxes plotted) or from ROI search (with smoothly curvered green search region plotted). 

There are a few tricks specific to the video pipeline that help identifiy lane lines.

#### 1. Tracking
a `Line()` class (in code cell #20) is defined to keep track of all the interesting parameters measured from frame to frame, such as the last several detections of the lane lines and what the curvature was. This helps to properly treat new detections in the next frame.

#### 2. Sanity Check
Confirm the detection of lines makes sense by considering:
* Both lane lines have similar curvature
* They are separated by approximately the right distance horizontally
* They are roughly parallel

#### 3. Look-Ahead Filter

#### 4. Reset

#### 5. Smoothing

#### 6. Link to the final video output.  

The pipeline performs very well on the entire project video, even during the shaddow area, or pavement color changes.
Here's a [link to my video result][video1]

---

### Discussion

#### 1. Problems and issues faced in the implementation of this project.  

#### 2. Where will the pipeline likely fail?  

#### 3. How to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.
