## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

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

The code for camera calibration used for compute camera matrix and distortion coefficients is in the code cell #2 - #7 of the IPython notebook located in "Advanced_Lane_Lines_Finding.ipynb". 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in the calibration chessboard images provided in the repository.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection with `cv2.findChessboardCorners()` function in code cell #2.

The output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in code cell #3. The camera calibration result is also saved in picckle file for later access.

This distortion correction is applied to the test image using the `cv2.undistort()` function in code cell #4 to obtaine this result: 

![Undistored Image][image1]

### Pipeline (single images)

#### 1. Distortion correction on road image

With the same camera calibration matrix and distortion coefficients calculated above from the chessboard images, use `cv2.undistort()` function again on the road image taken from the same camera mounted in the center of the car, straight lanes test images can be distortion corrected in the same manner, as shown below. The correction is successfully applied by obersaving the top right corner of the tree is more vertical than the un-corrected on. The car dashboard is also more (correctly) downward curvered on two sides in the corrected images.

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


#### 3. Thresholded binary image from color and gradients threshold methods

A combination of color and gradient thresholds is used to generate a binary image containing likely lane pixels (thresholding steps at code cell #11 through #12). 

Saturation (S) channel in the HSL color space with threshold between 170 and 255. The reason S channel is chosen is because it  does a fairly robust job of picking up the lines under very different color and contrast conditions.

By exploring different combination of color threshold and gradient threshold, it is found that the gradient threshold (specifically, Sobel operator in the x direction) can complement color threshold by picking up some portion of the lane lines. A combination of these two threhold is implemented in the final code  (code cell #11) to take advantage of both method.

Two examples of output are below. Most of the white pixels identified by color and gradients threshold methods are visual verified to be part of the lane lines, with some noise pixels which will be rejected later with some other techniques, such as sliding window search or region of interest search methods.

![Binary Example #1][image6]
![Binary Example #2][image7]

#### 4. Identified lane-line pixels and fit their positions with a polynomial

After applying calibration, a perspective transform, thresholding to a road image, we now have a binary image where the lane lines stand out clearly. Next, we need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line. 

One way to do that is to take a histogram along all the columns in the lower half of the image, and find the peaks in the histogram. The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. Use that as a starting point to use a sliding window search for the lines, then find and follow the lines up to the top of the frame. The python code for the sliding window search method is located in code cell #14.

Once we know where the left and right lane lines are, we may fit all identified pixels in each lane with two separate polynomials.

In the next frame of video no blind search is needed, since we already know where the lines are in the previous frame. We may just search in a margin around the previous line position. The python code for the region of interest search is located in code cell # 15. This should help us track the lanes through sharp curves and tricky conditions. If for some reason we lose track of the lines, just go back to start from scratch with the sliding windows search to rediscover the lane lines again. 

Below are the lane line search result with these two different methods:

Sliding window search method result:

![Sliding Window Fit][image8]

Region of Interest (ROI) search method result:

![ROI Fit][image9]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center

I did this in lines # through # in my code cell in `calc_curvature` and `calc_shift`.

The negative value for the curvature means the road lane is curving to the left, while the positive value means curving to the right.
For the car position relative to the center, negative value means the car is left to the center, while positive value means it's right to the center.

For the image above, the curverture for left lane is 290 meters curving to the left, the curverture for the right lane is 162 meters also curvigng to the left, the car is driving with 0.30 meters shift left to the center of the lanes.

#### 6. Plot lane area back down onto the road

I implemented this step in lines # through # in my code in `draw_road()` function.  Here is an example of my result on a test image:

![Lane Overlay Output][image10]

---

### Pipeline (video)

#### 1. Sanity Check

#### 2. Look-Ahead Filter

#### 3. Reset

#### 4. Smoothing

#### 5. Tracking

#### 6. Link to the final video output.  

The pipeline performs very well on the entire project video, even during the shaddow area.

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
