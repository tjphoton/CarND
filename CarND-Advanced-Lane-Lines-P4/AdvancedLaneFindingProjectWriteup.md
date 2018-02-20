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

[image1]: ./camera_cal/undist_calibration.png "Undistorted"
[image2]: ./examples/road_undistort1.png "Road Image Distortion Corrected #1"
[image3]: ./examples/road_undistort2.png "Road Image Distortion Corrected #2"
[image4]: ./examples/road_transformed1.jpg "Road Image Perspective Transformed #1"
[image5]: ./examples/road_transformed1.jpg "Road Image Perspective Transformed #2"

[image4]: ./examples/binary_combo_example.jpg "Binary Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

The code for camera calibration in the code cell #2 - #7 of the IPython notebook located in "Advanced_Lane_Lines_Finding.ipynb" is used for compute camera matrix and distortion coefficients.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in the calibration chessboard images provided in the repository.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection with `cv2.findChessboardCorners()` function in code cell #2.

The output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in code cell #3. The camera calibration result is also saved in picckle file for later access.

This distortion correction is applied to the test image using the `cv2.undistort()` function in code cell #4 to obtaine this result: 

![Undistored Image[image1]

### Pipeline (single images)

#### 1. Distortion correction on road image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the straight lanes test images:

![Road Image Distortion Corrected #1][image2]
![Road Image Distortion Corrected #2][image3]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `bird_view()`, which appears in the #8 code cell of the IPython notebook).  The `bird_view()` function takes as inputs an road image, outputs a perspective transformed bird view image. 
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

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Road Image Perspective Transformed #1][image4]
![Road Image Perspective Transformed #2][image5]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
