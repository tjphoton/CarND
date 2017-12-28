# **Finding Lane Lines on the Road**

Overview
---
The goal of this project is to develope an algorithm pipeline for a self-driving car that automatically detects lane lines on the road. In the first part this document, details of the algorithm pipeline will be presented with example image for each stage of the pipeline. The second part describes some shortcomings of the current pipeline. The last part will discuss possible improvement to make the lane line detection better in the future.

Lane lines detection pipeline
---
There are seven(7) steps in the pipeline, including grayscaling, Gaussian smoothing, color selection, region of interest selection, Canny Edge Detection, Hough Tranform line detection, and lines averaging and extropolation. 

An example of a typical colored road image with lane markings from a self-driving car would look like the one displayed below.
![Original Image][image1]

#### 1. Color to gray
The first step in the pipeline is to convert the color image to a grayscale one.
![Gray Image][image2]

#### 2. Gaussian smoothing
Then, Gaussian smoothing is appled to suppress noise.
![Blurred Image][image3]

#### 3. Colors of two lines - white and yellow
In the first practice images, both left and right side of the lanes are white, and lighting condition is good. Gray image with white color detection is sufficient to do a fairly good job to detect both side of lanes. But in the 2nd and 3rd vedio practice, the left side of the lane is solid yellow lines. In bright light condition, the yellow lines are hardly to distinguish from the background. 

To imporve line color detection efficiency, white and yellow lines are detected seperately, with white detect on gray image, and yellow on color image. 

cv2.inRange() function in the OpenCV libary is used to detect color ranges in the image. 
With some trial and error, the white color range is set between 140 and 255.
```
cv2.inRange(blur_gray, 140, 255)
```

For yellow lane color, the color range is set between [150, 10, 0] and [255, 255, 170]
```
lower_yellow = np.array([150, 10, 0], dtype = "uint8")
upper_yellow = np.array([255, 255, 170], dtype = "uint8")
```

#### 4. Canny edge detection
In addition to color selection, Canny algorithm is another great way to detect lane lines, for reason that lane lines are typically show up in the image as greatest pixel intensity changes compare to other area of the road. Canny edge detection is precisely useful to detect these edges!
![Canny Image][image4]

#### 5. ROI 
What we really need is the road lane lines. We don't care all other edge information, such as trees, buildings, bridges, etc. Let's focus on the area right in front of the self-driving car camera. This Region of Interset (ROI) area should be big enough to encompass sufficient length of the road lane lines, but small enough to exclude unncessary detected other edges.

With some tweaks on a polygon points, a four sided polygon shown below is defined to mask ROI.
![ROI Image][image5]

Image below shows the ROI mask applied to Canny detected edges
![Edges in ROI Image][image6]

#### 6. Hough transformation 
Hough transformation is another technique to detect lines from dots in an image. When the Canny edge detected image feed into a OpenCV libary function ```cv2.HoughLinesP()```, segmented lines (more precisely, numpy array of point pairs [x1, y1, x2, y2]) are returned, as shown below as red lines in the image.

![Lines Image][image7]

#### 7. Line extropolation
To get from the line segments we just detected from Hough transformation to map out the full extent of the lane and draw single solid lane lines on the left and right, the draw_lines() function is modified by first taking the average position of all the positive and negetive sloped line that are above the slope threshold (set at 0.45 for now). The slope itself is also averaged out to smooth out the curved line segement or outlier lines.

```
    // average over all x axis of lines
    lane_avg_x = np.mean([lines[:,0],lines[:,2]])
    // average over all y axis of lines
    lane_avg_y = np.mean([lines[:,1],lines[:,3]])
    // average over all slopes
    lane_avg_slope = np.mean((lines[:,3]-lines[:,1])/(lines[:,2]-lines[:,0]))
```

Solid lines are drawn from the bottom of the image to the top of the ROI, the x coordinate of these two intersecting points are calculated from the slope and averaged lane center point position. 
```
    // the coordinate of lines intersecting with the bottom of the image
    lane_bottom_x = int(lane_avg_x + (y_max - lane_avg_y)/lane_avg_slope)
    // the coordinate of lines intersecting with the top of the region of interest
    lane_top_x = int(lane_avg_x + ( y_min - lane_avg_y)/lane_avg_slope)
```

Finally, the fully extended solid line can be drawn on the image,
```
    cv2.line(img, (x1,lane_bottom_y), (x2,lane_top_y), color, thickness)
```

![Solid Lines][image8]

and overlay these left and right lane lines on the original color image:
![Lines on Image][image9]


[//]: # (Image References)

[image1]: ./test_images/solidWhiteRight.jpg "Original"
[image2]: ./test_images/gray.jpg "Grayscale"
[image3]: ./test_images/gray_blurred.jpg "Grayblurred"
[image4]: ./test_images/canny.jpg "Canny"
[image5]: ./test_images/mask.jpg "Mask"
[image6]: ./test_images/masked_edges.jpg "MaskedEdges"
[image7]: ./test_images/lines.jpg "Lines"
[image8]: ./test_images/lines_solid.jpg "SolidLines"
[image9]: ./test_images/lines_on_img.jpg "LinesOnImage"

Shortcomings with current pipeline
---
The above pipeline works well on the first two practice videos, and most of the last challenge one. 
There are still some potential shortcomings:
1. When the road is very bright, some noisy background somehow pass through all the above filters and end up in the line detection. The final lane lines are not properly detected and drawn.
2. The bright condition can also fail to detect left side yellow lanes.

Possible improvement
---
1. To solve the above two problems, improvement is possible on the color detection to fine tuning the lower and upper threshold of yellow color to make it detect yellow color more precisely, or successfully detect yellow lanes in wide varity lighting conditions.
2. Currently, ROI is hard coded. Ideally, it can be improved to be adjusted to the calibration of camera mouting position on car.
3. Extropolated lane lines are solid straight lines. In real condition, the road could be curved, the curvature on the far end may slightly tile the slope towards the far end curvature direction. One possible solution is to calculate the weighted average with more weight on the near line, and less weight on far lines. A even better solution is to fit with polinomial curve line to draw the curvature along the road.



