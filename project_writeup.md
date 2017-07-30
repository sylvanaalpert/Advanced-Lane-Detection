## **Advanced Lane Finding Project**
### By Sylvana Alpert

The goals of this project were:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit polynomial to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_undist.jpg "Undistorted"
[image2]: ./output_images/test_undist.jpg "Road Transformed"
[image3]: ./output_images/test_binary.jpg "Binary Example"
[image4]: ./output_images/test_pt.jpg "Warp Example"
[image5]: ./output_images/test_lines.jpg "Fit Visual"
[image6]: ./output_images/test_result.jpg "Output"
[video1]: ./project_video.mp4 "Video"


---

### Camera Calibration

The code for this step is contained in the file `camera_calibration.py`.  

The camera is calibrated by locating the corners of the chessboard images using `cv2.findChessboardCorners()`. These are the image points (2D coordinates). These points are mapped to object points, which are 3D coordinates of the same chessboard in the world. Assuming the chessboard is flat, we get that z=0 for all object points. Also, these points are the same for all images.
A list of image points and object points is compiled from all the images to be able to calculate the camera distortion at different locations in the field of view.  

Both lists are used to compute the camera calibration matrix and distortion coefficients using `cv2.calibrateCamera()`. Once these are known, we can undistort any other image captured with that camera using `cv2.undistort()`.

An example of such transformation can be seen here:

![alt text][image1]

### Pipeline - Test Images

The full pipeline is implemented in file `detection.py`, inside class `AdvancedLaneDetector` (function `pipeline_for_frame()`).

#### Camera Distortion

The camera calibration matrix and distortion coefficients are passed to the `AdvancedLaneDetector` constructor, which stores them and then applies them to any image processed through its pipeline. Here's an example of a test image that has been transformed with those parameters:

![alt text][image2]

#### Perspective Transform

The code used for transforming images into a birds-eye view image can be found on file `detection_utilities.py` in function `transform_perspective()`. This function takes an RGB image as an input and returns an RGB image as well. The source and destination points were found using images of straight road lines and hardcoded in the pipeline, assuming that the road is flat and the perspective transformation never changes. These points resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 590, 450      | 320, 0        |
| 690, 450      | 960, 0        |
| 1140, 719     | 960, 719      |
| 190, 719      | 320, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### RGB to Binary Image

I used a combination of color and gradient thresholds to generate a binary image. All thresholding steps are contained in file `detection_utilities.py` in function `combined_threshold()`. This was applied to the perspective transformed image. An example of this step is shown here:

![alt text][image3]

#### Road Line Pixel Identification

The pixels for each of the road lines were identified in two ways, depending on whether the location of those lanes could be inferred from found lanes in previous frames.

If the lanes had not been identified in the previous frame, a blind search for the lines was performed. This code can be found in `detection_utilities.py`, function `perform_blind_search()`. This method calculates an histogram of the image columns on the bottom half of the image and identifies its two peaks as starting x locations for the search of left and right lines. For each line, the x center of mass is calculated with a rectangular pixel mask around that location. That pixel mask slides through the image (upwards), always centered around the calculated center of mass. Pixels contained within the mask are labeled as belonging to the appropriate line.

In case the lines were identified in previous frames, we used a search for pixels around the known location and calculated a new fit for those new points.  
In case the lanes did not appear to be parallel, a blind search was triggered for that same frame again.

Here's an example of the identified pixels in a binary image:

![alt text][image5]

#### Radius of Curvature and Vehicle Center Offset Calculation

The radius of curvature is calculated in class `Line`, in function `update_fit()` which can be found in file `detection.py`. This class uses an average of the last five polynomial fits to calculate the best x coordinates of the line. The radius of curvature is calculated using the averaged fit to match the lines that are marked in the output image.

The vehicle offset in the lane is calculated in `AdvancedLaneDetector::search_lines()` (file `detection.py`), only if the lines are found to be parallel.

Both the radius of curvature and lane offset were calculated in meters using the conversion coefficients provided in the project directions.

#### Identified Lane Overlay on Road Image

This is implemented in `AdvancedLaneDetector::draw_lines()` (file `detection.py`).  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline - Video Output

Here's a [link to my video result](https://youtu.be/kKyUbv6CZHE).

---

### Discussion

#### Problems, Issues and Suggestions for Improvements

In order to calculate the binary image, I started using the S channel of the HLS image. This image did not contain all the information about the broken white lines on the right. For that reason, I had to apply a color filter to the R channel of the RGB image and combine both outputs. This color filter was more susceptible to changes in lighting in the road and shadows from trees, which nullified the advantages from using channel S.

Another issue was the delayed adaptation of the marked lines to the road curves, due to the smoothing of previously found lines. To overcome this, I used a weighted average which assigned more importance to the recent fits and  reduced the number of frames used for smoothing.

In order to make it more robust, I would probably modify the way images are thresholded and to avoid using the R channel directly. Possibly experimenting with other color spaces and image transformations such as gamma transform.  

Also, an automated calculation of the source points used in the perspective transform would allow using this detector in hilly terrains, not just flat roads.
