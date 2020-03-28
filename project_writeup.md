## Project Writeup 

---

**Advanced Lane Finding Project By Abdelkoddous Khamsi**

This writeup summarizes the main steps of the work I conducted within Udacity Self Driving Car Engineer Nanodegree 2nd project. For the implemetation details please refer to the jupyter notebook [P2.ipynb](P2.ipynb).

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds's eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image10]: ./writeup_images/chessBoardOriginal.jpg "Undistorted"
[image11]: ./writeup_images/chessBoardUndistorted.jpg "Undistorted"
[image12]: ./writeup_images/ActualRoadOriginal.jpg "Undistorted"
[image13]: ./writeup_images/ActualRoadUndistorted.jpg "Undistorted"

[image20]: ./writeup_images/SingleImgs0.jpg "HLS & Sobel Gradient"
[image21]: ./writeup_images/SingleImgs1.jpg "HLS & Sobel Gradient"
[image22]: ./writeup_images/SingleImgsdif0.jpg "HLS & Sobel Gradient"
[image23]: ./writeup_images/SingleImgsdif1.jpg "HLS & Sobel Gradient"

[image30]: ./writeup_images/RoI.jpg "Perspective Transform"
[image31]: ./writeup_images/RoI_Transformed.jpg "Perspective Transform"
[image32]: ./writeup_images/SingleImgs3.jpg "Perspective Transform"

[image40]: ./writeup_images/BirdseyeViewHistogram.jpg "Lane pixels detection"
[image41]: ./writeup_images/SingleImgs4.jpg "Lane pixels detection"

[image50]: ./writeup_images/SingleImgs5.jpg "inverse perspective transform"
[image51]: ./writeup_images/SingleImgs6.jpg "inverse perspective transform"

[image60]: ./writeup_images/naive_pipeline.gif "naive pipeline result"
[image61]: ./writeup_images/final_pipeline.gif "Final pipeline result"

---



### Camera Calibration



The goal in this step is to use a set of 20 chess boards images taken using the same camera providing the raod front view. In the real world space we know that these chess boards are square with no curve lines.

Thus, `objpoints` that was defined contains 3d coordinates of the corners of the boards in the real world (so the same for all pictures).
`imgpoints` has the 2d coordinates of the points/corners in that specific image, these are populted using  using the *findChessboardCorners* method.

In our case the corners were found in 17 out of 20 calibration images.

Using `objpoints` and `imgpoints` we can undistort all the chess board images. Here is an example of the result we get where we can see for example that the curvatures in the input image got straightened:

Input image             |  Undistorted chess board image
:-------------------------:|:-------------------------:
![alt text][image10]        |  ![alt text][image11]


## Single images Pipeline:

#### 1. Provide an example of a distortion-corrected image.

After the camera calebration step the idea is to use the same calibration coefficients to undistort our road images:


Input image             |  Undistorted chess board image
:-------------------------:|:-------------------------:
![alt text][image12]        |  ![alt text][image13]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

* The need to combine various thresholds here to come up with sth cool!!

As suggested in the course material I have explored the x gradient, the magnitude gradient and the direction gradient then tried to combine the different threshold to come up with a more reliable result. The finding of each gradient are detailed in the notebook.
The final combination consisted in pixel the activated pixel of the x gradient and the pixels where both the magnitude gradient and direction gradient are activated. The second part allowed to filter out the lines with small slopes form the magintude gradient.

``` python
def combinedGradient_thresholdedBinary(img):
    
    axis_binary_output = axis_gradient(img)
    mag_binary_output = magnitude_gradient(img)
    dir_binary_output = direction_gradient(img)
    
    combined_binary = np.zeros_like(axis_binary_output)
    combined_binary[ (axis_binary_output==1) | (mag_binary_output*dir_binary_output == 1) ] = 1

    return combined_binary
```

Performing this combined Gradient function directly on a gray scale image showed its limitations in difficult lighting conditions and on parts of the roads where the coloration changes.
Therefore I had my image converted into HLS color space and Where I do perform an anditionnal thresholding to isolates the yellow and white lanes as much as possible before running the `combinedGradient_thresholdedBinary` function. Details are in the function `to_HLS_Color_Channel`.

An example of the results I get from applying the combined gradient on the output of `to_HLS_Color_Channel` are is as follows:

Input image             |  HLS color Space
:-------------------------:|:-------------------------:
![alt text][image20]        |  ![alt text][image21]
![alt text][image22]        |  ![alt text][image23]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For my perspective transformation i have selected manually 4 source points and 4 destination points on the left and right lanes as in the figure below, then I hardcoded them to get the transformation matrix. then I applied `cv2.warpPerspective` method to perform the perspective transformation. See the potential improvements section for more comments about this. 



```python
src_p1 = (316,660)  # bottom left
src_p2 = (1050,660) # bottom right
src_p3 = (762,480)  # top right
src_p4 = (578,480)  # top left


dst_p1 = (300, height - 30)       # bottom left
dst_p2 = (width-330, height - 30) # bottom right
dst_p3 = (width - 250, 250)         # top right
dst_p4 = (300+70, 250)               # top left

```


![alt text][image30]

In the final pipeline I apply the perspective transformation step on the Binary thresholded image we got from the previous stage of the pipeline:

Input image             |  Binary thresholded image transform
:-------------------------:|:-------------------------:
![alt text][image21]        |  ![alt text][image32]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


Used the sliding windows approach implemented in the function `detectLanePixels(test_im, stats=True)` thats starts by checking the histogram of the bottom half of the frame to capture the 2 main peaks that will be the base positions for the left and right lanes.

![alt text][image40]

Building up the frame from these base positions we look for activated pixel within given windows for the left and right lanes. These windows are recentered whenever the average x position of the activated pixels in the current windows goes above a given threshold `minpix`.


![alt text][image41]

Then these activated data points that fall within these sliding windows ( Red pixelsfor the left lane and blue pixels for the right lane ) are used to fit a second order polynomial into the lanes.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

the computation of the curvature is then performed using the formula from the course material once the lane lines are detected reliably. I had to check that it is in meters and not in pixel scale.

I have also calculated the vehicule position from center by measuring far is the midpoint of the base for the left and right lanes from the center of the frame.


```python
    ######### Curvature radius ##########  
    
    ## fit poly in real world space
    y_eval = np.max(ploty)
    xm_per_pix = xm_ym_per_pix[0]
    ym_per_pix = xm_ym_per_pix[1]
    
    leftfit_rw = np.polyfit( lefty*ym_per_pix, leftx*xm_per_pix, 2 )
    rightfit_rw = np.polyfit( righty*ym_per_pix, rightx*xm_per_pix, 2) 
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*leftfit_rw[0]*y_eval*ym_per_pix + leftfit_rw[1])**2)**1.5) / np.absolute(2*leftfit_rw[0])
    right_curverad = ((1 + (2*rightfit_rw[0]*y_eval*ym_per_pix + rightfit_rw[1])**2)**1.5) / np.absolute(2*rightfit_rw[0])
    
    left_line.radius_of_curvature_array.append(left_curverad)
    right_line.radius_of_curvature_array.append(right_curverad)
    
    print("Curvature radius Real World - Left =>", left_curverad, "m - Right", right_curverad, "m")
    
    ########## Vehicle position ###########
   
    vehicle_pos = ( plot_rightx[-1] +  plot_leftx[-1] )//2 # midpoint of the two lines base
    
    vehicle_offset = xm_per_pix*( vehicle_pos - width//2 )
    
    #print("Vehicle position -->", vehicle_pos)
    
    print("Vehicle offset -->", vehicle_offset)
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


Finally an inverse transformation from bird's eye view to initial space was used to visualize the different results I got.


Input image             |  Inverse perspective transform
:-------------------------:|:-------------------------:
![alt text][image41]        |  ![alt text][image50]
![alt text][image22]        |  ![alt text][image51]


---

## Video Pipeline:

At first I decided to treat my video as a set of completely independant images. This means that I have only to apply the same pipeline that was designed in the previous part of the project on every single frame of the video. This did wotk to some extent as shown below, but it shows some major drawbacks:
1. If the pipeline comes to a "bad" or "difficult" frame the lane detection will be completely off.
2. The lane detection in a new frame starts all over a blind search even if we know that the location of the lanes in the current frame won't change that muche compared to where they were in the previous frame.
3. Even when the detection is accurate the lanes tend to consitently across the frames. 

![alt text][image60]

Here's a [link to the full video of the naive pipeline result](./final_pipeline_video.mp4)

The Second Pipeline's goal is to adress mainly those previous 3 points.

1. If the current frame is "bad" the lanes detection of the last frame is reused. For this a basic sanity check was impelemented to decide whether we accept or reject the lane detection, the criteria here was keeping a consistent distance between the left and right lanes along the frame in bird's eye view.
2. The actived pixels detection is performed locally around the polynomial fit in the previous frame.
3. The fitted polynomial in the current frame is averaged across the last `n` frames.

For these points's implemetation to be possible I had to track left and right lanes characteristics across the video stream. A snapshot of the result I get is the following:

![alt text][image61]

Here's a [link to my full video of the second pipeline result](./final_pipeline_video.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are several points I believe can be improved to have an more consistent lane finding pipeline:

* The combination of sobel different gradient and HLS color space can be tuned even more (even not manually) for a better isolation of the yellow and white lines.
* Hardcoded source and destination points for the perspective transform is definetly an optimal solution. Having a more elaborated algorithm to perform this task will result in a more reliable bird's eye view space, hence better polynomial fitting.
* With a small curvature radius road the sliding windows tend to reach to right and left edges of the frame in bird's eye view and the the pixels from the left and right lanes can get mixed up. 
* Improving the sanity check implementation for the detected lanes.
* Defining a better strategy to computed a weighted moving average of the lanes for the current frame across the previous ones. 