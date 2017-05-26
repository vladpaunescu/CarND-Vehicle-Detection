## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./assets/car_not_car.png
[image2]: ./assets/car_hog_Y_c.png
[image3]: ./assets/car_hog_Cr_c.png
[image4]: ./assets/car_hog_Cb_c.png
[image5]: ./assets/not_car_hog_Y_c.png
[image6]: ./assets/not_car_hog_Cr_c.png
[image7]: ./assets/not_car_hog_Cb_c.png
[image8]: ./assets/car_spatial_binning.png

[image9]: ./assets/histograms/car_y_color_hist.png
[image10]: ./assets/histograms/car_cr_color_hist.png
[image11]: ./assets/histograms/car_cb_color_hist.png
[image12]: ./assets/histograms/not_car_y_color_hist.png
[image13]: ./assets/histograms/not_car_cr_color_hist.png
[image14]: ./assets/histograms/not_car_cb_color_hist.png


[image15]: ./assets/search_windows/search_windows_far_no_overlap.png
[image16]: ./assets/search_windows/search_windows_far_overlap_09.png
[image17]: ./assets/search_windows/search_windows_near_no_overlap.png
[image18]: ./assets/search_windows/search_windows_near_overlap_09.png
[image19]: ./assets/search_windows/search_windows_all.png
[image20]: ./assets/search_windows/search_windows.png


[image21]: ./assets/detections/detections_all.png
[image22]: ./assets/detection_test_images/test1.png
[image23]: ./assets/detection_test_images/test2.png
[image25]: ./assets/detection_test_images/test5.png



[image26]: ./assets/heatmaps/test1.png
[image27]: ./assets/heatmaps/test2.png
[image28]: ./assets/heatmaps/test3.png
[image29]: ./assets/heatmaps/test4.png
[image30]: ./assets/heatmaps/test5.png
[image31]: ./assets/heatmaps/test6.png


[image32]: ./assets/labels/test1.png
[image33]: ./assets/labels/test2.png
[image34]: ./assets/labels/test3.png
[image35]: ./assets/labels/test4.png
[image36]: ./assets/labels/test5.png
[image37]: ./assets/labels/test6.png

[image38]: ./assets/detection_test_images/test6.png



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `features.py` file, in method ```get_hog_features()```.   

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces like ```RGB```, ```HSV``` and ```YCrCb``` and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 
I also used a more robust combined descriptor.

 I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of a car image using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Here is e negative example of a random image that is not a car:

![alt text][image5]
![alt text][image6]
![alt text][image7]

We can notice that the meaninguful information (at least for the human eye - for detecting shapes) resides in the Y channel.

####2. Explain how you settled on your final choice of HOG parameters.

I decided to use ```YCrCb``` for the reason it separates the luminance channel, and yields better quality for the HOG descriptor computed on Y channel.

I tried various combinations of parameters and noticed that the SVM classifier yielded hig accuracies (0.95). I decided to construct a combined classifier, and added spatial binning, and color histogram (on ```YCrCb```).

The HOG classifier was extracted on all the 3 channels of the image., and it had ```8 pixels per cell```, ```2 cells per block```, and ```9 orientations```.

After combining constructing the combined descriptor, the SVM reported accuracy was over 0.991 and the SVM trained in ~10 seconds.

The hard part was feature extraction that ammounted for ~300 seconds for the whole dataset (16 000 pictures).

The final HOG configuration is: 

```python
# HOG config
__C.PIX_PER_CELL = 8
__C.CELL_PER_BLOCK = 2
__C.ORIENT = 9
__C.HOG_CHANNEL = "ALL"
```

Here is an example of a spatial binned image with size (32,32):
![alt text][image8]

Here is an example of a color histogram computed on an image:

For the car class we have the following color histograms in ```YCrCb``` color space:

![alt text][image9]
![alt text][image10]
![alt text][image11]

And here is an example of a negative class (not car) color histograms:

![alt text][image12]
![alt text][image13]
![alt text][image14]

We can notice a different distribution mainly in Y channel.

The code for computing features is located in   ```features.py``` file.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the combined descriptor above:

* HOG on all channels as described above
* Color Histogram Features  with ```32 bins``` for all the 3 channels 
* Color spatial binning Features with spatial size of ```32x32```

The features were linearized and concatenated.
This resulted in a combined descriptor with a size of ```8460 features```.

The code for feature extraction resides in ```features.py``` file.
The code for linear SVM training resides in ```train.py```.

All files can run standalone, and cache their results into pickle objects to speedup later runs.

After combining the descriptor and extracting the features for all images, we used ```sklearn.preprocessing.StandardScaler``` to normalized the veature vectors.

The code can be found in ```train.py``` at line 15:

```python
def train_model(car_features, notcar_features):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
```

The labels were created in just one line of code, using ```numpy.hstack()``` at ```line 24``` in ```train.py```: 

```python
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
```

The train/test split was 80 % training, 20 % testing, with a random shuffle. I again used one line of code to accomplish this tasks (Python is beautiful). The  ```sklearn.sklearn.model_selection.train_test_split``` did the trick.
```Line 28``` in ```train.py``` contains the code for train/test splitting.

```python
from sklearn.model_selection import train_test_split
  
X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
        
```

I employed standard linear SVM from  `sklearn.svm.LinearSVC` to train the model. Code is at lines 32-35:

```python
  # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
 ```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

The idea was to sarch in the locations on the road, and to take into account that cars move on a ground plane so cars that ar distant from the camera, are smaller - they also appear higher in the image.

For generation search windows I used:

* 2 search scales
* overlap of 0.9 to maximize object detections, and remove spurious detections using voting (false positives was an issue).
* 2 Y areas and the right part of the image

In fact I only searched the bottom right of the image.

The far objects are searched in the area of ```y = (380, 550)```., and with a detection window of ```(120, 96)```.

The far objects are searched in the area of ```y = (380, 550)```., and with a detection window of ```(280, 224)```.

The overlap is 0.9.

Here are examples of the sliding windows used for detection:

Search windows in the far right side, with no overlap:

![alt text][image15]

Search windows in the far right side, with overlap 0.9:

![alt text][image16]

Search windows in near the camera, with no overlap:
![alt text][image17]

Search windows in near the camera, with overlap 0.9:

![alt text][image18]

Combined search windows on the bottom right side, overlap 0.9:

![alt text][image19]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The detection without any form of thersholding yilded many detections, and a couple of false positives. Here is a result of multiple detections, and false positives:

![alt text][image21]

I searched in the bottom right part of the image using 2 scales, and a combined descriptor of HOG on 3 channels, spatially binned colors, and color histogram for ```YCrCb``` color space. This provided better results.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image22]
![alt text][image23]
![alt text][image25]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./detections_video_final.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The code for detections and non maximum supression lies in ```detect.py``` file, in methods ```def add_heat(heatmap, bbox_list):``` and ```def apply_threshold(heatmap, threshold)```, and ```scipy.ndimage.measurements import label()```.

### Here are six frames and their corresponding heatmaps:

![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image32]
![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image38]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.


I had 2 challenges during this project:

* false positives - I minimized them using strategies from the lectures, like voting, and thresholding
* processing time/frame - I minimized it by limiting the search space of detection windows
 
  
Here are some future improvements to the detection pipeline. The robustness of the detection depends on 2 factors:

* feature descriptor
* classifier

The feature descriptor could be improved by identifiying most releveant features for the class, using PCA or some other form of data analysis.
Besides that dataset augmentation could be employed - like filpping, brightness and contrast augmentation. This will help the SVM better generalize.
Tracking accross time series of frames could help reduce the search space for the sliding windows.

We can exploit the data parallelism of detection across multiple windows. Because each classification of the search windows is independendent, we could speed-up the algorithm by moving feature extraction and classification onto the GPU (in CUDA or OpenCL).
Even a multithreaded approach in a fashion of replicated workers (map-reduce) could help us.



