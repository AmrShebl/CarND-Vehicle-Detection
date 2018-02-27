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
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the funtion get_hog_features().  

All `vehicle` and `non-vehicle` images are read in the function get_train_and_test_data(car_folder, non_car_folder). This function reads in all images and divides them into train and test datasets.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I applied hog on all the channels of the image after converting it to the YCrCb color space.

Here is an example of the output of HOG with parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried the parameters suggested in the provided material and they worked well. The only thing that I needed to change was the color space. I tried the Grayscale and the HLS but they didn't work well. Finally, I found that the YCrCb was the best color space.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

This is done in the function train_classifier. All car and non car images were loaded and seperated into training and test sets in the function get_train_and_test_data(car_folder,non_car_folder). The features were extracted from each image using the function extract_features(img). This function finds the following features and concatenates them side by side:
1- Color Bins in the YCrCb color space
2- Color histogram in the YCrCb color space
3- HOG features of all the channels of the YCrCb image
After parsing in all the training features, a standard scaler is fit to scale the features such that they all have a mean of zero and a standard deviation of one.
A linear Support Vector Machine, SVM, is then trained using the scaled training data. The features of the test set are also extracted and scaled using the same Standard Scaler. The accuracy of the classifier is then calculated and found to be 99.26%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the function find_cars_with_scale. I tried different scales and chose those that result in boxes that are the size of the cars in different test images. For window overlap, I used the value suggested in the provided materials.
I ended up using the scales 1, 1.25, 1.5, and 1.75

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The performance of the classifier was super good from the very beginning. Here are some examples of the outputs of the pipeline.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
Each frame is passed to the pipeline with scales 1, 1.25, 1.5, and 1.75. For each scale, the detected boxes are used to update a heatmap. This heatmap is then passed to a list that keeps track of the last 5 heatmaps corresponding to the last 5 frames. An average of this list of 5 heatmaps is used to detect vehicles. The average is first thresholded at 2. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main challenge I faced was chosing the right color space to go around the problem of the shadows. Since the pipeline takes a portion of the image, I am not sure how it would perform on steep roads  

