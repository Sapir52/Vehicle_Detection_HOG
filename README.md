# Vehicle_Detection_HOG

## Project
This project explores the problem of vehicle detection using a SVC implemented using Scikit-Learn and CNN using Keras.

## DataSet
In the project I used dataSets taken from the Kaggle website.

Link: https://www.kaggle.com/brsdincer/vehicle-detection-image-set


The vehicle detection image set contains two labels: Non-Vehicles and Vehicles. 

Total 17760 images.

## Pipeline

#### Train the classifier from the vehicle images
a. Load vehicle images.

b. Select required vehicle features (HOG and color features).

c. Train the classfier / cnn.


#### Detect vehicles on the test images
a. Do a sliding window search with different window scales tp detect bounding boxes.

b. Merge the bounding boxes using heat maps and thresholding.

c. Display bounding boxes on the images.
