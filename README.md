# Vehicle Detection HOG


## Project
This project explores the problem of vehicle detection using a SVC implemented using Scikit-Learn and CNN using Keras.

## DataSet
In the project I used dataSets taken from the Kaggle website.

Link: https://www.kaggle.com/brsdincer/vehicle-detection-image-set

The vehicle detection image set contains two labels: Non-Vehicles and Vehicles. 

Total 17760 images.

## Pipeline

#### Train the classifier from the vehicle images:
a. Load vehicle images.

b. Select required vehicle features (HOG and color features).

c. Train the classfier / cnn.


#### Detect vehicles on the test images:
a. Do a sliding window search with different window scales tp detect bounding boxes.

b. Merge the bounding boxes using heat maps and thresholding.

c. Display bounding boxes on the images.

##  Model Architecture & Training
#### For SVC:
1. The feature vectors for the training images were extracted using the extract_features().
2. The training data was then shuffled and 20% was split off to use as a test set after training. 
3.  The model selected for the classifier was Scikit-Learn's LinearSVC().
4.  Test accuracy of 98.9% was achieved. 
5.  The trained classifer was saved using Pickle.

#### For CNN:
The challenge in using a neural network for this task was to keep the time required for a forward pass as low as possible while maintaining a good level of accuracy.
The model was built using Keras and is described below.

![summery](https://user-images.githubusercontent.com/63209732/142011842-702a9d1a-5f24-45ed-b449-f3b774dc27e8.png)

The last FC layer has a single output node with a sigmoid activation function to obtain a probability of the classification result. 
An adam optimizer was used to initalize and gradually decrease the learning rate during training and a binary_crossentropy loss was used since this is a binary classification problem. 
The metric was set to accuracy.

The model was trained using a batch size of 512 images for 25 epochs and an EarlyStopping() callback.

20% of the data was split off and used as a validation set. 

The figure below shows the training and validation loss and accuracy:
![loss_and_accuracy_CNN](https://user-images.githubusercontent.com/63209732/142013620-00345a63-f0ce-4def-8e88-c02d3bebce1b.png)

 ##  Object Localization
 #### Sliding Windows:
A sliding window approach was then used to extract image patches and feed them into either the SVC / CNN to perform classification.
The search limits of the image were defined to keep the sliding windows within the roadway of the direction of travel of the vehicle to avoid any unnecessary false positives. 
 
## Search Optimization
#### HOG Subsampling:
The HOG subsampling approach allowed for efficient extraction of image patches and features from the test images. In particular, the HOG features were computed once for the entire frame and then array slicing was used to obtain the relevant HOG features. The same slicing parameters were then fed into the functions responsible for extracting the spatially binned and color histogram features. This process was repeated for the various scales of image windows required. It is important to note that the window scales need to be adjusted for HOG subsampling to maintain the same scales visualized above. This was necessary since the HOG subsampling approach downsamples the image while maintaining the same window size to achieve a scaling effect. 

#### Image Views & Batch Prediction:
The neural network approach also needed significant optimization to be practical. The unoptimized approach of extracting a single image patches and performing prediction on one sample at a time.

The neural net was optimized using two methods:

1. First, the Scikit-Image function view_as_windows() was used to create views of the image. This allowed for efficient extraction of the required patches from the images by simply defining the step sizes and search extents of the image. This function had to be called on each image channel at a time after which the results were stacked, reshaped into the correct dimensions and resized to 32x32 pixels for input into the classifier. Details of this processing can be found in the create_views() function.
2. The second optimization technique was to perform batch prediction on the views created for the entire frame at once as opposed to looping over each view one at a time. Details of the implementation can be found in the search_windows() function.

#### Heatmaps & False Positives:

he SVM and NN both correctly detected and classified the vehicles in the test images. Multiple detections were returned for each vehicle and these needed to be grouped into a single bounding box and thresholded to eliminate any potential false positives in the search area. Identical code was used for both approaches since this method was independent of the type of classifier used.

The add_heat() function was first used to convert the positive detections into a heatmap. A threshold was then applied using the apply_threshold() function to eliminate regions of the search results which may be outside the body of the vehicle. Finally, the thresholded heatmap was labeled using SciPy label() function and the results were plotted on the test image. 

## Results:

#### For SVC: https://github.com/Sapir52/Vehicle_detection_HOG/blob/main/VehicleDetectionSVC.ipynb

#### For CNN: https://github.com/Sapir52/Vehicle_detection_HOG/blob/main/VehicleDetectionCNN.ipynb
