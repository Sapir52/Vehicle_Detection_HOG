# Vehicle_Detection_HOG

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

 
 




