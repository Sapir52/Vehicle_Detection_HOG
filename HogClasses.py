import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import pickle
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn.utils import shuffle
from skimage.util.shape import view_as_windows
from keras.layers.core import Dense, Lambda, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import json

# -------------- Define image processing functions --------------
class ImageProcessingSVC():

    def bin_spatial(self, img, size=(32, 32)):
        # Spatial binning of image to reduce feature vector size 
        #Use inter-area interpolation to improve quality of down-sampled image
        features = cv2.resize(img, size, interpolation=cv2.INTER_AREA).ravel()
        return features

   
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute color histograms for image 
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features


    
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        # Compute hog features for image
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), 
                                      transform_sqrt=False, block_norm='L1', 
                                      visualize=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, 
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), 
                           transform_sqrt=False, block_norm='L1', 
                           visualize=vis, feature_vector=feature_vec)
            return features
    
   
    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        '''  Define a function to extract features from a list of images
             Have this function call bin_spatial() and color_hist() '''
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)      

            if spatial_feat == True:
                spatial_features = self.bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:,:,channel], 
                                            orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)        
                else:
                    hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, 
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap# Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    
    def draw_labeled_bboxes(self, img, labels, data):
        id_object=0
        # For use with labeled heatmaps
        draw_img = np.copy(img)
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(draw_img, bbox[0], bbox[1], (100,255,0), 3)
            data['vehicle'].append({'object_id':str(id_object),
                           'bbox0':str(bbox[0]),
                           'bbox1':str(bbox[1])})
            id_object+=1
        # Save object data from an image to a json file
        self.save_data_to_json(data)
        # Load object data from json file
        self.load_data_from_json('data_json')
        # Return the image
        return draw_img

    def save_data_to_json(self, data):
        # Save object data from an image to a json file
        with open('data_json.txt', 'w') as outfile:
            json.dump(data, outfile)
            
    def load_data_from_json(self, file_name):
        # Load object data from json file
        with open(file_name + '.txt') as json_file:
            data = json.load(json_file)
            for p in data['vehicle']:
                print('object_id: ' + p['object_id'])
                print('bbox0: ' + p['bbox0'])
                print('bbox1: ' + p['bbox1'])
                print()
    def find_cars(self, img, x_start_stop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, vis_bboxes = False):
        # Define a single function that can extract features using hog sub-sampling and make predictions
        draw_img = np.copy(img)    
        img_tosearch = img[ystart:ystop, x_start_stop[0]:x_start_stop[1], :]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step (2 originally)
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        on_windows = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = self.bin_spatial(subimg, size=spatial_size)
                hist_features = self.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    on_windows.append(((x_start_stop[0]+xbox_left, ytop_draw+ystart), 
                                       (x_start_stop[0]+xbox_left+win_draw, ytop_draw+win_draw+ystart)))

        return on_windows

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],  xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        random_color = False
        # Iterate through the bounding boxes
        for bbox in bboxes:
            if color == 'random' or random_color:
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                random_color = True
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def read_and_show_test_image(self, name_dir):
        test_files = glob.glob(name_dir)
        test_images = []
        #Use cv2.imread() to read files so that all files are scaled from 0-255
        for file in test_files:
            test_image = cv2.imread(file)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) 
            test_images.append(test_image)

        test_images = np.asarray(test_images)
        print("Test images shape is:", test_images.shape)

        fig, axes = plt.subplots(2, 2, figsize=(20,10))
        for idx, image in enumerate(test_images):
            plt.subplot(2,2, idx+1)
            plt.imshow(test_images[idx])
            plt.title("Test Image {:d}".format(idx+1))
        fig.tight_layout()
        plt.show()
        return test_images



class ImageProcessingNN():
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=3):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
        # Return the image copy with boxes drawn
        return imcopy
    
    
    def search_windows(self, views, model): 
        # Perform prediction on views generated from image 
        #Generate predictions for entire batch of views
        predictions = model.predict(views, batch_size = len(views))
        return predictions.reshape(-1)


    def create_views(self, img, window_size, xy_overlap, xlim, ylim, window_scale):
        '''
            Create views of image for input into the neural network
            http://scikit-image.org/docs/stable/api/skimage.util.html#view-as-windows
        '''
        views = []
        for idx, scale in enumerate(window_scale):
            #Determine step size based on overlap
            nx_pix_per_step = np.int(window_scale[idx]*window_size[0]*(1 - xy_overlap[0]))
            ny_pix_per_step = np.int(window_scale[idx]*window_size[1]*(1 - xy_overlap[1]))
            #Generate views one channel at a time
            image_views0 = view_as_windows(img[ylim[idx][0]:ylim[idx][1],xlim[idx][0]:xlim[idx][1],0], 
                                           (window_scale[idx]*window_size[0], window_scale[idx]*window_size[1]),
                                           (nx_pix_per_step, ny_pix_per_step))
            image_views1 = view_as_windows(img[ylim[idx][0]:ylim[idx][1],xlim[idx][0]:xlim[idx][1],1], 
                                           (window_scale[idx]*window_size[0], window_scale[idx]*window_size[1]),
                                           (nx_pix_per_step, ny_pix_per_step))
            image_views2 = view_as_windows(img[ylim[idx][0]:ylim[idx][1],xlim[idx][0]:xlim[idx][1],2], 
                                           (window_scale[idx]*window_size[0], window_scale[idx]*window_size[1]),
                                           (nx_pix_per_step, ny_pix_per_step))
            #Stack the channels back into an image
            image_views = np.stack((image_views0,image_views1,image_views2), -1)
            #Reshape the views to (n_samples, image height, image width, channels)
            image_views = np.reshape(image_views, (-1, int(window_scale[idx]*window_size[0]), 
                                                   int(window_scale[idx]*window_size[1]),3))
            #Equalize and resize the images
            image_views = self.resize(image_views)
            #Append the image view to the views array for each scale
            views.extend(image_views)
        return np.asarray(views)

    
    def resize(self, img_set):
        # Resize set of images to (32x32) for input into the neural network 
        resized_img_set = [cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA )  for img in img_set]
        return np.asarray(resized_img_set, dtype = np.uint8)

   
    def read_images(self, files, color_space='RGB'):
        # Read images given a color space 
        # Create aarray to hold images
        if color_space == 'GRAY':
            images = np.empty([len(files), 64, 64], dtype = np.uint8)
        else:
            images = np.empty([len(files), 64, 64, 3], dtype = np.uint8)
        # Iterate through the list of images
        for idx, file in enumerate(files):
            # Read in each one by one
            image = cv2.imread(file)
            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                elif color_space == 'GRAY':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else: feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
            images[idx] = feature_image
        return images


# -------------- Define ImageProcess Class --------------

class processDataSVC(ImageProcessingSVC): 
    
    def __init__(self, SVC, Scaler):
        
        self.Scaler = Scaler
        self.SVC = SVC
        #Variables for sliding windows
        self.window_scale = (1.25, 1.5625, 1.875) #window scales to use
        self.smooth_count = 10 # Number of frames to average over
        self.threshold = 0.73 #threshold for detection
        self.xy_window = (80, 80) #size of initial window
        self.xy_overlap = (0.75, 0.75) #overlap of search windows
        self.x_start_stop = [400, 1280] #start and stop x-coordinates to search
        self.y_start_stop = [[375, 520], [400, 580], [500, 700]] #start and stop y-coordinates to search
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 32    # Number of histogram bins
        self.heatmaps_list = deque(maxlen=self.smooth_count) #deque of heatmaps to smooth
        self.color_values = [(0,0,255), (0,255,0), (255,0,0)]
        

    def detect_heatmap(self, img, detection_windows):
        #Comment out this line to run the pipeline using heatmaps
        heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        heatmap = self.add_heat(heatmap ,detection_windows)
        heatmap = self.apply_threshold(heatmap, self.threshold)
        return heatmap
    
    
    def detection_windows(self, img):
        # Return the identified windows from the image
        detection_windows = []

        for i, scale in enumerate(self.window_scale):
            detection_windows.extend(self.find_cars(img, self.x_start_stop, self.y_start_stop[i][0], self.y_start_stop[i][1], 
                                               scale, self.SVC, self.Scaler, self.orient, self.pix_per_cell, 
                                               self.cell_per_block, self.spatial_size, self.hist_bins))

        return detection_windows
    
    def vehicle_detection(self, img, argument = "final_box"):
        data ={}
        data['vehicle'] =[]
        detection_windows=self.detection_windows(img)
        # Comment out this line to run the pipeline using heatmaps
        heatmap = self.detect_heatmap(img, detection_windows)
        labels = label(heatmap)

        if argument == 'heatmap':
            return heatmap
        elif argument == "boxes":
            return self.draw_boxes(img, detection_windows) 
        else:
            return  self.draw_labeled_bboxes(img, labels, data= data)
        
        
    def view_windows(self, test_images):
        # Show the windows on the image
        fig, axes = plt.subplots(2, 2, figsize=(20,10))
        for idx, image in enumerate(test_images):
            for i, scale in enumerate(self.window_scale):
                windows = self.slide_window(image, x_start_stop=self.x_start_stop, y_start_stop=self.y_start_stop[i], 
                                            xy_window=[int(dim*self.window_scale[i]) for dim in self.xy_window], xy_overlap=self.xy_overlap)
                image = self.draw_boxes(image, windows, self.color_values[i])
            plt.subplot(2,2,idx+1)
            fig.tight_layout()
            plt.imshow(image)
            plt.title("Test Image {:d}".format(idx+1))
        plt.savefig('output_images\\SVC_windows.png')
        plt.show()

    def visual_box(self,test_images, name='final_box', name_model=''):
        #Update the heatmaps section of the vehicle detection function before running
        fig, axes = plt.subplots(2, 2, figsize=(20,10))
        for idx, image in enumerate(test_images):
            output_image = self.vehicle_detection(image, name)
            plt.subplot(2,2,idx+1)
            plt.title("Test Image {:d}".format(idx+1))
            fig.tight_layout()
            plt.imshow(output_image)

        plt.savefig('output_images\\'+name_model+'_'+name+'_detections.png')
        plt.show()

class processDataNN(ImageProcessingNN, processDataSVC): 
    
    def __init__(self, model):
        
        self.model = model #model to use for predictions
        
        #Variables for sliding windows
        self.window_scale = (1.25, 1.5625) #window scales to use
        self.x_start_stop = [[575, 1280], [400, 1280]] #start and stop x-coordinates to search
        self.y_start_stop = [[375, 550], [450, 650]] #start and stop y-coordinates to search
        self.xy_window = (80, 80) #size of initial window
        self.xy_overlap = (0.75, 0.75) #overlap of search windows
        self.color_values = [(0,255,0), (255,0,0)]
        self.threshold = 1 #Thresholding for heatmap smoothing - use 1 for images and 9 for videos
        self.smooth_count = 15 # Number of frames to average over
        self.pred_threshold = 0.73 #Probability threshold for positive classification result
        self.heatmaps_list = deque(maxlen=self.smooth_count) #deque of heatmaps to smooth
    
    def detection_windows(self, img):
        # Return the identified windows from the image
        windows = []
        detection_windows = []
        
        trans_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        for i, scale in enumerate(self.window_scale):
        #Generate (x,y) coordinates for all windows that will be used
            windows.extend(self.slide_window(trans_img, x_start_stop=self.x_start_stop[i], y_start_stop=self.y_start_stop[i], 
                                   xy_window=[int(dim*scale) for dim in self.xy_window], xy_overlap=self.xy_overlap))
        windows = np.asarray(windows)

        #Generate views based on the same parameters as the slide_windows() function
        views = self.create_views(trans_img, self.xy_window, self.xy_overlap, self.x_start_stop, self.y_start_stop, self.window_scale)
        #Get predictions on all the views and reshape for boolean masking
        predictions = self.search_windows(views, self.model)
        #If detections are found, append the detected windows if the probability is greater than a threshold
        if len(predictions[predictions>=self.pred_threshold]) > 0:
            detection_windows.extend(windows[predictions>=self.pred_threshold])    
        
        return detection_windows

    def view_windows(self, test_images):
        # Show the windows on the image
        fig, axes = plt.subplots(2, 2, figsize=(20,10))
        for idx, image in enumerate(test_images):
            for i, scale in enumerate(self.window_scale):
                windows = self.slide_window(image, x_start_stop=self.x_start_stop[i], y_start_stop=self.y_start_stop[i], 
                                           xy_window=[int(dim*scale) for dim in self.xy_window], xy_overlap=self.xy_overlap)
                image = self.draw_boxes(image, windows, self.color_values[i])
            plt.subplot(2,2,idx+1)
            plt.title("Test Image {:d}".format(idx+1))
            fig.tight_layout()
            plt.imshow(image)

        plt.savefig('output_images\\CNN_windows.png')
        plt.show()


