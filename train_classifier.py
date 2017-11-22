from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# !!!!!!!!!!!!!!!!!!!!!!
# TODO : NEED TO CHANGE bins_range if reading .png files with mpimg!
# !!!!!!!!!!!!!!!!!!!!!!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of image locations
# This function could also be used to call bin_spatial() and color_hist() (as in the lessons) to extract
# flattened spatial color features and color histogram features and combine them all (making use of StandardScaler)
# to be used together for classification
def extract_features(imgs,
                     cspace='RGB',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9, 
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0,
                     spatial_feat=True,
                     hist_feat=True,
                     hog_feat=True,
                     flip_images=False
                    ):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        file_features = []
        if spatial_feat == True:
            # apply color conversion if other than 'RGB'
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

car_images = glob.glob('../vehicles/*/*.png')
noncar_images = glob.glob('../non-vehicles/*/*.png')
print('Number of cars image = '+str(len(car_images)))
print('Number of non-cars image = '+str(len(noncar_images)))

# Spacial feature configuration parameters
spatial_feat = False    # Spatial features on or off
spatial_size = (16, 16) # Spatial binning dimensions

# Histogram feature configuration parameters
hist_feat = False # Histogram features on or off
hist_bins = 32    # Number of histogram bins

# Feature extraction parameters
hog_feat       = True # HOG features on or off
colorspace     = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 11
pix_per_cell   = 16
cell_per_block =  2
hog_channel    = 'ALL' # Can be 0, 1, 2, or "ALL"

# TODO: Left-right flip images for training?
flip_images = False

car_features = extract_features(car_images,
                                cspace=colorspace,
                                spatial_size=spatial_size,
                                hist_bins=hist_bins,
                                orient=orient, 
                                pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel,
                                spatial_feat=spatial_feat,
                                hist_feat=hist_feat,
                                hog_feat=hog_feat
                               )
notcar_features = extract_features(noncar_images,
                                   cspace=colorspace,
                                   spatial_size=spatial_size,
                                   hist_bins=hist_bins,
                                   orient=orient, 
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, 
                                   hog_channel=hog_channel,
                                   spatial_feat=spatial_feat,
                                   hist_feat=hist_feat,
                                   hog_feat=hog_feat
                                  )
print('Feature vector length = ', len(car_features[0]))

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)  

# Fit a per-column scaler - this will be necessary if combining different types of features (HOG + color_hist/bin_spatial)
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# The false detections rapidly increase when X scaling is applied, even when using a
# single data type, HOG
# Disable for now
if hog_feat == True and spatial_feat == False and hist_feat == False :
  X_scaler = None
  scaled_X = X

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Save the output classifier and configuration to use in the video pipeline
svc_pickle = {}
svc_pickle['svc'] = svc
svc_pickle['X_scaler'] = X_scaler
svc_pickle['colorspace']=colorspace
svc_pickle['orient'] = orient
svc_pickle['pix_per_cell'] = pix_per_cell
svc_pickle['cell_per_block'] = cell_per_block
svc_pickle['hog_channel'] = hog_channel
svc_pickle['spatial_size'] = spatial_size
svc_pickle['hist_bins'] = hist_bins
svc_pickle['spatial_feat'] = spatial_feat
svc_pickle['hist_feat'] = hist_feat
svc_pickle['hog_feat'] = hog_feat
svc_pickle['flip_images'] = flip_images
pickle.dump(svc_pickle, open("trained_svc.p", "wb"))
