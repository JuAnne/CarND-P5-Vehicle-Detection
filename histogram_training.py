import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from lesson_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

# Divide up into cars and notcars
cars = glob.glob('../vehicles/*/*.png')
notcars = glob.glob('../non-vehicles/*/*.png')

# Spatial feature configuration parameters
spatial_cspace = 'RGB'    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat   = False    # Spatial features on or off
spatial_size   = (16, 16) # Spatial binning dimensions

# Histogram feature configuration parameters
hist_cspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hist_feat   = True  # Histogram features on or off
hist_bins   = 32    # Number of histogram bins

# HOG configuration parameters
hog_cspace     = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_feat       = False # HOG features on or off
orient         = 9     # HOG orientations
pix_per_cell   = 8     # HOG pixels per cell
cell_per_block = 4     # HOG cells per block
hog_channel    = 'ALL' # Can be 0, 1, 2, or "ALL"

# Include L-R flipped images in feature extraction?
flip_images = False

hist_modes = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
hist_sizes = [8, 16, 32, 64]
#cars = cars[:1500]
#notcars = notcars[:1500]
#cars = cars[1501:3000]
#notcars = notcars[1501:3000]

for hist_cspace in hist_modes :
  for hist_size in hist_sizes :
    print('Testing '+hist_cspace+' size '+str(hist_size))
    svc, X_scaler = extract_features_and_train(cars,
                                               notcars,
                                               spatial_cspace=spatial_cspace,
                                               hist_cspace=hist_cspace,
                                               hog_cspace=hog_cspace,
                                               spatial_size=spatial_size,
                                               hist_bins=hist_bins, 
                                               orient=orient,
                                               pix_per_cell=pix_per_cell, 
                                               cell_per_block=cell_per_block, 
                                               hog_channel=hog_channel,
                                               spatial_feat=spatial_feat, 
                                               hist_feat=hist_feat,
                                               hog_feat=hog_feat,
                                               flip_images=flip_images
                                              )
"""
Summary
HSV, LUV & HLS are the best
Larger sizes are better but take longer to compute

Results using images 0:1500
Testing RGB size 8
Test Accuracy of SVC =  0.4933
Testing RGB size 16
Test Accuracy of SVC =  0.4933
Testing RGB size 32
Test Accuracy of SVC =  0.48
Testing RGB size 64
Test Accuracy of SVC =  0.4778
Testing HSV size 8
Test Accuracy of SVC =  0.9911
Testing HSV size 16
Test Accuracy of SVC =  0.9933
Testing HSV size 32
Test Accuracy of SVC =  0.9956
Testing HSV size 64
Test Accuracy of SVC =  1.0
Testing LUV size 8
Test Accuracy of SVC =  0.9911
Testing LUV size 16
Test Accuracy of SVC =  0.9956
Testing LUV size 32
Test Accuracy of SVC =  0.9933
Testing LUV size 64
Test Accuracy of SVC =  0.9956
Testing HLS size 8
Test Accuracy of SVC =  0.9933
Testing HLS size 16
Test Accuracy of SVC =  0.9933
Testing HLS size 32
Test Accuracy of SVC =  1.0
Testing HLS size 64
Test Accuracy of SVC =  0.9933
Testing YUV size 8
Test Accuracy of SVC =  0.4867
Testing YUV size 16
Test Accuracy of SVC =  0.4889
Testing YUV size 32
Test Accuracy of SVC =  0.4911
Testing YUV size 64
Test Accuracy of SVC =  0.48
Testing YCrCb size 8
Test Accuracy of SVC =  0.4956
Testing YCrCb size 16
Test Accuracy of SVC =  0.4867
Testing YCrCb size 32
Test Accuracy of SVC =  0.48
Testing YCrCb size 64
Test Accuracy of SVC =  0.4889

Results using images 1501:3000
Testing RGB size 8
Test Accuracy of SVC =  0.4844
Testing RGB size 16
Test Accuracy of SVC =  0.4644
Testing RGB size 32
Test Accuracy of SVC =  0.4511
Testing RGB size 64
Test Accuracy of SVC =  0.4978
Testing HSV size 8
Test Accuracy of SVC =  0.98
Testing HSV size 16
Test Accuracy of SVC =  0.9778
Testing HSV size 32
Test Accuracy of SVC =  0.9756
Testing HSV size 64
Test Accuracy of SVC =  0.9822
Testing LUV size 8
Test Accuracy of SVC =  0.9933
Testing LUV size 16
Test Accuracy of SVC =  0.9867
Testing LUV size 32
Test Accuracy of SVC =  0.9867
Testing LUV size 64
Test Accuracy of SVC =  0.9867
Testing HLS size 8
Test Accuracy of SVC =  0.9867
Testing HLS size 16
Test Accuracy of SVC =  0.9844
Testing HLS size 32
Test Accuracy of SVC =  0.9867
Testing HLS size 64
Test Accuracy of SVC =  0.9867
Testing YUV size 8
Test Accuracy of SVC =  0.4511
Testing YUV size 16
Test Accuracy of SVC =  0.4756
Testing YUV size 32
Test Accuracy of SVC =  0.4889
Testing YUV size 64
Test Accuracy of SVC =  0.4978
Testing YCrCb size 8
Test Accuracy of SVC =  0.4889
Testing YCrCb size 16
Test Accuracy of SVC =  0.4956
Testing YCrCb size 32
Test Accuracy of SVC =  0.4822
Testing YCrCb size 64
Test Accuracy of SVC =  0.4867

Results using all images
Testing RGB size 8
Test Accuracy of SVC =  0.5079
Testing RGB size 16
Test Accuracy of SVC =  0.4906
Testing RGB size 32
Test Accuracy of SVC =  0.5146
Testing RGB size 64
Test Accuracy of SVC =  0.503
Testing HSV size 8
Test Accuracy of SVC =  0.8502
Testing HSV size 16
Test Accuracy of SVC =  0.8502
Testing HSV size 32
Test Accuracy of SVC =  0.8468
Testing HSV size 64
Test Accuracy of SVC =  0.8547
Testing LUV size 8
Test Accuracy of SVC =  0.8866
Testing LUV size 16
Test Accuracy of SVC =  0.8908
Testing LUV size 32
Test Accuracy of SVC =  0.8874
Testing LUV size 64
Test Accuracy of SVC =  0.9005
Testing HLS size 8
Test Accuracy of SVC =  0.8577
Testing HLS size 16
Test Accuracy of SVC =  0.8581
Testing HLS size 32
Test Accuracy of SVC =  0.8551
Testing HLS size 64
Test Accuracy of SVC =  0.8727
Testing YUV size 8
Test Accuracy of SVC =  0.4951
Testing YUV size 16
Test Accuracy of SVC =  0.5128
Testing YUV size 32
Test Accuracy of SVC =  0.5128
Testing YUV size 64
Test Accuracy of SVC =  0.5169
Testing YCrCb size 8
Test Accuracy of SVC =  0.4992
Testing YCrCb size 16
Test Accuracy of SVC =  0.5083
Testing YCrCb size 32
Test Accuracy of SVC =  0.5
Testing YCrCb size 64
Test Accuracy of SVC =  0.4947
"""
