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
spatial_feat   = True     # Spatial features on or off
spatial_size   = (16, 16) # Spatial binning dimensions

# Histogram feature configuration parameters
hist_cspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hist_feat   = False # Histogram features on or off
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

spatial_modes = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
spatial_sizes = [(8,8), (16,16), (32,32)]
#cars = cars[:1500]
#notcars = notcars[:1500]
#cars = cars[1501:3000]
#notcars = notcars[1501:3000]

for spatial_cspace in spatial_modes :
  for spatial_size in spatial_sizes :
    print('Testing '+spatial_cspace+' size '+str(spatial_size[0]))
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
YUV or LUV are the best color spaces
Larger sizes are better but take longer to compute

Results using images 0:1500
Testing RGB size 8
Test Accuracy of SVC =  0.96
Testing RGB size 16
Test Accuracy of SVC =  0.9756
Testing RGB size 32
Test Accuracy of SVC =  0.9711
Testing HSV size 8
Test Accuracy of SVC =  0.8889
Testing HSV size 16
Test Accuracy of SVC =  0.9556
Testing HSV size 32
Test Accuracy of SVC =  0.9378
Testing LUV size 8
Test Accuracy of SVC =  0.9667
Testing LUV size 16
Test Accuracy of SVC =  0.9756
Testing LUV size 32
Test Accuracy of SVC =  0.9778
Testing HLS size 8
Test Accuracy of SVC =  0.9022
Testing HLS size 16
Test Accuracy of SVC =  0.9178
Testing HLS size 32
Test Accuracy of SVC =  0.9444
Testing YUV size 8
Test Accuracy of SVC =  0.9622
Testing YUV size 16
Test Accuracy of SVC =  0.9756
Testing YUV size 32
Test Accuracy of SVC =  0.9867
Testing YCrCb size 8
Test Accuracy of SVC =  0.98
Testing YCrCb size 16
Test Accuracy of SVC =  0.9644
Testing YCrCb size 32
Test Accuracy of SVC =  0.9778

Results using images 1501:3000
Testing RGB size 8
Test Accuracy of SVC =  0.9511
Testing RGB size 16
Test Accuracy of SVC =  0.9444
Testing RGB size 32
Test Accuracy of SVC =  0.9844
Testing HSV size 8
Test Accuracy of SVC =  0.8867
Testing HSV size 16
Test Accuracy of SVC =  0.8622
Testing HSV size 32
Test Accuracy of SVC =  0.9333
Testing LUV size 8
Test Accuracy of SVC =  0.9289
Testing LUV size 16
Test Accuracy of SVC =  0.9533
Testing LUV size 32
Test Accuracy of SVC =  0.9644
Testing HLS size 8
Test Accuracy of SVC =  0.8933
Testing HLS size 16
Test Accuracy of SVC =  0.8867
Testing HLS size 32
Test Accuracy of SVC =  0.8978
Testing YUV size 8
Test Accuracy of SVC =  0.9378
Testing YUV size 16
Test Accuracy of SVC =  0.9622
Testing YUV size 32
Test Accuracy of SVC =  0.9556
Testing YCrCb size 8
Test Accuracy of SVC =  0.9444
Testing YCrCb size 16
Test Accuracy of SVC =  0.9489
Testing YCrCb size 32
Test Accuracy of SVC =  0.9667

Results using all images
Testing RGB size 8
Test Accuracy of SVC =  0.899
Testing RGB size 16
Test Accuracy of SVC =  0.9058
Testing RGB size 32
Test Accuracy of SVC =  0.9054
Testing HSV size 8
Test Accuracy of SVC =  0.8761
Testing HSV size 16
Test Accuracy of SVC =  0.8915
Testing HSV size 32
Test Accuracy of SVC =  0.8701
Testing LUV size 8
Test Accuracy of SVC =  0.893
Testing LUV size 16
Test Accuracy of SVC =  0.9238
Testing LUV size 32
Test Accuracy of SVC =  0.8964
Testing HLS size 8
Test Accuracy of SVC =  0.8686
Testing HLS size 16
Test Accuracy of SVC =  0.8795
Testing HLS size 32
Test Accuracy of SVC =  0.8735
Testing YUV size 8
Test Accuracy of SVC =  0.9024
Testing YUV size 16
Test Accuracy of SVC =  0.9223
Testing YUV size 32
Test Accuracy of SVC =  0.8956
Testing YCrCb size 8
Test Accuracy of SVC =  0.9043
Testing YCrCb size 16
Test Accuracy of SVC =  0.9294
Testing YCrCb size 32
Test Accuracy of SVC =  0.9118
"""
