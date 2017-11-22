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
hist_feat   = False # Histogram features on or off
hist_bins   = 32    # Number of histogram bins

# HOG configuration parameters
hog_cspace     = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_feat       = True  # HOG features on or off
orient         = 9     # HOG orientations
pix_per_cell   = 8     # HOG pixels per cell
cell_per_block = 4     # HOG cells per block
hog_channel    = 'ALL' # Can be 0, 1, 2, or "ALL"

# Include L-R flipped images in feature extraction?
flip_images = False

hog_modes = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
# Compare 0, 1, 2 & 3 to determine best HOG channel
# Compare 3, 4 & 5 to determine best orientations
# Compare 3, 6 & 7 to determine best pixels per cell
# Compare 3 & 8 to determine best cells per block
for hog_cspace in hog_modes :
  for x in range(9) :
    if x == 0 :
      hog_channel = 0
    elif x == 1 :
      hog_channel = 1
    elif x == 2 :
      hog_channel = 2
    elif x == 3 :
      hog_channel = 'ALL'
    elif x == 4 :
      orient = 5
    elif x == 5 :
      orient = 15
    elif x == 6 :
      orient = 9
      pix_per_cell = 4
    elif x == 7 :
      pix_per_cell = 16
    elif x == 8 :
      pix_per_cell = 8
      cell_per_block = 2

    print('Testing '+hog_cspace+' config '+str(x))
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
YUV & YCrCb color spaces are best
More gradient orientations are better
Pixels per cell has little impact
Cells per block has little impact
'ALL' channels are best

Results for images 0:1500
Testing RGB config 0
Test Accuracy of SVC =  0.9822
Testing RGB config 1
Test Accuracy of SVC =  0.9689
Testing RGB config 2
Test Accuracy of SVC =  0.9867
Testing RGB config 3
Test Accuracy of SVC =  0.98
Testing RGB config 4
Test Accuracy of SVC =  0.9778
Testing RGB config 5
Test Accuracy of SVC =  0.9933
Testing RGB config 6
Test Accuracy of SVC =  0.9911
Testing RGB config 7
Test Accuracy of SVC =  0.9711
Testing RGB config 8
Test Accuracy of SVC =  0.9911
Testing HSV config 0
Test Accuracy of SVC =  0.8911
Testing HSV config 1
Test Accuracy of SVC =  0.8822
Testing HSV config 2
Test Accuracy of SVC =  0.9822
Testing HSV config 3
Test Accuracy of SVC =  1.0
Testing HSV config 4
Test Accuracy of SVC =  0.9867
Testing HSV config 5
Test Accuracy of SVC =  0.9956
Testing HSV config 6
Test Accuracy of SVC =  0.9978
Testing HSV config 7
Test Accuracy of SVC =  0.9867
Testing HSV config 8
Test Accuracy of SVC =  0.9978
Testing LUV config 0
Test Accuracy of SVC =  0.9889
Testing LUV config 1
Test Accuracy of SVC =  0.9133
Testing LUV config 2
Test Accuracy of SVC =  0.8756
Testing LUV config 3
Test Accuracy of SVC =  0.9978
Testing LUV config 4
Test Accuracy of SVC =  0.9933
Testing LUV config 5
Test Accuracy of SVC =  1.0
Testing LUV config 6
Test Accuracy of SVC =  0.9978
Testing LUV config 7
Test Accuracy of SVC =  0.9956
Testing LUV config 8
Test Accuracy of SVC =  0.9978
Testing HLS config 0
Test Accuracy of SVC =  0.9067
Testing HLS config 1
Test Accuracy of SVC =  0.9889
Testing HLS config 2
Test Accuracy of SVC =  0.8778
Testing HLS config 3
Test Accuracy of SVC =  0.9978
Testing HLS config 4
Test Accuracy of SVC =  0.9956
Testing HLS config 5
Test Accuracy of SVC =  0.9978
Testing HLS config 6
Test Accuracy of SVC =  0.9911
Testing HLS config 7
Test Accuracy of SVC =  0.9889
Testing HLS config 8
Test Accuracy of SVC =  0.9889
Testing YUV config 0
Test Accuracy of SVC =  0.9822
Testing YUV config 1
Test Accuracy of SVC =  0.9222
Testing YUV config 2
Test Accuracy of SVC =  0.9267
Testing YUV config 3
Test Accuracy of SVC =  0.9978
Testing YUV config 4
Test Accuracy of SVC =  0.9911
Testing YUV config 5
Test Accuracy of SVC =  0.9933
Testing YUV config 6
Test Accuracy of SVC =  0.9911
Testing YUV config 7
Test Accuracy of SVC =  0.9844
Testing YUV config 8
Test Accuracy of SVC =  1.0
Testing YCrCb config 0
Test Accuracy of SVC =  0.9844
Testing YCrCb config 1
Test Accuracy of SVC =  0.9111
Testing YCrCb config 2
Test Accuracy of SVC =  0.9222
Testing YCrCb config 3
Test Accuracy of SVC =  0.9956
Testing YCrCb config 4
Test Accuracy of SVC =  0.9978
Testing YCrCb config 5
Test Accuracy of SVC =  0.9933
Testing YCrCb config 6
Test Accuracy of SVC =  0.9978
Testing YCrCb config 7
Test Accuracy of SVC =  0.9978
Testing YCrCb config 8
Test Accuracy of SVC =  0.9978

Results for images 1501:3000
Testing RGB config 0
Test Accuracy of SVC =  0.9244
Testing RGB config 1
Test Accuracy of SVC =  0.9356
Testing RGB config 2
Test Accuracy of SVC =  0.9289
Testing RGB config 3
Test Accuracy of SVC =  0.96
Testing RGB config 4
Test Accuracy of SVC =  0.9533
Testing RGB config 5
Test Accuracy of SVC =  0.9511
Testing RGB config 6
Test Accuracy of SVC =  0.9467
Testing RGB config 7
Test Accuracy of SVC =  0.9378
Testing RGB config 8
Test Accuracy of SVC =  0.9622
Testing HSV config 0
Test Accuracy of SVC =  0.9689
Testing HSV config 1
Test Accuracy of SVC =  0.9089
Testing HSV config 2
Test Accuracy of SVC =  0.9333
Testing HSV config 3
Test Accuracy of SVC =  0.9889
Testing HSV config 4
Test Accuracy of SVC =  0.98
Testing HSV config 5
Test Accuracy of SVC =  0.9911
Testing HSV config 6
Test Accuracy of SVC =  0.9911
Testing HSV config 7
Test Accuracy of SVC =  0.9956
Testing HSV config 8
Test Accuracy of SVC =  0.9911
Testing LUV config 0
Test Accuracy of SVC =  0.92
Testing LUV config 1
Test Accuracy of SVC =  0.9622
Testing LUV config 2
Test Accuracy of SVC =  0.9378
Testing LUV config 3
Test Accuracy of SVC =  0.9911
Testing LUV config 4
Test Accuracy of SVC =  0.9733
Testing LUV config 5
Test Accuracy of SVC =  0.9933
Testing LUV config 6
Test Accuracy of SVC =  0.9911
Testing LUV config 7
Test Accuracy of SVC =  0.9844
Testing LUV config 8
Test Accuracy of SVC =  0.9867
Testing HLS config 0
Test Accuracy of SVC =  0.9622
Testing HLS config 1
Test Accuracy of SVC =  0.9422
Testing HLS config 2
Test Accuracy of SVC =  0.9044
Testing HLS config 3
Test Accuracy of SVC =  0.9956
Testing HLS config 4
Test Accuracy of SVC =  0.9933
Testing HLS config 5
Test Accuracy of SVC =  0.9911
Testing HLS config 6
Test Accuracy of SVC =  0.9933
Testing HLS config 7
Test Accuracy of SVC =  0.9956
Testing HLS config 8
Test Accuracy of SVC =  0.9867
Testing YUV config 0
Test Accuracy of SVC =  0.9489
Testing YUV config 1
Test Accuracy of SVC =  0.9867
Testing YUV config 2
Test Accuracy of SVC =  0.9778
Testing YUV config 3
Test Accuracy of SVC =  0.9889
Testing YUV config 4
Test Accuracy of SVC =  0.9822
Testing YUV config 5
Test Accuracy of SVC =  1.0
Testing YUV config 6
Test Accuracy of SVC =  0.9956
Testing YUV config 7
Test Accuracy of SVC =  0.9978
Testing YUV config 8
Test Accuracy of SVC =  1.0
Testing YCrCb config 0
Test Accuracy of SVC =  0.9244
Testing YCrCb config 1
Test Accuracy of SVC =  0.9822
Testing YCrCb config 2
Test Accuracy of SVC =  0.9667
Testing YCrCb config 3
Test Accuracy of SVC =  0.9978
Testing YCrCb config 4
Test Accuracy of SVC =  0.9933
Testing YCrCb config 5
Test Accuracy of SVC =  1.0
Testing YCrCb config 6
Test Accuracy of SVC =  0.9933
Testing YCrCb config 7
Test Accuracy of SVC =  0.9978
Testing YCrCb config 8
Test Accuracy of SVC =  0.9956

Results for all images
Testing RGB config 0
"""
