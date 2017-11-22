from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob

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
def extract_features(file,
                     cspace='RGB',
                     orient=9, 
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0
                    ):
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

    _, hog_vis = get_hog_features(feature_image[:,:,hog_channel], orient, 
                pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    return image, feature_image, hog_vis

car_images = glob.glob('../vehicles/*/*.png')
noncar_images = glob.glob('../non-vehicles/*/*.png')
print('Number of cars image = '+str(len(car_images)))
print('Number of non-cars image = '+str(len(noncar_images)))

# Plot some of the car and non-car images
if False :
  fig, axs = plt.subplots(2,4)
  fig.subplots_adjust(hspace=.2, wspace=.01)
  axs = axs.ravel()
  for i in range(8) :
    img = mpimg.imread(car_images[7*i])
    axs[i].axis('off')
    axs[i].imshow(img)
  plt.savefig('output_images/example_cars.jpg')
  for i in range(8) :
    img = mpimg.imread(noncar_images[7*i])
    axs[i].axis('off')
    axs[i].imshow(img)
  plt.savefig('output_images/example_noncars.jpg')
  plt.close

# Feature extraction parameters
hog_feat       = True # HOG features on or off
colorspace     = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 11
pix_per_cell   = 16
cell_per_block =  2
hog_channel    =  0 # Can be 0, 1 or 2

car_file = car_images[36]
print('View HOG for file '+car_file)
noncar_file = noncar_images[36]
print('View HOG for file '+car_file)

fig, axs = plt.subplots(1,3)
fig.subplots_adjust(hspace=.2, wspace=.1)
axs = axs.ravel()

car_img, conv_img, car_hog_vis = extract_features(car_file,
                                                  cspace=colorspace,
                                                  orient=orient, 
                                                  pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block, 
                                                  hog_channel=hog_channel
                                                 )
axs[0].imshow(car_img)
axs[0].set_title('Source image')
axs[1].imshow(conv_img[:,:,hog_channel],cmap='gray')
axs[1].set_title('Color channel')
axs[2].imshow(car_hog_vis,cmap='gray')
axs[2].set_title('HOG')
plt.savefig('output_images/hog_vis_car.jpg')

noncar_img, nonconv_img, noncar_hog_vis = extract_features(noncar_file,
                                                           cspace=colorspace,
                                                           orient=orient, 
                                                           pix_per_cell=pix_per_cell,
                                                           cell_per_block=cell_per_block, 
                                                           hog_channel=hog_channel
                                                          )
axs[0].imshow(noncar_img)
axs[0].set_title('Source image')
axs[1].imshow(nonconv_img[:,:,hog_channel],cmap='gray')
axs[1].set_title('Color channel')
axs[2].imshow(noncar_hog_vis,cmap='gray')
axs[2].set_title('HOG')
plt.savefig('output_images/hog_vis_noncar.jpg')
