from moviepy.editor import VideoFileClip
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

tmp = pickle.load( open("trained_svc.p", "rb") )
svc = tmp['svc']
X_scaler = tmp['X_scaler']
colorspace = tmp['colorspace']
orient = tmp['orient']
pix_per_cell = tmp['pix_per_cell']
cell_per_block = tmp['cell_per_block']
hog_channel = tmp['hog_channel']
spatial_size = tmp['spatial_size']
hist_bins = tmp['hist_bins']
spatial_feat = tmp['spatial_feat']
hist_feat = tmp['hist_feat']
hog_feat = tmp['hog_feat']
flip_images = tmp['flip_images']

test_images = False
test_file   = 'test1'

test_clips  = False
full_movie  = True
dump_frame  = -1

if test_images :
  dump_file  = test_file
else :
  dump_file  = 'frame'+str(dump_frame)

# Min and max in x/y to search in slide_window()
# dist1 is close/big
# ...
# distX is far/small
dist1_x_start_stop = [280, None]
dist1_y_start_stop = [380, 690]
dist1_xy_window=(200,200)
dist1_xy_overlap=(0.8, 0.8) # 0.8
dist2_x_start_stop = [410, None]
dist2_y_start_stop = [380, 600]
dist2_xy_window=(140,140)
dist2_xy_overlap=(0.8, 0.8) # 0.65
dist3_x_start_stop = [470, None]
dist3_y_start_stop = [400, 520]
dist3_xy_window=(100,100)
dist3_xy_overlap=(0.8, 0.8) # 0.65
dist4_x_start_stop = [520, 1050]
dist4_y_start_stop = [400, 500]
dist4_xy_window=(60,60)
dist4_xy_overlap=(0.8, 0.8) # 0.50

heat_single_thr = 1

heat_history  = []
heat_hist_len =  5
heat_hist_thr =  5
hist_hist_idx =  0

frame_num = 0

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

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, rancol=False):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if rancol :
          color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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

def create_search_windows(srch_img) :
  dist1_windows = slide_window(srch_img,
                               x_start_stop=dist1_x_start_stop,
                               y_start_stop=dist1_y_start_stop, 
                               xy_window=dist1_xy_window,
                               xy_overlap=dist1_xy_overlap
                              )
  dist2_windows = slide_window(srch_img,
                               x_start_stop=dist2_x_start_stop,
                               y_start_stop=dist2_y_start_stop, 
                               xy_window=dist2_xy_window,
                               xy_overlap=dist2_xy_overlap
                              )
  dist3_windows = slide_window(srch_img,
                               x_start_stop=dist3_x_start_stop,
                               y_start_stop=dist3_y_start_stop, 
                               xy_window=dist3_xy_window,
                               xy_overlap=dist3_xy_overlap
                              )
  dist4_windows = slide_window(srch_img,
                               x_start_stop=dist4_x_start_stop,
                               y_start_stop=dist4_y_start_stop, 
                               xy_window=dist4_xy_window,
                               xy_overlap=dist4_xy_overlap
                              )
  #return [dist4_windows]
  return [dist1_windows, dist2_windows, dist3_windows, dist4_windows]

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img,
                        colorspace='RGB',
                        spatial_size=(32, 32),
                        hist_bins=32,
                        orient=9, 
                        pix_per_cell=8,
                        cell_per_block=2,
                        hog_channel=0,
                        spatial_feat=True,
                        hist_feat=True,
                        hog_feat=True
                       ) :    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    # Already converted the image
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(img, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(img, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.extend(get_hog_features(img[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(img[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def search_windows(img,
                   windows,
                   clf,
                   scaler,
                   colorspace='RGB',
                   spatial_size=(32, 32),
                   hist_bins=32, 
                   orient=9, 
                   pix_per_cell=8,
                   cell_per_block=2, 
                   hog_channel=0,
                   spatial_feat=True, 
                   hist_feat=True,
                   hog_feat=True
                  ) :
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                                       colorspace=colorspace,
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
        #5) Scale extracted features to be fed to classifier
        if scaler is None :
          test_features = np.array(features).reshape(1, -1)
        else :
          test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def draw_bw_heat_map(img,heat) :
  upscaled = np.copy(heat)
  upscaled = np.clip(20*upscaled, 0, 255)
  bwmap = cv2.resize(upscaled, (1280//4,960//4))
  h,w=bwmap.shape
  img[27:27+h, 5:5+w, 0] = bwmap
  img[27:27+h, 5:5+w, 1] = bwmap
  img[27:27+h, 5:5+w, 2] = bwmap
  cv2.putText(img, 'B/W heat map over '+str(heat_hist_len)+' last frames', (5,20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
  return img

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def update_heat_history(heatmap,frame) :
  global heat_history
  global hist_hist_idx
  
  if len(heat_history) == 0 :
    for x in range(heat_hist_len) :
      heat_history.append(np.zeros_like(heatmap).astype(np.float))

  heat_history[hist_hist_idx] = np.copy(heatmap)
  if hist_hist_idx == heat_hist_len-1 :
    hist_hist_idx = 0
  else :
    hist_hist_idx = hist_hist_idx + 1

  histmap = np.zeros_like(heatmap).astype(np.float)
  for x in range(heat_hist_len) :
    histmap = histmap + heat_history[x]

  return histmap

def draw_labeled_bboxes(img, labels, heatmap, heat_hist_len):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        tmp = heatmap[np.min(nonzeroy):np.max(nonzeroy), np.min(nonzerox):np.max(nonzerox)]
        if np.min(nonzerox) == np.max(nonzerox) or np.min(nonzeroy) == np.max(nonzeroy) :
          #print('Zero width or height box')
          continue

        if test_images == True or frame_num == dump_frame or heat_hist_len < 2 :
          conf_str = ''
          conf_col = (0,255,0)
        elif np.max(tmp) > 9 :
          conf_str = 'high'
          conf_col = (0,255,0)
        elif np.max(tmp) > 6 :
          conf_str = 'ave.'
          conf_col = (255,255,0)
        else :
          conf_str = 'low'
          conf_col = (255,0,0)

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], conf_col, 6)

        cv2.putText(img, '#'+str(car_number)+' '+conf_str, (np.min(nonzerox)-20, np.min(nonzeroy)-20), cv2.FONT_HERSHEY_DUPLEX, 0.9, conf_col, 2)
    # Return the image
    return img

def process_frame(img) :
  global frame_num
  frame_num += 1

  draw_img = np.copy(img)

  # apply color conversion if other than 'RGB'
  if colorspace != 'RGB':
      if colorspace == 'HSV':
          srch_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
      elif colorspace == 'LUV':
          srch_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
      elif colorspace == 'HLS':
          srch_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
      elif colorspace == 'YUV':
          srch_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
      elif colorspace == 'YCrCb':
          srch_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
  else: srch_img = np.copy(img)

  # Uncomment the following line if you extracted training
  # data from .png images (scaled 0 to 1 by mpimg) and the
  # image you are searching is a .jpg (scaled 0 to 255)
  srch_img = srch_img.astype(np.float32)/255

  srch_windows = create_search_windows(srch_img)
  x = 0
  srch_hits = []
  heat = np.zeros_like(srch_img[:,:,0]).astype(np.float)
  for wins in srch_windows :
    hits = search_windows(srch_img,
                          wins,
                          svc,
                          X_scaler,
                          colorspace=colorspace,
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
    srch_hits.append(hits)
    heat = add_heat(heat,hits)

    # Add some debug visualisations - Disabled
    if test_images or frame_num == dump_frame :
      if x == 0 :
        color = (255,0,0) # dist1
      elif x == 1 :
        color = (0,255,0) # dist2
      elif x == 2 :
        color = (0,0,255) # dist3
      else :
        color = (128,128,128) # dist4
      #draw_img = draw_boxes(draw_img,wins,color,rancol=True)
      draw_img = draw_boxes(draw_img,hits,color)
    x += 1

  if test_images == True or frame_num == dump_frame :
    plt.imshow(draw_img)
    plt.savefig('output_images/'+dump_file+'_hits.png')
    plt.close
    plt.imshow(heat,cmap='hot')
    plt.savefig('output_images/'+dump_file+'_heatmap.png')
    plt.close
    draw_img = np.copy(img)

  # Is the history of the heatmap enabled?
  if (test_images == False or frame_num != dump_frame) and heat_hist_len > 1 :
    # Use past history of hits to improve detections
    heat     = update_heat_history(heat,frame_num)
    heat_thr = heat_hist_thr
  else :
    heat_thr = heat_single_thr

  # Compute and draw the final detection boxes - Enable for submission
  if True :
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heat,heat_thr)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_img,labels,heat,heat_hist_len)
    if test_images == True or frame_num == dump_frame :
      plt.imshow(labels[0],cmap='gray')
      plt.savefig('output_images/'+dump_file+'_label.png')
      plt.close

  # Draw some debug information on screen
  # - black/white heat map
  # - frame number
  if test_images == False :
    draw_img = draw_bw_heat_map(draw_img,heat)
    cv2.putText(draw_img, 'Frame '+str(frame_num), (10,710), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)

  if test_images == True or frame_num == dump_frame :
    plt.imshow(draw_img)
    plt.savefig('output_images/'+dump_file+'_bboxes.png')
    plt.close

  return draw_img

if test_images :
  _ = process_frame(mpimg.imread('test_images/'+test_file+'.jpg'))
  test_images = False

if test_clips :
  mov = VideoFileClip("test_video.mp4")
  ann = mov.fl_image(process_frame)
  ann.write_videofile("test_video_vehicle_tracking.mp4", audio=False)
  # Empty freeway to test false positives
  mov = VideoFileClip("project_video.mp4").subclip(0,10)
  ann = mov.fl_image(process_frame)
  ann.write_videofile("project_video_vehicle_tracking.mp4", audio=False)

if full_movie :
  mov = VideoFileClip("project_video.mp4")
  #mov = VideoFileClip("project_video.mp4").subclip(9,9.2) # 250
  #mov = VideoFileClip("project_video.mp4").subclip(50,50.2) #1260
  ann = mov.fl_image(process_frame)
  ann.write_videofile("project_video_vehicle_tracking.mp4", audio=False)