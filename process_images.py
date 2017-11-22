import pickle
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from lesson_functions import *

tmp = pickle.load( open("trained_svc.p", "rb") )
svc = tmp['svc']
X_scaler = tmp['X_scaler']
spatial_cspace = tmp['spatial_cspace']
hist_cspace = tmp['hist_cspace']
hog_cspace = tmp['hog_cspace']
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

# Min and max in x/y to search in slide_window()
# dist1 is close/big
# ...
# dist4 is far/small
dist1_x_start_stop = [280, None]
dist1_y_start_stop = [380, 690]
dist1_xy_window=(220,180)
dist1_xy_overlap=(0.80, 0.80)
dist2_x_start_stop = [380, None]
dist2_y_start_stop = [380, 600]
dist2_xy_window=(160,128)
dist2_xy_overlap=(0.65, 0.65)
dist3_x_start_stop = [470, None]
dist3_y_start_stop = [400, 520]
dist3_xy_window=(120,80)
dist3_xy_overlap=(0.65, 0.65)
dist4_x_start_stop = [520, 1050]
dist4_y_start_stop = [400, 500]
dist4_xy_window=(60,60)
dist4_xy_overlap=(0.50, 0.50)

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

images = glob.glob('test_images/*.jpg')
#images = ['Clip_0_5_frames\\frame_in_7.jpg', 'Clip_0_5_frames\\frame_in_8.jpg', 'Clip_0_5_frames\\frame_in_9.jpg', 'Clip_0_5_frames\\frame_in_10.jpg', 'Clip_0_5_frames\\frame_in_11.jpg']
for image in images :
  tmp1 = image.split('\\')
  tmp2 = tmp1[1].split('.')
  img = mpimg.imread(image)

  # Uncomment the following line if you extracted training
  # data from .png images (scaled 0 to 1 by mpimg) and the
  # image you are searching is a .jpg (scaled 0 to 255)
  srch_img = img.astype(np.float32)/255

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
  #boxes_img = np.copy(img)
  #boxes_img = draw_boxes(boxes_img, dist1_windows, color=(0, 0, 255), thick=6)
  #boxes_img = draw_boxes(boxes_img, dist2_windows, color=(0, 255, 0), thick=6)
  #boxes_img = draw_boxes(boxes_img, dist3_windows, color=(255, 0, 0), thick=6)
  #boxes_img = draw_boxes(boxes_img, dist4_windows, color=(127, 127, 127), thick=6)
  #plt.imshow(boxes_img)
  #plt.show()
  #plt.savefig('output_images/'+tmp2[0]+'_srch_windows.png')

  dist1_hits = search_windows(srch_img,
                              dist1_windows,
                              svc,
                              X_scaler,
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
                              hog_feat=hog_feat
                             )                       
  dist2_hits = search_windows(srch_img,
                              dist2_windows,
                              svc,
                              X_scaler,
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
                              hog_feat=hog_feat
                             )                       
  dist3_hits = search_windows(srch_img,
                              dist3_windows,
                              svc,
                              X_scaler,
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
                              hog_feat=hog_feat
                             )                       
  dist4_hits = search_windows(srch_img,
                              dist4_windows,
                              svc,
                              X_scaler,
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
                              hog_feat=hog_feat
                             )                       
  hits_img = np.copy(img)
  hits_img = draw_boxes(hits_img, dist1_hits, color=(0, 0, 255), thick=6)
  hits_img = draw_boxes(hits_img, dist2_hits, color=(0, 255, 0), thick=6)
  hits_img = draw_boxes(hits_img, dist3_hits, color=(255, 0, 0), thick=6)
  hits_img = draw_boxes(hits_img, dist4_hits, color=(128, 128, 128), thick=6)
  oput = 'output_images/'+tmp2[0]+'_distx_hits.png'
  plt.imshow(hits_img)
  #plt.show()
  plt.savefig(oput)

  heat = np.zeros_like(srch_img[:,:,0]).astype(np.float)
  heat = add_heat(heat,dist1_hits)
  heat = add_heat(heat,dist2_hits)
  heat = add_heat(heat,dist3_hits)
  heat = add_heat(heat,dist4_hits)
  # Visualize the heatmap when displaying    
  heatmap = np.clip(heat, 0, 255)

  # Apply threshold to help remove false positives
  heat = apply_threshold(heat,2)

  # Find final boxes from heatmap using label function
  labels = label(heatmap)
  draw_img = draw_labeled_bboxes(np.copy(img), labels, heatmap)

  #oput = 'output_images/'+tmp2[0]+'_boxes_heatmap.png'
  #fig = plt.figure()
  #plt.subplot(121)
  #plt.imshow(draw_img)
  #plt.title('Car Positions')
  #plt.subplot(122)
  #plt.imshow(heatmap, cmap='hot')
  #plt.title('Heat Map')
  #fig.tight_layout()
  #plt.show()
  #plt.savefig(oput)
