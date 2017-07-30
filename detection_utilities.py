
import numpy as np
import cv2

#####################################
##########  Edge detection ##########
#####################################
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0,255)):
    x = 0
    y = 0
    if orient == 'x':
        x = 1
    elif orient == 'y':
        y = 1
    else:
        raise ValueError('Unrecognized input')
    
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    alpha = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(alpha)
    binary_output[(alpha > thresh[0]) & (alpha < thresh[1])] = 1
    return binary_output

#########################################
####### Color Thresholding ##############
#########################################
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

def combined_threshold(img):
    
    gray_S = hls_select(img)
    ksize = 11
    gradx = abs_sobel_thresh(gray_S, orient='x', sobel_kernel=ksize, thresh=(70, 255))
    grady = abs_sobel_thresh(gray_S, orient='y', sobel_kernel=ksize, thresh=(70, 255))
    mag_binary = mag_thresh(gray_S, sobel_kernel=ksize, mag_thresh=(50, 255))
    
    # Color mask
    yellow_mask = hls_select(img, thresh=(180, 255));
    white_mask = np.zeros_like(img[:, :, 0])
    white_mask[(img[:, :, 0] > 215) & (img[:, :, 0] < 255)] = 1
    
    binary = np.zeros_like(gray_S)
    binary[(white_mask == 1) | (yellow_mask == 1) | ((gradx == 1) & (grady == 1)) | ((mag_binary == 1))] = 1

    return binary


def transform_perspective(img):
    h, w = img.shape[:2]
    
    lefttop = [590, 450]
    righttop = [690, 450]
    leftbottom = [190, h-1]
    rightbottom = [1140, h-1]
    src = np.float32([leftbottom, lefttop, righttop, rightbottom])
    
    leftbottom = [320, h-1]
    lefttop = [320, 0]
    righttop = [960, 0]
    rightbottom = [960, h-1]
    dst = np.float32([leftbottom, lefttop, righttop, rightbottom])

    
        
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (w, h)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return M, Minv, warped


def perform_blind_search(img, color_img=''):
    
    dbg_img = np.dstack((img, img, img))*255
    left_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    right_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # Calculate histogram and its peaks
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Search windows definition
    nwindows = 10
    window_height = np.int(img.shape[0]/nwindows)
    margin = 80
    
    # Nonzero pixel locations
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Minimum number of pixels found to trigger re-centering of windows
    minpix = 20
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(dbg_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(dbg_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If enough pixels are found, re-center next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    if np.sum(left_lane_inds) > minpix:
        left_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None
    
    
    if np.sum(right_lane_inds) > minpix:    
        right_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]   
        # Fit a second order polynomial to each    
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None
    
    if color_img:
        colored_lines_img = left_img + right_img
        cv2.imwrite(color_img, colored_lines_img)
    
    return left_fit, right_fit


def search_previous_locations(img, prev_left_fit, prev_right_fit):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    
    left_min = (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] - margin)
    left_max = (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin)
    left_lane_inds = ((nonzerox > left_min) & (nonzerox < left_max))
    
    right_min = (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] - margin)
    right_max = (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin) 
    right_lane_inds = ((nonzerox > right_min) & (nonzerox < right_max))
    
    minpix = 20
    
    if np.sum(left_lane_inds) > minpix:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None
    
    if np.sum(right_lane_inds) > minpix:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None
    
    return left_fit, right_fit