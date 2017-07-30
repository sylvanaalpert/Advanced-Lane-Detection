import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from collections import deque
import detection_utilities as du
    
''' 
A class representing a road line and encapsulating all detection attributes
'''
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # The current x values
        self.current_x = []
        # x values of the last n fits of the line
        self.recent_xfitted = deque([])      
        # polynomial coefficients for the last n iterations
        self.last_fits = deque([])  
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # number of fits to average
        self.n = 5
        self.img_height = None
        self.img_width = None
        self.y = None
        
    def update_fit(self, newfit):
        
        if newfit is not None:
            self.detected = True
            self.diffs = np.array(self.current_fit) - np.array(newfit)
            self.current_fit = newfit
        
            if len(self.last_fits) == self.n:
                self.last_fits.popleft()
            self.last_fits.append(newfit)
        
            self.update_x()
            
            # update radius of curvature
            ym_per_pix = 30/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/700 # meters per pixel in x dimension
            y_eval = np.max(self.y)*ym_per_pix
            converted_fit = np.polyfit(self.y*ym_per_pix, self.get_best_x()*xm_per_pix, 2)
            self.radius_of_curvature = ((1 + (2*converted_fit[0]*y_eval + converted_fit[1])**2)**1.5) / np.absolute(2*converted_fit[0])
        else:
            self.detected = False
            
        
    
    def get_best_fit(self):
        array = np.array(self.last_fits)
        weights = np.linspace(0.1, 0.1*len(self.last_fits), num=len(self.last_fits))
        weights = weights / np.sum(weights)
        return np.average(array, axis=0, weights=weights)
    
    def get_best_x(self):
        best_fit = self.get_best_fit()
        best_x = best_fit[0]*self.y**2 + best_fit[1]*self.y + best_fit[2]
        return best_x
    
    def is_initialized(self):
        return self.img_height != None
    
    def set_image_size(self, h, w):
        self.img_height = h
        self.img_width = w
        self.y = np.linspace(0, h-1, h)
    
    def update_x(self):
        
        self.current_x = self.current_fit[0]*self.y**2 + self.current_fit[1]*self.y + self.current_fit[2]
        
        if len(self.recent_xfitted) == self.n:
            self.recent_xfitted.popleft()
        self.recent_xfitted.append(self.current_x)
        
        y_eval = np.max(self.y)
        x_eval = self.current_fit[0]*y_eval**2 + self.current_fit[1]*y_eval + self.current_fit[2]
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.line_base_pos = abs(x_eval - self.img_width/2) * xm_per_pix
            


''' 
A class that performs road line detection over a video
'''
class AdvancedLaneDetector(object):

    def __init__(self, camera_params):
        
        self.left = Line()
        self.right = Line()
        self.camera_params = camera_params
        self.center_offset = 0.0
    
    def process(self, input_video, output_video):
        original = VideoFileClip(input_video)
        marked = original.fl_image(self.pipeline_for_frame) 
        marked.write_videofile(output_video, audio=False)
        
    def undistort(self, img):
        undist = cv2.undistort(img, self.camera_params[0], self.camera_params[1], None, self.camera_params[0])
        return undist
    
    def search_lines(self, img):
        left_fit, right_fit = None, None
        parallel = False
        parallel_thr = 0.2
            
        if (self.left.detected == False | self.right.detected == False): 
            left_fit, right_fit = du.perform_blind_search(img)   
            parallel = np.linalg.norm(left_fit[:2] - right_fit[:2]) < parallel_thr if (left_fit is not None and right_fit is not None) else False
            
        else: 
            left_fit, right_fit = du.search_previous_locations(img, self.left.current_fit, self.right.current_fit)
            
            # If lines are not parallel, redo blind search
            parallel = np.linalg.norm(left_fit[:2] - right_fit[:2]) < parallel_thr if (left_fit is not None and right_fit is not None) else False
            
            if  (not parallel):
                left_fit, right_fit = du.perform_blind_search(img)
                
        if parallel: 
            self.left.update_fit(left_fit)
            self.right.update_fit(right_fit)
            self.center_offset = self.left.line_base_pos - self.right.line_base_pos
        else: 
            self.left.detected = False
            self.right.detected = False
        
    def draw_lines(self, undistorted_img, warped_binary, Minv):
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left.get_best_x(), self.left.y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right.get_best_x(), self.right.y])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
        
        # Add curvature measures and offset from the center. 
        cv2.putText(result, "Left curvature: %.1f m" % self.left.radius_of_curvature, (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(result, "Right curvature: %.1f m" % self.right.radius_of_curvature, (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(result, "Vehicle Lane Offset: %.2f m" % self.center_offset, (50, 170), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        return result
        
    
        
    def pipeline_for_frame(self, image):
        
        corrected_rgb = self.undistort(image)
        M, Minv, warped = du.transform_perspective(corrected_rgb)
        #return warped

        binary = du.combined_threshold(warped)
        
        #return np.dstack((binary, binary, binary)) * 255
        
        # Search lines
        if (self.left.is_initialized() == False | self.right.is_initialized() == False):
            h, w = binary.shape[:2]
            self.left.set_image_size(h, w)
            self.right.set_image_size(h, w)
            
        self.search_lines(binary)
        
        output = self.draw_lines(corrected_rgb, binary, Minv)

        
        return output