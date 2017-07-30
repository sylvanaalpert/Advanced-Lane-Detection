import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from camera_calibration import CameraCalibration
from detection import AdvancedLaneDetector
import os
import numpy as np
import detection_utilities as du

import cv2


if __name__ == "__main__":
    
    print_img = False
    img_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Advanced-Lane-Lines/camera_cal'
    c = CameraCalibration(img_dir, 9, 6)
    
    d = AdvancedLaneDetector(c.get_params())
    
    output_img_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Advanced-Lane-Lines/output_images'
    
    if print_img:
        
        # Camera Calibration Example
        cal_img_orig = os.path.join(img_dir, 'calibration1.jpg')
        cal_img_undis = os.path.join(output_img_dir, 'calibration1_undist.jpg')
        cal_img_in = cv2.cvtColor(cv2.imread(cal_img_orig), cv2.COLOR_BGR2RGB)
        cal_img_out = d.undistort(cal_img_in)
        
        plt.figure
        plt.subplot(1, 2, 1)
        plt.imshow(cal_img_in)
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(cal_img_out)
        plt.title('Undistorted')
        plt.savefig(cal_img_undis, bbox_inches = 'tight', pad_inches = 0)
        
        
        # Pipeline
        test_img_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Advanced-Lane-Lines/test_images'
        
        test_img_curve = os.path.join(test_img_dir, 'test2.jpg')
        test_img_straight = os.path.join(test_img_dir, 'straight_lines1.jpg')
        
        
        # Undistortion
        test_img_undis = os.path.join(output_img_dir, 'test_undist.jpg')
        test_img_in = cv2.cvtColor(cv2.imread(test_img_curve), cv2.COLOR_BGR2RGB)
        test_img_out = d.undistort(test_img_in)
        
        plt.figure
        plt.subplot(1, 2, 1)
        plt.imshow(test_img_in)
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(test_img_out)
        plt.title('Undistorted')
        plt.savefig(test_img_undis, bbox_inches = 'tight', pad_inches = 0)
        
        
        # Perspective Transform
        test_img_pt = os.path.join(output_img_dir, 'test_pt.jpg')
        test_img_in = cv2.cvtColor(cv2.imread(test_img_straight), cv2.COLOR_BGR2RGB)
        undist = d.undistort(test_img_in)
        M, Minv, test_img_out = du.transform_perspective(undist)
        
        lefttop = [590, 450]
        righttop = [690, 450]
        leftbottom = [190, 719]
        rightbottom = [1140, 719]
        src = np.float32([leftbottom, lefttop, righttop, rightbottom])
        pts = np.int32(src.reshape((-1,1,2)))
        cv2.polylines(undist, [pts],True,(255,0,0), 3)
        
        leftbottom = [320, 719]
        lefttop = [320, 0]
        righttop = [960, 0]
        rightbottom = [960, 719]
        dst = np.float32([leftbottom, lefttop, righttop, rightbottom])
        pts = np.int32(dst.reshape((-1,1,2)))
        cv2.polylines(test_img_out, [pts],True,(255,0,0), 3)
        
        
        plt.figure
        plt.subplot(1, 2, 1)
        plt.imshow(undist)
        plt.title('Undistorted')
        plt.subplot(1, 2, 2)
        plt.imshow(test_img_out)
        plt.title('Transformed')
        plt.savefig(test_img_pt, bbox_inches = 'tight', pad_inches = 0)
        
        # RGB to Binary
        test_img_bin = os.path.join(output_img_dir, 'test_binary.jpg')
        test_img_in = cv2.cvtColor(cv2.imread(test_img_curve), cv2.COLOR_BGR2RGB)
        undist = d.undistort(test_img_in)
        M, Minv, pt = du.transform_perspective(undist)
        binary = du.combined_threshold(pt)
        test_img_out = np.dstack((binary, binary, binary)) * 255
        
        plt.figure
        plt.subplot(1, 2, 1)
        plt.imshow(pt)
        plt.title('After Perspective Transform')
        plt.subplot(1, 2, 2)
        plt.imshow(test_img_out)
        plt.title('Binary')
        plt.savefig(test_img_bin, bbox_inches = 'tight', pad_inches = 0)
        
        
        # Identified Lines
        test_img_lines = os.path.join(output_img_dir, 'test_lines.jpg')
        left_fit, right_fit = du.perform_blind_search(binary, color_img=test_img_lines)
        
        # Drawn Lines
        test_img_overlay = os.path.join(output_img_dir, 'test_result.jpg')
        test_img_in = cv2.cvtColor(cv2.imread(test_img_curve), cv2.COLOR_BGR2RGB)
        test_img_out = cv2.cvtColor(d.pipeline_for_frame(test_img_in), cv2.COLOR_RGB2BGR)
        cv2.imwrite(test_img_overlay, test_img_out)
        
        
    else:
    
        video_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Advanced-Lane-Lines/'
        input_name = 'project_video.mp4'
        output_name = 'processed.mp4'
    
        in_file = os.path.join(video_dir, input_name)
        out_file = os.path.join(video_dir, output_name)
        d.process(in_file, out_file)
    

            
           