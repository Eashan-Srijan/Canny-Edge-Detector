####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 8th October 2021                ##
####################################################
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
from convoluion import SeConvolve

GRSC_PATH = 'grsc.PNG'

class CannyEdgeDetector:
    
    def __init__(self, image_path):
        # Path of the image, on which canny detector will be applied
        self.image_path = image_path
        # Read Image
        self.image_read()
        # Gaussian Kernel used for smoothing
        self._gaussian_kernel = [[1, 1, 2, 2, 2, 1, 1], [1, 2, 2, 4, 2, 2, 1], [2, 2, 4, 8, 4, 2, 2], [2, 4, 8, 16, 8, 4, 2], [2, 2, 4, 8, 4, 2, 2], [1, 2, 2, 4, 2, 2, 1], [1, 1, 2, 2, 2, 1, 1]]
        # Gradient x and y operations Prewitt's Operators
        self._convolution_matrix_gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        self._convolution_matrix_gy = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
        # Output of step 1
        self._smoothed_image = None
        # Output of step 2
        ## Normalized Output
        self._gradient_x = None
        self._gradient_y = None
        self._magnitude = None
        ## Normalized Output
        self._gradient_x_norm = None
        self._gradient_y_norm = None
        self._magnitude_norm = None
        ## Angle Output
        self._angle = None
        self._edge_angle = None
        # Output of step 3
        self._non_max_output = None
        # Output of step 4
        self._threshold_output = None
    
    #######################
    ## Getter and Setter ##
    #######################
    
    @property
    def gaussian_kernel(self):
        return self._gaussian_kernel

    @property
    def convolution_matrix_gx(self):
        return self._convolution_matrix_gx
    
    @property
    def convolution_matrix_gy(self):
        return self._convolution_matrix_gy
    
    @property
    def image_matrix(self):
        return self._image_matrix
    
    @property
    def smoothed_image(self):
        return self._smoothed_image
    
    #######################
    #######################
        
    def image_read(self):
        src = cv2.imread(self.image_path)
        self.img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        now_time = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")

        plt.imsave('grsc/' + now_time + GRSC_PATH, self.img, cmap='gray')

        self.covert_to_matrix('grsc/' + now_time + GRSC_PATH)

    
    def covert_to_matrix(self, path):

        gsrc = cv2.imread(path, 0)
        
        matrix = list()
        
        for row_index in range(0, gsrc.shape[0]):

            row = []
            
            for column_index in range(0, gsrc.shape[1]):
            
                pixel = gsrc.item(row_index, column_index)
            
                row.append(pixel)
            
            matrix.append(row)
        
        self._image_matrix = matrix
    
    # Main procedure
    def canny_detector(self):
        pass

    # Step 1: Gaussian Smoothing
    def gaussian_smoothing(self):
        pass

    # Step 2: Gradient Operation
    def gradient_operation(self):
        pass

    # Step 3: Non-Maxima Suppression
    def non_max_suppression(self):
        pass

    # Step 4 Thresholding
    def thresholding(self):
        pass