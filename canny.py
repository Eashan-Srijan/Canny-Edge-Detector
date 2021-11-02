####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 8th October 2021                ##
####################################################
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
from convolution import SeConvolve
import math
import copy

GRSC_PATH = 'grsc.PNG'

class CannyEdgeDetector:
    
    def __init__(self, image_path):
        # Path of the image, on which canny detector will be applied
        self.image_path = image_path
        # Read Image
        self.image_read()
        # Gaussian Kernel used for smoothing
        self._gaussian_kernel = np.array([[1.0,1.0,2.0,2.0,2.0,1.0,1.0],[1.0,2.0,2.0,4.0,2.0,2.0,1.0],[2.0,2.0,4.0,8.0,4.0,2.0,2.0],[2.0,4.0,8.0,16.0,8.0,4.0,2.0],[2.0,2.0,4.0,8.0,4.0,2.0,2.0],[1.0,2.0,2.0,4.0,2.0,2.0,1.0],[1.0,1.0,2.0,2.0,2.0,1.0,1.0]])
        # Gradient x and y operations Prewitt's Operators
        self._convolution_matrix_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self._convolution_matrix_gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
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
        self._threshold_output_25 = None
        self._threshold_output_50 = None
        self._threshold_output_75 = None
    
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
        
    @property
    def gradient_x(self):
        return self._gradient_x
            
    @property
    def gradient_y(self):
        return self._gradient_y
    
    @property
    def magnitude(self):
        return self._magnitude
        
    @property
    def gradient_x_norm(self):
        return self._gradient_x_norm
    
    @property
    def gradient_y_norm(self):
        return self._gradient_y_norm
    
    @property
    def magnitude_norm(self):
        return self._magnitude_norm
    
    @property
    def angle(self):
        return self._angle
    
    @property
    def edge_angle(self):
        return self._edge_angle
    
    @property
    def non_max_output(self):
        return self._non_max_output
    
    @property
    def threshold_output_25(self):
        return self._threshold_output_25
    
    @property
    def threshold_output_50(self):
        return self._threshold_output_50
        
    @property
    def threshold_output_75(self):
        return self._threshold_output_75
    
    #######################
    #######################
        
    #### Check out: img = np.array(Image.open('path_to_file\file.bmp'))
    def image_read(self):
        src = cv2.imread(self.image_path)
        self.img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        now_time = datetime.datetime.now().strftime("%d%m%Y%H%M%S")

        plt.imsave(now_time + '-' + GRSC_PATH, self.img, cmap='gray')

        self.covert_to_matrix(now_time + '-' + GRSC_PATH)
  
    def covert_to_matrix(self, path):

        gsrc = cv2.imread(path, 0)
        
        matrix = list()
        
        for row_index in range(0, gsrc.shape[0]):

            row = []
            
            for column_index in range(0, gsrc.shape[1]):
            
                pixel = gsrc.item(row_index, column_index)
            
                row.append(pixel)
            
            matrix.append(row)
        
        self._image_matrix = np.array(matrix)
    
    # Main procedure
    def canny_detector(self):
        
        self.gaussian_smoothing()
        self.gradient_operation()
        self.non_max_suppression()
        self.thresholding()

    # Step 1: Gaussian Smoothing
    def gaussian_smoothing(self):
        
        smoothing = SeConvolve(self._image_matrix, self._gaussian_kernel)

        self._smoothed_image = smoothing.convolution()

    # Step 2: Gradient Operation
    def gradient_operation(self):
        
        gradient_x = SeConvolve(self._smoothed_image, self._convolution_matrix_gx, mode='gradient')
        self._gradient_x = gradient_x.convolution()
        
        
        gradient_x = SeConvolve(self._smoothed_image, self._convolution_matrix_gy, mode='gradient')
        self._gradient_y = gradient_x.convolution()
        
        
        self._magnitude = self.calcuate_magnitude(self._gradient_x, self._gradient_y)
        self._angle, self._edge_angle = self.calculate_angle(self._gradient_x, self._gradient_y)
    
    # TODO: Magnitude, Angle and edge angle
    def calcuate_magnitude(self, gradient_x, gradient_y):
        height, width = gradient_x.shape

        magnitude = np.zeros((height - 8, width - 8))

        for i in range(4,height - 4):
            for j in range(4,width - 4):
                temp = (gradient_x[i, j] ** 2) + (gradient_y[i, j] ** 2)
                
                magnitude[i - 4, j - 4] = math.sqrt(temp)
        
        magnitude = np.pad(magnitude, 4, mode='constant')

        return magnitude
    
    def calculate_angle(self, gradient_x, gradient_y):
        
        height, width = gradient_x.shape
        
        angle = np.zeros((height - 8, width - 8))
        edge_angle = np.zeros((height - 8, width - 8))
        
        for i in range(4,height - 4):
            for j in range(4,width - 4):
                if gradient_x[i, j] != 0:
                    angle[i - 4, j - 4] = math.degrees(math.atan((gradient_y[i, j] / gradient_x[i, j])))
                    edge_angle[i - 4, j - 4] = angle[i - 4, j - 4] + 90
        
        angle = np.pad(angle, 4, mode='constant')
        edge_angle = np.pad(angle, 4, mode='constant')

        return angle, edge_angle

    # Step 3: Non-Maxima Suppression
    def non_max_suppression(self):
        angle = self._angle + 360
        magnitude = copy.deepcopy(self._magnitude)

        height, width = magnitude.shape
        
        for i in range(4,height - 4):
            for j in range(4,width - 4):
                current_sector = self.sector(angle[i, j])
                check_one, check_two = self.check(current_sector)
                check_one_x, check_one_y = check_one
                check_two_x, check_two_y = check_two

                if not(magnitude[i, j] > magnitude[check_one_x, check_one_y] and magnitude[i, j] > magnitude[check_two_x, check_two_y]):
                    magnitude[i, j] = 0
        
        self._non_max_output = magnitude

    
    def sector(self, angle):
        pass

    def check(self, current_sector):
        pass

    # Step 4 Thresholding
    def thresholding(self):

        temp_magnitude = copy.deepcopy(self._magnitude)
        temp_magnitude = temp_magnitude[4: temp_magnitude.shape[0] - 4, 4: temp_magnitude.shape[1] - 4].flatten()

        
        magnitude = copy.deepcopy(self._magnitude)
        percentile = np.percentile(magnitude, 25)
        self.threshold_output_25 = self.threshold(magnitude, percentile)

                
        magnitude = copy.deepcopy(self._magnitude)
        percentile = np.percentile(magnitude, 50)
        self.threshold_output_50 = self.threshold(magnitude, percentile)

                
        magnitude = copy.deepcopy(self._magnitude)
        percentile = np.percentile(magnitude, 75)
        self.threshold_output_75 = self.threshold(magnitude, percentile)
    
    def threshold(magnitude, T):
        
        height, width = magnitude.shape

        for i in range(4,height - 4):
            for j in range(4,width - 4):

                if magnitude[i,j] < T:
                    magnitude[i,j] = 0
        
        return magnitude