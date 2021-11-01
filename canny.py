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
    def threshold_output(self):
        return self._threshold_output
    
    #######################
    #######################
        
    #### Check out: img = np.array(Image.open('path_to_file\file.bmp'))
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
        
        self.gaussian_smoothing()
        self.gradient_operation()
        self.non_max_suppression()
        self.thresholding()

    # Step 1: Gaussian Smoothing
    def gaussian_smoothing(self):
        smoothing = SeConvolve()

        self._smoothed_image = smoothing.convolution(self._image_matrix, self._gaussian_kernel)

    # Step 2: Gradient Operation
    def gradient_operation(self):
        
        gradient = SeConvolve()
        self._gradient_x = gradient.convolution(self._smoothed_image, self._convolution_matrix_gx, mode='gradient')
        self._gradient_y = gradient.convolution(self._smoothed_image, self._convolution_matrix_gy, mode='gradient')

    # Step 3: Non-Maxima Suppression
    def non_max_suppression(self):
        pass

    # Step 4 Thresholding
    def thresholding(self):
        pass


    # Junk:


    # def prewittop(b):
    #   grey = np.array(Image.open(b)).astype(np.uint8)
    #   print("The values of the read image are ")
    #   print(grey)
    #   height,width=grey.shape
    #   Px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    #   Py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    #   for i in range(5,height-5):
    #     for j in range(5,width-5):
    #        Sx =     (Sx[0, 0] * grey[i - 1, j - 1]) + \
    #                 (Sx[0, 1] * grey[i - 1, j]) + \
    #                 (Sx[0, 2] * grey[i - 1, j + 1]) + \
    #                 (Sx[1, 0] * grey[i, j - 1]) + \
    #                 (Sx[1, 1] * grey[i, j]) + \
    #                 (Sx[1, 2] * grey[i, j + 1]) + \
    #                 (Sx[2, 0] * grey[i + 1, j - 1]) + \
    #                 (Sx[2, 1] * grey[i + 1, j]) + \
    #                 (Sx[2, 2] * grey[i + 1, j + 1])

    #        Sy =     (Sy[0, 0] * grey[i - 1, j - 1]) + \
    #                 (Sy[0, 1] * grey[i - 1, j]) + \
    #                 (Sy[0, 2] * grey[i - 1, j + 1]) + \
    #                 (Sy[1, 0] * grey[i, j - 1]) + \
    #                 (Sy[1, 1] * grey[i, j]) + \
    #                 (Sy[1, 2] * grey[i, j + 1]) + \
    #                 (Sy[2, 0] * grey[i + 1, j - 1]) + \
    #                 (Sy[2, 1] * grey[i + 1, j]) + \
    #                 (Sy[2, 2] * grey[i + 1, j + 1])
    #        ngx=Sxgrad
    #        ngy=Sygrad
    #        if(ngx[i,j]==0):
    #          tan[i,j]=90.0
    #        else:
    #         tan[i,j]=math.degrees(math.atan(ngy[i,j]/ngx[i,j]))
    #         if (tan[i,j]<0):
    #             tan[i,j]= tan[i,j] + 360

    #         magnitude = np.sqrt(pow(Sxgrad, 2.0) + pow(Sygrad, 2.0))
    #         nim[i - 1, j - 1] = mag
    #     pass