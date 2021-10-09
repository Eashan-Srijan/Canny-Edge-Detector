import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

GRSC_PATH = 'grsc.PNG'

class CannyEdgeDetector:
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_read()
        self._gaussian_kernel = None
        self._convolution_matrix_gx = None
        self._convolution_matrix_gy = None
        self._convolved_image = None
    
    @property
    def gaussian_kernel(self):
        return self._gaussian_kernel

    @property
    def convolution_matrix_gx(self):
        return self._convolution_matrix_gx
    
    @convolution_matrix_gx.setter
    def convolution_matrix_gx(self, matrix_gx):
        self._convolution_matrix_gx = matrix_gx
    
    @property
    def convolution_matrix_gy(self):
        return self._convolution_matrix_gy
    
    @convolution_matrix_gy.setter
    def convolution_matrix_gy(self, matrix_gy):
        self._convolution_matrix_gy = matrix_gy
    
    @property
    def image_matrix(self):
        return self._image_matrix
    
    @property
    def convolved_image(self):
        return self._convolved_image
        
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

        self.convolution(self._image_matrix, )
    
    def convolution(self, self._im):

        nrows, ncols = self._image_matrix.shape
        new_image = np.zeros((nrows - 1, ncols - 1))

        # TODO: Write convolution code

        self._convolved_image = new_image
        
        
        
        
