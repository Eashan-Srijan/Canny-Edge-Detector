####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 29th October 2021               ##
####################################################

import numpy as np

class SeConvolve:

    def __init__(self, image_matrix, kernel):
        self.image_matrix = image_matrix
        self.kernel = kernel
        self._output = None
        self._output_norm = None
        self.convolution()

    #######################
    ## Getter and Setter ##
    #######################

    @property
    def output(self):
        return self._output

    @property
    def output_norm(self):
        return self._output_norm

    #######################
    #######################

    def convolution(self):
    # function to perform gaussian smoothing
    # TODO: 
    # 1. remove '\'
    # 2. replace mask with self.kernel
    # 3. replace gray_img/ grey to self.image_matrix
      
      height, width = self.image_matrix.shape
      
      for i in range(3,height-3):
        for j in range(3,width-3):
          
          self._output[i,j] = (mask[0, 0] * gray_img[i - 3, j - 3]) + \
                              (mask[0, 1] * grey[i - 3, j - 2]) + \
                              (mask[0, 2] * grey[i - 3, j - 1]) + \
                              (mask[0, 3] * grey[i - 3, j]) + \
                              (mask[0, 4] * grey[i - 3, j + 1]) + \
                              (mask[0, 5] * grey[i - 3, j + 2]) + \
                              (mask[0, 6] * grey[i - 3, j + 3]) + \
                              (mask[1, 0] * grey[i - 2, j - 3]) + \
                              (mask[1, 1] * grey[i - 2, j - 2]) + \
                              (mask[1, 2] * grey[i - 2, j - 1]) + \
                              (mask[1, 3] * grey[i - 2, j]) + \
                              (mask[1, 4] * grey[i - 2, j + 1]) + \
                              (mask[1, 5] * grey[i - 2, j + 2]) + \
                              (mask[1, 6] * grey[i - 2, j + 3]) + \
                              (mask[2, 0] * grey[i - 1, j - 3]) + \
                              (mask[2, 1] * grey[i - 1, j - 2]) + \
                              (mask[2, 2] * grey[i - 1, j - 1]) + \
                              (mask[2, 3] * grey[i - 1, j]) + \
                              (mask[2, 4] * grey[i - 1, j + 1]) + \
                              (mask[2, 5] * grey[i - 1, j + 2]) + \
                              (mask[2, 6] * grey[i - 1, j + 3]) + \
                              (mask[3, 0] * grey[i, j - 3]) + \
                              (mask[3, 1] * grey[i, j - 2]) + \
                              (mask[3, 2] * grey[i, j - 1]) + \
                              (mask[3, 3] * grey[i, j]) + \
                              (mask[3, 4] * grey[i, j + 1]) + \
                              (mask[3, 5] * grey[i, j + 2]) + \
                              (mask[3, 6] * grey[i, j + 3]) + \
                              (mask[4, 0] * grey[i + 1, j - 3]) + \
                              (mask[4, 1] * grey[i + 1, j - 2]) + \
                              (mask[4, 2] * grey[i + 1, j - 1]) + \
                              (mask[4, 3] * grey[i + 1, j]) + \
                              (mask[4, 4] * grey[i + 1, j + 1]) + \
                              (mask[4, 5] * grey[i + 1, j + 2]) + \
                              (mask[4, 6] * grey[i + 1, j + 3]) + \
                              (mask[5, 0] * grey[i + 2, j - 3]) + \
                              (mask[5, 1] * grey[i + 2, j - 2]) + \
                              (mask[5, 2] * grey[i + 2, j - 1]) + \
                              (mask[5, 3] * grey[i + 2, j]) + \
                              (mask[5, 4] * grey[i + 2, j + 1]) + \
                              (mask[5, 5] * grey[i + 2, j + 2]) + \
                              (mask[5, 6] * grey[i + 2, j + 3]) + \
                              (mask[6, 0] * grey[i + 3, j - 3]) + \
                              (mask[6, 1] * grey[i + 3, j - 2]) + \
                              (mask[6, 2] * grey[i + 3, j - 1]) + \
                              (mask[6, 3] * grey[i + 3, j]) + \
                              (mask[6, 4] * grey[i + 3, j + 1]) + \
                              (mask[6, 5] * grey[i + 3, j + 2]) + \
                              (mask[6, 6] * grey[i + 3, j + 3])
      
      self.normalize()

    def normalize(self):
        self._output_norm = self.output / np.sum(self.kernel)
