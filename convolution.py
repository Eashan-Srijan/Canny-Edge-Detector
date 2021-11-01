####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 29th October 2021               ##
####################################################

import numpy as np

class SeConvolve:

    def __init__(self, image_matrix, kernel, mode='smoothing'):
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

          self._output[i,j] = (self.kernel[0, 0] * self.image_matrix[i - 3, j - 3]) +
                              (self.kernel[0, 1] * self.image_matrix[i - 3, j - 2]) +
                              (self.kernel[0, 2] * self.image_matrix[i - 3, j - 1]) +
                              (self.kernel[0, 3] * self.image_matrix[i - 3, j]) +
                              (self.kernel[0, 4] * self.image_matrix[i - 3, j + 1]) +
                              (self.kernel[0, 5] * self.image_matrix[i - 3, j + 2]) +
                              (self.kernel[0, 6] * self.image_matrix[i - 3, j + 3]) +
                              (self.kernel[1, 0] * self.image_matrix[i - 2, j - 3]) +
                              (self.kernel[1, 1] * self.image_matrix[i - 2, j - 2]) +
                              (self.kernel[1, 2] * self.image_matrix[i - 2, j - 1]) +
                              (self.kernel[1, 3] * self.image_matrix[i - 2, j]) +
                              (self.kernel[1, 4] * self.image_matrix[i - 2, j + 1]) +
                              (self.kernel[1, 5] * self.image_matrix[i - 2, j + 2]) +
                              (self.kernel[1, 6] * self.image_matrix[i - 2, j + 3]) +
                              (self.kernel[2, 0] * self.image_matrix[i - 1, j - 3]) +
                              (self.kernel[2, 1] * self.image_matrix[i - 1, j - 2]) +
                              (self.kernel[2, 2] * self.image_matrix[i - 1, j - 1]) +
                              (self.kernel[2, 3] * self.image_matrix[i - 1, j]) +
                              (self.kernel[2, 4] * self.image_matrix[i - 1, j + 1]) +
                              (self.kernel[2, 5] * self.image_matrix[i - 1, j + 2]) +
                              (self.kernel[2, 6] * self.image_matrix[i - 1, j + 3]) +
                              (self.kernel[3, 0] * self.image_matrix[i, j - 3]) +
                              (self.kernel[3, 1] * self.image_matrix[i, j - 2]) +
                              (self.kernel[3, 2] * self.image_matrix[i, j - 1]) +
                              (self.kernel[3, 3] * self.image_matrix[i, j]) +
                              (self.kernel[3, 4] * self.image_matrix[i, j + 1]) +
                              (self.kernel[3, 5] * self.image_matrix[i, j + 2]) +
                              (self.kernel[3, 6] * self.image_matrix[i, j + 3]) +
                              (self.kernel[4, 0] * self.image_matrix[i + 1, j - 3]) +
                              (self.kernel[4, 1] * self.image_matrix[i + 1, j - 2]) +
                              (self.kernel[4, 2] * self.image_matrix[i + 1, j - 1]) +
                              (self.kernel[4, 3] * self.image_matrix[i + 1, j]) +
                              (self.kernel[4, 4] * self.image_matrix[i + 1, j + 1]) +
                              (self.kernel[4, 5] * self.image_matrix[i + 1, j + 2]) +
                              (self.kernel[4, 6] * self.image_matrix[i + 1, j + 3]) +
                              (self.kernel[5, 0] * self.image_matrix[i + 2, j - 3]) +
                              (self.kernel[5, 1] * self.image_matrix[i + 2, j - 2]) +
                              (self.kernel[5, 2] * self.image_matrix[i + 2, j - 1]) +
                              (self.kernel[5, 3] * self.image_matrix[i + 2, j]) +
                              (self.kernel[5, 4] * self.image_matrix[i + 2, j + 1]) +
                              (self.kernel[5, 5] * self.image_matrix[i + 2, j + 2]) +
                              (self.kernel[5, 6] * self.image_matrix[i + 2, j + 3]) +
                              (self.kernel[6, 0] * self.image_matrix[i + 3, j - 3]) +
                              (self.kernel[6, 1] * self.image_matrix[i + 3, j - 2]) +
                              (self.kernel[6, 2] * self.image_matrix[i + 3, j - 1]) +
                              (self.kernel[6, 3] * self.image_matrix[i + 3, j]) +
                              (self.kernel[6, 4] * self.image_matrix[i + 3, j + 1]) +
                              (self.kernel[6, 5] * self.image_matrix[i + 3, j + 2]) +
                              (self.kernel[6, 6] * self.image_matrix[i + 3, j + 3])

      if self.mode == 'smoothing':
        self.normalize()
        return self._output_norm

      return self._output

    def normalize(self):
        self._output_norm = self.output / np.sum(self.kernel)
