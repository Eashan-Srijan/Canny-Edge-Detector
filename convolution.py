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
        self.mode = mode
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

      height, width = self.image_matrix.shape

      if self.mode == 'smoothing':
        self._output = np.zeros((height - 6, width - 6))
        for i in range(3,height-3):
          for j in range(3,width-3):

    # martix multiplication for convolution

            self._output[i - 3,j - 3] = (self.kernel[0, 0] * self.image_matrix[i - 3, j - 3]) + (self.kernel[0, 1] * self.image_matrix[i - 3, j - 2]) + (self.kernel[0, 2] * self.image_matrix[i - 3, j - 1]) + \
            (self.kernel[0, 3] * self.image_matrix[i - 3, j]) + (self.kernel[0, 4] * self.image_matrix[i - 3, j + 1]) + (self.kernel[0, 5] * self.image_matrix[i - 3, j + 2]) + \
            (self.kernel[0, 6] * self.image_matrix[i - 3, j + 3]) + (self.kernel[1, 0] * self.image_matrix[i - 2, j - 3]) + (self.kernel[1, 1] * self.image_matrix[i - 2, j - 2]) + \
            (self.kernel[1, 2] * self.image_matrix[i - 2, j - 1]) + (self.kernel[1, 3] * self.image_matrix[i - 2, j]) + (self.kernel[1, 4] * self.image_matrix[i - 2, j + 1]) + \
            (self.kernel[1, 5] * self.image_matrix[i - 2, j + 2]) + (self.kernel[1, 6] * self.image_matrix[i - 2, j + 3]) + (self.kernel[2, 0] * self.image_matrix[i - 1, j - 3]) + \
            (self.kernel[2, 1] * self.image_matrix[i - 1, j - 2]) + (self.kernel[2, 2] * self.image_matrix[i - 1, j - 1]) + (self.kernel[2, 3] * self.image_matrix[i - 1, j]) + \
            (self.kernel[2, 4] * self.image_matrix[i - 1, j + 1]) + (self.kernel[2, 5] * self.image_matrix[i - 1, j + 2]) + (self.kernel[2, 6] * self.image_matrix[i - 1, j + 3]) + \
            (self.kernel[3, 0] * self.image_matrix[i, j - 3]) + (self.kernel[3, 1] * self.image_matrix[i, j - 2]) + (self.kernel[3, 2] * self.image_matrix[i, j - 1]) + \
            (self.kernel[3, 3] * self.image_matrix[i, j]) + (self.kernel[3, 4] * self.image_matrix[i, j + 1]) + (self.kernel[3, 5] * self.image_matrix[i, j + 2]) + \
            (self.kernel[3, 6] * self.image_matrix[i, j + 3]) + (self.kernel[4, 0] * self.image_matrix[i + 1, j - 3]) + (self.kernel[4, 1] * self.image_matrix[i + 1, j - 2]) + \
            (self.kernel[4, 2] * self.image_matrix[i + 1, j - 1]) + (self.kernel[4, 3] * self.image_matrix[i + 1, j]) + (self.kernel[4, 4] * self.image_matrix[i + 1, j + 1]) + \
            (self.kernel[4, 5] * self.image_matrix[i + 1, j + 2]) + (self.kernel[4, 6] * self.image_matrix[i + 1, j + 3]) + (self.kernel[5, 0] * self.image_matrix[i + 2, j - 3]) + \
            (self.kernel[5, 1] * self.image_matrix[i + 2, j - 2]) + (self.kernel[5, 2] * self.image_matrix[i + 2, j - 1]) + (self.kernel[5, 3] * self.image_matrix[i + 2, j]) + \
            (self.kernel[5, 4] * self.image_matrix[i + 2, j + 1]) + (self.kernel[5, 5] * self.image_matrix[i + 2, j + 2]) + (self.kernel[5, 6] * self.image_matrix[i + 2, j + 3]) + \
            (self.kernel[6, 0] * self.image_matrix[i + 3, j - 3]) + (self.kernel[6, 1] * self.image_matrix[i + 3, j - 2]) + (self.kernel[6, 2] * self.image_matrix[i + 3, j - 1]) + \
            (self.kernel[6, 3] * self.image_matrix[i + 3, j]) + (self.kernel[6, 4] * self.image_matrix[i + 3, j + 1]) + (self.kernel[6, 5] * self.image_matrix[i + 3, j + 2]) + \
            (self.kernel[6, 6] * self.image_matrix[i + 3, j + 3])

    # function to find the gradients 
      elif self.mode == 'gradient':
        self._output = np.zeros((height - 8, width - 8))

        for i in range(4,height - 4):
          for j in range(4,width - 4):
            self._output[i - 4,j - 4] = (self.kernel[0, 0] * self.image_matrix[i - 1, j - 1]) + \
                      (self.kernel[0, 1] * self.image_matrix[i - 1, j]) + \
                      (self.kernel[0, 2] * self.image_matrix[i - 1, j + 1]) + \
                      (self.kernel[1, 0] * self.image_matrix[i, j - 1]) + \
                      (self.kernel[1, 1] * self.image_matrix[i, j]) + \
                      (self.kernel[1, 2] * self.image_matrix[i, j + 1]) + \
                      (self.kernel[2, 0] * self.image_matrix[i + 1, j - 1]) + \
                      (self.kernel[2, 1] * self.image_matrix[i + 1, j]) + \
                      (self.kernel[2, 2] * self.image_matrix[i + 1, j + 1])

      self.normalize()

      return self._output_norm

    # normalize
    def normalize(self):
    # normalize using sum of all values
      if self.mode == 'smoothing':
        self._output_norm = self.output / np.sum(self.output)
        self._output_norm = np.pad(self._output_norm, 3, mode='constant')
    # normalize using absolute values
      elif self.mode == 'gradient':
        temp_output = np.absolute(self.output)
        self._output_norm = self.output / np.sum(temp_output)
        self._output_norm = np.pad(self._output_norm, 4, mode='constant')
