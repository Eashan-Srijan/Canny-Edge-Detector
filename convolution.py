####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 29th October 2021               ##
####################################################

import numpy as np

class SeConvolve:

    def __init__(self, image_matrix, kernel, mode='smoothing'):
        # input image matrix
        self.image_matrix = image_matrix
        # inpur kernel
        self.kernel = kernel
        # output after convolution with the kerne
        self._output = None
        # normalized output
        self._output_norm = None
        # mode smoothing or gradient
        self.mode = mode
        # calling the convolution function
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

      height, width = self.image_matrix.shape
      # if mode == smoothing
      if self.mode == 'smoothing':
        self._output = np.zeros((height - 6, width - 6))
        # code to perform gaussian smoothing
        # looping over the desired matrix
        for i in range(3,height-3):
          for j in range(3,width-3):
            # martix multiplication for convolution
            # 7 x 7 gaussian smoothing leads to loss of 3 rows and 3 columns and thats why we start from i-3 and j-3
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
      # if mode == gradient
      elif self.mode == 'gradient':
        
        # code to find the gradients 
        self._output = np.zeros((height - 8, width - 8))
        
        # looping over the desired matrix
        for i in range(4,height - 4):
          for j in range(4,width - 4):
            # martix multiplication for gradient computation
            # prewit convolution after gaussian smoothing leads to loss of 4 rows and 4 columns thats why we start from i-4 and j-4
            self._output[i - 4,j - 4] = (self.kernel[0, 0] * self.image_matrix[i - 1, j - 1]) + \
                      (self.kernel[0, 1] * self.image_matrix[i - 1, j]) + \
                      (self.kernel[0, 2] * self.image_matrix[i - 1, j + 1]) + \
                      (self.kernel[1, 0] * self.image_matrix[i, j - 1]) + \
                      (self.kernel[1, 1] * self.image_matrix[i, j]) + \
                      (self.kernel[1, 2] * self.image_matrix[i, j + 1]) + \
                      (self.kernel[2, 0] * self.image_matrix[i + 1, j - 1]) + \
                      (self.kernel[2, 1] * self.image_matrix[i + 1, j]) + \
                      (self.kernel[2, 2] * self.image_matrix[i + 1, j + 1])

      # we call the normalize function to normalize the output
      self.normalize()

      return self._output_norm

    # normalize
    def normalize(self):
      if self.mode == 'smoothing':
        # normalize using sum of all values
        self._output_norm = self.output / np.sum(self.output)
        self._output_norm = np.pad(self._output_norm, 3, mode='constant')
      elif self.mode == 'gradient':
        # normalize using sum of absolute values
        temp_output = np.absolute(self.output)
        self._output_norm = self.output / np.sum(temp_output)
        self._output_norm = np.pad(self._output_norm, 4, mode='constant')
