####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 8th October 2021                ##
####################################################
class SeConvolve:

    def __init__(self, image_matrix, kernel):
        self.image_matrix = image_matrix
        self.kernel = kernel
        self._output = None
        self._output_norm = None
    
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
        # TODO: write here put 0 instead of dec size
        pass

    def normalize(self):
        # TODO: normalize _output
        pass