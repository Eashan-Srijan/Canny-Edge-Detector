####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 29th October 2021               ##
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
    # function to perform gaussian smoothing
      mask=np.array([1.0,1.0,2.0,2.0,2.0,1.0,1.0]
                    ,[1.0,2.0,2.0,4.0,2.0,2.0,1.0],
                    [2.0,2.0,4.0,8.0,4.0,2.0,2.0],
                    [2.0,4.0,8.0,16.0,8.0,4.0,2.0],
                    [2.0,2.0,4.0,8.0,4.0,2.0,2.0],
                    [1.0,2.0,2.0,4.0,2.0,2.0,1.0],
                    [1.0,1.0,2.0,2.0,2.0,1.0,1.0])
      grey=np.array(Image.open(a)).astype(np.uint8)
      height, width = grey.shape
      for i in range(3,height-3):
        for j in range(3,width-3):
          gimage[i,j]=(mask[0, 0] * gray_img[i - 3, j - 3]) + \
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
          gimage[i,j] = gimage[i,j]/140
      new_im = Image.fromarray(gimage)
      new_im.save("gaussian.bmp",)
        pass

    def prewittop(b):
      grey = np.array(Image.open(b)).astype(np.uint8)
      print("The values of the read image are ")
      print(grey)
      height,width=grey.shape
      Px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
      Py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
      for i in range(5,height-5):
        for j in range(5,width-5):
           Sx =     (Sx[0, 0] * grey[i - 1, j - 1]) + \
                    (Sx[0, 1] * grey[i - 1, j]) + \
                    (Sx[0, 2] * grey[i - 1, j + 1]) + \
                    (Sx[1, 0] * grey[i, j - 1]) + \
                    (Sx[1, 1] * grey[i, j]) + \
                    (Sx[1, 2] * grey[i, j + 1]) + \
                    (Sx[2, 0] * grey[i + 1, j - 1]) + \
                    (Sx[2, 1] * grey[i + 1, j]) + \
                    (Sx[2, 2] * grey[i + 1, j + 1])

           Sy =     (Sy[0, 0] * grey[i - 1, j - 1]) + \
                    (Sy[0, 1] * grey[i - 1, j]) + \
                    (Sy[0, 2] * grey[i - 1, j + 1]) + \
                    (Sy[1, 0] * grey[i, j - 1]) + \
                    (Sy[1, 1] * grey[i, j]) + \
                    (Sy[1, 2] * grey[i, j + 1]) + \
                    (Sy[2, 0] * grey[i + 1, j - 1]) + \
                    (Sy[2, 1] * grey[i + 1, j]) + \
                    (Sy[2, 2] * grey[i + 1, j + 1])
           ngx=Sxgrad
           ngy=Sygrad
           if(ngx[i,j]==0):
             tan[i,j]=90.0
           else:
            tan[i,j]=math.degrees(math.atan(ngy[i,j]/ngx[i,j]))
            if (tan[i,j]<0):
                tan[i,j]= tan[i,j] + 360

            magnitude = np.sqrt(pow(Sxgrad, 2.0) + pow(Sygrad, 2.0))
            nim[i - 1, j - 1] = mag
        pass

    def normalize(self):
        # TODO: normalize _output
        pass
