####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 8th October 2021                ##
####################################################
from canny import CannyEdgeDetector

if __name__ == '__main__':
    detector = CannyEdgeDetector('Images/Test-patterns.bmp')
    detector.canny_detector()

    detector = CannyEdgeDetector('Images/House.bmp')
    detector.canny_detector()