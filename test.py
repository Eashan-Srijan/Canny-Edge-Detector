####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 8th October 2021                ##
####################################################
from canny import CannyEdgeDetector

if __name__ == '__main__':
    detector = CannyEdgeDetector('C:/Users/Eashan/Desktop/canny-edge-detector/Images/Test-patterns.bmp')
    detector.canny_detector()

    detector = CannyEdgeDetector('C:/Users/Eashan/Desktop/canny-edge-detector/Images/House.bmp')
    detector.canny_detector()