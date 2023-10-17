import numpy as np
import sys
import cv2
from imutils.video import VideoStream
import imutils
import time

# Configuration variables and constants
prototxtPath = "deploy.prototxt.txt"
caffemodelPath = "res10_300x300_ssd_iter_140000.caffemodel"
conf = 0.30
thickness = 2
blue = (247, 173, 62)
white = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
meanValues = (104.0, 177.0, 124.0)
