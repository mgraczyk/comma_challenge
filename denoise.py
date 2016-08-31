import numpy as np
import cv2

def denoise_colored(frame):
  dst = cv2.fastNlMeansDenoisingColored(frame, h=0.5)
  return dst

def denoise_gray(frame):
  dst = cv2.fastNlMeansDenoising(frame, h=0.8)
  return dst
