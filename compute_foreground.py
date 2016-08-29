import numpy as np
import cv2

def compute(image_paths):
  # Adapted from
  # http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html

  # Work around: github.com/opencv/opencv/issues/5667
  cv2.ocl.setUseOpenCL(False)

  fgbg = cv2.createBackgroundSubtractorMOG2()
  fgbg.setDetectShadows(False)
  fgbg.setBackgroundRatio(1e-3)
  fgbg.setVarThreshold(50)

  frame = cv2.imread(image_paths[0])
  avg = np.zeros_like(frame, dtype=float)
  mask = fgbg.apply(frame)
  fgmask = mask
  frame_prev = frame

  for path in image_paths[1:]:
    frame = cv2.imread(path)
    fgbg.apply(frame, mask, 1e-3)
    fgmask = np.maximum(mask, fgmask)

    avg += abs(frame - frame_prev)
    frame_prev = frame
  cv2.ocl.setUseOpenCL(True)

  # res = np.round(avg / len(image_paths)).astype(np.uint8)
  # # res = cv2.convertScaleAbs(avg, alpha=1/len(image_paths))
  # cv2.imshow('avg', res)
  # cv2.waitKey(0)

  kernel = np.ones((5,5),np.uint8)
  fgmask = cv2.morphologyEx(fgmask, op=cv2.MORPH_OPEN, kernel=kernel)
  fgmask = cv2.morphologyEx(fgmask, op=cv2.MORPH_CLOSE, kernel=kernel)

  return fgmask
  # return np.round(fgmask / len(image_paths)).astype(np.uint8)
