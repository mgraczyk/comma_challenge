import numpy as np
import cv2

def compute(images):
  # Adapted from
  # http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html

  # Work around: github.com/opencv/opencv/issues/5667
  cv2.ocl.setUseOpenCL(False)

  fgbg = cv2.createBackgroundSubtractorMOG2()
  fgbg.setDetectShadows(False)
  fgbg.setBackgroundRatio(1e-3)
  fgbg.setVarThreshold(50)
  # fgbg.setComplexityReductionThreshold(0)

  frame = images[0].frame
  avg = np.zeros_like(frame, dtype=float)
  mask = fgbg.apply(frame)
  fgmask = mask
  frame_prev = frame

  for image in images[1:]:
    frame = image.frame
    fgbg.apply(frame, mask, 1e-3)
    fgmask = np.maximum(mask, fgmask)
    # fgmask += mask;

    avg += abs(frame - frame_prev)
    frame_prev = frame
  cv2.ocl.setUseOpenCL(True)

  # res = np.round(avg.min(2) / (len(image_paths))).astype(np.uint8)
  # fgmask = np.round(fgmask / len(image_paths)).astype(np.uint8)

  # TODO(mgraczyk): Scale by image width.
  kernel = np.ones((5, 5),np.uint8)
  fgmask = cv2.morphologyEx(fgmask, op=cv2.MORPH_DILATE, kernel=kernel, iterations=10)
  fgmask = cv2.morphologyEx(fgmask, op=cv2.MORPH_ERODE, kernel=kernel, iterations=30)

  return fgmask
  # return np.round(fgmask / len(image_paths)).astype(np.uint8)
