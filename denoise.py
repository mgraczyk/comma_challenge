import numpy as np
import cv2
# import image_data

# image_paths = image_data.find_image_paths('./pics')[1]

# # create a list of first 5 frames
# imgs = map(cv2.imread, image_paths)

# # Denoise 3rd frame considering all the 5 frames
# # dst = cv2.fastNlMeansDenoisingColoredMulti(imgs, 2, 2, None, 4, 7, 35)

# # convert all to grayscale
# gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]

# # convert all to float64
# gray = [np.float64(i) for i in gray]

# # Denoise 3rd frame considering all the 5 frames
# dst = cv2.fastNlMeansDenoisingColoredMulti(imgs, 2, 5, None, 4, 7, 35)

# plt.subplot(211), plt.imshow(imgs[2],'gray')
# plt.subplot(212), plt.imshow(dst,'gray')
# plt.show()

def denoise_colored(frame):
  dst = cv2.fastNlMeansDenoisingColored(frame, h=1)
  return dst

def denoise_gray(frame):
  dst = cv2.fastNlMeansDenoising(frame, h=1)
  return dst
