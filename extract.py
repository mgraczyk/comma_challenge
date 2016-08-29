import cv2
import numpy as np

img = cv2.imread('./pics/0_0.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray, None)

img=cv2.drawKeypoints(gray,kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

gray_dither = np.uint8(np.minimum(255, np.maximum(0, np.int8(10*np.random.randn(*gray.shape)) + gray)))

kp = sift.detect(gray_dither, None)
img=cv2.drawKeypoints(gray_dither, kp,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints_dither.jpg',img)
