import cv2
import numpy as np

img = cv2.imread('canyon.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.where(gray_img < 80, 0., 1.)
cv2.imwrite('mask.png', mask*255.)
np.save('mask.npy', mask)
masked_img = img * np.expand_dims(mask,-1)
cv2.imwrite('masked_img.png', masked_img)
