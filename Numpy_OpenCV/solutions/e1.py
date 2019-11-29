import cv2
import numpy as np

img = cv2.cvtColor(cv2.imread('canyon.png'), cv2.COLOR_BGR2RGB)
red = img[:,:,0:1]
green = img[:,:,1:2]
blue = img[:,:,2:3]
zeros = np.zeros_like(red, np.float32)
red = np.concatenate([red, zeros, zeros], axis=-1)
green = np.concatenate([zeros, green, zeros], axis=-1)
blue = np.concatenate([zeros, zeros, blue], axis=-1)
red = cv2.cvtColor(red, cv2.COLOR_RGB2BGR)
green = cv2.cvtColor(green, cv2.COLOR_RGB2BGR)
blue = cv2.cvtColor(blue, cv2.COLOR_RGB2BGR)
cv2.imwrite('red.png', red)
cv2.imwrite('green.png', green)
cv2.imwrite('blue.png', blue)
assert np.array_equal(blue + red + green, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
