import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('canyon.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

values = []
for i in range(256):
    n = np.where(gray_img == i, 1., 0.).sum()
    values.append(n)

plt.bar(range(256), height=values, width=1.)
plt.xlabel('intensity')
plt.ylabel('pixels')
plt.savefig('chart.png')