import cv2
import numpy as np
import matplotlib.pyplot as plt

# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
image = cv2.imread('global_assets/Currency/note500.jpg',0)
original = image
canvas = np.ones(image.shape)
# noise reduction
image = cv2.blur(image, (5, 5))
ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

img, contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dummy = np.append(contour[15], contour[16], axis=0)
# dummy = np.append(dummy, contour[17], axis=0)

print(dummy)

cv2.drawContours(canvas, dummy, -1, (0,0,0), 2)

# get bounding rectangle
x, y, dx, dy = cv2.boundingRect(dummy)
cv2.rectangle(canvas, (x,y), (x+dx, y+dy), (0,0,0), 2)

plt.subplot(221)
plt.imshow(original, cmap='gray')
plt.title('Original Image')

plt.subplot(222)
plt.imshow(image, cmap='gray')
plt.title('Reduced Noise')
plt.axis('off')


plt.subplot(223)
plt.imshow(canvas, cmap='gray')
plt.title('Contour')

plt.subplot(224)
plt.axis('off')
plt.title('Aspect Ratio :\n ' + str(dy/dx))
# plt.imshow(thresh)
plt.show()
# cv2.waitKey(0)
