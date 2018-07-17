import cv2
import numpy as np

canvas = np.ones((725, 725))*255
points = np.array(
    [
        [[4,4]],
        [[4,256]],
        [[256,256]],
        [[256,4]]
        
    ]
)

cv2.drawContours(canvas, [points], -1, (0,0,0), 2)
cv2.imshow('image', canvas)
cv2.waitKey(0)

# ((193.01063537597656, 346.7585144042969), (293.0352783203125, 600.0025634765625), -0.09744171053171158)
# <class 'tuple'>
# [[647  47]
#  [ 47  45]
#  [ 46 339]
#  [646 340]]