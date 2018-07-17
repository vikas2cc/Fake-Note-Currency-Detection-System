import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

############

def order_points(pts):
        rect = np.zeros((4, 2), np.float32)
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
##############
    
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH))

#######################

def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
 
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
 
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        if (maxHeight-maxWidth>0):
                image=warped
                warped = imutils.rotate_bound(image, -90)
        return warped
                
'''#################################'''#####################################

#image = cv2.imread('10_1.jpeg')
image = cv2.imread('50_1.jpeg')
#image = cv2.imread('100_1.jpeg')
#image = cv2.imread('10_1.jpeg')
#image = cv2.imread('500_1.jpeg')

'''#################################'''#####################################
print(type(image))

ratio = image.shape[0] / 500.0
orig = image
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray,9,75,75)
cv2.imshow('image blur',gray) 
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 5, 200)


print ("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

                  
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


for c in cnts:
        peri = cv2.3(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
                screenCnt = approx
                break
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

print("perspective transform applied")
cv2.imshow("Original", imutils.resize(orig, width =500))
warped=imutils.resize(warped,width = 1000)
cv2.imshow("croped image",warped )

cv2.waitKey(0)
cv2.destroyAllWindows()


# orb = cv2.ORB_create()
# (kp1,des1)=orb.detectAndCompute(warped, None)
# training_set = ['20.jpg','50.jpg','100.jpg','500.jpg']

# max_val = 8
# max_pt = -1
# max_kp = 0
# for i in range(0, len(training_set)):
# 	train_img = cv2.imread(training_set[i])
# 	train_img = imutils.resize(train_img,width = 1000)
# 	cv2.imshow("traing set",train_img)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
	
# 	(kp2, des2) = orb.detectAndCompute(train_img, None)
# 	# brute force matcher
# 	bf = cv2.BFMatcher()
# 	all_matches = bf.knnMatch(des1, des2, k=2)

# 	good = []
# 	for (m, n) in all_matches:
# 		if m.distance < 0.9 * n.distance:
# 			good.append([m])
# 	if len(good) > max_val:
# 		max_val = len(good)
# 		max_pt = i
# 		max_kp = kp2
# 	print(i, ' ', training_set[i], ' ', len(good))
# ################################################################################
	
# if max_val >= 60:
# 	print('scanned image is -->>',training_set[max_pt])
# 	print('\ntotal good matches ', max_val)
# 	train_img = cv2.imread(training_set[max_pt])
# 	train_img = imutils.resize(train_img,width = 1000)
# 	img3 = cv2.drawMatchesKnn(warped, kp1, train_img, max_kp, good, 4)
# 	note = str(training_set[max_pt])[0:-4]
# 	print('\nnote is of : "', note,' Rs "')
# 	(plt.imshow(img3), plt.show())
# else:
# 	print('try again, it is suspicious note')
                    

                    
