import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# get file list from folder
def getFiles(_root):
    return next(os.walk(_root))[2]

# take image input
def takeImageInput(_root, _path):
    _path = _root + _path
    colored = cv2.imread(_path)
    _grayscale = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (colored, _grayscale)


# image preprocessing and noise reduction
def initialTransformations(_image):
    global thresh, blur, canny, filters
    _image = cv2.bilateralFilter(_image, 9, 50 ,50 )
    filters = _image
    _image = cv2.GaussianBlur(_image,(11,11) , 4)
    blur = _image
    _thresh = cv2.adaptiveThreshold(_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2 )
    _thresh  = cv2.medianBlur(_thresh, 11)
    thresh = _thresh
    _image = _thresh
    # _image = cv2.Canny(_image,5,10)
    canny = _image
    return _image

# returns set of points to be enclosed by bounding rectangle
def maxContour(_contour):
    global imageArea
    pointList = np.array(_contour[0][:,0])
    _maxContourArea = 0
    for cnt in _contour:
        area = cv2.contourArea(cnt)
        if area/imageArea < 0.9:
            pointList = np.append(pointList, cnt[:,0], 0)
    return pointList

def dist(_x,_y):
    return ((_x[0] - _y[0])**2 + (_x[1]-_y[1])**2)**0.5

def getFit(_image, box, clip=None):
    br = box[0]
    bl = box[1]
    tl = box[2]
    tr = box[3]

    _width = dist(br, bl)
    _height = dist(bl, tl)
    
    _pts1 = np.float32([br, bl, tl, tr])
    _pts2 = np.float32([[_width,_height],[0,_height], [0,0],[_width,0]])

    transformationMatrix = cv2.getPerspectiveTransform(_pts1, _pts2)
    transImage = cv2.warpPerspective(_image, transformationMatrix, (int(_width), int(_height)))
    if(clip!=None):
        transImage = transImage[clip[1]:int(_height)-clip[3], clip[0] : int(_width)-clip[2]]
    if(transImage.shape[0] > transImage.shape[1]):
        M = cv2.getRotationMatrix2D((transImage.shape[0]/2,transImage.shape[0]/2),-90,1)
        transImage = cv2.warpAffine(transImage,M,(transImage.shape[0],transImage.shape[1]))
    return transImage

# match two images and generate descriptors
def imageMatcher(_standardImage, _sampleImage):
    _orb = cv2.ORB_create()
    try:
        _keyPoint_1, _descriptors_1 = _orb.detectAndCompute(_standardImage, None)
        _keyPoint_2, _descriptors_2 = _orb.detectAndCompute(_sampleImage, None)
        # Create a brute force matcher
        bruteForceMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Matches descriptors of images
        _matches = bruteForceMatcher.match(_descriptors_1, _descriptors_2)
        # sort matches wrt distance: less distance => more accurate match
        _matches = sorted(_matches, key=lambda x: x.distance)
        return (_matches, (_keyPoint_1, _keyPoint_2))
    except Exception as e:
        print(e.args)
        return [None,(None, None)]

# detemines accuracy points on the matches
def determineAccuracy(_matches, _limit = None):
    if _limit == None or _limit ==0 :
        _limit = 2
    _sum = 0
    _limit = int(len(_matches)/_limit)
    for _ in _matches[:_limit]:
        _sum += _.distance
    _avg = float(0)
    try:
        _avg = _sum/_limit
    except Exception as e:
        print(e.args)
        return 0
    return _avg
# draw matches to a new image
def drawMatcher(_standardImage, _sampleImage, _keyPoints, _matches):
    result  = None
    result = cv2.drawMatches(_standardImage,_keyPoints[0],_sampleImage,
    _keyPoints[1],_matches,result, flags=2)
    return result

def getFeatureImage(_fullImage, _feature):
    _height = _fullImage.shape[0]
    _width = _fullImage.shape[1]
    _x= int(_height * _feature[0])
    _y= int(_width *_feature[1])
    _dx = int(_height*_feature[2])
    _dy = int(_width*_feature[3])
    
    return _fullImage[_x:_x + _dx , _y : _y + _dy]


# ----------STARTS---HERE------------------------#

# root = "global_assets/Currency/"
# path = "note10/2.jpg"

# -----ROOT - folder containing all the test note images-------------
# -----Pass root to getFiles() to get list of files in that folder---
#----------DATA SET STARTS------------

notes = ['10','20','50','100', '200','500']
artio = [2.075723367, 2.212180173, 1.9959253487, 2.10026738, 2.1148898655, 2.16519103]
tolerance = [0.004586393, 0.006160723, 0.008263391, 0.024120747, 0.019034105, 0.020002683]
feature_set = [
    [0.2331274,0.0096001,0.2236,0.0644],
    [0.66313,0.7095218,0.1455,0.17838],
    [0.401345291,0.167958656,0.293721973,0.069767442],
    [0.243820225,0.947776629,0.235955056,0.051706308],
    [0.570786517,0.88262668,0.265168539,0.084281282],
    [0.08688764,0.097207859,0.120224719,0.257497415],
    [0.18362069,0.543632075,0.735344828,0.037735849],
    [0.100482759,0.577044025,0.099827586,0.32468535],
    [0.495689655,0.29009434,0.132758621,0.027122642]
]

feature_set_old = [
    [0.324112769, 0.385028302, 0.2039801, 0.2044009434],
    [0.247159451, 0.843353597, 0.161372756, 0.111560284],
    [0.055966209, 0.072921986, 0.173522703, 0.133981763],
    [0.067581837, 0.322695035, 0.1731151, 0.339148936],
    [0.610538543, 0.024822695, 0.299271383, 0.086119554],
    [0.648363252, 0.873353597, 0.173178458, 0.088652482]
]



#-----------DATA SET ENDS-------------

root ="../Image/new_m/"
fileList = getFiles(root)
fileList = sorted(fileList)
for i in range(0,21):
    path =  fileList[i]

    # print(path)
    coloredImage, grayscale = takeImageInput(root, path)

    height = coloredImage.shape[0]
    width = coloredImage.shape[1]
    imageArea = float(height * width)

    # prepares image for contour detectection
    image = initialTransformations(grayscale)

    img, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # canvas = np.ones((height, width, 3))

    pointList = maxContour(contours)
    rect = cv2.minAreaRect(pointList)
    rect = cv2.boxPoints(rect) # 
    rect = np.int0(rect)
    # print(rect)  # all the points of the bounding rectangle
    # cv2.drawContours(coloredImage, [rect], -1, (0,255,0),3)

    ss = getFit(coloredImage, rect)
    aratio = ss.shape[1]/ss.shape[0]
    title = ''
    for x,t in enumerate(notes):
        if(aratio <= artio[x] + tolerance[x] and aratio >= artio[x] - tolerance[x]):  
            title = t
    # plt.subplot(3,3,i+1)
    height = ss.shape[0]
    width = ss.shape[1]
    # for j in feature_set:

    # x= int(height * 0.648363252 )
    # y= int(width * 0.873353597)
    # dx = int(height* 0.173178458)
    # dy = int(width* 0.088652482)

    # cv2.rectangle(ss, (y,x), (y+dy,x+dx), (0,255,0),2)
    cv2.namedWindow(str(i), cv2.WINDOW_KEEPRATIO)
    cv2.imshow(str(i), ss)
    # plt.title("Note: "+title)
    # if i == 1:
        # print(rect)
        # print(path)
        # break
    # plt.subplot(212)
    # plt.imshow(coloredImage, cmap='Greys_r')
# plt.show()
cv2.waitKey(0)
    # cv2.imshow("Image",coloredImage)
    # cv2.waitKey(0)