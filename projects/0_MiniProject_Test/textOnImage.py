import numpy as np
import cv2

print_list = []

def print_f(sen):
    global print_list
    print_list = print_list + [sen]

def print_s(_header, DefaultCanvas=None):
    if DefaultCanvas !=  None :
        print_list = DefaultCanvas
    else:
        global print_list
    x = len(print_list)
    canvas = np.ones((20*x+10,512))*255
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(0,x):
        cv2.putText(canvas,print_list[i] ,(10,(i+1)*20), font, 1,(0,255,0),1,cv2.LINE_AA)
    cv2.imshow(_header,canvas)


print_f('Hello World')
print_f('Shit down')
print_f('I can\'t believe it works just fine')
print_s('Some Heading')
cv2.waitKey(0)