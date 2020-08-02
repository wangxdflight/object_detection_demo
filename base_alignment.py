from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def test1():
    img = cv.imread('ftc_base.jpg', 1)

    h, w, c = img.shape
    print(h, w, c)

    edges = cv.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show();


def make_image():
    #img = cv.imread('ftc_base.jpg', 1)
    img = cv.imread('skystones_2.jpg', 1)
    print(type(img))
    h, w, c = img.shape
    print(h, w, c)
    #img = np.zeros((500, 500), np.uint8)
    black, white = 0, 255
    for i in xrange(6):
        dx = int((i%2)*250 - 30)
        dy = int((i/2.)*150)

        if i == 0:
            for j in xrange(11):
                angle = (j+5)*np.pi/21
                c, s = np.cos(angle), np.sin(angle)
                x1, y1 = np.int32([dx+100+j*10-80*c, dy+100-90*s])
                x2, y2 = np.int32([dx+100+j*10-30*c, dy+100-30*s])
                cv.line(img, (x1, y1), (x2, y2), white)

        cv.ellipse( img, (dx+150, dy+100), (100,70), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+115, dy+70), (30,20), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+185, dy+70), (30,20), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+115, dy+70), (15,15), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+185, dy+70), (15,15), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+115, dy+70), (5,5), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+185, dy+70), (5,5), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+150, dy+100), (10,5), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+150, dy+150), (40,10), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+27, dy+100), (20,35), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+273, dy+100), (20,35), 0, 0, 360, white, -1 )
    return img

def main():
    img = make_image()
    #img = edges
    h, w = img.shape[:2]

    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    cv.imshow("Resized image", resized)
    cv.waitKey(0)
    img=resized;

    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.subplot(131), plt.imshow(imgray)
    plt.subplot(132), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    plt.subplot(133), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    plt.show()

    ret, blackAndWhiteImage = cv.threshold(imgray, 127, 255, 0)
    plt.imshow(blackAndWhiteImage)
    plt.show()

    contours0, hierarchy = cv.findContours( blackAndWhiteImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

    def update(levels):
        vis = np.zeros((height, width, 3), np.uint8)
        levels = levels - 3
        cv.drawContours( vis, contours, (-1, 2)[levels <= 0], (128,255,255),
            3, cv.LINE_AA, hierarchy, abs(levels) )
        cv.imshow('contours', vis)
    update(3)
    cv.createTrackbar( "levels+3", "contours", 3, 7, update )
    cv.imshow('image', img)
    cv.waitKey()
    print('Done')
    cv.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
