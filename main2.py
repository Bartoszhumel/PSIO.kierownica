import cv2
import sys
import numpy as np
import time
from tracker import EuclideanDistTracker
from math import atan2,degrees
import math
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
def AngleBtw2Points(pointA, pointB):
  changeInX = pointB[0] - pointA[0]
  changeInY = pointB[1] - pointA[1]
  return degrees(atan2(changeInY,changeInX))
def gradient(pt1,pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
def getAngle(pointsList,frame):
    pt1,pt2,pt3 = pointsList
    m1 = gradient(pt1,pt2)
    m2 = gradient(pt1,pt3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))
    angD = round(math.degrees(angR))
    cv2.putText(frame,str(angD),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2)
def findClosest(array,point):
    result = (0,0)
    mindist = 1000000
    for b in array:
        dist = ((b[0] - point[0])**2 + (b[1] - point[1])**2)**0.5
        if mindist>dist:
            mindist = dist
            result = b
    print(mindist)
    return result,mindist
device=0
cap = cv2.VideoCapture('HG16MEU.mp4')
pos_frame = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while not cap.isOpened():
    cap = cv2.VideoCapture('HG16MEU.mp4')
    cv2.waitKey(2000)
    print ("Czekam na wideo")

break_flag=False
start_flag=True
previous_frame = None
tracker = cv2.legacy.TrackerCSRT_create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking",img,False)
pt2 = [0, 0]
pt2[0], pt2[1] , stw, sth = bbox
tracker.init(img,bbox)
while True:
    t = time.time()
    for i in range(25):
        flag, frame = cap.read()
        if flag:
            todisplay = frame
            # ramka jest gotowa
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            obraz_sz = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obraz_fil = cv2.GaussianBlur(obraz_sz, (15, 15), 0)
            # obraz_kr = cv2.Canny(obraz_sz, 60, 150)
            circles=cv2.HoughCircles(obraz_fil, cv2.HOUGH_GRADIENT, 1, 20,minRadius=100,maxRadius=700)
            # object_detector = cv2.createBackgroundSubtractorMOG2()
            # x= object_detector.apply(frame)


            if circles is not None:
                maks=0
                x1=0
                y1=0
                r1=0
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                    if(r>maks):
                        maks=r
                        x1=x
                        y1=y
                        r1=r

                kierownicaobrys=cv2.circle(frame, (x1, y1), r1, (0, 255, 0), 4)
                cv2.rectangle(frame, (x1 - 5, y1 - 5), (x1 + 5, y1 + 5), (0, 128, 255), -1)
                # cv2.imshow('image', kierownica)
                # cv2.waitKey(1000)
                # h, w = frame.shape[:2]
                # if x1!=0 and y1!=0 and r1!=0:
                #     circlemask = create_circular_mask(h,w,(x1,y1),r1)
                #     frame[~circlemask] = 0
                # mask = object_detector.apply(frame)
                # _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
                # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #
                # for cnt in contours:
                #     x ,y , w, h =cv2.boundingRect(cnt)
                #     print(w,h)
                #     if((w>15 and h>15) and (w<20 and h<20)):
                #         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                # cv2.imshow("mask", mask)
                # success, bbox = tracker.update(frame)
                # if success:
                #     x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), -1)
                # mdist = findClosest(array, bbox)
                # if mdist[1] < 50:
                #     bbox = mdist[0]
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + 100, bbox[1] + 100), (0, 255, 0), 3)
                ################################-------------------------------------------------------------------------------wykrywanie ruchu
                if (previous_frame is None):
                        # First frame; there is no previous one yet
                    previous_frame = obraz_fil
                    # previous_frame[~circlemask] = 0
                    continue
                # calculate difference and update previous frame
                diff_frame = cv2.absdiff(src1=previous_frame, src2=obraz_fil)
                previous_frame = obraz_fil
                # previous_frame[~circlemask] = 0
                # diff_frame[~circlemask] = 0
                    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
                kernel = np.ones((5, 5))
                diff_frame = cv2.dilate(diff_frame, kernel, 1)

                    # 5. Only take different areas that are different enough (>20 / 255)
                thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

                kierownica= thresh_frame[y1-r1:(y1+2*r1), x1-r1:(x1+2*r1)]


                contours, _ = cv2.findContours(image=kierownica, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), -1)
                    lastbbox= bbox
                else:
                    array = []
                    for cnt in contours:
                        array.append(cv2.boundingRect(cnt))
                    if array != []:
                        mdist = findClosest(array, bbox)
                        bbox = mdist[0]
                    else:
                        bbox=lastbbox
                    if len(bbox) == 2:
                        bbox = (bbox[0], bbox[1], stw, sth)
                    tracker.init(frame, bbox)
                    # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+100, bbox[1]+100), (0, 255, 0), 3)
                    ################################---
                angle = 180 - AngleBtw2Points((bbox[0], bbox[1]), (x1, y1))
                print("angle:", angle)
                getAngle(((x1, y1), pt2,(bbox[0],bbox[1])), todisplay)
            cv2.imshow("Obraz", todisplay)
            if start_flag:
                cv2.setWindowProperty('Obraz',cv2.WND_PROP_TOPMOST,1)
                cv2.moveWindow('Obraz',20,20)
                start_flag=False
        else:
            # ramka nie jest gotowa
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print ("ramka nie jest gotowa")
            cv2.waitKey(1000)
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break_flag=True
            break
    t = time.time()-t
    if break_flag:
        cap.release() 
        break
    print(int(25/t))