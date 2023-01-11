import cv2
import sys
import numpy as np
import time
import math

import skimage


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
def findclosest(contours,point):
    xn, yn, _, _ = cv2.boundingRect(point)
    closest=contours[0]
    mindist=100000
    for cnt in contours:
        # Calculate area and remove small elements                    a
        x, y, _, _ = cv2.boundingRect(cnt)
        dist=math.dist((x,y),(xn,yn))
        if(dist<mindist):
            mindist=dist
            closest=cnt
    return closest
device=0
cap = cv2.VideoCapture('short.mp4')
pos_frame = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while not cap.isOpened():
    cap = cv2.VideoCapture('short.mp4')
    cv2.waitKey(2000)
    print ("Czekam na wideo")
break_flag=False
start_flag=True
previous_frame = None
ocx = 0
ocy = 0
while True:
    t = time.time()
    for i in range(25):
        flag, frame = cap.read()
        if flag:
            # ramka jest gotowa
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            obraz_sz = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obraz_fil = cv2.GaussianBlur(obraz_sz, (15, 15), 0)
            black= np.zeros(obraz_fil.shape)
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

                ################################-------------------------------------------------------------------------------wykrywanie ruchu
                if (previous_frame is None):
                        # First frame; there is no previous one yet
                    previous_frame = obraz_fil
                    continue
                # calculate difference and update previous frame
                diff_frame = cv2.absdiff(src1=previous_frame, src2=obraz_fil)
                previous_frame = obraz_fil

                    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
                kernel = np.ones((5, 5))
                diff_frame = cv2.dilate(diff_frame, kernel, 1)

                    # 5. Only take different areas that are different enough (>20 / 255)
                thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

                # thresh_frame=skimage.morphology.remove_small_objects(thresh_frame,100)

                kierownica=black.astype('uint8')
                kierownica[y1-r1:(y1+2*r1), x1-r1:(x1+2*r1)]= thresh_frame[y1-r1:(y1+2*r1), x1-r1:(x1+2*r1)]

                contours, _ = cv2.findContours(image=kierownica, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)
                if(contours):
                    point=contours[0]
                    # for cnt in contours:
                    #     # Calculate area and remove small elements                    a
                    #     area = cv2.contourArea(cnt)
                    #     x, y, _, _ = cv2.boundingRect(cnt)
                    #     xn, yn, _, _ = cv2.boundingRect(point)
                    #
                    #     if(cv2.contourArea(point)<area and math.dist((x,y),(xn,yn))<100):
                    #         point=cnt
                    #     else:
                    point=findclosest(contours,point)
                    cv2.drawContours(frame, [point], -1, (0, 255, 0), 2)
                    cx, cy, _, _ = cv2.boundingRect(point)

                    # if best_cnt>1:

                    cv2.line(frame, (x1, y1), (cx, cy), (255, 255, 255), 5)
                    cv2.line(frame, (ocx, ocy), (cx, cy), (255, 0, 0), 5)

                    # out = cv2.add(blank, blur)
                    # if(math.dist((ocx,ocy),(cx,cy))>800):
                    ocx = cx
                    ocy = cy
                    if(ocx!=cx and ocy!=cy):
                        getAngle(((x1,y1),(ocx, ocy),(cx,cy)), frame)

                # cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                #                          lineType=cv2.LINE_AA)
                    ################################-----------------------------------------------------------------------------------------------------------

            cv2.imshow("Obraz", frame)
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