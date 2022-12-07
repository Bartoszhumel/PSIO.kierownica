import cv2
import sys
import numpy as np
import time

device=0
cap = cv2.VideoCapture('HG16MEU.mp4')
pos_frame = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while not cap.isOpened():
    cap = cv2.VideoCapture('HG16MEU.mp4')
    cv2.waitKey(2000)
    print ("Czekam na wideo")
break_flag=False
start_flag=True
while True:
    t = time.time()
    for i in range(25):
        flag, frame = cap.read()
        if flag:
            # ramka jest gotowa
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            obraz_sz = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obraz_fil = cv2.GaussianBlur(obraz_sz, (5, 5), 0)
            obraz_kr = cv2.Canny(obraz_fil, 60, 150)
            cv2.imshow("Obraz", frame)
            #cv2.imshow("Filtr", obraz_fil)
            cv2.imshow("krawedzie", obraz_kr)
            if start_flag:
                cv2.setWindowProperty('Obraz',cv2.WND_PROP_TOPMOST,1)
                cv2.moveWindow('Obraz',20,20)
                cv2.setWindowProperty('krawedzie',cv2.WND_PROP_TOPMOST,1)
                cv2.moveWindow('krawedzie',800,20)
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