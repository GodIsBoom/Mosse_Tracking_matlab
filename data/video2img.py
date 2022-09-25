#coding=utf-8
import cv2
import os

path = 'card'
video_list = os.listdir(path)
for video in video_list:
    print(str(video))
    video_path = os.path.join(os.getcwd(),path,video)
    vc=cv2.VideoCapture(video_path)
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False

    c =1
    while rval:
        rval,frame=vc.read()
        if rval==False:
            break
        cv2.imwrite('./card/'+str(c).zfill(4)+'.jpg',frame)
        if c%10 ==0:
            print(c)
        if c==100:
            cv2.imshow('100',frame)
        c=c+1
        cv2.waitKey(1)
    vc.release()
