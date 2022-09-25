import cv2

videoWriter=cv2.VideoWriter('res_card.avi',cv2.VideoWriter_fourcc(*'MJPG'),30,(1280,720))

for i in range(1,237):
    frame=cv2.imread('./results_card/'+str(i).zfill(4)+'.jpg')
    videoWriter.write(frame)
videoWriter.release()