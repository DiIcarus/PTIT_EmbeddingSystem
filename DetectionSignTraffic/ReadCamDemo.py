import cv2
import numpy as np
from Recognition import detection 
from detec import mainDetect
# url = 'rtsp://192.168.1.15:8554/mjpeg/1'
url = 'rtsp://192.168.0.105:8554/mjpeg/1'
cap = cv2.VideoCapture(url)
count = 0;
while(True):
  ret, frame = cap.read()
  im = mainDetect(frame)
  cv2.imshow("cam",frame)
  if im is None:
    count+=1
    # continue
  else:
    count+=1
    height, width = im.shape[:2]
    if(0.85<height/width and height/width<1.25):
      res = detection(im)
      cv2.imshow("demo1",im)
      print(res)
  print(count)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    break
'''
SIGN:
turn_right: turn_right
turn_left: turn_left
stop: stop
ahead only: ahead_only
speed_limit: 20,30,40,50,60,70,80,90,100
'''
# frame = cv2.imread('./img/sss.jpg')
# im = mainDetect(frame)
# cv2.imshow("demo",frame)
# if im is None:
#   print("None")
# else:
#   print(detection(im))