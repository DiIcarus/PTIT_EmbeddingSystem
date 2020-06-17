import socketio

sio = socketio.Client()

sio.event
def connect():
    print('connection established')
    
def message_handler(data):
    print('Received message: ', data)

sio.event
def my_message(data):
    print('message received with ', data)
    sio.emit('trafficSign', 
        {
            'sign': data
        })
    # sio.on('trafficSignReceive', message_handler)

sio.event
def disconnect():
    print('disconnected from server')

host = 'http://207.148.71.252:7050'
nhut = 'http://192.168.1.7:7050'
sio.connect("http://192.168.0.101:7050")
import cv2
import numpy as np
from Recognition import detection 
from detec import mainDetect
url = 'rtsp://192.168.0.105:8554/mjpeg/1'
cap = cv2.VideoCapture(url)
count = 0;
while(True):
  ret, frame = cap.read()
  im = mainDetect(frame)
  cv2.imshow("demo",frame)
  if im is None:
    count+=1
  else:
        height, width = im.shape[:2]
        if(0.85<height/width and height/width<1.25):
            res = detection(im)
            cv2.imshow("demo1",im)
            my_message(res)
            print(res)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    break
sio.wait()

