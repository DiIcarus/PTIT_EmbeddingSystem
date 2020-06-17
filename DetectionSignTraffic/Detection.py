import cv2
import numpy as np

def colorFilter(image,lower_color,upper_color):
  '''
  filfer Image by (lower_color,upper_color)
  return: 
    frame: numpyarray,
    mask: numpyarray,
    res: numpyarray,
  '''
  image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(image, lower_color, upper_color)
  res = cv2.bitwise_and(image,image, mask= mask)
  return [image,mask,res]
def blockDiagram(img):
  imgr_b = cv2.medianBlur(img,7);
  imgr_b = cv2.GaussianBlur(imgr_b,(5,5),0);
  imgr_b = cv2.cvtColor(imgr_b,cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(imgr_b,np.array([150,0,0]),np.array([200,255,255]))
  mask = cv2.medianBlur(mask,7);
  cimg = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
  circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=50,minRadius=50,maxRadius=1000)
  circles = np.uint16(np.around(circles))
  for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
  return cimg

def findThread(img):
  imgr_b = cv2.medianBlur(img,7);
  imgr_b = cv2.GaussianBlur(imgr_b,(5,5),0)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

  erosion = cv2.erode(imgr_b,kernel,iterations = 3)

  opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
  # opening = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
  opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
  opening = cv2.dilate(opening,(5,5),iterations = 1)
  edges = cv2.Canny(opening,300,300)
  return edges

def findLine(img):
  imgray = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
  imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(imgray, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # print(len(hierarchy[0]))
  # print(len(contours))
  # for i in range(len(contours)):
  #   if len(contours[i]) < 180 and len(contours[i]) > 0:
  #     cv2.drawContours(opening, [contours[i]], -1, (0,255,0), 3)
  #   else:
  #     cv2.drawContours(opening, [contours[i]], -1, (255,255,255), 3)
  cv2.drawContours(img, contours, -1, (0,255,0), 3)
  return [contours,hierarchy]

def findCircle():
  img = cv2.imread('houghcirclesjpg.jpg',0)
  img = cv2.medianBlur(img,5)
  cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
  circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=50,minRadius=30,maxRadius=1000)
  print(img)
  circles = np.uint16(np.around(circles))
  for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
  return cimg


# imgr = colorFilter(cv2.imread('./img/img1.png'),np.array([150,0,0]),np.array([200,255,255]))[0]
# imgb = colorFilter(cv2.imread('./img/img.jpg'),np.array([100,0,0]),np.array([255,255,255]))[0]

# Blurring
# Colour-based detection
# Shape-based detection
# Cropped and Extract features
# Separation

# cv2.imshow("demo",blockDiagram(cv2.imread('./img/imgTemp.png')))
# cv2.waitKey(5000)