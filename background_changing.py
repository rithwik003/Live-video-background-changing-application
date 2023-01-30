import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()
listImg = os.listdir("Backgrounds")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Backgrounds/{imgPath}')
    imgList.append(img)

indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)
    imgStack = cvzone.stackImages([img, imgOut], 2,1)
    cv2.imshow("Background_changer", imgStack)
    key = cv2.waitKey(1)
    #press 'd' for the next background
    if key == ord('d'):
        if indexImg>0:
            indexImg -=1
    #press 'a' for the previous background
    elif key == ord('a'):
        if indexImg<len(imgList)-1:
            indexImg +=1
    #press q to stop the execution
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
