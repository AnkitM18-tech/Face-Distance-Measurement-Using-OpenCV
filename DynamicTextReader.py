import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

# Capture Video
cap = cv2.VideoCapture(0)

# Detecting the face
detector = FaceMeshDetector(maxFaces=1)

textList = ["World Cup 2011","Man Of The","Tournament :", "Yuvraj Singh","Total Runs : 385","Total Wickets : 15", "Marvelous", "Performance","Absolute Legend"]

sen = 10 # scaling value -> more is less

# Showing the Image
while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img,faces = detector.findFaceMesh(img,draw=False)
    if faces:
        face = faces[0]
        # Detect landmark of eyes
        pointLeft = face[145]
        pointRight = face[374]
        # Finding the focal length
        w,_ = detector.findDistance(pointLeft,pointRight) #find distance between centre of eyes
        W = 6.3 #average actual distance between eyes
        f = 600
        d = (W*f)/w
        cvzone.putTextRect(img,f"Depth: {int(d)} cms",(face[10][0]-100,face[10][1]-50),scale=2)
        for i,text in enumerate(textList):
            singleLineHeight = 20 + int((int(d/sen)*sen)/4)
            scale = 0.4 + (int(d/sen)*sen)/80
            cv2.putText(imgText,text,(50,50+(i*singleLineHeight)),cv2.FONT_ITALIC,scale,(255,255,255),2) 

    imgStacked = cvzone.stackImages([img,imgText],2,1)
    cv2.imshow("Image",imgStacked)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break