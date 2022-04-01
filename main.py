import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# Capture Video
cap = cv2.VideoCapture(0)

# Detecting the face
detector = FaceMeshDetector(maxFaces=1)

# Showing the Image
while True:
    success, img = cap.read()
    img,faces = detector.findFaceMesh(img,draw=False)
    if faces:
        face = faces[0]
        # Detect landmark of eyes
        pointLeft = face[145]
        pointRight = face[374]
        cv2.line(img,pointLeft,pointRight,(0,200,0),3)
        cv2.circle(img,pointLeft,5,(255,0,255),cv2.FILLED)
        cv2.circle(img,pointRight,5,(255,0,255),cv2.FILLED)
        # Finding the focal length
        w,_ = detector.findDistance(pointLeft,pointRight) #find distance between centre of eyes
        # print(w)
        W = 6.3 #average actual distance between eyes
        d = 40 #distance between the camera and face
        # f = (w*d)/W # focal length
        # print(f)
        # Finding Distance
        f = 600
        d = (W*f)/w
        # print(d)
        cvzone.putTextRect(img,f"Depth: {int(d)} cms",(face[10][0]-100,face[10][1]-50),scale=2)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break