import numpy as np
import cv2
cap = cv2.VideoCapture(0)
##############################################
widthImg = 640
heightImg = 480
NumberPlateCascade = cv2.CascadeClassifier("Learn-OpenCV-in-3-hours/Resources/haarcascade_russian_plate_number.xml")
MinArea = 500
color = (0,255,0)
count = 0
##############################################
cap.set(3,widthImg)
cap.set(4,heightImg)
cap.set(10,200)
while True:
    success,img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    NumberPlates = NumberPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x,y,w,h) in NumberPlates:
        area = w*h
        if area > MinArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img,"NumberPlate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("NumberPlate",imgRoi)

    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("NumberPlatesDetected/NoPlate_"+str(count)+".jpg",imgRoi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,
                    2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count = count + 1
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
