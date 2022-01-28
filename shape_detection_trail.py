import numpy as np
import cv2
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    objCor = []
    approx_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            objCor.append(len(approx))
            approx_points.append(approx)
    return objCor,approx_points
cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    # preprocessing the image for to find edges and to find corner points of a shape present in image
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,50,50)
    objcor,approx_points = getContours(imgCanny)
    for o,a in zip(objcor,approx_points):
        x, y, w, h = cv2.boundingRect(a)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Categorise thier shapes
        if o == 3:
            objectType = "Tri"
        elif o == 4:
            # hear for square and rectangle will have 4 corners but square width and height are same
            # so we will divide width and height if the value is nearer to 1 then it is square or else rectangle
            aspRatio = w / float(h)
            if aspRatio > 0.95 and aspRatio < 1.05:
                objectType = "Square"
            else:
                objectType = "Rect"
        elif o > 4:
            objectType = "Circle"
        cv2.putText(img, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                (0, 255, 255), 2)
        cv2.imshow("Original",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break