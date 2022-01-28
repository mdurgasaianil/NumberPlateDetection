import numpy as np
import cv2
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500: # Giving threshold used to avoid detecting the noise data
            cv2.drawContours(imgBlank, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True) # calculating the curve length helps to find the approx corner points in contours
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) # finding the corner points
            # above True is because we are assuming all our shapes are closed
            print(len(approx))
            objcor = len(approx)
            # Drawing a bounding boxes around shapes
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgBlank,(x,y),(x+w,y+h),(0,255,0),2)
            # from this bounding boxes we get information like width and height of the shape and center point
            # of the shape
            # Categorise thier shapes
            if objcor == 3:
                objectType = "Tri"
            elif objcor == 4:
                # hear for square and rectangle will have 4 corners but square width and height are same
                # so we will divide width and height if the value is nearer to 1 then it is square or else rectangle
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rect"
            elif objcor > 4:
                objectType = "Circle"
            cv2.putText(imgBlank, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 255), 2)




img = cv2.imread('Learn-OpenCV-in-3-hours/Resources/shapes.png')
# preprocessing the image for to find edges and to find corner points of a shape present in image
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)
imgBlank = np.zeros_like(img)
getContours(imgCanny)
imgBlank1 = np.zeros_like(img)

final = stackImages(0.5,([img,imgGray,imgBlur],
                         [imgCanny,imgBlank,imgBlank1]))
cv2.imshow("Original",final)
cv2.waitKey(0)