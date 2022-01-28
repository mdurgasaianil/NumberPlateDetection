import numpy as np
import cv2
cap = cv2.VideoCapture(0)
widthImg = 640
heightImg = 480
cap.set(3,widthImg)
cap.set(4,heightImg)
cap.set(10,150)
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)

    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > 500: # Giving threshold used to avoid detecting the noise data
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True) # calculating the curve length helps to find the approx corner points in contours
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) # finding the corner points
            #print(area)
            #print(len(approx))
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print("add",add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    #print("diff",diff)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print("mypoints",myPoints)
    # print("newpoints",myPointsNew)
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgoutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return  imgoutput



while True:
    success,img = cap.read()
    img = cv2.resize(img,(widthImg,heightImg))
    imgContour = img.copy()
    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    if biggest.size != 0:
        imgWarped = getWarp(imgThres,biggest)
        cv2.imshow("Contours", imgWarped)
    else:
        cv2.imshow("Contours", imgThres)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break