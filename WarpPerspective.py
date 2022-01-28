import numpy as np
import cv2

# WARP PERSPECTIVE
img = cv2.imread('image2.jpg')
print(img.shape)
# img = cv2.resize(img,(500,400))
# print(img.shape)
width,height = 144,524
pts1 = np.float32([[848,226],[991,227],[848,750],[991,750]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgoutput = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow('image',img)
cv2.imshow('WarpImage',imgoutput)
cv2.waitKey(0)

