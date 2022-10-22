#https://blog.csdn.net/weixin_42279087/article/details/80450237
import cv2
import numpy as np
 
 
img = cv2.imread('varese.jpg')#读取要处理的图片
alg = input('Select an Alg --> ')
 
 
def fd(algorithm):
	if algorithm == 'SIFT':
		return cv2.xfeatures2d.SIFT_create()
	if algorithm == 'SURF':
		threshold = input('Enter a threshold --> ')#提示输入SURF算法所采用的阈值
		return cv2.xfeatures2d.SURF_create(float(threshold))
 
 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度图像
fd_alg = fd(alg)
keypoints,descriptor = fd_alg.detectAndCompute(gray,None)#进行算法的检测和计算
 
 
img = cv2.drawKeypoints(image = img,outImage = img,keypoints = keypoints,
						flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
						color = (51,163,236))
cv2.imshow('keypoints',img)
while(True): 
	if cv2.waitKey(30) & 0xff == ord('q'):
		break
cv2.destroyAllWindows()
