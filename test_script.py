import cv2
import numpy as np
import argparse

#Input args

parser = argparse.ArgumentParser()
parser.add_argument("--content_img")
parser.add_argument("--style_img")

args = parser.parse_args()

print(args.content_img,args.style_img)
content_image = cv2.imread(args.content_img)
style_image = cv2.imread(args.style_img)

print(content_image.size)
h,w,c = style_image.shape
#cv2.waitKey(0)
content_image = cv2.resize(content_image,(h,w))
#cv2.imshow("123",content_image)
#cv2.waitKey(0)
#print(content_image.size,style_image.size)
result = cv2.addWeighted(content_image,0.5,style_image,0.4,1.0)

cv2.imwrite("./result/result1.jpg",result)

cv2.imshow("123",result)
cv2.waitKey(0)