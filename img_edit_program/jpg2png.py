import cv2
import numpy as np

#画像読み込み
filename = "Munch.jpg"
imgCV = cv2.imread(filename)
cv2.imwrite('Munch.png',imgCV)
