import cv2
import numpy as np

# 画像の表示と保存はコメントアウトしている

#画像読み込み
filename = "Monna_Lisa.png"
imgCV = cv2.imread(filename)

#画像表示
cv2.namedWindow("PNG1", cv2.WINDOW_NORMAL)
#cv2.namedWindow("PNG2", cv2.WINDOW_NORMAL)
#cv2.imshow("PNG", imgCV)
#cv2.imshow("PNG2", imgCV)

#画像の保存
#cv2.imwrite('name.png', img)

#任意のキー入力で画像を閉じる
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#画像の縦と横のサイズを取得
h = imgCV.shape[0]
w = imgCV.shape[1]
#h,w = imgCV.shape[:2]

#画像のサイズ変更
imgCV_LARGE = cv2.resize(imgCV, (w*2, h*2)) #double
imgCV_SMALL = cv2.resize(imgCV, None, fx=0.5,fy=0.5) #half

"""
#サイズ変更画像の保存
cv2.imwrite('large.png', imgCV_LARGE)
cv2.imwrite('small.png', imgCV_SMALL)
"""

"""
#表示
cv2.imshow("PNG1", imgCV_LARGE)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("PNG1", imgCV_SMALL)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#画像の回転
img_90_clock = cv2.rotate(imgCV, cv2.ROTATE_90_CLOCKWISE)
img_90_cclock = cv2.rotate(imgCV, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_180 = cv2.rotate(imgCV, cv2.ROTATE_180)

#画像の上下と左右の反転
img_flp_ud = cv2.flip(imgCV, 0)
img_flp_lr = cv2.flip(imgCV, 1)
img_flp_udlr = cv2.flip(imgCV, -1)

"""
cv2.imwrite('90rot.png', img_90_clock)
cv2.imwrite('270rot.png', img_90_cclock)
cv2.imwrite('180rot.png', img_180)
cv2.imwrite('udflp.png', img_flp_ud)
cv2.imwrite('lrflp.png', img_flp_lr)
cv2.imwrite('udlrflp.png', img_flp_udlr)
"""

#画像の2値化 thはしきい値,im_thは画像
th, im_th = cv2.threshold(imgCV, 128, 255, cv2.THRESH_BINARY)
th, im_tz = cv2.threshold(imgCV, 128, 255, cv2.THRESH_TOZERO)

#画像のグレースケール化→2値化
im_gray = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
th, im_gray_th_otsu = cv2.threshold(im_gray, 128, 255, cv2.THRESH_OTSU)

"""
cv2.imshow("PNG1", im_gray_th_otsu)
#cv2.imshow("PNG2", im_tz)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#cv2.imwrite('binari.png', im_gray_th_otsu)

#２枚の画像の差分画像作成
flame1 = "basketball1.png"
flame2 = "basketball2.png"

img1 = cv2.imread(flame1)
img2 = cv2.imread(flame2)

#差分画像作成→グレースケール化→2値化
im_diff= cv2.absdiff(img1, img2)
im_diff_gray = cv2.cvtColor(im_diff, cv2.COLOR_BGR2GRAY)
th, im_diff_th = cv2.threshold(im_diff_gray, 50, 255, cv2.THRESH_BINARY)

"""
cv2.imshow("PNG1", im_diff_th)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#cv2.imwrite('diff.png', im_diff_th)








