import cv2
import numpy as np
from skimage.feature import hog  
from skimage import exposure 
from skimage import data, segmentation, color
from skimage import graph
from skimage.feature import daisy
import matplotlib.pyplot as plt 

#画像読み込み
filename = "Munch.png"
imgCV = cv2.imread(filename)
    #グレースケール
img = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)

#画像表示
def show_img(img_with_keypoints):
    cv2.namedWindow("Extract", cv2.WINDOW_NORMAL)
    cv2.imshow("Extract", img_with_keypoints)
    cv2.waitKey(0)  
    cv2.destroyAllWindows() 

#画像の特徴量抽出と図示
#SIFT特徴量
def ext_sift(img):
    sift = cv2.SIFT_create()  
    keypoints, descriptors = sift.detectAndCompute(img, None) 
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None) 
    show_img(img_with_keypoints)
    return

#ORB特徴量
def ext_orb(img):
    orb = cv2.ORB_create()   
    keypoints, descriptors = orb.detectAndCompute(img, None) 
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None) 
    show_img(img_with_keypoints)
    return

#AKAZE特徴量
def ext_akaze(img):
    akaze = cv2.AKAZE_create()   
    keypoints, descriptors = akaze.detectAndCompute(img, None) 
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None) 
    show_img(img_with_keypoints)
    return

#FAST特徴量
def ext_fast(img):
    fast = cv2.FastFeatureDetector_create()  
    keypoints = fast.detect(img, None) 
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None) 
    show_img(img_with_keypoints)
    return

#HOG特徴量
def ext_hog(img):
    features, hog_image = hog(img, pixels_per_cell=(16, 16),  
                            cells_per_block=(1, 1), visualize=True)  
    # HOG画像を適切に可視化  
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    show_img(hog_image_rescaled)
    return

#RAG特徴量
def ext_rag(img):
    labels1 = segmentation.slic(img, n_segments=400, compactness=30)
    out1 = color.label2rgb(labels1, img, kind='avg')
    #show_img(out1)
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg')
    show_img(out2)
    return

#DAISY特徴量
def ext_daisy(img):
    descs, descs_img = daisy(img, step=155, radius=58, rings=2, histograms=6, orientations=8, visualize=True)
    show_img(descs_img)
    return

#ヒストグラム(グレースケール)
def ext_hist_gray(img):
    gray_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure(figsize=(12, 6)) 
    plt.subplot(1, 2, 1)  
    plt.title('Grayscale Histogram')  
    plt.xlabel('Pixel Value')  
    plt.ylabel('Frequency')  
    plt.plot(gray_hist, color='black')  
    plt.tight_layout()  
    plt.show() 
    return

#ヒストグラム(RGB)
def ext_hist_rgb(imgCV):
    colors = ('b', 'g', 'r')  # OpenCVではBGRの順で格納されている  
    color_hist = {}  
    for i, color in enumerate(colors):  
        color_hist[color] = cv2.calcHist([imgCV], [i], None, [256], [0, 256]) 
    plt.figure(figsize=(12, 6))  
    # カラー画像のヒストグラムを表示  
    plt.subplot(1, 2, 2)  
    plt.title('Color Histogram')  
    plt.xlabel('Pixel Value')  
    plt.ylabel('Frequency')  
    for color in colors:  
        plt.plot(color_hist[color], color=color)  
    plt.xlim([0, 256]) 
    plt.tight_layout()  
    plt.show()
    return

def switch_extract(value):  
    match value:  
        case 1:  
            return ext_sift(img)  
        case 2:  
            return ext_orb(img) 
        case 3:  
            return ext_akaze(img)  
        case 4:  
            return ext_fast(img)  
        case 5:  
            return ext_hog(img) 
        case 6:  
            return ext_rag(imgCV) 
        case 7:  
            return ext_daisy(img)
        case 8:  
            return ext_hist_gray(img)
        case 9:  
            return ext_hist_rgb(imgCV)
        case _:  
            return
        


switch_extract(9)
