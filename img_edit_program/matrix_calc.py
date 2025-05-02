import numpy as np
#import torch

X = np.array([[2,-1],[-3,4]])
Y = np.array([[1,2],[3,4]])

# XとYの表示
print("X=",X,sep="\n")
print("Y=",Y,sep="\n")

# XとYの加減算と結果の表示
print("X+Y=",X+Y,sep="\n")
print("Y-X=",Y-X,sep="\n")

# 行列の掛け算
print("2*X=",2*X,sep="\n")  # スカラー倍
print("X*Y=",np.dot(X,Y),sep="\n")  # 行列同士の掛け算
#print("X*Y=",np.matmul(X,Y),sep="\n")
#print("X*Y=",X@Y,sep="\n")
print("Y*X=",np.dot(Y,X),sep="\n")  # X*Y とは異なる結果を示す

# 同じ行かつ同じ列の数同士で割り算(X[1][2]/Y[1][2], X[2][2]/Y[2][2])
print("X÷Y=",np.divide(X,Y),sep="\n")

# X の逆行列
#print(np.linalg.inv(X))

# 行列の割り算？逆行列を掛ける
    # X / X
print("X*X^-1=",X@np.linalg.inv(X),sep="\n")
#print("X*X^-1=",np.dot(X,np.linalg.inv(X)),sep="\n")

    # X / Y
print("X*Y^-1=",X@np.linalg.inv(Y),sep="\n")
