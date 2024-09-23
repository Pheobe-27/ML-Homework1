# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:13:22 2024

@author: 23290
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

data='HousingData(2).csv'  #数据只保留了每间住宅的平均房间数RM和以1000美元为单位的自住房屋的中位数价格MEDV
dataList=[]
with open(data,'r') as file:
    next(file) 
    for line in file:
        size,price=line.strip().split(',')
        dataList.append((float(size),float(price)))
#print(dataList)

#数据标准化    
sizes=[]
prices=[]
for size,price in dataList:
    sizes.append(size)
for size,price in dataList:
    prices.append(price)
avgSize=sum(sizes)/len(sizes)  #均值
stdSize=(sum((x-avgSize)**2 for x in sizes) / len(sizes))**0.5  #标准差
standardizedSize=[]
for size in sizes:
    standardizedSize.append((size-avgSize) / stdSize)

#梯度下降
weight=0 #权重初始化
bias=0 #偏差初始化
rate=0.01 #学习率
iterations=1000 #迭代次数
for i in range(iterations):
    Gweight=0
    Gbias=0
    for j in range(len(standardizedSize)):
        size=standardizedSize[j]
        price=prices[j]
        predict=weight*size+bias #y=θx+b
        Gweight+=-2*(price-predict)*size  #-2是导数导出来的
        Gbias+=-2*(price-predict)
    weight-=rate*Gweight/len(dataList) #在梯度反方向更新参数，找到损失函数的最小值
    bias-=rate*Gbias/len(dataList)

def predict(size,weight,bias):
    return weight*size+bias

plt.figure(figsize=(10,6))
x=standardizedSize
y=prices

plt.scatter(x,y,color='blue',label='实际房价数据点')

x_fit=[min(x),max(x)]
y_fit=[]
for size in x_fit:
    y_fit.append(predict(size,weight,bias))
plt.plot(x_fit,y_fit,color='orange',linewidth=4,label='拟合曲线')

plt.legend()
plt.xlabel('每间住宅平局房间数')
plt.ylabel('以1000美元为单位的自住房屋的中位数价格')
plt.title('用线性回归模型预测房价')
plt.show()