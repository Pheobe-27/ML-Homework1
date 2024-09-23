# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:29:13 2024

@author: 23290
"""

#①用线性回归求房价模型
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''数据点生成'''
a=0
b=2*np.pi
points=100
step=(b-a)/(points-1)
x=[]  #等间距的数据点list
y=[]  #计算对应的准确的sin(x)值
for i in range(points):
    x.append(a+i*step)
for val in x:
    y.append(np.sin(val))
    

'''初始化多项式'''
frequency=5 #5次多项式拟合
xita=[0]*(frequency+1) #多项式系数
rate=0.012 #学习率
iterations=200000 #迭代次数

'''采用梯度下降法'''
z=0.01 #引入正则化函数，避免出现overflow梯度爆炸
for k in range(iterations):
    preY=[] #y的预测值列表
    for i in x:
        predict=0
        for j in range(frequency+1):
            predict+=xita[j]*(i**j)  #相当于y=θ0+θ1*x+...+θ5*x^5
        preY.append(predict)
        
    gradients=[0]*(frequency+1)  #梯度初始化
    for i in range(points):
        for j in range(frequency+1):
            gradients[j]+=(preY[i]-y[i])*(x[i]**j) #计算损失函数关于第j项参数的梯度 ~贡献度
            
    for j in range(frequency+1):
        xita[j]-=rate*(gradients[j]/points+z*xita[j]) 

#绘图
plt.plot(x, y, label='sin(x)', color='blue')  # 原始 sin(x)
plt.plot(x, preY, label='采用梯度下降法拟合的曲线', color='red', linestyle='--')  # 拟合曲线
plt.scatter(x, y, color='orange', s=20, label='样本点')  
plt.xlabel('x')
plt.ylabel('y')
plt.title('用五次多项式θ0+θ1*x+...+θ5*x^5拟合sin(x)')
plt.legend()
plt.grid(True)
plt.show()
