# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 00:10:02 2019

@author: Mehwish Tariq Ameen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Linear_Reg:
#initialize the path,X,Y and theta for your model
    def __init__(self,path,x=0,y=0,theta=0):
        self.x=np.array(x)
        self.y=np.array(y)
        self.theta=np.array(theta)        
        self.data = pd.read_csv(path)
#function to plot X,Y values
    def plot(self,x,y):
        plt.scatter(x,y)
        plt.xlabel('Money spent on TV ads ($)')
        plt.ylabel('Sales ($)')
        self.draw_line()
        plt.show()
        
#define X from data that is the csv file                
    def define_x(self):
        X= np.array(self.data)
        self.x = X[:,1] 
        #self.m=np.shape(self.x)[0]
        return self.x
    
#reshaping X     
    def reshape_x(self):
        return self.x.reshape(-1,1)
    
#reshaping Y    
    def reshape_y(self):
        return self.y.reshape(-1,1)
    
#define X from data that is the csv file         
    def define_y(self):
        Y = np.array(self.data)
        self.y = Y[:,-1] 
        return self.y
  
#function to draw the line to fit the data
    def draw_line(self):
        reg = LinearRegression()
        reg.fit(self.reshape_x(),self.reshape_y())
        predictions = reg.predict(self.reshape_x())
        plt.plot(self.x,predictions,c='black',linewidth=3)
        plt.show()
        
#creating objet of class and pass the path for your csv file      
line = Linear_Reg("data\Advertising.csv")
#plotting the data
line.plot(line.define_x(),line.define_y())
