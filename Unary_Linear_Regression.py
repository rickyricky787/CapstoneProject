# Jiayao Zhou
# 2021/5/21
# This is the code applies our Unary Linear Regression
import pandas as pd
import numpy as np
import pylab


def plot_data(data, b, m,datax,datay):
    x = datax
    y = datay
    y_predict = m*x + b
    print(y_predict)
    data['Date']=pd.to_datetime(data['Date'])
    pylab.plot(data['Date'], y_predict, '-b')
    pylab.plot(data['Date'], data["Close"], '-r')
    pylab.show()

def linear_regression(data, learning_rate, times,datax,datay):
     b = 0.0
     m = 0.0
     #The datax and datay means two values from the data
     x = datax
     y = datay
     n = float(len(data))
     for i in range(times):
         db = -(1/n)*(y - m*x-b)
         db = np.sum(db, axis=0)
         dw = -(1/n)*x*(y - m*x - b)
         dw = np.sum(dw)
         sb = b - (learning_rate*db)
         sw = m - (learning_rate*dw)
         b = sb
         m = sw
         if i % 100 == 0:
             j = (np.sum((y - m*x - b)**2))/n
    #the return values b and m means the line y=mx+b
     return [b, m]




if __name__ == '__main__':
     data = pd.read_csv("AMC.csv")
     x=data["Open"]
     y=data['Close']
     learning_rate = 0.001
     times = 1000
     b, m = linear_regression(data, learning_rate, times,x,y)
     plot_data(data, b, m,x,y)

