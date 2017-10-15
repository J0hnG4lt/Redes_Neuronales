#!/usr/bin/python3

"""
Estudiante: Georvic Tur
Carnet: 12-11402
"""

from random import uniform
import pprint
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from mlp import MLP, sigmoid, dsigmoid, lineal, derivada, htan, dhtan

modelo = lambda x : (233.846*(1 - np.exp(-0.00604*x)))


if __name__ == "__main__" :
    
    training = pd.read_csv("rabbit.csv", names = ["x","y"])
    
    scaler = MinMaxScaler()
    training = pd.DataFrame(scaler.fit_transform(training), columns=training.columns)
    
    testing = training.sample(frac = 0.2)
    print(testing.index)
    testing.sort_index(inplace=True)
    print(testing.index)
    training = training.drop(testing.index)
    
    print(len(testing))
    print(len(training))
    
    
    x = training["x"].values
    x = np.column_stack((x,np.ones(len(x))))
    y = training["y"].values.reshape((len(training["y"]),1))
    
    x_t = testing["x"].values
    x_t = np.column_stack((x_t,np.ones(len(x_t))))
    y_t = testing["y"].values.reshape((len(testing["y"]),1))
    
    mlp = MLP([1,2,2,1],[sigmoid,sigmoid,lineal],[dsigmoid,dsigmoid, derivada], 0.1)      
    mlp.entrenar(x,y,x_t,y_t,1000)
