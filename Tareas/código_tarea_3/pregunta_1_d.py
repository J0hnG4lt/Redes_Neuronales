#!/usr/bin/python3

"""
Estudiante: Georvic Tur
Carnet: 12-11402
"""

from mlp import MLP, htan, lineal, derivada, dhtan
from adaline import adaline_interpolation
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__" :
    
    training = pd.read_csv("reglin_train.csv", names = ["x","y"])
    testing = pd.read_csv("reglin_test.csv",names = ["x","y"])
    
    
    #from sklearn.utils import shuffle
    #training = shuffle(training)
    
    #p = training.plot(x="x",y="y")
    #plt.show()
    #p = testing.plot(x="x",y="y")
    #plt.show()
    
    scaler = MinMaxScaler()
    training = pd.DataFrame(scaler.fit_transform(training), columns=training.columns)
    testing = pd.DataFrame(scaler.fit_transform(testing), columns=training.columns)
    
    x = training["x"].values
    x = np.column_stack((x,np.ones(len(x))))
    y = training["y"].values.reshape((len(training["y"]),1))
    
    x_test = testing["x"].values
    x_test = np.column_stack((x_test,np.ones(len(x_test))))
    y_test = testing["y"].values.reshape((len(testing["y"]),1))
    
    for grado in [2,4,6,10,20,40] :
        print("Grado del polinomio: ", grado)
        adaline_interpolation(x,y,x_test,y_test,epocas=700,grado=grado,learning=0.01)
    
    
