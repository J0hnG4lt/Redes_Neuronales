#!/usr/bin/python3

"""
Estudiante: Georvic Tur
Carnet: 12-11402
"""

from mlp import MLP, htan, lineal, derivada, dhtan
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
    
    
    for num_neuronas_ocultas in [1, 2, 3, 4, 6, 8, 12, 20, 40] :
        print("Numero de neuronas ocultas: ", num_neuronas_ocultas)
        mlp = MLP([1,num_neuronas_ocultas,1],[htan,lineal],[dhtan, derivada],0.1)      
        mlp.entrenar(x,y,x_test,y_test,700)
        del mlp
