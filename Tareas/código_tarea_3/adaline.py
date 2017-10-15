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


class ArgumentosFaltantes(Exception) :
    pass


def adaline_interpolation(x=None,y=None,x_test=None,y_test=None,epocas=700,grado=1,learning=0.1) :
    
    if  any([x is None,y is None,x_test is None,y_test is None]) :
        raise ArgumentosFaltantes("Los datasets de entrenamiento (x con y) y los de validación (x_test con y_test) son necesarios")
    
    pesos = [0.2 for i in range(grado+1)]
    
    errores = []
    errores_prediccion = []
    epocas_fin = epocas
    diez_malos = 0
    
    for epoca in range(epocas) :
        
        print("Epocas: ", epoca)
        
        deltas = [0.0 for i in pesos]
        
        error_cuadratico_medio = 0.0
        
        for num_fila in range(len(x)) :
            
            monomios = [x[num_fila][0]**e for e in range(grado+1)]
            
            prediccion = np.dot(monomios,pesos)
            
            error = y[num_fila][0] - prediccion
            
            for num_delta in range(len(deltas)) :
                
                deltas[num_delta] = learning * error * monomios[num_delta]  
        
            for num_peso in range(len(pesos)) :
                
                pesos[num_peso] += deltas[num_peso]
            
            error_cuadratico_medio += error**2
        
        error_cuadratico_medio /= len(x)
        
        errores.append(error_cuadratico_medio)
        

    
        predicciones = []
        
        error_cuadratico_medio = 0.0
        
        for num_fila in range(len(x_test)) :
            
            monomios = [x_test[num_fila][0]**e for e in range(grado+1)]
            
            prediccion = np.dot(monomios,pesos)
            
            predicciones.append(prediccion)
            
            error_cuadratico_medio += (prediccion-y_test[num_fila][0])**2
    
        error_cuadratico_medio /= len(y_test)
        
        errores_prediccion.append(error_cuadratico_medio)
        
        if len(errores_prediccion) > 1 and errores_prediccion[-1] > errores_prediccion[-2] :
            diez_malos += 1
        else :
            diez_malos = 0
        
        if diez_malos >= 10:
            epocas_fin = epoca +1
            break
        
    plt.figure()    
    plt.plot(range(epocas_fin),errores, label="Entrenamiento")
    plt.plot(range(epocas_fin), errores_prediccion, label="Validación")
    min_p = errores_prediccion[np.argmin(errores_prediccion)]
    plt.plot(range(epocas_fin), [min_p for i in range(epocas_fin)], label="Error mínimo de validación")
    plt.ylim(0.0, 0.1)
    plt.yticks(np.arange(0.0, 0.1, 0.01))
    plt.title("Error grado "+str(grado))
    plt.xlabel("Epoca")
    plt.ylabel("Error cuadratico medio")
    plt.legend(loc=4, framealpha=0.6)
    
    plt.savefig("Error_grado_"+str(grado)+"_epocas_usadas_"+str(epocas_fin)+"_.jpg")
    
    plt.figure()
    plt.plot(x_test[:,0],y_test[:,0], label="Datos de Validación")
    plt.plot(x_test[:,0], predicciones, label="Predicciones")
    plt.title("Polinomio grado "+str(grado))
    plt.xlabel("Dominio")
    plt.ylabel("Rango")
    plt.legend(loc=4, framealpha=0.6)
    
    plt.savefig("Funcion_grado_"+str(grado)+"_epocas_usadas_"+str(epocas_fin)+"_.jpg")
    

    
    
