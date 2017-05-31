#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación de un perceptron multi capas

"""
__author__ = "Georvic Tur 12-11402"
__email__ = "alexanderstower@gmail.com"


import pandas as pd

import numpy as np

import sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.linear_model import Perceptron # Para comparar

class DimensionesDistintas(Exception):
    pass

class NoSeHaEntrenadoElPerceptron(Exception):
    pass

class Mi_Perceptron :
    
    def __init__(self) :
        
        self._eta = 1e-1
        self._max_iters = 5
        self._b0 = 0.0
        
        self._w = pd.DataFrame()
        self._clases = []
        self._obtener_clase = dict()
        
        self._tam_t_set = 0
        self._t_set_correctos = 0
        
    
    
    def entrenar_mi_perceptron(self,x,d,b=0.0,eta=1e-1,max_iters=5,clases=[]) :
        """
        Perceptrón basado en aprendizaje competitivo. La regla de aprendizaje
        es la actualización con constante de aprendizaje
        
        Entrada
            x : features no aumentados
            d : clase
            eta : tasa de aprendizaje
            max_iters : número máximo de iteraciones
            b : sesgo
            clases : np array de clases en d
            pocket : si se usa un refuerzo positivo
        Salida
            w : matriz de coeficientes. Cada columna es de una neurona.
        
        Bibliografía: 
            Neural Networks: A Systematic Introduction de Raúl Rojas (p. 104)
        
        """
        
        self._max_iters = max_iters
        self._eta = eta
        self._b0 = b
        
        (x_dim, y_dim) = x.shape
        if (x_dim, 1) != d.shape :
            raise DimensionesDistintas("Las filas de x ({}) y d ({}) no concuerdan".format(x.shape,d.shape))
        
        # Añado la columna de bias inicializada con valores de 1 al input
        x = pd.concat([x,pd.DataFrame( np.ones( ( x_dim , 1 ) ), columns=["BIAS"] ) ], axis=1, join="inner")
        (x_dim, y_dim) = x.shape
        
        if len(clases) == 0:
            clases = d.CLASE.unique()
            clases.sort()
        self._clases = clases
        
        # Creo un mapa de clases
        obtener_clase = dict( zip( clases , [i for i in range( len(clases) )] ) )
        self._obtener_clase = obtener_clase
        
        # Inicializo la matriz de pesos: una fila por neurona/clase
        w = pd.DataFrame( np.random.random( ( len(clases), y_dim ) ), columns=x.columns.values.tolist() )
        w["BIAS"] = b
        
        d.columns = ["CLASE"]
        
        mejor_w = w
        numero_correctos = 0
        for _ in range(max_iters) :
            for index, entradas in x.iterrows() :
                ppunto = -sys.maxsize-1
                maxClase = 0
                
                # Winner takes all
                for numClase, pesos in w.iterrows() :
                    ppunto_t = np.dot(entradas.values, pesos.values)
                    (ppunto,maxClase) = (ppunto_t,numClase) if ppunto_t > ppunto else (ppunto,maxClase)
                claseEsperada = d.iloc[index][0]
                if obtener_clase[claseEsperada] != maxClase :
                    w.iloc[obtener_clase[claseEsperada]] += eta * entradas
                    w.iloc[maxClase] -= eta * entradas
                
            self._w = w
        
        return w

    def evaluar_mi_perceptron(self,x) :
        """
        ENTRADA
            x : instancia de features no aumentados
            w : matriz de pesos aumentados
        SALIDA
            return (y,argmax)
                y : evaluación
                argmax : clase (index de w) 
        """
        w = self._w
        
        x["BIAS"] = 1.0
        
        if w.shape[1] != x.size :
            raise DimensionesDistintas("Las columnas de x ({}) y w ({}) no concuerdan".format(x.shape,w.shape))
        
        y = -sys.maxsize - 1
        argmax = 0
        yt=None
        for index, item in w.iterrows() :
            yt = np.dot(item, x)
            (y, argmax) = (yt, index) if yt > y else (y, argmax)
        
        return (y, argmax)
    
    @property
    def w(self):
        """
        Getter que amplía la matriz de pesos del perceptron con las clases
        """
        if self._w.empty :
            raise NoSeHaEntrenadoElPerceptron("Hay que inicializar los pesos del perceptron")
        w_con_clase = pd.concat([self._w, pd.DataFrame( self._clases, columns=["CLASE"] ) ], axis=1, join="inner")
        return w_con_clase
    
    def test_perceptron(self, x_t, d_t) :
        """
        ENTRADA
            x_t : testing feature dataset
            d_t : testing class dataset
        SALIDA
            return r
                r : lista de booleanos con 1 en i si la clase de d_t[i] es correcta
        """
        
        if self._w.empty :
            raise NoSeHaEntrenadoElPerceptron("Hay que inicializar los pesos del perceptron")
        
        if (x_t.shape[0], 1) != d_t.shape :
            raise DimensionesDistintas("Las filas de x_t ({}) y d_t ({}) no concuerdan".format(x_t.shape,d_t.shape))
        
        r = pd.DataFrame( list( map( bool, np.zeros( ( d_t.shape ) ) ) ), columns=["CLASE"], index= x_t.index )
        
        for index, entrada in x_t.iterrows() :
            (y, argmax) = self.evaluar_mi_perceptron(entrada)
            r.ix[index] = (argmax == self._obtener_clase[d_t.ix[index][0]])
        
        self._tam_t_set = r.shape[0]
        self._t_set_correctos = r.loc[r["CLASE"] == True].shape[0]
        
        return r
    

if __name__ == "__main__":

    dataset = pd.read_csv("4D.csv")
    x = dataset[[0,1,2,3]]
    
    x = dataset[[0,1,2,3]]
    y = dataset[[4]]

    clases = y.CLASE.unique()
    clases.sort()
    
    kf = KFold(n_splits=3)
    k = 0
    mejorPerceptron = None
    mejorScore = 0
    score = 0.0
    scoreSKA = 0.0
    
    eta = 0.5
    max_iters = 5
    
    print()
    print("TASA DE APRENDIZAJE: {} \nNÚMERO DE ITERACIONES: {}".format(eta,max_iters))
    
    for train, test in kf.split(dataset):
        x_e = dataset.iloc[ train ][ [0,1,2,3] ]
        y_e = dataset.iloc[ train ][ [4] ]
        x_t = dataset.iloc[ test ][ [0,1,2,3] ]
        y_t = dataset.iloc[ test ][ [4] ]
        
        print("\nCROSS-VALIDATION fold: {}".format(k))
        miP = Mi_Perceptron()
        miP.entrenar_mi_perceptron(x=x_e,
                                   d=y_e,
                                   clases=clases,
                                   eta=eta,
                                   max_iters=max_iters)
        print("Vectores de pesos por cada clase")
        print(miP.w)
        miP.test_perceptron(x_t=x_t, d_t=y_t)
        print("CORRECTOS: ",miP._t_set_correctos)
        print("INCORRECTOS: ",miP._tam_t_set-miP._t_set_correctos)
        print("SCORE DE MI PERCEPTRON: ",miP._t_set_correctos / miP._tam_t_set)
        (mejorScore, mejorPerceptron) = (miP._t_set_correctos, miP) if miP._t_set_correctos > mejorScore else (mejorScore, mejorPerceptron)
        score += (miP._t_set_correctos / miP._tam_t_set)
        
        cls = Perceptron(eta0=0.5,n_iter=5)
        cls.fit(x_e,np.ravel(y_e))
        scoreSK = cls.score(x_t,np.ravel(y_t))
        scoreSKA += scoreSK
        print("PERCEPTRON DE SKLEARN")
        print("SCORE DE SKLEARN: {}".format(scoreSK))
        k+=1
        
    meanScore = score / float(k)
    meanScoreSK = scoreSKA / float(k)
    print()
    print("EXACTITUD PROMEDIO DE MI PERCEPTRON: {}".format(meanScore))
    print("EXACTITUD PROMEDIO DE SKLEARN: {}".format(meanScoreSK))
    print()
    
    



