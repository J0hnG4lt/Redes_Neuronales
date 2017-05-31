#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación de una red de asociadores lineales basado en el algoritmo
de Oja y la arquitectura de Sanger
"""

__author__ = "Georvic Tur 12-11402"
__email__ = "alexanderstower@gmail.com"



import pandas as pd

import numpy as np

def enconrar_componentes_principales(x,eta=1e-1,max_iters=5,num_clusters=2) :
    """
    Programa que encuentra componentes principales múltiples tomando en cuenta
    la arquitectura de red de Sanger y el algoritmo de Oja.
    
    ENTRADA
        x : matriz de features no aumentados
        eta : tasa de aprendizaje
        max_iters : número máximo de iteraciones
        num_clusters : número de componentes a extraer
    SALIDA
        return (phi, w)
            phi : matriz de datos vs componentes
            w : matriz de pesos
    BIBLIOGRAFÍA
        T. Sanger,"Optimal unsupervised learning in a single-layer linear 
        feedforward neural network", Neural Networks, vol. 2, no. 6, pp. 459-473, 
        1989. [ONLINE]. 
        Disponible en: https://pdfs.semanticscholar.org/709b/4bfc5198336ba5d70da987889a157f695c1e.pdf. 
        [Visto: 30/05/2017 ]
    """
    (x_dim, y_dim) = x.shape
    w = pd.DataFrame( np.random.random( ( num_clusters, y_dim ) ), columns=x.columns.values.tolist() )
    phi = pd.DataFrame( np.random.random( ( x_dim, num_clusters ) ), columns=range(num_clusters) )
    
    for num_componente in range(num_clusters):
        for index, entrada in x.iterrows() :
        
            phi.iloc[index][num_componente] = np.dot(entrada, w.iloc[num_componente])
            w.iloc[num_componente] = w.iloc[num_componente] \
                                + eta*phi.iloc[index][num_componente]*(entrada - w.iloc[num_componente]\
                                *phi.iloc[index][num_componente])
            
    return (phi, w)


if __name__ == "__main__":

    dataset = pd.read_csv("4D.csv")
    x = dataset[[0,1,2,3]]
    
    # Normalizo
    x = x.div(x.sum(axis=1), axis=0)
    
    # Calculo componentes
    (phi, w) = enconrar_componentes_principales(x=x,num_clusters=4)
    
    # Reconstruyo datos iniciales y calculo error
    x_aprox = np.dot( phi , w )
    error = x - x_aprox
    
    error_t = np.sqrt( np.square( error ).sum() )
    error_t = np.sqrt( np.square( error_t ).sum() )
    print("ERROR TOTAL: {}".format( error_t ))

