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

lineal = lambda x : x
derivada = lambda x : 1.0

sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))
dsigmoid = lambda x : sigmoid(x) * ( 1.0 - sigmoid(x) )

htan = lambda x : ((1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x)))
dhtan = lambda x : 1.0 - ( htan(x) ) ** 2 

class ArgumentosFaltantes(Exception) :
    pass

class MLP :
    
    
    def __init__(self,
                 neuronas_por_capa = None, 
                 activacion_por_capa = None,
                 derivada_por_capa = None,
                 learning = 0.5) :
        """
        ENTRADA
            neuronas_por_capa : lista de numeros que indican la cantidad de neuronas (sin bias)
            por cada capa (incluida la inicial y final)
            activacion_por_capa : lista de funciones de activacion por capa
            derivada_por_capa : lista de derivadas de las funciones de activacion.
            Su posicion debe coincidir con su senda derivada en activacion_por_capa
        SALIDA
            self.pesos : tensor con los pesos inicializados de la red
        """
        
        if neuronas_por_capa is None :
            raise ArgumentosFaltantes("neuronas_por_capa debe ser dado como argumento.")

        if activacion_por_capa is None :
            raise ArgumentosFaltantes("activacion_por_capa debe ser dado como argumento.")
        
        if derivada_por_capa is None :
            raise ArgumentosFaltantes("derivada_por_capa debe ser dado como argumento.")
        
        # Contamos los sesgos 
        
        for capa in range(len(neuronas_por_capa)-1) :
            
            neuronas_por_capa[capa] += 1
        
        # Ahora se asume que la ultima neurona de cada capa no final es un sesgo
        
        self.neuronas_por_capa = neuronas_por_capa
        self.activacion_por_capa = activacion_por_capa
        self.derivada_por_capa = derivada_por_capa
        self.learning = learning
        
        # Inicializamos los pesos
        
        pesos = []
        
        for entre_capas in range(1,len(neuronas_por_capa)) :
            
            pesos_entre_capa = []
            
            # La ultima capa no tiene sesgo
            
            if entre_capas < len(neuronas_por_capa) - 1 :
            
                # Los sesgos (ultima neurona de la capa) no reciben pesos
            
                for neurona in range(neuronas_por_capa[entre_capas] - 1 ) : 
                    
                    # Aqui si se incluye el sesgo de la capa anterior
                    
                    pesos_neurona = [0.5 for entrada in range(neuronas_por_capa[entre_capas - 1])]
                    pesos_entre_capa.append(pesos_neurona)
            
            elif entre_capas == len(neuronas_por_capa) - 1 :
            
                for neurona in range(neuronas_por_capa[entre_capas]) :
                    
                    # Aqui si se incluye el sesgo de la capa anterior
            
                    pesos_neurona = [0.0 for entrada in range(neuronas_por_capa[entre_capas - 1])]
                    pesos_entre_capa.append(pesos_neurona)
            
            pesos.append(pesos_entre_capa)
        
        self.pesos = pesos
        #pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(pesos)
        
    def propagate_forward(self, sample_input) :
        """
        ENTRADA
            sample_input : fila del dominio del training set con bias al final
            self.pesos : pesos inicializados
        SALIDA
            return : salida de la ultima capa
        """
        salidas = []
        entrada = sample_input
        salidas.append(list(sample_input))
        
        for capa, pesos_capa in enumerate(self.pesos) :
            
            salidas_capa = []
            funcion_activacion = self.activacion_por_capa[capa]
            
            
            for pesos_neurona in pesos_capa :
                
                salida_neurona = funcion_activacion(np.dot(pesos_neurona, entrada))
                
                salidas_capa.append(salida_neurona)
                
            # El sesgo no tiene entradas segun self.pesos
            # La ultima capa no tiene sesgos
            
            if capa < (len(self.pesos)-1) :
            
                # Sesgo
                
                salidas_capa.append(-1.0)
            
            salidas.append(salidas_capa)
            
            entrada = salidas_capa
                
        
        #print("SALIDAS: ")
        pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(salidas)
        
        self.salidas = salidas
        
        return salidas

    def predecir(self, sample_input) :
        """
        ENTRADA
            sample_input : fila del dominio del training set con bias al final
            self.pesos : pesos inicializados
        SALIDA
            return : salida de la ultima capa
        """
        salidas = []
        entrada = sample_input
        salidas.append(list(sample_input))
        
        for capa, pesos_capa in enumerate(self.pesos) :
            
            salidas_capa = []
            funcion_activacion = self.activacion_por_capa[capa]
            
            
            for pesos_neurona in pesos_capa :
                
                salida_neurona = funcion_activacion(np.dot(pesos_neurona, entrada))
                
                salidas_capa.append(salida_neurona)
                
            # El sesgo no tiene entradas segun self.pesos
            # La ultima capa no tiene sesgos
            
            if capa < (len(self.pesos)-1) :
            
                # Sesgo
                
                salidas_capa.append(-1.0)
            
            salidas.append(salidas_capa)
            
            entrada = salidas_capa
                
        
        #print("SALIDAS: ")
        pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(salidas)
        
        #self.salidas = salidas
        
        return salidas_capa
  
    def propagate_backwards(self, sample_output) :
        """
        ENTRADA
            sample_output : salida esperada den training set para una fila
        SALIDA
            self.errores
        """
        
        #print("ANTES: ")
        #pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(self.pesos)
        
        sample_output = list(sample_output)
        
        gradientes = deque()
        
        # Capa de salida
        
        error_salida = []
        error_cuadratico_medio = 0.0
        
        derivada_activacion = self.derivada_por_capa[-1]
        
        for salida, valor_salida in enumerate(self.salidas[-1]) :
            
            error = sample_output[salida] - valor_salida
            
            estimulo_neurona = np.dot(self.salidas[-2],self.pesos[-1][salida])
            
            gradiente_neurona = error * derivada_activacion(estimulo_neurona)
            
            error_salida.append(gradiente_neurona)
            
            error_cuadratico_medio += error ** 2
            
            # Actualizar pesos
            
            #for pos, peso_salida in enumerate(self.pesos[-1]) :
            #    
            #    delta = self.learning * gradiente_neurona * self.salidas[-2][pos]
            #    print(delta)
            #    self.pesos[-1][pos] += delta
        
        gradientes.appendleft(error_salida)
        
        error_cuadratico_medio /= len(self.salidas[-1]) 
        
        self.ultimo_error = error_cuadratico_medio
        
        # Capas Ocultas
        
        for capa_oculta in range(len(self.salidas) - 2,0,-1) :
            
            gradientes_derechos = gradientes[0]
            
            derivada_activacion = self.derivada_por_capa[capa_oculta]
            
            error_salida = []
            
            for neurona in range(len(self.salidas[capa_oculta])-1) :
                
                estimulo_neurona = np.dot(self.salidas[capa_oculta - 1], self.pesos[capa_oculta-1][neurona])
                
                suma = np.dot(gradientes_derechos, [peso[neurona] for peso in self.pesos[capa_oculta]])
                
                gradiente_neurona = derivada_activacion(estimulo_neurona) * suma
                
                error_salida.append(gradiente_neurona)
                
                # Actualizar Pesos
                
                #for pos, peso_salida in enumerate(self.pesos[capa_oculta]) :
                #    
                #    delta = self.learning * gradiente_neurona * self.salidas[capa_oculta-1][pos]
                #    
                #    self.pesos[capa_oculta-1][pos] += delta
                    
                    
                
            gradientes.appendleft(error_salida)
        
        # Actualizar pesos
        
        for num_entre_capas, entre_capas in enumerate(self.pesos) :
            
            for num_neurona, neurona in enumerate(entre_capas) :
                
                for num_peso_entrada, peso_entrada in enumerate(neurona) :
                    
                    delta = \
                     self.learning * gradientes[num_entre_capas][num_neurona] * self.salidas[num_entre_capas][num_peso_entrada]
                    
                    self.pesos[num_entre_capas][num_neurona][num_peso_entrada] += delta
        
        #print("DESPUES: ")
        #pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(self.pesos)

    def entrenar(self, x, y, x_test, y_test, epocas = 700) :
        """
        Entrena el modelo con x e y. Luego lo valida con x_test e y_test.
        
        Se genera un gráfico de los datos y el modelo aprendido
        """
        
        print("Entrenamiento: ")
        
        self.error_entrenamiento = []
        self.error_prueba = []
        diez_malos = 0
        
        epoca_fin = epocas
        
        for epoch in range(epocas) :
        
            print("Epoca: ", epoch)
            
            #indices = np.random.permutation(len(x))
            
            for num_fila in range(len(x)) :
            
                self.propagate_forward(x[num_fila])
                self.propagate_backwards(y[num_fila])
            
            self.error_entrenamiento.append(self.ultimo_error)
            print("Error de entrenamiento: ", self.ultimo_error)
            self.probar(x_test, y_test)
            
            if len(self.error_prueba) > 1 and self.error_prueba[-1] > self.error_prueba[-2] :
                diez_malos += 1
            else :
                diez_malos = 0
            
            if diez_malos >= 10 :
                epoca_fin = epoch+1
                break
        
        
        plt.plot(x_test[:,0],y_test[:,0],label="Verdadera función")
        plt.plot(x_test[:,0],self.ultima_prueba,label="Modelo de la última validación")
        plt.title("Función "+str(self.neuronas_por_capa[1]-1))
        plt.xlabel("Muestra")
        plt.ylabel("Rango")
        plt.legend(loc=4, framealpha=0.6)
        plt.savefig("Funcion_"+str(self.neuronas_por_capa[1]-1)+" epocas_usadas_"+str(epoca_fin)+"_.jpg")
        
        plt.figure()
        
        plt.plot(range(epoca_fin), self.error_entrenamiento, label="Entrenamiento")
        plt.plot(range(epoca_fin), self.error_prueba, label="Validación")
        min_p = self.error_prueba[np.argmin(self.error_prueba)]
        plt.plot(range(epoca_fin), [min_p for i in range(epoca_fin)], label="Error mínimo de validación")
        plt.ylim(0.0, 0.1)
        plt.yticks(np.arange(min(self.error_prueba), max(self.error_prueba), 0.01))
        plt.title("Error "+str(self.neuronas_por_capa[1]-1))
        plt.xlabel("Épocas")
        plt.ylabel("MSE")
        plt.legend(loc=4, framealpha=0.6)
        plt.savefig("Error_"+str(self.neuronas_por_capa[1]-1)+" epocas_usadas_"+str(epoca_fin)+"_.jpg")
        
        plt.figure()
        
    def probar(self, x, y) :
        
        error_cuadratico_medio = 0.0
        self.ultima_prueba = []
        
        
        for num_fila in range(len(x)) :
            
            salida_final = self.predecir(x[num_fila])
            
            error_cuadratico_medio_ = 0.0
            
            for num_salida in range(len(y[num_fila])) :
                
                error_cuadratico_medio_ += (salida_final[num_salida]-y[num_fila][num_salida])**2
            
            error_cuadratico_medio_ /= len(y[num_fila])
            
            error_cuadratico_medio += error_cuadratico_medio_
            
            self.ultima_prueba.append(salida_final[0]) # 1dim
            
        error_cuadratico_medio /= len(y)
        
        self.error_prueba.append(error_cuadratico_medio)
        print("Error cuadratico medio test: ",error_cuadratico_medio)


    
    
