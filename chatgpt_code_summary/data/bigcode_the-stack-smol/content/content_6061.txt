# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 20:21:07 2019
Tecnológico Nacional de México (TECNM)
Tecnológico de Estudios Superiores de Ixtapaluca (TESI)
División de ingeniería electrónica
Introducción a la librería Numpy 2
M. en C. Rogelio Manuel Higuera Gonzalez
"""
import numpy as np
##################################################################################
ages = np.array([34,14,37,5,13]) #Crea un arreglo de edades
sorted_ages = np.sort(ages) #Acomoda los elementos del arreglo ages del menor al mayor
#ages.sort() #Acomoda los elementos del arreglo original ages del menor al mayor
argages = ages.argsort() #Indica el indice que clasifica a cada uno de los elementos del arreglo ages (del menor al mayor)
ages1 = ages[ages.argsort()] #Crea un arreglo ages ordenado dependiendo de su indice
##################################################################################
persons = np.array(['Johnny','Mary','Peter','Will','Joe'])
heights = np.array([1.76,1.2,1.68,0.5,1.25])
sort_indices = np.argsort(ages) #Realiza una clasificación basada en edades
#print(persons[sort_indices]) #Imprime la lista de personas clasificadas por su edad
#print(heights[sort_indices]) #Imprime la lista de altura clasificadas por su esdad
#print(ages[sort_indices]) #Imprime la lista de edad clasificadas por su edad
sort_indices1 = np.argsort(persons)
#print(persons[sort_indices1])
#print(ages[sort_indices1])
#print(heights[sort_indices1])
#Para ordenar en orden desendente las estaturas usar la notación en Python [::-1]
sort_indices2 = np.argsort(heights)[::-1]
#print(persons[sort_indices2])
#print(ages[sort_indices2])
#print(heights[sort_indices2])
##################################################################################
list1 = [[1,2,3,4],[5,6,7,8]]
a1 = np.array(list1)
a2 = a1
a2[0][0] = 11 #Hacer un cambio en a2 afecta  a a1
a1.shape = 1,-1 #a2 tambien cambia su forma
##################################################################################
list2 = [[10,11,12,13],[14,15,16,17]]
a3 = np.array(list2)
a4 = a3.view() #Copia superficial, cuando cambias la forma de a3, a4 no es afectado
a3.shape = 1,-1
##################################################################################
list3 = [[20,21,22,23],[24,25,26,27]]
a5 = np.array(list3)
a6 = a5.copy() #La función copy() crea una copia profunda del arreglo 
a5[0][0] = 10 #El cambio no es reflejado en a6
a5.shape = 1,-1 #a6 no cambia su forma 

