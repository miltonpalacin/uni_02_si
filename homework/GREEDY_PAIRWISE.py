# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:06:15 2020

@author: EDUARDO
"""
from datetime import *
import random,os,copy,time
import itertools
import numpy as np
from itertools import chain, combinations, product

#VARIBALES GLOBALES
set_factor = [['a1','a2','a3'],['b1','b2','b3'],['c1','c2','c3'],['d1','d2','d3']]
#set_factor = [['Master','Visa'],['Iphone','Blackberry'],['Iplanet','Apache'],['Chome','Explorer','Firefox'],['SQL','Oracle','Acces']]
'''set_factor = [["Destacados","Recientes","Menor Precio","Mayor Precio", "Nombre A-Z"],
              ["Portabilidad","Linea Nueva"],
              ["Aple_SI","Aple_NO"],
              ["Bmovile_SI","Bmovile_NO"],
              ["EKS_SI","EKS_NO"],
              ["Huawey_SI","Huawey_NO"],
              ["LG_SI","LG_NO"],
              ["Motorala_SI","Motorala_NO"],
              ["Movistar_SI","Movistar_NO"],
              ["Xiaomy_SI","Xiaomy_NO"],
              ["ZTE_SI","ZTE_NO"],
              ["35.9","45.9","55.9","65.9","75.9","89.9","109.9","149.9"],
              ["0-50","51-150","151-300","301-1000","Más de 1000"]]'''
              
Allcombinations = []
Allpairwise = []
TS = []
k = 4
#k=5
#k=13
random.seed(442)

#genera todas las combinaciones de opciones
def generateAllCombinations():
    newlist = []
    for x in eval('itertools.product'+str(tuple(set_factor))):
        lista = list(x)
        newlist.append(lista)
    return newlist

#compara dos lista para ver si tienen al menos 2 elementos en la misma posición para hacer el fitness
def patronequivalente(a,b): # argumento listas
    contador = 0
    for k in range(len(a)):
        if a[k] == b[k]:
            contador+=1
    if contador==2:
        return True
    else:
        return False

### funcion para generar el esquema generadas por un test
def schemaTest(testgeneradop): # argumento lista
    lista1_generados = []
    for con in itertools.combinations(testgeneradop,2):
        p1 = con[0]
        p2 = con[1]
        componente=[]
        for s in range(0,k):
            componente.append(None)
        indexp1 = testgeneradop.index(p1)
        indexp2 = testgeneradop.index(p2)
        componente[indexp1] = p1
        componente[indexp2] = p2
        lista1_generados.append(componente)
    return lista1_generados, len(lista1_generados)

#genera las cobinaciones 2-way
def generatorPairwiseSet(sequences):
    listpairwise = []
    nseen = list(chain.from_iterable(product(*i) for i in combinations(sequences, 2)))
    for pair in nseen:
        pair1 = pair[0]
        pair2 = pair[1]
        pairunfixed = []
        for factor in sequences:
            if pair1 in factor:
                pairunfixed.append(pair1)
            elif pair2 in factor:
                pairunfixed.append(pair2)
            else:
                pairunfixed.append(None)
        listpairwise.append(pairunfixed)
    return listpairwise

#Elminar combinaciones generadas por un test de la lista general
def dropCobinationCovering(testi):
    schema, lenp = schemaTest(testi)
    for element in schema:
        if element in Allpairwise:
            #print(True)
            Allpairwise.remove(element)
    Allcombinations.remove(testi)

#generar un conjunto aleatorio de candidatos    
def generateCandidates(m):
    PS = random.sample(Allcombinations,m)
    return PS

#medir la calidad de un candidato
def calidad(testi):
    contador = 0
    schema, lenp = schemaTest(testi)
    for m in schema:
        if m in Allpairwise:
             contador +=1
    return contador

#seleccionar al mejor candidato en base a la calidad
def selectBestCandidate(PS):
    best = PS[0]
    for k in PS:
        if calidad(k) > calidad(best):
            best = k
    #print(calidad(best))
    return best

#funcion prinicipal donde se aplica greedy
def greedy_pairwise(m):
    time_star = time.time()
    while len(Allpairwise)> 0:
        PS = generateCandidates(m)
        best = selectBestCandidate(PS)
        TS.append(best)
        dropCobinationCovering(best)
    time_final = time.time()-time_star
    print("Soluciones")
    print(len(TS))
    #pintando datos
    for test in TS:
        print(test)
    print(f"Tiempo final {time_final:.10f} segundos")
#llamando a ejecucion
Allcombinations = []
Allpairwise = []
TS = []
Allcombinations = generateAllCombinations()
Allpairwise = generatorPairwiseSet(set_factor)
greedy_pairwise(70) #200
#print(len(Allcombinations))



