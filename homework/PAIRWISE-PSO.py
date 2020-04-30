# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:48:04 2020

@author: EDUARDO
"""
from datetime import *
import random,os,copy,time
import itertools
import numpy as np
from itertools import chain, combinations, product

#VARIABLES GLOBALES
### lista de factores con sus respectivos niveles
#set_factor = [['a1','a2','a3'],['b1','b2','b3'],['c1','c2','c3'],['d1','d2','d3']] #1
#set_factor = [['Master','Visa'],['Iphone','Blackberry'],['Iplanet','Apache'],['Chome','Explorer','Firefox'],['SQL','Oracle','Acces']]#2
set_factor = [["Destacados","Recientes","Menor Precio","Mayor Precio", "Nombre A-Z"],
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
              ["0-50","51-150","151-300","301-1000","Más de 1000"]]
### lista de soluciones
TS = []
### todos los test pairwise generados
Allpairwise = []
#limite de nivel de cada factor
#li = [3,3,3,3] #1
#li=[2,2,2,3,3]
li =[5,2,2,2,2,2,2,2,2,2,2,8,5] #2
#numero de factores = # Di
#k = 4 # 1
#k=5
k=13
#### parametros para actualizar velocidad y posicion
w = 0.9
c1 = c2 = 2
r1 = np.random.uniform(0,1)
r2 = np.random.uniform(0,1)
random.seed(442)

class particle:    
    def __init__(self):
        self.k= k            
        self.velocity = []
        self.position = []
        self.pbesti = []
    #actualizar la velocidad
    def update_velocity(self,newvel):
        self.velocity= newvel
    # actualizar la posicion
    def update_position(self,newposit):
        self.position= newposit
    # actualizar la mejor posicion local de la particula
    def update_pbesti(self,newpbesti):
        self.pbesti= newpbesti
    # generar una lista con las velocidades
    def generate_listVelocity(self):
        return self.velocity
    ## generar una lista con posiciones
    def generate_listPosition(self):
        return self.position
    def generate_listbestPosition(self):
        return self.pbesti
    
#generar un test en base a una lista de posiciones
def derivateTest_x(x):
    testp = []
    for i in range(0,k):
        m = int(x[i])-1
        testp.append(set_factor[i][m])
    return testp

## funcion regularizadora para la posicion de la particula no salga del limite para cada componente de una nueva posicion
def f(x,i): 
    if x > li[i]:
        return li[i]
    elif x < 1:
        return 1
    else:
        return x

# funcion para ajustar el valor de la velocidad segun formula: trabaja con un componente determinado
def ajustvel(v,i): 
    if v < -li[i]:
        return -li[i]
    elif v > li[i]:
        return li[i]
    else:
        return v

##funcion para generar todos los posibles pairwise en base al set_factor: lista de listas con posiciones none para rellenar
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

### funcion para generar el esquema o las combinatorias generada por una particula p
def schemap(positionp): # argumento lista
    testgeneradop = derivateTest_x(positionp)
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
    
#fitness para evaluar al mejor candidato
# TS es una lista de listas con los test ya definidos, empieza en nulo
'''def fitness(positionp):
    contador = 0
    if len(positionp) == 0:
        return 0
    else:
        schema, lenp = schemap(positionp)
        for k in schema:
            for ts in TS:
                if patronequivalente(k,ts):
                    contador+=1
        return lenp - contador'''
def fitness(positionp):
    contador = 0
    if len(positionp) == 0:
        return 0
    else:
        schema, lenp = schemap(positionp)
        for m in schema:
            if m in Allpairwise:
                 contador +=1
    return contador
#actualizar una lista con otra emepezando de null
def actualizarList(a,b): #listas 
    if len(a)==0:
        for i in b:
            a.append(i)
    else:
        for j in range(0,len(b)):
            a[j]=b[j]
    return a

##eliminar todas las combinaciones cubiertas por un test generado de una particula p
def dropCobinationCovering(testi):
    contador=0
    schema, lenp = schemaTest(testi)
    for element in schema:
        if element in Allpairwise:
            #print(True)
            Allpairwise.remove(element)
            contador +=1
    return contador

#funcion que genera aleatoriamente un simple test.
def single_test_pso(m):  
    NCmax = 30
    bestTest = None
    NC = 0
    #print(sss)
    particles_list= []
    for i in range(0,m):
        #generando y definiendo aleatoriamente la posicion de las m particulas
        vel_ini=[]
        post_ini=[]
        pbest_ini=[]
        p1 = None
        for j in range(0,k):
            vel_ini.append(random.randint(-li[j],li[j]))
            post_ini.append(random.randint(1,li[j]))
        p1 = particle() # cuidado
        p1.update_velocity(vel_ini)
        p1.update_position(post_ini)
        p1.update_pbesti(post_ini)
        #print(p1.generate_listVelocity())
        particles_list.append(p1)
    #gBest=[0,0,0,0]
    #gBest=[0,0,0,0,0]
    gBest = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    while NC < NCmax:
        #evaluar el fitness de las particulas
        for i in range(0,m):
            pp = particles_list[i].generate_listPosition()
            pb = particles_list[i].generate_listbestPosition()
            if fitness(pp) > fitness(pb):
                particles_list[i].update_pbesti(pp)
            if fitness(pp) > fitness(gBest):
                gBest= pp
            if fitness(pp) == k*(k-1)/2:
                bestTest = derivateTest_x(pp)
                return bestTest
        #actualizar velocidad y posicion de las particulas
        for i in range(0,m):
            for s in range(0,k):
                particles_list[i].velocity[s] = (w*particles_list[i].velocity[s]  + 
                                                     c1*r1*(particles_list[i].pbesti[s] - particles_list[i].position[s])+ 
                                                     c2*r2*(gBest[s]- particles_list[i].position[s]))
                particles_list[i].velocity[s] = ajustvel(particles_list[i].velocity[s],s)
        for i in range(0,m):
            for s in range(0,k):
                particles_list[i].position[s] = particles_list[i].position[s] + particles_list[i].velocity[s]
                particles_list[i].position[s] = f(particles_list[i].position[s],s)        
        NC = NC + 1
    bestTest = derivateTest_x(gBest)
    return bestTest # retornar test con todos los factores corregidos


#funcion que genera todos los datos
def PSO():
    #print("Combinaciones Pairwise iniciales:")
    #print(len(Allpairwise))
    #i=0
    neliminados=0
    time_star = time.time()
    while len(Allpairwise)>0:
        testi = single_test_pso(80)
        neliminados = dropCobinationCovering(testi)
        if neliminados > 0:
            TS.append(testi)
            #print(len(Allpairwise))
       # i=i+1;
    time_final = time.time()-time_star
    print("Conbinaciones Conjunto Final TS:")
    print(len(TS))
    for test in TS:
        print(test)
    print(f"Tiempo final {time_final:.10f} segundos")

Allpairwise = generatorPairwiseSet(set_factor)

#llamando a la funcion principal
PSO()


    
