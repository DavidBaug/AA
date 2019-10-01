# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:57:27 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy.lib import scimath

#np.random.seed(1)
#random.seed(1)


def label_data(x1, x2):
    y = np.sign((x1-0.2)**2 + x2**2 - 0.6)
    idx = np.random.choice(range(y.shape[0]), size=(int(y.shape[0]*0.1)), replace=True)
    y[idx] *= -1
    
    return y


def simula_unif(N=2, dims=2, size=(0, 1)):
    m = np.random.uniform(low=size[0], 
                          high=size[1], 
                          size=(N, dims))
    
    return m


def simula_gaus(size, sigma, media=None):
    media = 0 if media is None else media
    
    if len(size) > 2:
        N = size[0]
        size_sub = size[1:]
        
        out = np.zeros(size, np.float64)
        
        for i in range(N):
            out[i] = np.random.normal(loc=media, scale=sigma, size=size_sub)
    
    else:
        out = np.random.normal(loc=media, scale=sigma, size=size)
    
    return out


def simula_recta(intervalo=(-1,1), ptos = None):
    if ptos is None: 
        m = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    
    a = (m[0,1]-m[1,1])/(m[0,0]-m[1,0]) # Calculo de la pendiente.
    b = m[0,1] - a*m[0,0]               # Calculo del termino independiente.
    
    return a, b


def line2coef(a, b):
    w = np.zeros(3, np.float64)
    #w[0] = a/(1-a-b)
    #w[2] = (b-b*w[0])/(b-1)
    #w[1] = 1 - w[0] - w[2]
    
    #Suponemos que w[1] = 1
    w[0] = a
    w[1] = 1.0
    w[2] = b
    
    return w


def plot_datos_recta(X, y, a, b, title='Point clod plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    w = line2coef(a, b)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], -a*grid[:, 0]-b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    ax.legend()
    plt.title(title)
    plt.show()


###############################################################################

##############################  EJERCICIO 1   #################################

###############################################################################
    
#Funcion que añade una etiqueta dependiendo de la distancia del punto a la recta
def aniade_etiquetas(recta, X, Y):
    w = []
    for x,y in zip(X,Y):
        etiqueta = np.sign(y-(recta[0]*x)-recta[1])
        w.append([x,y,etiqueta])
        
    
    return np.asarray(w)
    
#Funcion que cambia el porcentaje de etiquetas de un array
def aniade_ruido(w, percent=10):
    if percent < 0 or percent > 100:
        percent = 10
        print('Percent set to 10%')
    
    c1 = int(np.ceil((w == 1).sum()*(percent/100)))
    c2 = int(np.ceil((w == -1).sum()*(percent/100)))
    
    subarray = w[:,2]
    
    index = np.argsort(subarray)
    
    res = w[index]
    
    #print("sorted w: ",res)
    
    res[:c2,2]*=-1
    res[-c1:,2]*=-1
        
    #print("sorted w: ",res)
    
    return res


    
def funcion1(x):
    return np.asarray(np.float64(20-np.sqrt(-(x**2)+20*x+300)))

def funcion11(x):
    return np.asarray(np.float64(np.sqrt(-(x**2)+20*x+300)+20))

def funcion2(x):
    return np.asarray(np.float64((1/2)*(40-(np.sqrt(2)*np.sqrt(-(x**2)-20*x+700)))))

def funcion22(x):
    return np.asarray(np.float64((1/2)*(40+(np.sqrt(2)*np.sqrt(-(x**2)-20*x+700)))))

def funcion3(x):
    return np.asarray(np.float64(-1*((np.sqrt(x**2-20*x-700)/np.sqrt(2)))-20))

def funcion33(x):
    return np.asarray(np.float64(((np.sqrt(x**2-20*x-700)/np.sqrt(2)))-20))

def funcion4(x):
    return np.asarray(np.float64(20*(np.power(x,2))+(5*x)-3))

# .............................................................................

#Creamos datos para ejercicio 1.2

N = 444
dim = 2
rango = [-50,50]

recta = []

a, b = simula_recta()

recta.append(a)
recta.append(b)

del a
del b


unif = simula_unif(N, dim, rango)

x = unif[:,0]
y = unif[:,1]    

w = aniade_etiquetas(recta, x,y)


w_ruido = aniade_ruido(w.copy(), 10)


def ejercicio1_1():
    print()
    print ("--------------------------- EJERCICIO 1 -----------------------------")
    print()    
        
    # Apartado 1
    # -----------------------------------------------------------------------------
        
    print ("---------------------------- APARTADO 1 -----------------------------")
    print ("---------------------------------------------------------------------")
    print()
    
    #N = 444
    N = 50
    dim = 2
    rango = [-50,50]
    sigma = [5,7]
        
    N1 = (50,2)
    #N1 = (444,2)
        
    unif = simula_unif(N, dim, rango)
    print('Distribución Uniforme creada con N ={}, dim={}, y rango = {}'.format(N,dim,rango))
    
    
    x = unif[:,0]
    y = unif[:,1]
    
    
    plt.clf()
    plt.title("Distribución Uniforme")
    plt.scatter(x, y,c=y, s=50, linewidth=2, cmap="rainbow_r", edgecolor='white', label='Distribución Uniforme')
    plt.show()
    
    
    gaus = simula_gaus(N1, dim, sigma)
    print('Distribución Gaussiana creada con N ={}, dim={}, y sigma = {}'.format(N,dim,sigma))
    
    x = gaus[:,0]
    y = gaus[:,1]
    
    plt.clf()
    plt.title("Distribución Gaussiana")
    plt.scatter(x, y,c=y, s=50, linewidth=2, cmap="rainbow_r", edgecolor='white', label='Distribución Gaussiana')
    plt.show()
    

def ejercicio1_2(w,w_ruido,recta):
    
    # Apartado 2
    # -----------------------------------------------------------------------------
    print()
    print ("---------------------------- APARTADO 2 -----------------------------")
    print ("---------------------------------------------------------------------")
    print()
    
        
    print('Recta')
    print ("a : ", recta[0])
    print ("b : ", recta[1])
    
    
    print()
    print("Creado conjunto de distribución uniforme con etiquetas medidas mediante la distancia del punto a la recta.")
    print()
    
    print('w : ', w)
    
    #print ("w : ",w[:,:2])
    #print ("w etiquetas: ", (w[:,2]))
    
    label = w.copy()
    label = label[:,2]
    label[(label == -1)] = 0
    
    print('w label : ', w)
    
    xx = np.arange(np.floor(np.min(w[:,0])),np.ceil(np.max(w[:,0])))
    yy = recta[0]*xx+recta[1]
    
    plt.clf()
    plt.plot(xx,yy,'k')
    plt.axis([np.min(w[:,0]),np.max(w[:,0]),np.min(w[:,1]),np.max(w[:,1])])
    plt.title('Clasificación sin ruido')
    plt.scatter(w[:,0], w[:,1], c=label, cmap='rainbow_r', edgecolor='white', label='Clasificación sin ruido')
    plt.show()
    
    
    
    print()
    print()
    print('Añadimos 10% de ruido a las muestras')
    print()
    print('w : ', w_ruido)
    
    
    label = w_ruido.copy()
    label = label[:,2]
    label[(label == -1)] = 0
    
    xx = np.arange(np.floor(np.min(w_ruido[:,0])),np.ceil(np.max(w_ruido[:,0])))
    yy = recta[0]*xx+recta[1]
    
    plt.clf()
    plt.plot(xx,yy,'k')
    plt.axis([np.min(w_ruido[:,0]),np.max(w_ruido[:,0]),np.min(w_ruido[:,1]),np.max(w_ruido[:,1])])
    plt.title('Clasificación con 10% ruido')
    plt.scatter(w_ruido[:,0], w_ruido[:,1], c=label, cmap='rainbow_r', edgecolor='white', label='Clasificación con 10% ruido')
    plt.show()
    
    
    
    
def ejercicio1_3(w,w_ruido,recta):
    
    # Apartado 3
    # -----------------------------------------------------------------------------
    print()
    print ("---------------------------- APARTADO 3 -----------------------------")
    print ("---------------------------------------------------------------------")
    print()
    
    
    print('Recta')
    print ("a : ", recta[0])
    print ("b : ", recta[1])
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
        
    print('A partir de los puntos iniciales clasificamos según las otras funciones.')
    
    print()
    print('Primera función de evaluación')
    
    #Escogemos la primera caracteristtca
    x = np.asarray(np.arange(np.floor(np.min(w_ruido[:,0])),np.ceil(np.max(w_ruido[:,0]))),dtype=np.float64)
    
    label = w_ruido.copy()
    label = label[:,2]
    label[(label == -1)] = 0
    
    plt.clf()
    plt.plot(x,[funcion1(i) for i in x],'k')
    plt.plot(x,[funcion11(i) for i in x],'k')
    plt.axis([np.min(w_ruido[:,0]),np.max(w_ruido[:,0]),np.min(w_ruido[:,1]),np.max(w_ruido[:,1])])
    plt.title('Primera función')
    plt.scatter(w_ruido[:,0], w_ruido[:,1], c=label, cmap='rainbow_r', edgecolor='white', label='Primera función')
    plt.show()
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    print()
    print('Segunda función de evaluación')
    
    label = w_ruido.copy()
    label = label[:,2]
    label[(label == -1)] = 0
    
    plt.clf()
    plt.plot(x,[funcion2(i) for i in x],'k')
    plt.plot(x,[funcion22(i) for i in x],'k')
    plt.axis([np.min(w_ruido[:,0]),np.max(w_ruido[:,0]),np.min(w_ruido[:,1]),np.max(w_ruido[:,1])])
    plt.title('Segunda función')
    plt.scatter(w_ruido[:,0], w_ruido[:,1], c=label, cmap='rainbow_r', edgecolor='white', label='Segunda función')
    plt.show()
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    print()
    print('Tercera función de evaluación')
    
    
    label = w_ruido.copy()
    label = label[:,2]
    label[(label == -1)] = 0
    
    plt.clf()
    plt.plot(x,[funcion3(i) for i in x],'k')
    plt.plot(x,[funcion33(i) for i in x],'k')
    plt.axis([np.min(w_ruido[:,0]),np.max(w_ruido[:,0]),np.min(w_ruido[:,1]),np.max(w_ruido[:,1])])
    plt.title('Tercera función')
    plt.scatter(w_ruido[:,0], w_ruido[:,1], c=label, cmap='rainbow_r', edgecolor='white', label='Tercera función')
    plt.show()
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    print()
    print('Cuarta función de evaluación')
    
    label = w_ruido.copy()
    label = label[:,2]
    label[(label == -1)] = 0
    
    plt.clf()
    plt.plot(x,[funcion4(i) for i in x],'k')
    plt.axis([np.min(w_ruido[:,0]),np.max(w_ruido[:,0]),np.min(w_ruido[:,1]),np.max(w_ruido[:,1])])
    plt.title('Cuarta función')
    plt.scatter(w_ruido[:,0], w_ruido[:,1], c=label, cmap='rainbow_r', edgecolor='white', label='Cuarta función')
    plt.show()
    
    plt.clf()
    plt.plot(x,[funcion4(i) for i in x],'k')
    #plt.axis([np.min(w_ruido[:,0]),np.max(w_ruido[:,0]),np.min(w_ruido[:,1]),np.max(w_ruido[:,1])])
    plt.title('Cuarta función')
    plt.scatter(w_ruido[:,0], w_ruido[:,1], c=label, cmap='rainbow_r', edgecolor='white', label='Cuarta función')
    plt.show()
    
    print()
    
###############################################################################

##############################  EJERCICIO 2  ##################################

###############################################################################

def plot_data(X, y, w, titulo):
    #Preparar datos
    a, b = coef2line(w)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel='Intensidad promedio', ylabel='Simetria')
    ax.legend()
    plt.title(titulo)
    plt.show()
    
    
def coef2line(w):
    if(len(w)!= 3):
        raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')
    
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    
    return a, b


# ------------------------------------------------------------------------


def ajusta_PLA(datos, label, max_iter, vini):                   
        
    w = vini.copy()
    
    iteraciones = max_iter
    
    for i in range(max_iter):
        w_copy = w.copy()
        
        
        for X,y in zip(datos,label):         
            
            if(np.sign(np.dot(X,w)) != y):
                w += X*y
                
        if np.all(w_copy == w):
            iteraciones = i+1
            break
            
    print('El número de iteraciones ha sido {}'.format(iteraciones))
    return w, iteraciones
    
    
def classificationError(w, X, y):
    h = np.sign(np.dot(X,w))
    
    fails = np.int(0)
    
    for h, y in zip(h,y):
        if (h != y):
            fails += 1
    
    print ("De un conjunto de datos de {} elementos ha fallado en predecir {}".format(len(X),fails))
    print ("ErrorClasificacion = {}".format((fails/len(X))))

    
def ejercicio2_1(w, w_ruido, recta):
    print()
    print ("--------------------------- EJERCICIO 2 -----------------------------")
    print()
    
    # Apartado 1
    # -----------------------------------------------------------------------------
    print()
    print ("--------------------------- APARTADO 1A -----------------------------")
    print()
    
        
    print('Recta')
    print ("a : ", recta[0])
    print ("b : ", recta[1])
    
    X = w.copy()
    X[:,2] = 1
    
    y = w.copy()
    y = y[:,2]
    
    #print('x:',X)
    
    
    model = np.array([0,0,0],np.float64)
    print('Con vector inicial = ', model)
    print()
    
    model, ite = ajusta_PLA(X,y, 100, model)
    print('Modelo obtenido = ', model)
    print()
    
    classificationError(model, X, y)
    
    plot_data(X,y,model,'PLA muestra sin ruido')    
    plt.clf()
    
    print()
    print()
    print()
    print ("--------------------------- APARTADO 1B -----------------------------")
    print()
    
    iters = []
    
    pruebas = 10
    
    for i in range(pruebas):
        print()
        np.random.seed(i)
        a = np.random.random()
        b = np.random.random()
        c = np.random.random()
        
        vini = [a,b,c]
        
        print('Con vector inicial {} = {}'.format(i, vini))
        
        model, ite = ajusta_PLA(X,y, 100, vini)
        iters.append(ite)
        
        
        print('Modelo obtenido = ', model)
        
        print()
        classificationError(model, X, y)
        print('----------------------------------------------------------------')
        

    print("Para {} vectores aleatorios obtenemos una media de iteraciones de {}".format(pruebas,np.mean(iters)))

    #print('v ini:', v)
    
    print()
    print()
    print()
    print ("--------------------------- APARTADO 1C -----------------------------")
    print()
    
    X = w_ruido.copy()
    X[:,2] = 1
    
    y = w_ruido.copy()
    y = y[:,2]
    
    #print('x:',X)
    
    
    model = np.array([0,0,0],np.float64)
    print('Con vector inicial = ', model)
    print()
    
    model, ite = ajusta_PLA(X,y, 100, model)
    print('Modelo obtenido = ', model)
    print()
    
    classificationError(model, X, y)
    
    plt.clf()
    plot_data(X,y,model,'PLA muestra con 10% ruido')
    plt.clf()
    
    
###############################################################################    
    
def regresion_logistica(model, X_train, y_train, iters, minibatch_size, n=0.01):
    
    model = np.array([0,0,0],np.float64)
    
    print("\nCalculando regresión logística para {} iteraciones. \nPlease, do not turn off your computer\n".format(iters))
  
    iteraciones = 0
    
    for i in range(iters):
        iteraciones = i+1
        
        # Permutation
        aux = list(zip(X_train,y_train))        
        random.shuffle(aux)
        X_train, y_train = zip(*aux)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        model_anterior = model.copy()            
        
        for j in range(0, X_train.shape[0], minibatch_size): #For minibatch
            X = X_train[j:j+minibatch_size]
            y = y_train[j:j+minibatch_size]
            
            model = minibatch_step(model, X, y, n)

    
        if np.sqrt(np.sum((model_anterior - model)**2)) < 0.01:
            break
    
    print('Modelo terminado con {} iteraciones'.format(iteraciones))                

    return model


    
#Función que calcula el gradiente de un conjunto de datos
def minibatch_step(w, X_train, y_train, n):
    for X,y in zip(X_train,y_train):
        w -= n*gradient(w,X,y)
                
    return w


# Función sigmoide para clasificar
def sigmoide(x):
    print('sigma: ',1/1+(np.exp(-x)))
    return 1/1+(np.exp(-x))

def gradient(w, X, y):
    return (-y*X)/(1+np.exp(y*X.dot(w)))
    #return (-y*X)/(1+np.exp(y*X*w))
    #return np.float64(2/len(X))*X*((w.dot(X))-y)


def error_logistica(w, X, y):
    error = 0
    
    etiqueta = sigmoide(X.dot(w))
    
    print('etiqueta: ', etiqueta)
    
    etiqueta[(etiqueta > 0.5)] = 1    
    etiqueta[(etiqueta < 0.5)] = -1
    
    for e, y in zip(etiqueta, y):
        if e != y:
            error += 1
        
    print('Error obtenido = {}'.format(error/X.shape[0]))
        
    
    

###############################################################################

def ejercicio2_2():
    
    # Apartado 2
    # -----------------------------------------------------------------------------
    print()
    print ("--------------------------- APARTADO 2A -----------------------------")
    print()
    
    pts = simula_unif()
    
    a, b = simula_recta(intervalo=(0,2))
    
    print()
    print("Recta\na = {}\nb = {}\n".format(a,b))
            
    #Muestra de 100 puntos
    X = simula_unif(N=100, size=(0,2))
    
    
    w = aniade_etiquetas([a,b], X[:,0],X[:,1])
    
    label = w.copy()
    label = label[:,2]
    label[(label == -1)] = 0

    #print('w:',w)

    plt.clf()
    plt.title('Clasificación inicial')
    plt.plot(range(3), [ a*i+b for i in range(3)], 'k')
    plt.axis([np.min(w[:,0]),np.max(w[:,0]),np.min(w[:,1]),np.max(w[:,1])])
    plt.scatter(w[:,0], w[:,1], c=label, cmap='rainbow_r', edgecolor='white')
    plt.show()
    
    
    #Ajustar datos para entrenar
    X = w.copy()
    X[:,2] = 1
    
    y = w.copy()
    y = y[:,2]
    
    
    
    
    
    model = np.array([0,0,0],np.float64)
    
    print('Modelo inicial: ',model)
    
    iteraciones = 1000
    
    model = regresion_logistica(model, X, y, iteraciones, 10)
 
    print('Modelo tras entrenar: ',model)

    error_logistica(model, X, y)

    plot_data(X,y,model,'Regresión Logística')
    
    
        # Apartado 2
    # -----------------------------------------------------------------------------
    print()
    print ("--------------------------- APARTADO 2B -----------------------------")
    print()
    
    
    X = simula_unif(N=1234, size=(0,2))
    
    
    w = aniade_etiquetas([a,b], X[:,0],X[:,1])
    
    
    
    
    
# Ejecución ejercicios   
    
#ejercicio1_1()
#ejercicio1_2(w,w_ruido,recta)
#ejercicio1_3(w,w_ruido,recta)
#ejercicio2_1(w,w_ruido,recta)       
ejercicio2_2()       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        