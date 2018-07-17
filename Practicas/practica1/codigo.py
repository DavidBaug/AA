#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:41:51 2018

@author: David
"""

""" INFO

David Gil Bautista

1) Para ejecutar el código se recomienda ejecutar el código por celdas
2) He cambiado el número de iteraciones para que no tarde tanto. En la memoria
    he hecho todo con más iteraciones y he añadido ejemplos cambiando los datos.


"""


import numpy as np
import matplotlib.pyplot as plt
import random


#Fijación semilla aleatoria
np.random.seed(0)
random.seed(0)

###############################################################################

##############################  EJERCICIO 1   #################################

###############################################################################



print("\n")

print(" ------------------------------------------------- ")
print(" ------------------ EJERCICIO 1 ------------------ ")
print(" ------------------------------------------------- \n")

print("\n")


#Función de coste
def cost(function,w):
    return (function(w[-2],w[-1]))
   
    
#Función y derivadas parciales - apartado 1
    
def function(x,y):
    return np.float64((np.float64(x)**3*np.exp(np.float64(y)-2) - 4*np.float64(y)**3*np.exp(-np.float64(x)))**2)

def gradientU(u,v):
    return np.float64(2*(np.exp(-2 + np.float64(v))*np.float64(u)**3 - 4*np.exp(-np.float64(u))*np.float64(v)**3)*(3*np.exp(-2 + np.float64(v))*np.float64(u)**2 + 4*np.exp(-np.float64(u))*np.float64(v)**3))
    
def gradientV(u,v):
    return np.float64(2*(np.exp(-2 + np.float64(v))*np.float64(u)**3 - 12*np.exp(-np.float64(u))*np.float64(v)**2)*(np.exp(-2 + np.float64(v))*np.float64(u)**3 - 4*np.exp(-np.float64(u))*np.float64(v)**3))


#Gradiente de la función
def gradient(data, w):
    grU = gradientU(w[-2],w[-1])
    grV = gradientV(w[-2],w[-1])
    
    return np.array([grU,grV],np.float64)



#Función iterativa para el cálculo de los valores mínimos de la función
def gradientDescent(iters, stop, minError, boolMinError,  n, w, gradient, function, graph):
   
    #Inicializar lista de Error y lista con iteraciones
    E = [cost(function,w)]

    print("Initial w = ( {} , {} )".format(w[0],w[1]))
    
    for i in range (iters):
               
        w -= n*gradient(0,w)
        
        E.append(cost(function,w))
         
        if (E[-1] < 0):
            E[-1]*=-1
        
        if (E[-1] <= minError and boolMinError):
            print("Error: {} -> iter: {}".format(E[-1],i))
            print("Sol (u,v) = ( {} , {} ) with {} iterations and n = {}.\n".format(w[0],w[1],i,n))
            break
        
        if (i >= stop):
            print("Error: {} -> iter: {}".format(E[-1],i))
            print("Max iters({}) with n = {} -> Sol (u,v) = ( {} , {} )\n".format(stop,n,w[0],w[1]))
            break
    
    # plot2d iters + E
    if(graph):    
        x = range(0,len(E))
        plt.plot(x, E, label='n = {}'.format(n))   
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.legend()
        plt.title('Gradient Descent')
        
    del E
    
    return w
        



print(" ------------------------------------------------- ")
print(" --------------- PRIMER APARTADO ----------------- ")
print(" ------------------------------------------------- \n")



#Set datos

# Vector de pesos
w = np.array([1, 1], np.float64)

#Learning rate
n = np.float64(0.05)

# Min err es el error de parada, cuando el error sea menor a él se devuelve la w
minErr = 1e-14

#Numero de iteraciones
n_iters = 5000000

#Maximo de iteraciones
stop = 3000000
        
        
w = gradientDescent(n_iters, stop, minErr, bool(1), n, w, gradient, function, bool(0))



#%%


###############################################################################

############################  SEGUNDO APARTADO   ##############################

###############################################################################



print("\n")

print(" ------------------------------------------------- ")
print(" ----------------SEGUNDO APARTADO----------------- ")
print(" ------------------------------------------------- \n")


#Función y derivadas parciales - apartado 2
def function2(x,y):
    return np.float64((x - 2)**2 + 2*(y + 2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))


#derivada parcial de x
def gradientU2(x,y):
    return np.float64(4*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*x) + 2*(x - 2))
    
#derivada parcial de y
def gradientV2(x,y):
    return np.float64(4*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + y + 2)
    
#Gradiente de la función
def gradient2(data, w):
    grU = gradientU2(w[-2],w[-1])
    grV = gradientV2(w[-2],w[-1])
    
    return np.array([grU,grV],np.float64)

#Set datos

w = np.array([1, 1], np.float64)
w0 = np.array([1, 1], np.float64)


n1 = np.float64(0.01)
n2 = np.float64(0.1)

n_iters = 1000
stop = 50

# Min err es el error de parada, cuando el error sea menor a él se devuelve la w
minErr = 0

w = gradientDescent(n_iters, stop, minErr, bool(0), n1, w, gradient2, function2, bool(1))
w0 = gradientDescent(n_iters, stop, minErr, bool(0), n2, w0, gradient2, function2, bool(1))

plt.show()

del w, w0, n1, n2, n_iters, stop, minErr


#%%

""" Segundo apartado """

print(" ------------------------------------------------- ")
print(" ------------- Cambiando w inicial --------------- ")
print(" ------------------------------------------------- \n")


n1 = np.float64(0.01)
n2 = np.float64(0.1)

n_iters = 300000
stop = 299999


minErr = 1e-5

# Calculo del gradiente con distintos valores de entrada

w1 = np.array([2.1, -2.1], np.float64)
values1 = gradientDescent(n_iters, stop, minErr, bool(1), n2, w1.copy(), gradient2, function2, bool(0))


w2 = np.array([3, -3], np.float64)
values2 = gradientDescent(n_iters, stop, minErr, bool(1), n2, w2.copy(), gradient2, function2, bool(0))


w3 = np.array([1.5, 1.5], np.float64)
values3 = gradientDescent(n_iters, stop, minErr, bool(1), n2, w3.copy(), gradient2, function2, bool(0))


w4 = np.array([1, -1], np.float64)
values4 = gradientDescent(n_iters, stop, minErr, bool(1), n2, w4.copy(), gradient2, function2, bool(0))


#Tabla con las w finales 

tabla = []
tabla.append(values1.copy())
tabla.append(values2.copy())
tabla.append(values3.copy())
tabla.append(values4.copy())

print (" ----------------------------------------------------- \n")

print("Initial")


print ("w1 ( {} , {} )".format(w1[0],w1[1]))
print ("w2 ( {} , {} )".format(w2[0],w2[1]))
print ("w3 ( {} , {} )".format(w3[0],w3[1]))
print ("w4 ( {} , {} )\n".format(w4[0],w4[1]))

print("After gradient")

print ("w1 ( {} , {} )".format(tabla[0][0],tabla[0][1]))
print ("w2 ( {} , {} )".format(tabla[1][0],tabla[1][1]))
print ("w3 ( {} , {} )".format(tabla[2][0],tabla[2][1]))
print ("w4 ( {} , {} )".format(tabla[3][0],tabla[3][1]))

print("\n")




del w1, w2, w3, w4, values1, values2, values3, values4, n1, n2, n_iters, stop, minErr

#%%
###############################################################################

##############################  EJERCICIO 2   #################################

###############################################################################

print("\n")

print(" ------------------------------------------------- ")
print(" ------------------ EJERCICIO 2 ------------------ ")
print(" ------------------------------------------------- \n")

print("\n")



print(" ------------------------------------------------- ")
print(" --------------- PRIMER APARTADO ----------------- ")
print(" ------------------------------------------------- \n")

print(" ------ GRADIENTE DESCENDENTE ESTOCÁSTICO -------- ")

print("\n")
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# Mi función de gradiente se llama sgd() para abreviar

def coef2line(w):
    if(len(w)!= 3):
        raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')
    
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    
    return a, b


'''
    Pinta los datos con su etiqueta y la solución definida por w.
    X: Datos (Intensidad promedio, Simetría).
    y: Etiquetas (-1, 1).
    w: Lista o array de numpy (1d) con 3 valores. w[2] ha de ser el témino 
       independiente.
'''
def plot_data(X, y, w, tipo):
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
    plt.title(tipo)
    plt.show()
 



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------



# Nums 1 y 5
def setData(X_train, y_train, X_test, y_test):
    
    X_train.tolist()    
    y_train.tolist()
    X_test.tolist()
    y_test.tolist()

    X_train = X_train[(y_train==5) + (y_train==1)] #XOR
    y_train = y_train[(y_train==5) + (y_train==1)]

    X_test = X_test[(y_test==5) + (y_test==1)]
    y_test = y_test[(y_test==5) + (y_test==1)]

    y_train[y_train==5] = -1
    y_test[y_test==5] = -1

    X_train = [list(tup)+[np.float64(1)] for tup in X_train]
    X_test= [list(tup)+[np.float64(1)] for tup in X_test]

    X_train = np.array(X_train,np.float64)    
    y_train = np.array(y_train,np.float64)
    X_test = np.array(X_test,np.float64)
    y_test = np.array(y_test,np.float64)
    
    
    return X_train, y_train, X_test, y_test


# Stochastic Gradient Descent
# Función iterativa que calcula el gradiente en distintos batches

# Si tuvieramos un error de parada podríamos devolver la solución obtenida
# en uno de los batches. Es decir, si llega al mínimo en un batch no tendría
# que recorrer los otros.
    
# Como en el enunciado no se indica nada mi algoritmo tan solo usa el tamanio del
# minibatch para recorrer los entrenamientos, el bucle for de j es totalmente 
# innecesario y se podría sustituir por la función minibatch_step().
    
def sgd(model, X_train, y_train, iters, minibatch_size, n):
    
    tam = len(X_train)
    Error = []
    err = []
    
    print("Loading... \nPlease, do not turn off your computer\n")
    
    for i in range(iters):
        #print('Iteration {}'.format(i))

        aux = list(zip(X_train,y_train))
        
        random.shuffle(aux)

        X_train, y_train = zip(*aux) #Random datos
        
        for j in range(0, tam, minibatch_size): #For minibatch
            X = X_train[j:j+minibatch_size]
            y = y_train[j:j+minibatch_size]
            
            model, err = minibatch_step(model, X, y, n)
            Error += err.copy()
                    
    return model, Error

# w = w - E(h(x)-y)
    
#Función que calcula el gradiente de un conjunto de datos
def minibatch_step(w, X_train, y_train, n):
    
    E = []
    
    for X,y in zip(X_train, y_train):
        w -= n*gradient(w,X,y)
        err = error(w,X,y)

        E.append(err)
        
        
    return w, E

# Derivada mínimos cuadrados
def gradient(w, X, y):
    return np.float64(2/len(X))*X*((w.dot(X))-y)
    

#Error de clasificación de las muestras
def classificationError(w, X, y):
    h = np.sign(np.dot(X,w))
    
    fails = np.int(0)
    
    for h, y in zip(h,y):
        if (h != y):
            fails += 1
    
    print ("De un conjunto de datos de {} elementos ha fallado en predecir {}".format(len(X),fails))
    print ("ErrorClasificacion = {}".format((fails/len(X))))


#Error w*x = y
def error(w, X, y):
    res = np.dot(X,w)
    error = (res-y)**2
        
    return error




n = np.float64(0.1)

max_iters = np.int32(234)

X_train = np.load("./datos/X_train.npy")
X_test = np.load("./datos/X_test.npy")

y_train = np.load("./datos/y_train.npy")
y_test = np.load("./datos/y_test.npy")

minibatch_size = np.int(np.ceil(np.log2(len(X_train))*5))


Error = []

#Dejar 1 y 5 + aniadir columna de unos
X_train, y_train, X_test, y_test = setData(X_train, y_train, X_test, y_test)

model = np.array([1, 1, 1], np.float64)
model, Error = sgd(model, X_train, y_train, max_iters, minibatch_size, n)


print ("stochastic gradient descent -> Minibatch size : {}".format(minibatch_size))

print("w = {}".format(model))


# Calculo error train

ErrorTrain = np.float64(np.mean(Error))
print("Error train = {}\n".format(ErrorTrain))

print ("Error de clasificación en el train")
classificationError(model, X_train, y_train)

plot_data(X_train, y_train, model, 'Gradiente descendente estocástico - Train')

print ("")


# Calculo error test

ErrorTest = np.float64(np.mean(error (model, X_test, y_test)))
print("Error test = {}\n".format(ErrorTest))

print ("Error de clasificación en el test")
classificationError(model, X_test, y_test)

plot_data(X_test, y_test, model, 'Gradiente descendente estocástico - Test')


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

del  X_train, y_train, X_test, y_test, ErrorTest, ErrorTrain, Error, model, minibatch_size, max_iters, n







#%%

print("\n")

print(" ---------------- PSEUDO INVERSA ----------------- ")

print("\n")

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


# Función que calcula la pseudo inversa
def pseudoInverse(X_train, y_train):
    
    Xtranspose = np.transpose(X_train.copy())
    
    Xt = Xtranspose.copy()
    
    Xx = np.dot(Xt,X_train)
    
    Xx = np.linalg.inv(Xx)
    
    
    return np.dot(np.dot(Xx,Xt),y_train)


# Cargar datos
X_train = np.load("./datos/X_train.npy")
X_test = np.load("./datos/X_test.npy")

y_train = np.load("./datos/y_train.npy")
y_test = np.load("./datos/y_test.npy")


# Adaptamos datos para dejar 1 y 5 y aniadir término independiente
X_train, y_train, X_test, y_test = setData(X_train, y_train, X_test, y_test)

#calcular pseudoinversa
w = pseudoInverse(X_train, y_train)

print("w = {}".format(w))


#Calcular error train
Err = error(w, X_train,y_train)

ErrorTrain = np.float64(np.mean(Err))

print ("Error train = {}\n".format(ErrorTrain))

print ("Error de clasificación en el train")
classificationError(w, X_train, y_train)


plot_data(X_train, y_train, w, ' Pseudo inversa - Train')


print("")


#Calcular error test
Err = error(w, X_test,y_test)

ErrorTest = np.float64(np.mean(Err))

print ("Error test = {}\n".format(ErrorTest))


print ("Error de clasificación en el test")
classificationError(w, X_test, y_test)

plot_data(X_test, y_test, w, 'Pseudo inversa - Test')


print("\n")
print("\n")


del Err, ErrorTrain, ErrorTest, w


#%%

print(" ------------------------------------------------- ")
print(" ----------------SEGUNDO APARTADO----------------- ")
print(" ------------------------------------------------- \n")






def simula_unif(N=2, dims=2, size=(0, 1)):
    m = np.random.uniform(low=size[0], 
                          high=size[1], 
                          size=(N, dims))
    
    return m


def label_data(x1, x2):
    y = np.sign((x1-0.2)**2 + x2**2 - 0.6)
    idx = np.random.choice(range(y.shape[0]), size=(int(y.shape[0]*0.1)), replace=True)
    y[idx] *= -1
    
    return y


#Generate data
X = simula_unif(N=1000, dims=2, size=(-1, 1))
y = label_data(X[:, 0], X[:, 1])

#Plot data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


#Averiguamos que clases hay en el dataset.
clases = np.unique(y)

print("Clases : {}".format(clases))

#Separamos los datos según su clase
#Estoy usando una forma de crear una lista con un bucle que permite hacer 
#python, puede sustituirse por un bucle for con un append a la lista.
X_class = [X[y==c_i] for c_i in clases]

#Separamos en train y test.
trainX_class = [Xi[:int(Xi.shape[0]*0.8)] for Xi in X_class]
testX_class = [Xi[int(Xi.shape[0]*0.8):] for Xi in X_class]


#Calculamos el nuevo tamaño.
sizes_train = [tX.shape[0] for tX in trainX_class]
sizes_test = [tX.shape[0] for tX in testX_class]

#Concatenamos
trainX = np.concatenate(trainX_class, axis=0)
testX = np.concatenate(testX_class, axis=0)

#Creamos trainY y testY
trainY = np.zeros(trainX.shape[0], y.dtype)
testY = np.zeros(testX.shape[0], y.dtype)
pos_train = pos_test = 0

#El comando zip permite empaquetar listas de la misma longitud para recorrerlas
#a la vez.
for c_i, size_train, size_test in zip(clases, sizes_train, sizes_test):
    end_train = pos_train+size_train
    end_test = pos_test+size_test
    
    trainY[pos_train:end_train] = c_i
    testY[pos_test:end_test] = c_i

    pos_train = end_train
    pos_test = end_test

#Eliminamos lo que sobra (no es necesario).
del X
del y
del sizes_train
del sizes_test
del pos_train
del pos_test


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# Derivada minimos cuadrados
def gradient(w, X, y):
    return np.float64(2/len(X))*X*((w.dot(X))-y)
    


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Set datos
minibatch_size = np.int(np.ceil(np.log2(len(trainX))*5))
max_iters = 100

#Aniadir columna de 1 en X
trainX = [[np.float64(1)]+list(tup) for tup in trainX]
testX = [[np.float64(1)]+list(tup) for tup in testX]

trainX = np.array(trainX, np.float64)
trainY = np.array(trainY,np.float64)
testX = np.array(testX,np.float64)
testY = np.array(testY,np.float64)

#Learning rate
n = np.float64(1e-1)

E = []
values = np.array([1,1,1],np.float64)

values, E = sgd(values, trainX, trainY, max_iters, minibatch_size, n)

print ("stochastic gradient descent -> Minibatch size : {}".format(minibatch_size))

print("w = {}".format(values))

print ("Learning rate : {}".format(n))

# Calculo error train
ErrorTrain = np.float64(np.mean(E))
print("Error train = {}\n".format(ErrorTrain))

print ("Error de clasificación en el train")
classificationError(values, trainX, trainY)


print ("")

# Calculo error test

ErrorTest = np.float64(np.mean(error (values, testX, testY)))
print("Error test = {}\n".format(ErrorTest))

print ("Error de clasificación en el test")
classificationError(values, testX, testY)