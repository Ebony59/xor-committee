import numpy as np
from scipy.optimize import minimize

global C
global K

C = 4
K = 2

def Gaussian(Xi, axis=0, mean=0, var=1):
    pdf = np.exp(-.5*(Xi-mean)**2/var) / np.sqrt(2*np.pi*var)
    return np.prod(pdf,axis=axis)

def gaussian(x, mean=0, var=1):
    if type(x) == np.ndarray:
        x = x.flatten()
        prod = 1
        for i in range(len(x)):
            prod *= np.exp(-.5*(x[i]-mean)**2/var) / np.sqrt(2*np.pi*var)
        return prod
    else:
        return np.exp(-.5*(x-mean)**2/var) / np.sqrt(2*np.pi*var)

def activation(x):
    return np.tanh(x)

def dactivation(x):
    return 1/np.cosh(x) ** 2

def sigma_bar(x):
    # return x
    return activation(x[0])*activation(x[1])

def l11(y,x):
    return (1/np.cosh(x[0])**4) * (np.tanh(x[1])**2) + 2 * np.tanh(x[0]) * np.tanh(x[1]) * (1/np.cosh(x[0])**2) * (y-np.tanh(x[0])*np.tanh(x[1]))
def l12(y,x):
    return (1/np.cosh(x[0])**2) * (1/np.cosh(x[1])** 2) * (2 * np.tanh(x[0]) * np.tanh(x[1]) - y)
def l22(y,x):
    return (1/np.cosh(x[1])**4) * (np.tanh(x[0])**2) + 2 * np.tanh(x[1]) * np.tanh(x[0]) * (1/np.cosh(x[1])**2) * (y-np.tanh(x[0])*np.tanh(x[1]))

def l2x(y,x):
    # x = np.array([x1,x2])
    return np.array([l11(y,x),l12(y,x),l12(y,x),l22(y,x)])

def lxx(y,h,integral_steps):
    # input x has shape (2,itera)
    l2x_vec = np.vectorize(l2x,excluded=['y'],signature='(m)->(n)')
    lhh = l2x_vec(y=y,x=h.T)
    lhh = np.moveaxis(l2x_vec(y=y,x=h.T).reshape(integral_steps,K,K),0,-1)
    return lhh

def loss(y,x):
    return 0.5 * (y-sigma_bar(x))**2

def proxh_given_w(x,y,w,V,Q,m):
    return 0.5 * np.dot((x-w) , np.matmul(np.linalg.inv(V),x-w)) + loss(y,x)

def proxh_given_w_K2(w1, w2, c, V, Q, m):
    w = np.array([w1,w2])
    center = w
    if c == 0 or c == 1:
        y = 1
    else:
        y = -1

    eigvalues_V, eig_V = np.linalg.eig(V)
    minimas = []
    values = []
    minimas.append(minimize(proxh_given_w, center, args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[0],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]+eigvalues_V[0]*eig_V[0][0],center[1]+eigvalues_V[0]*eig_V[0][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[1],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]+eigvalues_V[1]*eig_V[1][0],center[1]+eigvalues_V[1]*eig_V[1][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[2],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]-eigvalues_V[0]*eig_V[0][0],center[1]-eigvalues_V[0]*eig_V[0][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[3],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]-eigvalues_V[1]*eig_V[1][0],center[1]-eigvalues_V[1]*eig_V[1][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[4],y,w,V,Q,m))

def hmin_given_w_K2(w1, w2, c, V, Q, m):
    w = np.array([w1,w2])
    center = w
    if c == 0 or c == 1:
        y=1
    else:
        y=-1

    eigvalues_V, eig_V = np.linalg.eig(V)
    minimas = []
    values = []

    minimas.append(minimize(proxh_given_w, center, args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[0],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]+eigvalues_V[0]*eig_V[0][0],center[1]+eigvalues_V[0]*eig_V[0][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[1],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]+eigvalues_V[1]*eig_V[1][0],center[1]+eigvalues_V[1]*eig_V[1][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[2],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]-eigvalues_V[0]*eig_V[0][0],center[1]-eigvalues_V[0]*eig_V[0][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[3],y,w,V,Q,m))
    minimas.append(minimize(proxh_given_w, [center[0]-eigvalues_V[1]*eig_V[1][0],center[1]-eigvalues_V[1]*eig_V[1][1]], args=(y,w,V,Q,m)).x)
    values.append(proxh_given_w(minimas[4],y,w,V,Q,m))

    minindex = np.argmin(values)
    minima = minimas[minindex]

    return minima

def get_diff(t,V,Q,M):
    '''
    Compute differencial between step t+1 and t.
    '''
    diff = np.linalg.norm(V[t+1]-V[t]) + np.linalg.norm(Q[t+1] - Q[t]) + np.linalg.norm(M[t+1] - M[t])

    return diff

def phi(omega):
    tanh1 = np.tanh(omega[0]) #shape (C,itera_err)
    tanh2 = np.tanh(omega[1])
    return np.sign(np.einsum('im,im->im',tanh1,tanh2)) #shape (C,itera_err)
