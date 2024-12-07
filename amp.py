import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pandas as pd
from scipy.special import erf
from sklearn.metrics import zero_one_loss
# import torch

from scipy.optimize import minimize, brute, root

import argparse

import warnings
warnings.filterwarnings('ignore')

global dimension
global delta
global lamb
global seeds
global w_damping

parser = argparse.ArgumentParser(description='Input variables.')
parser.add_argument('--start_alpha', type=float, 
                    help='starting alpha')
parser.add_argument('--end_alpha', type=float,
                    help='ending alpha')
parser.add_argument('--alpha_step',type=float,default=0.5,help='step for each alpha')
parser.add_argument('--test_iterate',type=bool, default=False, help='run test script')
parser.add_argument('--out_csv',help='output csv path each step',default='test_output/amp_xor_default.csv')
parser.add_argument('--out_csv_final',help='output csv path final',default='test_output/amp_xor_final_default.csv')
parser.add_argument('--out_txt',help='output txt path',default='test_output/amp_xor_default.txt')
parser.add_argument('--dimension',help='dimension of the samples',type=int,default=100)
parser.add_argument('--delta',help='delta for the variance of the cluster',type=float,default=0.1)
parser.add_argument('--lamb',help='lambba for the l2 regularization strength',type=float,default=0.1)
parser.add_argument('--max_step',help='max iteration step',type=int,default=1000)
parser.add_argument('--damping',help='damping of iteration',type=float,default=0.9)
parser.add_argument('--w_damping',help='damping of the weights in iteration',type=float,default=0)
parser.add_argument('--tol',help='tolerance for convergence',type=float,default=0.001)
parser.add_argument('--seeds',help='number of trials for each alpha',type=int,default=5)
parser.add_argument('--printout',help='wether or not to print out details at each step',type=bool,default=False)
parser.add_argument('--printout_csv',help='csv to printout proximals at each step',default='test_output/amp_xor_printout.csv')
parser.add_argument('--intercept',help='weather or not include bias',type=lambda x: (str(x).lower() == 'true'),default=True)
parser.add_argument('--separate_branch',help='weather or not separate informative and uninformative branch',type=bool,default=False)
parser.add_argument('--out_upper_csv',help='output csv path each step for the uninformative branch',default='test_output/amp_xor_upper_default.csv')
parser.add_argument('--out_lower_csv',help='output csv path each step for the informative branch',default='test_output/amp_xor_lower_default.csv')

args = parser.parse_args()

dimension = args.dimension
delta = args.delta
lamb = args.lamb
seeds = args.seeds
w_damping = args.w_damping

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_means(dimension=500):
    means = np.zeros((4, dimension))
    
    means[0,0] = 1
    means[1,0] = -1
    means[2,1] = 1
    means[3,1] = -1

    return means


def sample_xor_data(dimension=500, intercept = False, *, samples, variance):
    '''
    Samples XOR dataset, with means lying on the x and y axis and uniform diagonal covariance.
    
    args:
        - dimension: input dimension
        - samples: total number of samples (sum of all clouds)
        - variance: variance of each cloud.
    '''
    
    n = int(samples/4)

    ni = [n,n,n,n]

    if 4*n < samples:
        for i in range(samples-4*n):
            tag = np.random.choice(range(4))
            ni[tag] += 1
    
    means = get_means(dimension=dimension)
    
    X1 = means[0] + np.sqrt(variance) * np.random.normal(0,1, (ni[0], dimension))
    X2 = means[1] + np.sqrt(variance) * np.random.normal(0,1, (ni[1], dimension))
    X3 = means[2] + np.sqrt(variance) * np.random.normal(0,1, (ni[2], dimension))
    X4 = means[3] + np.sqrt(variance) * np.random.normal(0,1, (ni[3], dimension))
    
    y1, y2, y3, y4 = np.ones(ni[0]), np.ones(ni[1]), -np.ones(ni[2]), -np.ones(ni[3])
    
    X, y = np.concatenate([X1, X2, X3, X4], axis=0), np.concatenate([y1, y2, y3, y4], axis=0) 
    
    
    if intercept:
        X = np.c_[X, np.ones(samples)]

    X /= np.sqrt(dimension)
    # print('no. of samples, sample dimension:',X.shape)
        
    return unison_shuffled_copies(X, y)

def get_errors(train_first_layer=True, *, data, labels, weights, a, regularisation):
    predictions = predictor(data=data, 
                            weights=weights,
                            a=a)
    
    loss = .5 * np.mean((labels - predictions)**2)
    # error = .25 * np.mean((labels - np.sign(predictions))**2)
    error = zero_one_loss(labels,np.sign(predictions))
    
    
    if train_first_layer:
        risk = .5 * (np.mean((labels - predictions)**2) + 
                     regularisation * (np.linalg.norm(weights)**2 + np.linalg.norm(a)**2))
    else: 
        risk = .5 * (np.mean((labels - predictions)**2) + 
                     regularisation * np.linalg.norm(weights)**2 / len(labels))       
    
    return loss, risk, error

def get_oracle(*, variance):
    return .5 * (1-erf(1/(2*np.sqrt(variance)))**2)

def get_overlaps(intercept=False, *, weights):
    if intercept:
        weights = weights[:, :-1]
    
    _, dimension = weights.shape
    means = get_means(dimension=dimension)
        
    q = weights @ weights.T / dimension
    m = means @ weights.T / np.sqrt(dimension)
        
    return m, q
    
def predictor(*, data, weights, a):
    Wx = weights @ data.T #shape (k,n)
    return np.tanh(Wx[0]) * np.tanh(Wx[1]) #shape (n)

def moreau_loss(h, y, V, w, a):
    '''
    Loss to be minimised in Moreau envelope.
    args:
        h: proximal, shape (samples, width) or flattened 
        y: labels, shape (samples, )
        V: overlap, shape (samples, width, width)
        f: denoiser, shape (samples, width)
        w: argument, shape (samples, width)
        a: last layer weights, shape (width, )
    
    returns:
        float
    '''

    #reshape if h is flattened
    if len(h.shape) == 1:
        h = h.reshape(w.shape)
    sigmax = np.tanh(h[:,0]) * np.tanh(h[:,1])
    Vinv = np.linalg.inv(V)
    
    return .5 * np.einsum('nk,nkl,nl->n',h-w,Vinv,h-w) + .5 * (y-sigmax)**2

def moreau_loss_single(h,y,V,w,a):
    '''
    Loss to be minimised in Moreau envelope.
    args:
        h: proximal, shape (width,)
        y: label
        V: overlap, shape (width, width)
        w: denoiser, shape (width,)
        a: last layer weights, shape (width, )
    
    returns:
        float
    '''
    sigmax = np.tanh(h[0]) * np.tanh(h[1])
    Vinv = np.linalg.inv(V)
    
    return .5 * np.einsum('k,kl,l->',h-w,Vinv,h-w) + .5 * (y-sigmax)**2

def lxx(y,h):
    '''
    l11 derivative of loss wrt to h.
    args:
        y: labels, np.array of shape (samples, )
        h: proximal, shape (samples, width) 
    returns:
        shape (samples, width, width)
    ''' 
    samples, width = h.shape
    l11 = (1/np.cosh(h[:,0])**4) * (np.tanh(h[:,1])**2) + 2 * np.tanh(h[:,0]) * np.tanh(h[:,1]) * (1/np.cosh(h[:,0])**2) * (y-np.tanh(h[:,0])*np.tanh(h[:,1])) #shape (samples,)
    l12 = (1/np.cosh(h[:,0])**2) * (1/np.cosh(h[:,1])** 2) * (2 * np.tanh(h[:,0]) * np.tanh(h[:,1]) - y) #shape (samples,)
    l22 = (1/np.cosh(h[:,1])**4) * (np.tanh(h[:,0])**2) + 2 * np.tanh(h[:,1]) * np.tanh(h[:,0]) * (1/np.cosh(h[:,1])**2) * (y-np.tanh(h[:,0])*np.tanh(h[:,1])) #shape (samples,)

    lhh = np.stack((l11,l12,l12,l22)).T.reshape(samples,width,width)
    
    return lhh

def get_df(y,h,V,a):
    '''
    Derivative of proximal wrt to omega.
    args:
        y: labels, np.array of shape (samples, )
        h: proximal, shape (samples, width) 
        V: V overlap, shape (samples, width, width)
        a: last layer weights, shape (width, )
    
    returns:
        shape (samples, width, width)
    '''
    samples, width = h.shape
    lhh = lxx(y,h) #shape (samples, width, width)

    Id = np.array(samples * [np.identity(width)]) #shape (samples, width, width)

    Inversa = Id + np.einsum('nij,njk->nik',lhh,V) #shape (samples, width, width)
    try:
        Inversa = np.linalg.inv(Inversa)
    except np.linalg.LinAlgError:
        print('singular matrix Inversa:',Inversa)
        Inversa = np.linalg.inv(Inversa)

    return -np.einsum('nij,njk->nik',Inversa,lhh)

def prior(b, A, regularisation=0.1):
    d, width = b.shape
    denominator = np.linalg.inv(np.stack(d * [regularisation * np.identity(width)], axis=0) + A)

    return np.einsum('ijk,ik->ij', denominator, b), denominator

def minimize_moreau_loss_single(y,V,w,a):
    '''
    Loss to be minimised in Moreau envelope.
    args:
        y: label
        V: overlap, shape (width, width)
        w: argument, shape (samples, width) 
        a: last layer weights, shape (width, )
    
    returns:
        float h: minimised solution
    '''
    width = len(w)
    eigvalues_V, eig_V = np.linalg.eig(V)
    minimas = []
    values = []
    minimas.append(minimize(moreau_loss_single, w, args=(y,V,w,a)).x)
    values.append(moreau_loss_single(minimas[0],y,V,w,a))
    minimas.append(minimize(moreau_loss_single, [w[0]+eigvalues_V[0]*eig_V[0,0],w[1]+eigvalues_V[0]*eig_V[0,1]], args=(y,V,w,a)).x)
    values.append(moreau_loss_single(minimas[1],y,V,w,a))
    minimas.append(minimize(moreau_loss_single, [w[0]+eigvalues_V[1]*eig_V[1,0],w[1]+eigvalues_V[1]*eig_V[1,1]], args=(y,V,w,a)).x)
    values.append(moreau_loss_single(minimas[2],y,V,w,a))
    minimas.append(minimize(moreau_loss_single, [w[0]-eigvalues_V[0]*eig_V[0,0],w[1]-eigvalues_V[0]*eig_V[0,1]], args=(y,V,w,a)).x)
    values.append(moreau_loss_single(minimas[3],y,V,w,a))
    minimas.append(minimize(moreau_loss_single, [w[0]-eigvalues_V[1]*eig_V[1,0],w[1]-eigvalues_V[1]*eig_V[1,1]], args=(y,V,w,a)).x)
    values.append(moreau_loss_single(minimas[4],y,V,w,a))

    minindex = np.argmin(values)
    minima = minimas[minindex]

    return minima

def minimize_moreau_loss(y,V,w,a):
    '''
    Loss to be minimised in Moreau envelope.
    args:
        y: label
        V: overlap, shape (samples,width, width)
        f: denoiser, shape (width,)
        a: last layer weights, shape (samples,width)
    
    returns:
        float h: minimised solution, shape (samples, width)
    '''
    samples, width = w.shape
    vfunc = np.vectorize(minimize_moreau_loss_single,excluded=['a'],signature='(),(n,n),(n)->(n)')
    h = vfunc(y=y,V=V,w=w,a=a) #shape (samples,width)

    return h

def likelihood(y, w, V, a):
    '''
    Likelihood associated with the loss.
    args:
        y: labels, shape (samples, )
        w: argument, shape (samples, width) 
        V: overlap, shape (samples, width, width)
        a: last layer weights, shape (width, )
    
    returns:
        f: denoiser, shape (samples, width)
        df: derivative of denoiser, shape (samples, width, width)
    '''
    samples, width = w.shape

    eps=0.001
    error=1
    r = np.sqrt(1./np.min(np.abs(np.linalg.eigvals(V))))
    try:
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        print('LinAlgError error, reset')
        print('singular V:',V)
        Vinv = np.array(samples * [np.identity(width)])
            
    f = w
    h = w
    lossold = moreau_loss(h,y,V,w,a)

    # for t in range(10):
        
    htmp = minimize_moreau_loss(y,V,w,a)

    ftmp = np.einsum('nkl,nl->nk',Vinv,htmp-w)
    
    losstmp=moreau_loss(htmp, y, V, w, a)
    
    f=np.where((losstmp<lossold)[:, None],ftmp,f)
    h=np.where((losstmp<lossold)[:, None],htmp,h)
    lossold=np.where(losstmp<lossold,losstmp,lossold)

    f = ftmp.copy()
    h = htmp.copy()
        
    Id = np.array(samples * [np.identity(width)])

    df = get_df(y,h,V,a) #shape (samples, width, width)
    dh = np.einsum('nij,njk->nik',V,df) + Id #shape (samples, width, width)
        
    return f, df

def damp(damping=0.5, *, new, old):
    return (1-damping) * new + damping * old

def iterate_amp(tolerance = 1e-5, 
               max_steps = int(1e4),
               width = 2,
               damping = 0,
               verbose = True,
               printout = False,
               regularisation = 0,
               intercept=False,
               train_first_layer=False,
               *, data, labels):

    # pre-processing
    samples, dimension = data.shape    

    X2 = data**2 #shape (samples, dimension)
    status = 0
    loss, risk, error, m, q = (np.zeros(max_steps+1), 
                               np.zeros(max_steps+1), 
                               np.zeros(max_steps+1), 
                               np.zeros((max_steps+1, 4, width)), 
                               np.zeros((max_steps+1, width, width)))

    
    
    # initialise
    #random
    weights = np.random.normal(0, 1, (dimension, width))
    weights[-1,:] = np.array([0,0])
    weights_norms = np.linalg.norm(weights,axis=0)
    weights = weights / weights_norms

    # weights = np.random.normal(0, 1, (dimension, width)) / np.sqrt(dimension) #shape (dimension,width)
    
    #informative
    # weights = np.zeros([dimension,width])
    # weights[0,0] = 1
    # weights[0,1] = 1
    # weights[1,0] = 1
    # weights[1,1] = -1
    # weights_norms = np.linalg.norm(weights,axis=0)
    # weights = weights / weights_norms

    a = np.ones(width) / width #shape (width)
    
    weights_old = np.zeros([dimension,width])
    
    g = np.zeros((samples, width))
    chat = np.stack(dimension*[np.identity(width)], axis=0) #shape (dimension, width, width)
    
    loss[0], risk[0], error[0] = get_errors(data = data, 
                                        labels = labels, 
                                        weights = weights.T,
                                        a = a,
                                        regularisation = regularisation,
                                        train_first_layer = train_first_layer)
    
    m[0], q[0] = get_overlaps(weights=weights.T,
                              intercept=intercept)

    diffs = []

    with open(args.printout_csv,'a') as f:
        f.write(f'{0},')
        f.write(f'{error[0]},{loss[0]},')
        f.write(f'{q[0][0][0]},{q[0][0][1]},{q[0][1][0]},{q[0][1][1]},')
        f.write(f'{m[0][0][0]},{m[0][1][0]},{m[0][2][0]},{m[0][3][0]},')
        f.write(f'{m[0][0][1]},{m[0][1][1]},{m[0][2][1]},{m[0][3][1]},')
        f.write(f'{weights[-1,0]},{weights[-1,1]}\n')
        f.close()
    
    for t in np.arange(max_steps):
        if t==0:
            V = np.tensordot(X2, chat, axes=([1,0]))

            omega = (np.tensordot(data, weights, axes=([1,0])) -
                     np.einsum('ijk,ik->ij', V, g)) #shape (dimension, width)

            g, dg = likelihood(labels, omega, V, a) #shape g(samples,width) dg(samples,width,width)

            A = - np.tensordot(X2, dg, axes=([0, 0])) #shape (dimension, width, width)

            b = (np.einsum('ijk,ik->ij', A, weights) +
                 np.tensordot(data, g, axes=([0, 0]))) #shape (dimension, width)

            weights, chat = prior(b, A, regularisation=regularisation)

        else:
            V_tmp = np.tensordot(X2, chat, axes=([1,0]))
            V = damp(damping=damping, new=V_tmp, old=V)
            
            omega = (np.tensordot(data, weights, axes=([1,0])) -
                     np.einsum('ijk,ik->ij', V, g))

            g_tmp, dg_tmp = likelihood(labels, omega, V, a)

            g, dg = damp(damping=damping,new=g_tmp, old=g), damp(damping=damping,new=dg_tmp, old=dg)

            A = -np.tensordot(X2, dg, axes=([0, 0])) #shape (d, width, width)

            b = (np.einsum('ijk,ik->ij', A, weights) +
                 np.tensordot(data, g, axes=([0, 0]))) #shape (d,width)

            weights, chat = prior(b, A, regularisation=regularisation)

            weights = damp(damping=w_damping,new=weights, old=weights_old)
            # chat = damp(damping=damping, new=chat_tmp, old=chat)
        
        m[t+1], q[t+1] = get_overlaps(weights=weights.T,
                                      intercept=intercept)
        
        diff = np.mean(np.abs(weights - weights_old))

        Vnew = np.tensordot(X2,chat,axes=([1,0]))
        # diff = np.linalg.norm(Vnew-V) / samples + np.linalg.norm(q[t+1] - q[t]) + np.linalg.norm(m[t+1] - m[t])
        diffs.append(diff)
        
        loss[t+1], risk[t+1], error[t+1] = get_errors(data = data, 
                                              labels = labels, 
                                              weights = weights.T,
                                              a=a,
                                              regularisation=regularisation,
                                              train_first_layer = train_first_layer)
        
        if diff<tolerance:
            if verbose:
                print('\t\t AMP converged with {} steps, diff : {}'.format(t+1, diff))
                print('\t\t weights:',weights)
            if printout:
                with open(args.out_txt,'a') as f:
                    f.write('\t\t AMP converged with {} steps\n'.format(t+1))
                    f.write('\t\t Step {}, diff: {}, loss: {}, risk: {}, error: {}\n'.format(t+1, diff, loss[t+1], risk[t+1], error[t+1]))
                    f.write(f'\t\t m {m[t+1]}, q {q[t+1]}, b {weights[:,-1]}\n')
                    f.write(f'\t\t V {Vnew}\n')
                    f.write(f'\t\t weights: {weights}\n')
                    f.write(f'\t\t chat:{chat}\n')
                    f.close()
                with open(args.printout_csv,'a') as f:
                    f.write(f'{t+1},')
                    f.write(f'{error[t+1]},{loss[t+1]},')
                    f.write(f'{q[t+1][0][0]},{q[t+1][0][1]},{q[t+1][1][0]},{q[t+1][1][1]},')
                    f.write(f'{m[t+1][0][0]},{m[t+1][1][0]},{m[t+1][2][0]},{m[t+1][3][0]},')
                    f.write(f'{m[t+1][0][1]},{m[t+1][1][1]},{m[t+1][2][1]},{m[t+1][3][1]},')
                    f.write(f'{weights[-1,0]},{weights[-1,1]}\n')
                    f.close()
            status = 1
            break
        
        if verbose & (t % 1 == 0):
            print('\t\t Step {}, diff: {}, loss: {}, risk: {}, error: {}'.format(t+1, diff, loss[t+1], risk[t+1], error[t+1]))
        
        if printout & (t % 1 == 0):
            with open(args.out_txt,'a') as f:
                f.write('\t\t Step {}, diff: {}, loss: {}, risk: {}, error: {}\n'.format(t+1, diff, loss[t+1], risk[t+1], error[t+1]))
                f.write(f'\t\t m {m[t+1]}, q {q[t+1]}, b {weights[:,-1]}\n')
                f.write(f'\t\t V {Vnew}\n')
                f.write(f'\t\t weights: {weights}\n')
                f.write(f'\t\t chat:{chat}\n')
                f.close()

            with open(args.printout_csv,'a') as f:
                f.write(f'{t+1},')
                f.write(f'{error[t+1]},{loss[t+1]},')
                f.write(f'{q[t+1][0][0]},{q[t+1][0][1]},{q[t+1][1][0]},{q[t+1][1][1]},')
                f.write(f'{m[t+1][0][0]},{m[t+1][1][0]},{m[t+1][2][0]},{m[t+1][3][0]},')
                f.write(f'{m[t+1][0][1]},{m[t+1][1][1]},{m[t+1][2][1]},{m[t+1][3][1]},')
                f.write(f'{weights[-1,0]},{weights[-1,1]}\n')
                f.close()
        
        weights_old = weights.copy()
        
        # stop if diff unpexpectedly explodes
        if loss[t+1] > 0.9 or diff>1000:
            print("\t\t Explosion, broken seed")

            weights = weights_old
            status = -1
            break
        
    if t == max_steps-1:
        print("AMP didn't converge after {} steps, keeping last value.".format(max_steps))
        status = -1
            
    return weights.T, a, loss[:t+1], risk[:t+1], error[:t+1], m[:t+1], q[:t+1], t+1, status

def simulate(dimension = 500, 
             tolerance = 1e-5, 
             max_steps = int(1e5),
             width = 2,
             verbose_short = True,
             verbose = False,
             printout = False,
             regularisation = 0,
             seeds=10,
             train_first_layer = False,
             intercept=False,
             damping=0,
             separate_branch = False,
             *, 
             sample_complexity, 
             variance):
    
    result = {}
    
    samples = int(sample_complexity * dimension)
    
    gen_error, gen_loss, train_error, train_loss, m_tab, q_tab = [],[],[],[],[],[]

    if separate_branch:
            gen_error_upper, gen_loss_upper, train_error_upper, train_loss_upper = [],[],[],[]
            gen_error_lower, gen_loss_lower, train_error_lower, train_loss_lower = [],[],[],[]
    
    converged = 0
    for seed in np.arange(seeds):
        if (verbose == True) or (verbose_short==True):
            print('\t Simulating seed: {}'.format(seed+1))
        
        if printout == True:
            with open(args.out_txt,'a') as f:
                f.write('\t Simulating seed: {}\n'.format(seed+1))
                f.close()
            
        X_train, y_train = sample_xor_data(dimension=dimension, 
                                           samples=samples,
                                           variance=variance,
                                           intercept=intercept)
        
        X_test, y_test = sample_xor_data(dimension=dimension, 
                                           samples=samples,
                                        #    samples = dimension,
                                           variance=variance,
                                           intercept=intercept)

        
        try:
            weights, a, loss, risk, error, m, q, t, status = iterate_amp(tolerance=tolerance,
                                                                        intercept=intercept,
                                                                        max_steps=max_steps,
                                                                        width=width,
                                                                        verbose = verbose,
                                                                        printout = printout,
                                                                        regularisation=regularisation,
                                                                        data = X_train, 
                                                                        labels = y_train,
                                                                        train_first_layer = train_first_layer,
                                                                        damping=damping)
        
        except np.linalg.LinAlgError:
            print('LinAlgError error, skipping this seed')
            continue
        
        if status == -1:
            continue
        
        if status == 1:
            converged += 1
        
        #loss, risk, error
        egl, _, eg =  get_errors(data = X_test, 
                                 labels = y_test, 
                                 weights = weights,
                                 a=a,
                                 regularisation=regularisation,
                                 train_first_layer=train_first_layer)
        
        if (verbose == True) or (verbose_short==True):
            print('\t Training done with t={}. Train loss: {}, train error: {}, test loss: {}, test error: {}'.format(t, loss[-1], error[-1], egl, eg))
            print('\t weights:',weights)
            if intercept:
                print('\t bias:',weights[:,-1])

        if printout == True:
            with open(args.out_txt,'a') as f:
                f.write('\t Training done with t={}. Train loss: {}, train error: {}, test loss: {}, test error: {}\n'.format(t, loss[-1], error[-1], egl, eg))
                f.close()
        
        #discard broken cases
        if loss[-1] > 0.6:
            continue
        
        m_tab.append(m[-1])
        q_tab.append(q[-1])
        train_error.append(error[-1])
        train_loss.append(loss[-1])
        gen_error.append(eg)
        gen_loss.append(egl)

        if separate_branch:
            if eg>0.4:
                train_error_upper.append(error[-1])
                train_loss_upper.append(loss[-1])
                gen_error_upper.append(eg)
                gen_loss_upper.append(egl)
            else:
                train_error_lower.append(error[-1])
                train_loss_lower.append(loss[-1])
                gen_error_lower.append(eg)
                gen_loss_lower.append(egl)
        
    result = {
        'test_error': np.mean(gen_error),
        'test_error_std': np.std(gen_error),
        'test_loss': np.mean(gen_loss),
        'test_loss_std': np.std(gen_loss), 
        'train_error': np.mean(train_error),
        'train_error_std': np.std(train_error),
        'train_loss': np.mean(train_loss),
        'train_loss_std': np.std(train_loss),
        'm': np.mean(m_tab, axis=0),
        'm_std': np.std(m_tab, axis=0),
        'q': np.mean(q_tab, axis=0),
        'q_std': np.std(q_tab, axis=0),
        'seeds': converged
             }
    
    if (verbose == True) or (verbose_short==True):
        print('\t Training done for alpha={}. train error: {}, test error: {}, train std: {}, test std: {}'.format(sample_complexity, np.mean(train_error), np.mean(gen_error), np.std(train_error), np.std(gen_error)))
        print('\t result:',result)

    if printout == True:
        with open(args.out_txt,'a') as f:
            f.write('\t Training done for alpha={}. train error: {}, test error: {}\n'.format(sample_complexity, np.mean(train_error), np.mean(gen_error)))
            f.close()
    
    #seperate branch:
    if separate_branch:
        with open(args.out_upper_csv,'a') as f:
            f.write(f"{sample_complexity},{len(train_error_upper)},{np.mean(train_error_upper)},{np.std(train_error_upper)},{np.mean(train_loss_upper)},{np.std(train_loss_upper)},{np.mean(gen_error_upper)},{np.std(gen_error_upper)},{np.mean(gen_loss_upper)},{np.std(gen_loss_upper)}\n")
        with open(args.out_lower_csv,'a') as f:
            f.write(f"{sample_complexity},{len(train_error_lower)},{np.mean(train_error_lower)},{np.std(train_error_lower)},{np.mean(train_loss_lower)},{np.std(train_loss_lower)},{np.mean(gen_error_lower)},{np.std(gen_error_lower)},{np.mean(gen_loss_lower)},{np.std(gen_loss_lower)}\n")

    with open(args.out_csv,'a') as f:
        f.write(f"{sample_complexity},{np.mean(train_error)},{np.std(train_error)},{np.mean(train_loss)},{np.std(train_loss)},{np.mean(gen_error)},{np.std(gen_error)},{np.mean(gen_loss)},{np.std(gen_loss)}\n")

    return result

def simulate_range(dimension = 500, 
                   tolerance = 1e-5, 
                   max_steps = int(1e5),
                   width = 2,
                   regularisation = 0,
                   verbose_short = True,
                   verbose = False,
                   printout = False,
                   seeds=10,
                   variance = 0.1,
                   train_first_layer = False,
                   intercept=False,
                   damping=0,
                   separate_branch = False,
                   *, 
                   sc_range):

    result = {
        'sample_complexity': list(sc_range),
        'intercept': [intercept] * len(sc_range),
        'regularisation': [regularisation] * len(sc_range),
        'width': [width] * len(sc_range),
        'dimension': [dimension] * len(sc_range), 
        'variance': [variance] * len(sc_range),
        'train_first_layer': [train_first_layer] * len(sc_range),
        'damping': [damping] * len(sc_range),
        'test_error': [],
        'test_error_std': [],
        'test_loss': [],
        'test_loss_std': [],
        'train_error': [],
        'train_error_std': [],
        'train_loss': [],
        'train_loss_std': [],
        'seeds': []
             }

    for k in np.arange(width):
        for l in np.arange(k+1):
            result['q{}{}'.format(l+1, k+1)] = []
            result['q{}{}_std'.format(l+1, k+1)] = []

        for l in np.arange(4):
            result['m{}{}'.format(l+1, k+1)] = []
            result['m{}{}_std'.format(l+1, k+1)] = []

    if printout:
        with open(args.out_txt,'w') as f:
                f.write('Training with sc_range={}, dimension={}, tolerance={}\n'.format(sc_range, dimension,tolerance))
                f.close()

    for sample_complexity in sc_range:
        if (verbose == True) or (verbose_short==True):
            print('Simulating sample complexity: {}'.format(sample_complexity))

        if printout==True:
            with open(args.out_txt,'a') as f:
                f.write('Simulating sample complexity: {}\n'.format(sample_complexity))
                f.close()
                
        res = simulate(dimension = dimension,
                       tolerance=tolerance,
                       seeds=seeds,
                       max_steps=max_steps,
                       width=width,
                       regularisation=regularisation,
                       variance = variance,
                       sample_complexity=sample_complexity,
                       verbose_short = verbose_short,
                       verbose=verbose,
                       printout = printout,
                       train_first_layer = train_first_layer,
                       intercept=intercept,
                       separate_branch=separate_branch,
                       damping=damping)
        
        for key, val in res.items():
            if key not in ['q', 'm', 'q_std', 'm_std']:
                result[key].append(val)
                
        for k in np.arange(width):
            for l in np.arange(k+1):
                result['q{}{}'.format(l+1, k+1)].append(res['q'][l,k])
                result['q{}{}_std'.format(l+1, k+1)].append(res['q_std'][l,k])

            for l in np.arange(4):
                result['m{}{}'.format(l+1, k+1)].append(res['m'][l,k])
                result['m{}{}_std'.format(l+1, k+1)].append(res['m_std'][l,k])
                        
    return pd.DataFrame.from_dict(result)

def test_iterate(dimension = 500, 
                   tolerance = 1e-5, 
                   max_steps = int(1e5),
                   width = 2,
                   regularisation = 0,
                   verbose_short = True,
                   verbose = False,
                   printout = False,
                   variance = 0.1,
                   train_first_layer = False,
                   intercept=False,
                   damping=0,
                   *, 
                   sample_complexity):

    samples = int(dimension * sample_complexity)
    print('intercept:',intercept)
    
    X_train, y_train = sample_xor_data(dimension=dimension, 
                                           samples=samples,
                                           variance=variance,
                                           intercept=intercept)

    print('samples:',len(y_train))

    weights, a, loss, risk, error, m, q, tmax, status = iterate_amp(tolerance=tolerance,
                                                                        intercept=intercept,
                                                                        max_steps=max_steps,
                                                                        width=width,
                                                                        verbose = verbose,
                                                                        printout = printout,
                                                                        regularisation=regularisation,
                                                                        data = X_train, 
                                                                        labels = y_train,
                                                                        train_first_layer = train_first_layer,
                                                                        damping=damping)
    
    if status == 1:
        X_plus = X_train[y_train==1]
        X_minus = X_train[y_train==-1]

        xmin, xmax = X_train[:,0].min()-1, X_train[:,0].max()+1
        ymin, ymax = X_train[:,1].min()-1, X_train[:,1].max()+1

        X_test, y_test = sample_xor_data(dimension=dimension, 
                                            samples=samples,
                                            #    samples = dimension,
                                            variance=variance,
                                            intercept=intercept)

        pre_y_test = predictor(data = X_test, 
                            weights = weights,
                            a=a)
        pre_y_test = np.sign(pre_y_test)

        # Retrieve the model parameters.
        w = weights[:,:2]
        if intercept:
            b = weights[:,-1]

        # Calculate the intercept and gradient of the decision boundary.
        slope = -w[:,0]/w[:,1]
        if intercept:
            c = -b/w[:,1]

        print('slope:',slope)

        # Plot the data and the classification with the decision boundary.
        xmin, xmax = X_train[:,0].min(), X_train[:,0].max()
        ymin, ymax = X_train[:,1].min(), X_train[:,1].max()

        print("train loss, risk, error:",get_errors(train_first_layer = train_first_layer, data=X_train, labels=y_train, weights=weights,a=a,regularisation = regularisation))
        print("test loss, risk, error:",get_errors(train_first_layer = train_first_layer, data=X_test, labels=y_test, weights=weights,a=a,regularisation = regularisation))

        # pre_y_train = np.sign(predictor(data=X_train,weights=weights,a=a))

        X_plus = X_train[y_train==1]
        X_minus = X_train[y_train==-1]

        # Retrieve the model parameters.
        w = weights[:,:2]
        if intercept:
            b = weights[:,-1]

        # Calculate the intercept and gradient of the decision boundary.
        slope = -w[:,0]/w[:,1]
        if intercept:
            c = -b/w[:,1]

        # Plot the data and the classification with the decision boundary.
        xmin, xmax = X_train[:,0].min(), X_train[:,0].max()
        ymin, ymax = X_train[:,1].min(), X_train[:,1].max()

        y_diff = pre_y_test * y_test
        X_wrong = X_test[y_diff == -1]

        xd = np.array([xmin, xmax])

        for index in range(len(slope)):
            if intercept:
                yd = slope[index]*xd + c[index]
            else:
                yd = slope[index]*xd

            print('xd:',xd)
            print('yd:',yd)

            plt.plot(xd, yd, lw=1, ls='--', color='black')

            plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
            plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
            
            plt.scatter(w[index,0], w[index,1], c='black')


        plt.scatter(X_plus[:, 0], X_plus[:, 1], c='blue',label='True y=1')
        plt.scatter(X_minus[:, 0], X_minus[:, 1], c='orange',label='True y=-1')

        plt.scatter(X_wrong[:, 0], X_wrong[:, 1], c='red',label='missed')

        plt.scatter(w[0,0], w[0,1], c='black')
        plt.scatter(w[1,0], w[1,1], c='black')

        print('4')

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.ylabel(r'$x_2$')
        plt.xlabel(r'$x_1$')
        plt.title('Training result')
        plt.legend()

        plt.show()

        print('5')
    else:
        print('Process failed')
    
    return 0


def main():
    
    if (args.start_alpha == args.end_alpha) or (args.alpha_step) == 0:
        sc_range = [args.start_alpha]
    else:
        sc_range = np.arange(args.start_alpha,args.end_alpha,args.alpha_step)

    if args.test_iterate:
        sample_complexity = args.start_alpha
        test_iterate(dimension = args.dimension, 
                                tolerance = args.tol, 
                                max_steps = args.max_step,
                                width = 2,
                                regularisation = lamb,
                                verbose_short=True,
                                verbose = True,
                                printout = args.printout,
                                intercept = args.intercept,
                                variance = args.delta,
                                damping=args.damping,
                                sample_complexity=sample_complexity)
    else:
        with open(args.out_csv,'w') as f:
            f.write('alpha,train_error,train_error_std,train_loss,train_loss_std,gen_error,gen_error_std,gen_loss,gen_loss_std\n')
            f.close()
        
        if args.separate_branch:
            with open(args.out_upper_csv,'w') as f:
                f.write('alpha,seeds,train_error,train_error_std,train_loss,train_loss_std,gen_error,gen_error_std,gen_loss,gen_loss_std\n')
                f.close()
            with open(args.out_lower_csv,'w') as f:
                f.write('alpha,seeds,train_error,train_error_std,train_loss,train_loss_std,gen_error,gen_error_std,gen_loss,gen_loss_std\n')
                f.close()

        if args.printout:
            with open(args.printout_csv,'w') as f:
                f.write('t,train_error,train_loss,q00,q01,q10,q11,m00,m10,m20,m30,m01,m11,m21,m31,b0,b1\n')
                f.close()
        simulation = simulate_range(dimension = args.dimension, 
                                tolerance = args.tol, 
                                max_steps = args.max_step,
                                width = 2,
                                regularisation = lamb,
                                verbose_short=True,
                                verbose = True,
                                printout = args.printout,
                                seeds=seeds,
                                intercept=args.intercept,
                                variance = args.delta,
                                damping=args.damping,
                                separate_branch=args.separate_branch,
                                sc_range=sc_range)
        
        simulation.to_csv(args.out_csv_final, index=False)
        

main()
