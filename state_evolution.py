import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import zero_one_loss

from utils import *

global C
global K
global d

C = 4
K = 2
d = 100

class StateEvolution():
    def __init__(self, delta, lamb, tol, integral_steps, max_steps,
                 initV,initQ,initM,initb,
                 out_step_csv: str, out_final_csv: str,
                 out_txt='test_output/out.txt', printout_csv='test_output/printout.csv',
                 damping=0.9, printout=False, error_evaluation=False, resume=False):
        self.delta = delta
        self.lamb = lamb
        self.tol = tol
        self.integral_steps = integral_steps
        self.max_steps = max_steps
        self.initV = initV
        self.initQ = initQ
        self.initM = initM
        self.initb = initb
        self.out_step_csv = out_step_csv
        self.out_final_csv = out_final_csv
        self.out_txt = out_txt
        self.printout_csv = printout_csv
        self.damping = damping
        self.printout = printout
        self.error_evaluation = error_evaluation
        self.resume = resume

    @property
    def mu(self):
        mu = np.zeros([C,d])
        mu[0][0] = 1
        mu[1][0] = -1
        mu[2][1] = 1
        mu[3][1] = -1
        return mu
    
    @property
    def mumu(self):
        return self.mu @ self.mu.T

    @property
    def rho(self):
        return np.array([0.25,0.25,0.25,0.25])

    @property
    def prob(self):
        return np.diag(self.rho)

    def initialisation(self,initV,initQ,initM,initb):
        V = np.zeros([self.max_steps + 1,K,K])
        M = np.zeros([self.max_steps + 1,K,C])
        Q = np.zeros([self.max_steps + 1,K,K])
        b = np.zeros([self.max_steps + 1,K])

        V[0] = initV
        M[0] = initM
        Q[0] = initQ
        b[0] = initb

        return V, Q, M, b 

    def update_hatoverlaps(self,alpha,V,Q,M):
        sqQ=np.real(sqrtm(Q))
        Xi = np.random.normal(0,1,(K,C,self.integral_steps)) #shape (K,C,integral_steps)
        omega = np.einsum('ij,jsk -> isk',sqQ, Xi) + M[:,:,np.newaxis]

        vfunc = []
        hs = []
        for i in range(C):
            vfunc.append(np.vectorize(hmin_given_w_K2,excluded=['c','V','Q','m'],signature='(),()->(n)'))
            hs.append(np.moveaxis(vfunc[i](w1=omega[0][i],w2=omega[1][i],c=i,V=V,Q=Q,m=M[:,i]),0,-1))        
        
        h = np.stack(hs, axis=0)
        h = np.swapaxes(h,0,1) #shape (K,C,integral_steps)
        f = np.einsum('ij,jsk -> isk',np.linalg.inv(V),h-omega) #shape (K,C,integral_steps)
        
        # Vhat as E(dwf)
        lhh0 = lxx(1,hs[0],self.integral_steps) #shape (K,K,integral_steps)
        lhh1 = lxx(1,hs[1],self.integral_steps)
        lhh2 = lxx(-1,hs[2],self.integral_steps)
        lhh3 = lxx(-1,hs[3],self.integral_steps)
        lhh = np.stack([lhh0,lhh1,lhh2,lhh3],axis=0) 
        lhh = np.moveaxis(lhh,0,2) #shape (K,K,C,integral_steps)
        
        IDEK = np.tensordot(np.identity(K),np.ones((C,self.integral_steps)),axes=0)
        Inversa = IDEK + np.einsum('ijcm,jk ->ikcm',lhh,V) #shape (K,K,C,integral_steps)
        Inversa = np.linalg.inv(Inversa.T).T #shape (K,K,C,integral_steps)
        Vhat = alpha * np.einsum('c,ijcm,jkcm->ikm',self.rho,Inversa,lhh).mean(axis=-1) 
        Qhat = alpha * np.einsum('ijm,jk,lkm->ilm',f,self.prob,f).mean(axis=-1) #shape (K,K)
        Mhat = alpha * np.einsum('ijm,jk->ikm',f,self.prob).mean(axis=-1) #shape (K,C)
        b = np.einsum('ijm,jk,k->im',h-M[:,:,np.newaxis],self.prob,np.ones(C)).mean(axis=-1) #shape (K)

        return Vhat, Qhat, Mhat, b

    def update_overlaps(self, Vhat, Qhat, Mhat):
        inv_hat = np.linalg.inv(self.lamb*np.identity(K) + self.delta*Vhat)
        V = self.delta * inv_hat
        Q = self.delta * inv_hat @ (self.delta * Qhat + Mhat @ self.mumu @ Mhat.T) @ inv_hat
        M = inv_hat @ Mhat @ self.mumu
        return V, Q, M
    
    def update_se(self,alpha,V, Q, M):
        # hnew = update_h(u, V, Q, M)
        Vhat, Qhat, Mhat, b = self.update_hatoverlaps(alpha,V, Q, M)
        Vnew, Qnew, Mnew = self.update_overlaps(Vhat, Qhat, Mhat)
        Vnew = self.damp(Vnew,V)
        Qnew = self.damp(Qnew,Q)
        Mnew = self.damp(Mnew,M)
        return Vnew, Qnew, Mnew, b
    
    def damp(self, new, old):
        '''
        Damping function.
        '''
        return (1-self.damping) * new + self.damping * old
    
    def train_error_loss(self,V,Q,M,b):
        sqQ=np.real(sqrtm(Q))
        omega = np.random.normal(0,1,(K,C,self.integral_steps))
        omega = np.einsum('ij,jsk -> isk',sqQ, omega) + ((M.T+b).T)[:,:,np.newaxis]
        
        vfunc0 = np.vectorize(hmin_given_w_K2,excluded=['c','V','Q','m'],signature='(),()->(n)')
        vfunc1 = np.vectorize(hmin_given_w_K2,excluded=['c','V','Q','m'],signature='(),()->(n)')
        vfunc2 = np.vectorize(hmin_given_w_K2,excluded=['c','V','Q','m'],signature='(),()->(n)')
        vfunc3 = np.vectorize(hmin_given_w_K2,excluded=['c','V','Q','m'],signature='(),()->(n)')
        h0 = np.moveaxis(vfunc0(w1=omega[0][0],w2=omega[1][0],c=0,V=V,Q=Q,m=M[:,0]),0,-1)
        h1 = np.moveaxis(vfunc1(w1=omega[0][1],w2=omega[1][1],c=1,V=V,Q=Q,m=M[:,1]),0,-1)
        h2 = np.moveaxis(vfunc2(w1=omega[0][2],w2=omega[1][2],c=2,V=V,Q=Q,m=M[:,2]),0,-1)
        h3 = np.moveaxis(vfunc3(w1=omega[0][3],w2=omega[1][3],c=3,V=V,Q=Q,m=M[:,3]),0,-1)
        h = np.stack([h0,h1,h2,h3],axis=0)
        
        h = np.swapaxes(h,0,1) #shape (K,C,integral_steps)

        aux =np.tensordot(np.array([1,1,-1,-1]),np.ones(self.integral_steps),axes=0)
        lamLoss = (phi(h)==aux).mean(axis=1)
        train_error = 1-np.dot(self.rho,lamLoss)
        train_loss = loss(aux,h).mean(axis=1)
        train_loss = np.dot(self.rho,train_loss)
        
        return train_error, train_loss
    
    def test_error(self,V,Q,M,b):
        sqQ=np.real(sqrtm(Q))
        omega = np.random.normal(0,1,(K,C,1000000))
        omega = np.einsum('ij,jsk -> isk',sqQ, omega) + ((M.T+b).T)[:,:,np.newaxis]

        aux =np.tensordot(np.array([1,1,-1,-1]),np.ones(1000000),axes=0)
        lamLoss = (phi(omega)==aux).mean(axis=1)

        return 1-np.dot(self.rho,lamLoss)
    
    def print_step_result(self,t,diff,V,Q,M,b):
        if self.error_evaluation:
            tmp_train_error, tmp_train_loss = self.train_error_loss(V[t+1], Q[t+1], M[t+1], b[t+1])
            tmp_test_error = self.test_error(V[t+1], Q[t+1], M[t+1], b[t+1])
        
        if self.printout:
            with open(self.out_txt,'a') as f:
                    f.write(f'diff: {diff}\n')
                    f.write('V:'+str(V[t+1])+'\n')
                    f.write('Q:'+str(Q[t+1])+'\n')
                    f.write('M:'+str(M[t+1])+'\n')
                    f.write('b:'+str(b[t+1])+'\n')
                    if self.error_evaluation:
                        f.write(f'train_error:{tmp_train_error}, train_loss:{tmp_train_loss}, test_error:{tmp_test_error}\n')
                        print("train_error:",tmp_train_error,"train_loss:",tmp_train_loss,"test_error:",tmp_test_error)
                    f.close()
            print('step:',t+1," diff:",diff)

        else:
            with open(self.out_txt,'a') as f:
                f.write(f'diff: {diff}\n')
                if self.error_evaluation:
                    f.write(f'train_error:{tmp_train_error}, train_loss:{tmp_train_loss}, test_error:{tmp_test_error}\n')
                f.close()
    
    def iteration(self, temp_alpha, initV, initQ, initM, initb):
        alpha = temp_alpha
        converge = False
        print('alpha:',alpha)
        print("error_evaluation:",self.error_evaluation)

        V, Q, M, b = self.initialisation(initV=initV,initQ=initQ,initM=initM,initb=initb)

        with open(self.out_txt,'w') as f:
            f.write(f'alpha = {alpha}\n')
            f.write(f'max_steps = {self.max_steps}\n')
            f.close()

        for t in tqdm(range(self.max_steps)):
            
            with open(self.out_txt,'a') as f:
                f.write(f'step: {t+1}\n')
                f.close()
            
            V[t+1], Q[t+1], M[t+1], b[t+1] = self.update_se(alpha,V[t],Q[t],M[t])

            if np.linalg.norm(V[t+1]) > 200:
                diff = get_diff(t,0.01 * V,Q,M)
            else:
                diff = get_diff(t,V,Q,M)

            self.print_step_result(t,diff,V,Q,M,b)

            if diff < self.tol:
                converge = True
                print('converge!')
                break

        Q_final = Q[t+1]
        V_final = V[t+1]
        M_final = M[t+1]
        b_final = b[t+1]
        
        tmp_test_error = self.test_error(V_final,Q_final,M_final, b_final)
        tmp_train_error, tmp_train_loss = self.train_error_loss(V_final, Q_final, M_final, b_final)
        print(f'\t\t train error:{tmp_train_error}; train loss:{tmp_train_loss}; test error:{tmp_test_error};')

        if self.printout:
            if not converge:
                print('not converge!')

            print('diff:',diff)
            print('V:',V_final)
            print('Q:',Q_final)
            print('M:',M_final)
            print('b:',b_final)

            with open(self.printout_csv, 'a') as f:
                f.write(f'{alpha},{tmp_train_error},{tmp_train_loss},{tmp_test_error},')
                f.write(f'{V_final[0][0]},{V_final[0][1]},{V_final[1][0]},{V_final[1][1]},')
                f.write(f'{Q_final[0][0]},{Q_final[0][1]},{Q_final[1][0]},{Q_final[1][1]},')
                f.write(f'{M_final[0][0]},{M_final[0][1]},{M_final[0][2]},{M_final[0][3]},')
                f.write(f'{M_final[1][0]},{M_final[1][1]},{M_final[1][2]},{M_final[1][3]},')
                f.write(f'{b_final[0]},{b_final[1]}\n')
                f.close()
            
            with open(self.out_txt,'w') as f:
                f.write('')
                f.close()

        return V_final,Q_final,M_final,b_final,tmp_test_error, tmp_train_error, tmp_train_loss

    def iterate_alphas(self, alphas):
        test_err = []
        train_err = []
        train_loss = []
        Vs = []
        Qs = []
        Ms = []
        bs = []

        print('initV:',self.initV)
        print('initQ:',self.initQ)
        print('initM:',self.initM)
        print('initb:',self.initb)

        if not self.resume:
            with open(self.out_step_csv,'w') as f:
                f.write('alpha,train_error,train_loss,test_error\n')
                f.close()
            
            if self.printout:
                with open(self.printout_csv,'w') as f:
                    f.write('alpha,train_error,train_loss,test_error,V00,V01,V10,V11,Q00,Q01,Q10,Q11,M00,M01,M02,M03,M10,M11,M12,M13,b0,b1\n')
                    f.close()
            
            with open(self.out_txt,'w') as f:
                f.write('')
                f.close()
        
        for temp_alpha in alphas:
            tmpV,tmpQ,tmpM,tmpb,temp_test_error, temp_train_err, temp_train_loss = self.iteration(temp_alpha,self.initV,self.initQ,self.initM,self.initb)
            test_err.append(temp_test_error)
            train_err.append(temp_train_err)
            train_loss.append(temp_train_loss)
            Vs.append(tmpV)
            Qs.append(tmpQ)
            Ms.append(tmpM)
            bs.append(tmpb)
            with open(self.out_step_csv,'a') as f:
                f.write(f'{temp_alpha},{temp_train_err},{temp_train_loss},{temp_test_error}\n')
                f.close()
        df_theory = pd.DataFrame({'alpha':alphas, 'train_error':train_err,'train_loss':train_loss,'test_error':test_err,'V':Vs,'Q':Qs,'M':Ms,'b':bs})
        df_theory = df_theory.loc[:, ~df_theory.columns.str.contains('^Unnamed')]
        df_theory.to_csv(self.out_final_csv)

    def iterate_follow(self,alphas):
        print("resume:",self.resume)
        if not self.resume:
            with open(self.out_step_csv,'w') as f:
                f.write('alpha,train_error,train_loss,test_error\n')
                f.close()

            if self.printout:
                with open(self.printout_csv,'w') as f:
                    f.write('alpha,train_error,train_loss,test_error,V00,V01,V10,V11,Q00,Q01,Q10,Q11,M00,M01,M02,M03,M10,M11,M12,M13,b0,b1\n')
                    f.close()

        test_err = []
        train_err = []
        train_loss = []
        Vs = []
        Qs = []
        Ms = []
        bs = []

        initV, initQ, initM, initb = self.initV, self.initQ, self.initM, self.initb

        for temp_alpha in alphas:
            tmpV,tmpQ,tmpM,tmpb,tmp_test_error, tmp_train_error, tmp_train_loss = self.iteration(temp_alpha,initV=initV,initQ=initQ,initM=initM,initb=initb)
            test_err.append(tmp_test_error)
            train_err.append(tmp_train_error)
            train_loss.append(tmp_train_loss)
            Vs.append(tmpV)
            Qs.append(tmpQ)
            Ms.append(tmpM)
            bs.append(tmpb)

            initV, initQ, initM, initb = tmpV, tmpQ, tmpM, tmpb

            with open(self.out_step_csv,'a') as f:
                f.write(f'{temp_alpha},{tmp_train_error},{tmp_train_loss},{tmp_test_error}\n')
                f.close()

            print('tmpV:',tmpV)
            print("tmpQ:",tmpQ)
            print("tmpM:",tmpM)
            print("tmpb:",tmpb)
        
        df_theory = pd.DataFrame({'alpha':alphas, 'train_error':train_err,'train_loss':train_loss,'test_error':test_err,'V':Vs,'Q':Qs,'M':Ms,'b':bs})
        df_theory = df_theory.loc[:, ~df_theory.columns.str.contains('^Unnamed')]
        df_theory.to_csv(self.out_final_csv)


                
                
