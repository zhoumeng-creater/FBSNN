"""
@author: Maziar Raissi
Modified for TensorFlow 2
"""

import numpy as np
import tensorflow as tf
from FBSNNs import FBSNN  
import matplotlib.pyplot as plt
from plotting import newfig, savefig

class HamiltonJacobiBellman(FBSNN):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):
        
        super().__init__(Xi, T,
                         M, N, D,
                         layers)
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return tf.reduce_sum(Z**2, axis=1, keepdims=True) # M x 1
    
    def g_tf(self, X): # M x D
        x2 = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)          # ||X||²
        return tf.math.log1p(0.5 * x2 - 0.5)                             # log(1 + 0.5||X||² - 0.5); M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return tf.sqrt(2.0)*super().sigma_tf(t, X, Y) # M x D x D
    
    ###########################################################################


if __name__ == "__main__":
    np.random.seed(123)
    tf.random.set_seed(123)
    
    M = 100 # number of trajectories (batch size)
    N = 50 # number of time snapshots
    D = 100 # number of dimensions
    
    layers = [D+1] + 4*[256] + [1]

    Xi = np.zeros([1,D], dtype=np.float32)
    T = 1.0
         
    # Training
    model = HamiltonJacobiBellman(Xi, T,
                                  M, N, D,
                                  layers)
        
    model.train(N_Iter = 2*10**4, learning_rate=1e-3)
    model.train(N_Iter = 3*10**4, learning_rate=1e-4)
    model.train(N_Iter = 3*10**4, learning_rate=1e-5)
    model.train(N_Iter = 2*10**4, learning_rate=1e-6)
    
    
    t_test, W_test = model.fetch_minibatch()
    
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)
    
    def g(X): # MC x NC x D
        return np.log(0.5 + 0.5*np.sum(X**2, axis=2, keepdims=True)) # MC x N x 1
        
    def u_exact(t, X): # NC x 1, NC x D
        MC = 10**5
        NC = t.shape[0]
        
        W = np.random.normal(size=(MC,NC,D)) # MC x NC x D
        
        A = -g(X + np.sqrt(2.0 * np.abs(T - t)) * W)           # MC × NC × 1
        A_max = np.max(A, axis=0, keepdims=True)               # NC × 1 做基准
        return -(np.log(np.mean(np.exp(A - A_max), axis=0)) + A_max[0]) #防止大数溢出（本处为AI建议）
        
    Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:])
    
    Y_test_terminal = np.log(0.5 + 0.5*np.sum(X_pred[:,-1,:]**2, axis=1, keepdims=True))
    
    plt.figure()
    plt.plot(t_test[0:1,:,0].T,Y_pred[0:1,:,0].T,'b',label='Learned $u(t,X_t)$')
    #plt.plot(t_test[1:5,:,0].T,Y_pred[1:5,:,0].T,'b')
    plt.plot(t_test[0,:,0].T,Y_test[:,0].T,'r--',label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1,-1,0],Y_test_terminal[0:1,0],'ks',label='$Y_T = u(T,X_T)$')
    #plt.plot(t_test[1:5,-1,0],Y_test_terminal[1:5,0])
    plt.plot([0],Y_test[0,0],'ko',label='$Y_0 = u(0,X_0)$')
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    plt.legend()
    
    # savefig('./figures/HJB_Apr18_50', crop = False)
    
    denom  = np.maximum(np.abs(Y_test), 1e-12)
    errors = np.abs(Y_test - Y_pred[0,:,:]) / denom   #原式相对误差分母若接近 0 会爆炸（此处为AI建议）

    plt.figure()
    plt.plot(t_test[0,:,0],errors,'b')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    # plt.legend()
    
    # savefig('./figures/HJB_Apr18_50_errors', crop = False)