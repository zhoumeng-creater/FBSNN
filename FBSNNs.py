"""
@author: Maziar Raissi
Modified for TensorFlow 2 by Meng
Modified to use Deep BSDE architecture from solver.py
"""

import numpy as np
import tensorflow as tf
import time
from abc import ABC, abstractmethod

DELTA_CLIP = 50.0  # From solver.py


class FeedForwardSubNet(tf.keras.Model):
    """Feed-forward subnet used at each time step (from solver.py)"""
    def __init__(self, dim, num_hiddens):
        super(FeedForwardSubNet, self).__init__()
        
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training=True):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training=training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training=training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)
        return x


class FBSNN(ABC): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T,
                       M, N, D,
                       layers,  # Kept for compatibility, but not used
                       num_hiddens=[50, 50, 50],
                       y_init_range=[-0.5, 0.5]):
        
        self.Xi = Xi # initial point
        self.T = T # terminal time
        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions
        
        self.delta_t = T / N
        self.num_time_interval = N
        
        # layers - kept for compatibility but using num_hiddens instead
        self.layers = layers
        self.num_hiddens = num_hiddens
        
        # Initialize trainable parameters (from solver.py)
        self.y_init = tf.Variable(
            tf.random.uniform([1], y_init_range[0], y_init_range[1]),
            dtype=tf.float32,
            name='y_init'
        )
        self.z_init = tf.Variable(
            tf.random.uniform([1, D], -0.1, 0.1),
            dtype=tf.float32,
            name='z_init'
        )
        
        # Create N-1 subnets with non-shared weights (from solver.py)
        self.subnet = [FeedForwardSubNet(D, num_hiddens) for _ in range(N-1)]
        
        # Collect all trainable variables
        self.trainable_variables = [self.y_init, self.z_init]
        for subnet in self.subnet:
            self.trainable_variables.extend(subnet.trainable_variables)
        
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8)
        
        # Convert Xi to tensor
        self.Xi_tensor = tf.constant(self.Xi, dtype=tf.float32)
        
        # Time stamps
        self.time_stamp = np.arange(0, self.num_time_interval) * self.delta_t
    
    # Kept for compatibility but not used
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    @tf.function
    def net_u(self, t, X): # M x 1, M x D
        # Kept for compatibility but not used
        with tf.GradientTape() as tape:
            tape.watch(X)
            u = self.neural_net(tf.concat([t,X], 1), self.weights, self.biases) # M x 1
        Du = tape.gradient(u, X) # M x D
        
        return u, Du

    @tf.function
    def Dg_tf(self, X): # M x D
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)
        return tape.gradient(g, X) # M x D
    
    @tf.function
    def forward_pass(self, dw, x, training=True):
        """
        Forward pass using Deep BSDE method (from solver.py)
        dw: M x D x N
        x: M x D x (N+1)
        """
        all_one_vec = tf.ones(shape=[self.M, 1], dtype=tf.float32)
        y = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)
        
        # Time marching (from solver.py)
        for t in range(0, self.num_time_interval-1):
            # BSDE: y = y - dt * f(t,x,y,z) + z * dW
            y = y - self.delta_t * self.phi_tf(
                tf.constant([[self.time_stamp[t]]], dtype=tf.float32),
                x[:, :, t], y, z
            ) + tf.reduce_sum(z * dw[:, :, t], axis=1, keepdims=True)
            
            # Update z using subnet
            z = self.subnet[t](x[:, :, t + 1], training=training) / self.D
        
        # Terminal time
        y = y - self.delta_t * self.phi_tf(
            tf.constant([[self.time_stamp[-1]]], dtype=tf.float32),
            x[:, :, -2], y, z
        ) + tf.reduce_sum(z * dw[:, :, -1], axis=1, keepdims=True)
        
        return y
    
    @tf.function
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        """
        Loss function based on terminal condition (from solver.py)
        """
        # Generate paths
        X_list = []
        
        t0 = t[:,0,:]
        W0 = W[:,0,:]   
        X0 = tf.tile(Xi,[self.M,1]) # M x D
        X_list.append(X0)
        
        # Generate X path and collect dW
        dW_list = []
        for n in range(0,self.N):
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]
            
            # Placeholder Y and Z for path generation
            Y_temp = tf.zeros([self.M, 1])
            Z_temp = tf.zeros([self.M, self.D])
            
            # Calculate dW
            dW = W1 - W0
            dW_list.append(dW)
            
            # Update X
            sigma = self.sigma_tf(t0, X0, Y_temp)
            dW_expanded = tf.expand_dims(dW, -1)
            X1 = X0 + self.mu_tf(t0, X0, Y_temp, Z_temp) * (t1 - t0) + \
                 tf.squeeze(tf.matmul(sigma, dW_expanded), axis=[-1])
            
            X_list.append(X1)
            
            t0 = t1
            W0 = W1
            X0 = X1
        
        # Prepare data in correct format
        X = tf.stack(X_list, axis=2)  # M x D x (N+1)
        dW = tf.stack(dW_list, axis=2)  # M x D x N
        
        # Forward pass
        y_terminal = self.forward_pass(dW, X, training=True)
        
        # Terminal condition loss (from solver.py)
        delta = y_terminal - self.g_tf(X[:, :, -1])
        # Use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, 
                                      tf.square(delta),
                                      2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        
        # For compatibility, return additional values
        X_return = tf.transpose(X, [0, 2, 1])  # M x (N+1) x D
        Y_return = tf.zeros([self.M, self.N+1, 1])  # Placeholder
        Y0_return = self.y_init[0]
        
        return loss, X_return, Y_return, Y0_return
    
    @tf.function
    def train_step(self, Xi, t_batch, W_batch, learning_rate):
        
        with tf.GradientTape() as tape:
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, Xi)
        
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)  #防止梯度爆炸
        self.optimizer.learning_rate.assign(learning_rate)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss, Y0_pred

    def fetch_minibatch(self):
        T = self.T
        
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D
        
        dt = T/N
        
        Dt[:,1:,:] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D)).astype(np.float32)    #采样时直接生成 float32
        
        t = np.cumsum(Dt,axis=1) # M x (N+1) x 1
        W = np.cumsum(DW,axis=1) # M x (N+1) x D
        
        return t, W
    
    def train(self, N_Iter, learning_rate, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
            
        start_time = time.time()
        for it in range(N_Iter):
            
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
            
            # Convert to tensors
            t_batch_tf = tf.constant(t_batch, dtype=tf.float32)
            W_batch_tf = tf.constant(W_batch, dtype=tf.float32)
            
            loss_value, Y0_value = self.train_step(self.Xi_tensor, t_batch_tf, W_batch_tf, learning_rate)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' % 
                      (it, loss_value.numpy(), Y0_value.numpy(), elapsed, learning_rate))
                start_time = time.time()
                
    
    def predict(self, Xi_star, t_star, W_star):
        
        Xi_star_tf = tf.constant(Xi_star, dtype=tf.float32)
        t_star_tf = tf.constant(t_star, dtype=tf.float32)
        W_star_tf = tf.constant(W_star, dtype=tf.float32)
        
        _, X_star, Y_star, _ = self.loss_function(t_star_tf, W_star_tf, Xi_star_tf)
        
        return X_star.numpy(), Y_star.numpy()
    
    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x1
    
    @abstractmethod
    def g_tf(self, X): # M x D
        pass # M x 1
    
    @abstractmethod
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return tf.zeros([M,D], dtype=tf.float32) # M x D
    
    @abstractmethod
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.linalg.diag(tf.ones([M,D], dtype=tf.float32)) # M x D x D
    ###########################################################################