import pandas as pd
import numpy as np


class my_svm:
    def __init__(self):
        self.w = None
        self.b = 0

    def loss_function(self, X, y, C):
        # Samples
        global n
        n = X.shape[0]

        # Equation for geometric Margines
        weight = (np.dot(self.w, self.w)) / 2

        fi = 0
        # Hinge Loss
        for i in range(X.shape[0]):
            fi += max(0, 1 - y[i] * (np.dot(self.w, X[i]) + self.b))
        

        # Retturning whole Loss
        loss = weight + (C * fi)
        return loss
    

    
    def fit(self, X, y, C, epochs, lr):
        # Gradient descent
        if self.w is None:
            self.w = np.zeros(X.shape[1])
        
        loss = self.loss_function(X, y, C)
        
        for batch in range(epochs):
            grad_w = 0
            grad_b = 0

            for i in range(n):
                loss = self.loss_function(X, y, C)
                margin = y[i] * np.dot(self.w, X[i].T) + self.b

                if margin < 1:
                    grad_w = self.w - C * y[i] * X[i]
                    grad_b = -C * y[i]
                
                else:
                    grad_w = self.w
                    grad_b = 0


            self.w = self.w - (lr * grad_w)
            self.b = self.b - (lr * grad_b)

            if batch % 10 == 0:
                print(f'Loss for {i}th batch: {loss}, Current w: {self.w}, Current b: {self.b}')


            
