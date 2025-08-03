import pandas as pd
import numpy as np


class my_svm:
    def __init__(self):
        self.w = None
        self.b = 0

    def loss_function(self, X, y, C):
        # Samples
        n = X.shape[0]

        # Equation for geometric Margines
        weight = (np.dot(self.w, self.w)) / 2

        fi = 0
        # Hinge Loss
        for i in range(n):
            fi += max(0, 1 - y[i] * (np.dot(self.w, X[i]) + self.b))
        

        # Retturning whole Loss
        loss = weight + (C * fi)
        return loss
    

    
    def fit(self, X, y, C, epochs, lr):
        n = X.shape[0]
        # Gradient descent
        if self.w is None:
            self.w = np.zeros(X.shape[1])
        
        loss = self.loss_function(X, y, C)
        
        for batch in range(epochs + 1):
            grad_w = np.zeros_like(self.w)
            grad_b = 0

            for i in range(n):
                margin = y[i] * np.dot(self.w, X[i] + self.b)

                if margin < 1:
                    grad_w += -C * y[i] * X[i]
                    grad_b += -C * y[i]
                
            grad_w += self.w


            self.w = self.w - (lr * grad_w)
            self.b = self.b - (lr * grad_b)

            if batch % 10 == 0:
                loss = self.loss_function(X, y, C)
                print(f'Loss for {batch}th batch: {loss}, Current w: {np.round(self.w, 2)}, Current b: {round(self.b, 2)}')

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
        
