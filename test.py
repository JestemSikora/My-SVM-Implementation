# Importing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from my_svm import my_svm
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score


# Generating Dataset, Splitting

X, y = datasets.make_blobs(
            n_samples = 800,
            n_features=2,
            cluster_std=1,
            centers=2,
            random_state=123
)

y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=123)
X_train_1 = X_train[:,0]
X_train_2 = X_train[:,1]


model_1 = my_svm()

# Model training process
print(model_1.fit(X_train, y_train, 1.0, 100, 0.1))

# Predicting 
prediction = model_1.predict(X_test)
print(f'Example prediction: {prediction[:5]}')
print(f'Accuracy: {accuracy_score(prediction, y_test)}')
