
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data_dict = {}
# Importing Data
df = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\Implementations of ML\SVM_implementation\breast_cancer_datset.csv')


# Taking only two features and true classification
# For making implementation a little less complex
# And to achive nice visualization
cols_to_keep = df.columns.isin(["diagnosis","radius_mean","texture_mean"])
df = df.loc[:, cols_to_keep]     


# Setting X's and y's (train & test)
y = df["diagnosis"]
X = df.drop("diagnosis", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)


# Visualization

X_1_train = X_train["radius_mean"]
X_2_train = X_train["texture_mean"]

y_train_numbers = pd.factorize(y_train)[0]

plt.scatter(X_1_train, X_2_train, c=y_train_numbers, s=10)

plt.show()