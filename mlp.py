import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

data = pd.read_csv('data.csv', header=None)
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state= 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# classifier.add(Dropout(p=0.1))