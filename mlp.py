import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
data = pd.read_csv('data.csv', header=None)
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state= 0)
