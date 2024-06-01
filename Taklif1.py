from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#---------------------------------
#Taklif1: Load Dataset
url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv(url1, names=names)
#---------------------------------
# Taklif1:1
print(dataset.shape)
#---------------------------------
# Taklif1:2
dataset.isnull().sum()
#---------------------------------
# Taklif1:3
np.max(dataset.sepal_length)
np.min(dataset.sepal_length)
np.mean(dataset.sepal_length)
np.median(dataset.sepal_length)
np.var(dataset.sepal_length)
np.percentile(dataset.sepal_length, [25, 50, 75])
#---------------------------------
# Taklif1:4
dataset.columns
#---------------------------------
# Taklif1:5
petal=dataset[["petal_length", "petal_width"]]
#---------------------------------
# Taklif1:6
dataset.hist()
pd.plotting.scatter_matrix(dataset, alpha=0.2)
#---------------------------------
# Taklif1:7
dataset.head(3)
dataset.tail(3)
#---------------------------------
# Taklif1:8
dataset.sort_values(by='sepal_width', ascending=False)
#---------------------------------
# Taklif1:9
np.cov(dataset.sepal_length, dataset.sepal_width)
#---------------------------------