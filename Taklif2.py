from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
#---------------------------------
#Taklif2: Load DataSet
url2="F:\data\weight-height.csv"
#https://github.com/kaboutari/itcsam_data_science_1/blob/master/weight-height.csv
dataset2 = pd.read_csv(url2)
#---------------------------------
height=dataset2.Height.array.reshape(-1, 1)
weight=dataset2.Weight.array.reshape(-1, 1)
model=LinearRegression()
model.fit(X=height, y=weight)
w=model.predict([[66.461]])
print(w)
#---------------------------------
plt.xlabel('Height in Inches')
plt.ylabel('Weight in Pounds')
plt.plot(height, weight)
plt.grid(True)
plt.show()
#---------------------------------
