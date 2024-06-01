from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
#---------------------------------
#Taklif2: Load DataSet
url2="F:\data\weight-height.csv"
https://gist.github.com/nstokoe/7d4717e96c21b8ad04ec91f361b000cb#file-weight-height-csv
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