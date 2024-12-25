import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


plt.close("all")

# load in data 
df = pd.read_csv('real estate.csv')

# drop rows that have missing values 
df = df.dropna(axis = 0)

# get correlation data 
corr = df.corr()



##plots 

# heatmap for correlation matrix 
plt.figure(1)
sns.heatmap(corr, annot = True)


# show house price per unit area over house age 
plt.figure(2)
sns.scatterplot(data = df,
                x = 'X2 house age',
                y = 'Y house price of unit area')



# show house price per unit over distance to the nearest MRT station 
plt.figure(3)
sns.scatterplot(data = df,
                x = 'X3 distance to the nearest MRT station',
                y = 'Y house price of unit area')




# show house price per unit over distance to the nearest MRT station 
plt.figure(4)
sns.scatterplot(data = df,
                x = 'X4 number of convenience stores',
                y = 'Y house price of unit area')




# show house price per unit over latitude and longitude  

fig= plt.figure(5)

ax = fig.add_subplot(projection='3d')

h = ax.scatter(df['X5 latitude'],
               df['X6 longitude'],
               df['Y house price of unit area'])


plt.colorbar(h)

ax.set_xlabel("latitude")
ax.set_ylabel("longitude")
ax.set_zlabel('house price of unit area')





## make a prediction model that predicts the house price of unit area

# Input parameters
X = df[['X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores', 
        'X5 latitude', 
        'X6 longitude']]

# Out parameter
y = df['Y house price of unit area']


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                               
                                                    random_state=42)


# define a model 
clf = MLPRegressor()

#Define the parameter grid to search over
param_grid = {'hidden_layer_sizes': [(10,10),(10,10,10),
                                     (100,100),(100,100,100)],
              'n_iter_no_change':   [1000],
              'max_iter': [200]}






# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=clf, 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1,
                           scoring = 'r2')


# train the model with different parameters
grid_search.fit(X_train, y_train)


# get the best model 
best = grid_search.best_estimator_

# predict on the test portion dataset 
y_pred = best.predict(X_test)


# calculate mean absolut error 
mae = mean_absolute_error(y_test, y_pred)

# calculate r2 score 
r2 = r2_score(y_test, y_pred)


# plot results 
fig2,ax2 = plt.subplots(1,1)

ax2.scatter(y_pred,y_test,label = 'Predictions')

ax2.set_xlabel("predicted")

ax2.set_ylabel("true")

ax2.grid()




ax2.set_xlim([0, max(max(y_test), max(y_pred))])
   
ax2.set_ylim([0, max(max(y_test), max(y_pred))])
 
ax2.plot(y_test,y_test,color = 'red',label = "True")

ax2.legend()

ax2.set_title("Comparision Prediction and true House Prices")

plt.text(1, 0.1,
         "R2 Score: " + str(round(r2,2)), 
         horizontalalignment='center',
         verticalalignment='center', 
         transform=ax.transAxes)