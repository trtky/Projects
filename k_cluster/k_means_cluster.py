import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

import plotly as py
import plotly.graph_objs as go

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')





df = pd.read_excel('segmented_customers.xlsx')




df = df.dropna()


sns.scatterplot(data = df, x = 'Age', y = 'Spending Score (1-100)')




X = np.array([df['Age'],df['Spending Score (1-100)']])

X = np.transpose(X)

intertias = []

my_k = []

for k in range(2,8): 

    clf = KMeans(n_clusters = k) 
   
    clf.fit(X)

    my_k.append(k)
    intertias.append(clf.inertia_)



fig,ax = plt.subplots(1,1)

ax.plot(my_k,intertias)
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
ax.grid()
ax.set_title("k-means cluster")


# we pick k = 4 


clf_picked =  KMeans(n_clusters = 4) 

clf_picked.fit(X)


labels1 = clf_picked.labels_



fig2,ax2 = plt.subplots()


colors = {0:'red',
          1:'blue',
          2:'green',
          3: 'yellow'}


my_labels = []

for i in range(len(labels1)):
    
    
    
    if labels1[i] not in my_labels:
        
        ax2.scatter(X[i,0],X[i,1],label = labels1[i],color = colors[labels1[i]] )
    
        my_labels.append(labels1[i])
    
    else:
        ax2.scatter(X[i,0],X[i,1],color = colors[labels1[i]] )



ax2.legend()
ax2.grid()
ax2.set_xlabel("Age")
ax2.set_ylabel("Spending Score (1-100)]")