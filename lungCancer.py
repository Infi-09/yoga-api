import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/lungCancer.csv")

"""
    
"""

X = df[["AGE", "SMOKING", 'ANXIETY', 'CHRONIC DISEASE', 'WHEEZING']]
y = df['LUNG_CANCER']
 
modelLC = DecisionTreeClassifier() 
modelLC.fit(X, y)

filename = 'modelLC.sav'
pickle.dump(modelLC, open(filename, 'wb'))
   