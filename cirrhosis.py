import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv("data/cirrhosis.csv")

"""
    
    platelets = Platelets in the blood (kiloplatelets/mL)
"""

mean1 = np.mean(data['Cholesterol'])
data['Cholesterol'] = data['Cholesterol'].fillna(mean1)

mean2 = np.mean(data['Platelets'])
data['Platelets'] = data['Platelets'].fillna(mean2)

data['Stage'].fillna(2, inplace=True)

X = data[["Age", 'Bilirubin', 'Cholesterol', 'Albumin', 'Platelets']]
y = data["Stage"]

modelCH = SVC()
modelCH.fit(X, y)

filename = 'modelCH.sav'
pickle.dump(modelCH, open(filename, 'wb'))