import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

diabetes = pd.read_csv("data/diabetes.csv")

"""
    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    BloodPressure: Diastolic blood pressure (mm Hg)
    Insulin: 2-Hour serum insulin (mu U/ml)
    BMI: Body mass index (weight in kg/(height in m)^2) DiabetesPedigreeFunction: Diabetes pedigree function
    Age: Age (years)
"""

diabetes.drop(["Pregnancies", "SkinThickness", "DiabetesPedigreeFunction"], axis=1, inplace=True)

X = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

tree = DecisionTreeClassifier()
tree.fit(X, y)

filename = 'modelDB.sav'
pickle.dump(tree, open(filename, 'wb'))