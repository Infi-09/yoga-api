import pickle 
import numpy as np
from flask import Flask, request, redirect, url_for

dbPath = 'model/modelDB.sav'
chPath = 'model/modelCH.sav'
lcPath = 'model/modelLC.sav'

modelDB = pickle.load(open(dbPath, 'rb'))
modelCH = pickle.load(open(chPath, 'rb'))
modelLC = pickle.load(open(lcPath, 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello"

@app.route("/predictDiabetes", methods=["POST", "GET"])
def predictDiabetes():
    if request.method == "POST":
        content = request.json

        data = np.array([content['glucose'], content['bloodpressure'],content['insulin'], content['bmi'],content['age']])
        data = data.reshape(1, 5) 
        output = modelDB.predict(data)
        output = output.tolist()
        
        return {"diabetes": output}    

@app.route("/predictCirrhosis", methods=["POST", "GET"])
def predictCirrhosis():
    if request.method == "POST":
        content = request.json

        data = np.array([content['age'], content['bilirubin'],content['cholesterol'], content['albumin'],content['platelets']])
        data = data.reshape(1, 5) 
        output = modelCH.predict(data)
        output = output.tolist()
        
        return {"cirrhosis": output} 

@app.route("/predictLungCancer", methods=["POST", "GET"])
def predictCirrhosis():
    if request.method == "POST":
        content = request.json

        data = np.array([content['age'], content['smoking'],content['anxiety'], content['chronic'],content['wheezing']])
        data = data.reshape(1, 5) 

        output = modelLC.predict(data)
        output = output.tolist()
        
        return {"Lung Cancer": output} 

if __name__ == "__main__":
    app.run(debug=True)        