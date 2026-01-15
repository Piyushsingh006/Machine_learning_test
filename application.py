from flask import Flask,request,render_template,jsonify
import numpy as np
import sklearn.preprocessing as Standardscaler
import pandas as pd
import pickle

application = Flask(__name__)
app = application

# import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
Standardscaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/Predictdata",methods=['GET','POST'])
def Predict_datapoint():
    if request.method=="POST":
       Temp = float(request.form.get("Temp"))
       RH = float(request.form.get("RH"))
       Ws = float(request.form.get("Ws"))
       Rain = float(request.form.get("Rain"))
       FFMC = float(request.form.get("FFMC"))
       DMC = float(request.form.get("DMC"))
       DC = float(request.form.get("DC"))
       ISI = float(request.form.get("ISI"))
       BUI = float(request.form.get("BUI"))

       new_scaled_data=Standardscaler.transform([[Temp,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI]])
       result=ridge_model.predict(new_scaled_data)
       return render_template('home.html',results=result[0])
    else:
         return render_template('home.html')
    
     


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
