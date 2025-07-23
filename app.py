
from flask import *
import pickle
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV
import xgboost
from xgboost import XGBClassifier

app=Flask(__name__)


def make_pred(inps):
    oe=pickle.load(open('ordinal_encoder.pkl','rb'))
    best_model=pickle.load(open('best_model.pkl','rb')) 

    data = np.array(inps).reshape(1,-1)
    prediction=best_model.predict(data)[0]
    
    if prediction==1:
        return f'Customer is likely to churn.'
    else:
        return f'Customer is likely to stay'

    



@app.route("/")
def home_fun():

    return render_template("main.html")


@app.route("/pred_link",methods=["POST"])
def check_fun():
    Geography=request.form["Geography"]
    Gender=request.form["Gender"] 
    CreditScore=float(request.form["CreditScore"])
    Age=float(request.form["Age"])
    Tenure=float(request.form["Tenure"])
    Balance=float(request.form["Balance"])
    NumOfProducts=float(request.form["NumOfProducts"])
    HasCrCard=float(request.form["HasCrCard"])
    IsActiveMember=float(request.form["IsActiveMember"])
    EstimatedSalary=float(request.form["EstimatedSalary"])

    oe=pickle.load(open('ordinal_encoder.pkl','rb'))
    encoded = oe.transform([[Geography, Gender]])[0]
    Geography_encoded = float(encoded[0])
    Gender_encoded = float(encoded[1])

    
    inps=[Geography_encoded, Gender_encoded, CreditScore, Age, Tenure, Balance,	NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
    result=make_pred(inps)
    return render_template("display.html",prediction=result)



if (__name__=="__main__"):
    app.run(debug=True)
