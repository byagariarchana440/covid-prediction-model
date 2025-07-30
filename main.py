import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


data=joblib.load("D:\demo-covid\ml_source\covid_diag.pkl")

class imp_data(BaseModel):
    Age:int
    Gender:int
    fever:int
    Cough:int
    Fatigue:int
    Breathlessness:int
    comorbidity:int
    stage:int
    Type:int
    Tumor_size:float

    app=FastAPI()

    @app.get("/")
    def root_msg():
        return {"Message": "Welcome to KariKalan magic show"}


@app.post("/predict")
def prediction(Data:inp_data):
    #inp=pd.datFrame([[Data.dict()]])
    inp=pd.DataFrame([data.dict()])
    prdd=data.predict(inp)
    return{"prediction":prdd}