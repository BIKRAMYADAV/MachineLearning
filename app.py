from fastapi import FastAPI
import numpy as np
import pickle 

app = FastAPI()

with open("model.pkl","rb") as f:
    model = pickle.load(f)
with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

@app.get("/")
def home():
    return {"message": "health risk predictor for diabetes"}

@app.post("/predict")
def predict(data:dict):
    features = np.array([
          data["Pregnancies"],
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["DiabetesPedigreeFunction"],
        data["Age"]
    ]).reshape(1,-1)

    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "risk_probabiltiy": float(probability)
    }
