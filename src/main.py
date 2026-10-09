from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "model", "titanic_model.pkl")
)

print("Loading model from:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

# Create FastAPI application
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML API for predicting Titanic passenger survival",
    version="1.0.0"
)


# Input data structure
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running",
        "version": "1.0.0"
    }


# Prediction endpoint
@app.post("/predict")
def predict(passenger: Passenger):

    # Convert input into DataFrame
    input_data = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Sex": passenger.Sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked": passenger.Embarked
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability
    probability = model.predict_proba(input_data)[0]

    # Convert prediction to readable output
    if prediction == 1:
        result = "Survived"
    else:
        result = "Did not survive"

    return {
        "prediction": int(prediction),
        "result": result,
        "survival_probability": round(float(probability[1]), 4)
    }