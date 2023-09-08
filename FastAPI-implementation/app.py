# Importing important libraries
import pickle # for deserialization of saved model
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel # Helps parse & validate payload content.

import warnings
warnings.filterwarnings("ignore")

# Creating instance of FASTAPI.
app = FastAPI()

# deserializing saved model.
model_file_path = "../final_model.pkl"
with open(model_file_path, "rb") as file:
    model = pickle.load(file)

# Defining JSON payload input structure.
class PatientData(BaseModel):
    no_times_pregnant: int
    glucose_concentration: float 
    blood_pressure: float
    skin_fold: float
    serum_insulin : float
    bmi : float
    pedigree : float
    age : int


# Async function for endpoint.
@app.post("/diabetes-predictor-FASTAPI")
async def create_item(data: PatientData):
    
    # Extracting each variable from PatientData class.
    no_times_pregnant = data.no_times_pregnant
    glucose_concentration = data.glucose_concentration
    blood_pressure = data.blood_pressure
    skin_fold = data.skin_fold
    serum_insulin = data.serum_insulin
    bmi = data.bmi
    pedigree = data.pedigree
    age = data.age

    # Storing variables in list for model prediction
    variables = [no_times_pregnant, glucose_concentration, blood_pressure, skin_fold, serum_insulin, bmi, pedigree, age]

    # Passing variables into saved model for prediction. Remember to pass the variables in the right shape.
    result = model.predict([variables])
    
    # Formatting result
    if result[0] == 0:
            status = "Negative"
    else:
        status = "Positive"

    # Defining output payload
    json_response = {
        "Prediction": str(result[0]),
        "Result": status
    }
    

    return json_response

if __name__ == "__main__":
    uvicorn.run(app)