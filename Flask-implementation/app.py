# Importing important libraries
import pickle # for deserialization of saved model
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings("ignore")

# Creating a flask instance.
app = Flask(__name__)

# deserializing saved model.
model_file_path = "../final_model.pkl"
with open(model_file_path, "rb") as file:
    model = pickle.load(file)

# Creating API endpoint.
@app.post("/diabetes-predictor/") # API endpoint for POST requests only.
def predict_diagnosis(): # <-- Endpoint function
    """
    This function extracts payload posted to endpoint, feeds the data to the model
    and returns the result payload in JSON format.
    """
    # Collecting JSON payload data.
    json_data = request.json

    # Extracting each variable from JSON above.
    no_times_pregnant = json_data.get("no_times_pregnant")
    glucose_concentration = json_data.get("glucose_concentration")
    blood_pressure = json_data.get("blood_pressure")
    skin_fold = json_data.get("skin_fold")
    serum_insulin = json_data.get("serum_insulin")
    bmi = json_data.get("bmi")
    pedigree = json_data.get("pedigree")
    age = json_data.get("age")

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

    # Converting response dictionary to JSON and returning.
    return jsonify(json_response)


if __name__ == "__main__":
     app.run()