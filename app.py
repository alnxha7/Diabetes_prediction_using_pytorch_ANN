from flask import Flask, render_template, request
import pickle
import numpy as np
import torch
import torch.nn as nn
from model import ANN_Model  # Ensure the model class is imported

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("diabetes.html")

# Predictor function
def ValuePredictor(to_predict_list, size):
    to_predict = torch.tensor(to_predict_list, dtype=torch.float).reshape(1, size)

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Perform inference with the loaded model
    with torch.no_grad():
        result = loaded_model(to_predict)

    # Assuming the result is a tensor, convert to a Python-compatible format
    predicted_class = result.argmax().item()  # Get the predicted class

    return predicted_class

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = list(map(float, request.form.to_dict().values()))

        if len(to_predict_list) == 8:
            result = ValuePredictor(to_predict_list, 8)

            if result == 1:
                prediction = "Sorry, you have chances of getting the disease. Please consult a doctor."
            else:
                prediction = "No need to fear! You don't have dangerous symptoms of the disease."
            
            # Render the result.html template with the prediction text
            return render_template("result.html", prediction_text=prediction)
        else:
            return "Invalid input size. Provide 8 inputs.", 400

    return "Invalid request method. Use POST for predictions.", 405

if __name__ == "__main__":
    app.run(debug=True)
