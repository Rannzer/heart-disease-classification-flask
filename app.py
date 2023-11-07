from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("Heart-Disease-Prediction-Model.pkl","rb"))

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    output = prediction

    if(output == 2):
        return render_template("index.html",result = "The patient has Heart Disease")
    else:
        return render_template("index.html",result = "The patient does not have Heart Disease")


if __name__ == "__main__":
    app.run(debug=True)