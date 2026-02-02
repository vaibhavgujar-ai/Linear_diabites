import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

model = pickle.load(open("C:\\Users\\hp\\Linear_diabites\\linear_regression_model_Diabites.pkl", "rb"))
scaler = pickle.load(open("C:\\Users\\hp\\Linear_diabites\\scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    data_array = np.array(list(data.values())).reshape(1,-1)
    scaled_data = scaler.transform(data_array)
    output = model.predict(scaled_data)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    output = model.predict(final_input)[0]

    return render_template('home.html',
    prediction_text=f'The Diabetes Prediction is {output:.2f}')

if __name__ == "__main__":
    app.run(debug=True)