from flask import Flask, request, jsonify
from flask_cors import CORS  # ← Add this line
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # ← Allow cross-origin requests

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Credit Card Approval Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        features = [
            float(data['CODE_GENDER']),
            float(data['Car_Owner']),
            float(data['Propert_Owner']),
            float(data['CNT_CHILDREN']),
            float(data['Annual_income']),
            float(data['NAME_INCOME_TYPE']),
            float(data['EDUCATION']),
            float(data['NAME_HOUSING_TYPE']),
            -1 * 365 * float(data['AGE_YEARS']),
            365 * float(data['EMP_YEARS']),
            float(data['CNT_FAM_MEMBERS']),
            float(data['paid_off']),
            float(data['past_dues']),
            float(data['no_loan'])
        ]

        prediction = model.predict([features])[0]
        result = "Approved" if prediction == 1 else "Rejected"

        return jsonify({'approval': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
