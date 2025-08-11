from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('adult_income_logistic_regression_model.pkl')

selected_features = [
    "age",
    "education_num",
    "hours_per_week",
    "capital_gain",
    "sex_Male",
    "marital_status_Married-civ-spouse"
]

placeholders = {
    "age": 37,
    "education_num": 10,
    "hours_per_week": 40,
    "capital_gain": 0
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        age = float(request.form["age"])
        education_num = float(request.form["education_num"])
        hours_per_week = float(request.form["hours_per_week"])
        capital_gain = float(request.form["capital_gain"])

        sex_value = request.form["sex"]
        marital_value = request.form["marital_status"]

        sex_male = 1 if sex_value == "Male" else 0
        married_civ_spouse = 1 if marital_value == "Married-civ-spouse" else 0

        input_vector = [age, education_num, hours_per_week, capital_gain, sex_male, married_civ_spouse]

        pred = model.predict([input_vector])[0]
        prob = model.predict_proba([input_vector])[0][pred] * 100

        prediction = ">50K Income" if pred == 1 else "<=50K Income"
        probability = round(prob, 2)

    return render_template('index.html',
                           placeholders=placeholders,
                           prediction=prediction,
                           probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
