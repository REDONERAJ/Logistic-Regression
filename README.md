# Adult Income Prediction (Logistic Regression + Flask)

This project is a **web-based machine learning application** that predicts whether an individual's annual income is **more than $50K** or **less/equal to $50K**, using a **Logistic Regression** model trained on the **Adult Income (Census) dataset** from the UCI Machine Learning Repository.

The app provides a user-friendly form with both **numeric fields** and **radio buttons** for categorical data, making it intuitive for non-technical users.

---

##  Features
- Logistic Regression model trained on curated, important features
- Tidy and clear input interface:
  - Numeric inputs for quantitative values
  - Radio buttons for categorical choices
- Outputs income prediction category with a **confidence score**
- Responsive and minimalistic Flask-based frontend

---

## Project Structure
```adult-income-logistic/
│
├── app.py # Flask application
├── model.py # Logistic Regression training script
├── adult_income_logistic_regression_model.pkl # Trained model file
├── templates/
│ └── index.html # HTML template with numeric + radio inputs
├── requirements.txt # Dependencies
└── README.md # Project documentation

```

---

## Dataset
- **Source:** [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Rows:** 48,842
- **Target:**  
  - **1** → Income > $50K  
  - **0** → Income ≤ $50K  

**Features used in the model:**
1. **Age** – Numeric (e.g., 37)  
2. **Education Number** – Numeric representation of education level (e.g., 10)  
3. **Hours Per Week** – Average working hours per week (e.g., 40)  
4. **Capital Gain** – Annual capital gain from investments/assets (e.g., 0)  
5. **Sex** – Radio option: Male / Female  
6. **Marital Status** – Radio option: Married-civ-spouse / Other  

---

## Prediction Output
The model predicts:
- **">50K Income"** – Likely earning over $50K annually  
- **"<=50K Income"** – Likely earning $50K or less annually  
- Displays **Confidence %** alongside the prediction

---

## Requirements
Flask
scikit-learn
pandas
numpy
joblib

---

## Screenshot
<img width="1366" height="640" alt="Screenshot 2025-08-11 112202" src="https://github.com/user-attachments/assets/3d3441da-1676-4800-b805-7cb24b94b1f5" />
<img width="1366" height="644" alt="Screenshot 2025-08-11 112229" src="https://github.com/user-attachments/assets/3bb6c230-5be4-4789-9207-3c4c5b0f05d7" />
<img width="1366" height="640" alt="Screenshot 2025-08-11 112306" src="https://github.com/user-attachments/assets/551deb33-ccde-42de-9334-8c1be8657407" />
