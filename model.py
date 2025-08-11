import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
             "occupation", "relationship", "race", "sex", "capital_gain",
             "capital_loss", "hours_per_week", "native_country", "income"]

df = pd.read_csv(url, header=None, names=col_names, na_values=" ?")
df.dropna(inplace=True)

df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

df['sex_Male'] = df['sex'].apply(lambda x: 1 if x.strip() == 'Male' else 0)
df['marital_status_Married-civ-spouse'] = df['marital_status'].apply(
    lambda x: 1 if x.strip() == 'Married-civ-spouse' else 0
)


features = [
    "age",
    "education_num",
    "hours_per_week",
    "capital_gain",
    "sex_Male",
    "marital_status_Married-civ-spouse"
]

X = df[features]
y = df['income']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


joblib.dump(model, 'adult_income_logistic_regression_model.pkl')
