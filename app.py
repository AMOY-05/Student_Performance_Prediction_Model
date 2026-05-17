import pandas as pd
import numpy as np
#import matplotlib as plt

df = pd.read_csv("student-mat.csv", sep=';')

print(df.columns.tolist())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Now this will work perfectly
#X = df[['sex', 'reason', 'guardian', 'paid', 'internet']]

X = df[['school', 'sex', 'reason', 'guardian', 'paid', 'internet', 'studytime', 'age', 'health', 'absences']]
y = df[['failures']]

X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))

import pickle
pickle.dump(model, open("model.pkl", "wb"))

from flask import Flask, request, jsonify

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "Student Predictor API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['sex'], data['reason'], data['guardian'], data['paid'], data['internet']]])
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
