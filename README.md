# Student-Performance-Prediction-Model
📊 Student Performance Prediction (Logistic Regression)
This project focuses on predicting student academic failure using a Logistic Regression model. The model classifies students into two categories:

0 → Pass
1 → Fail

The prediction is based on selected socio-demographic and academic-related features:
sex
reason
guardian
paid
internet

🎯 Objective
The main goal of this project is to:
Build a classification model that predicts student failure
Understand the impact of key factors on student performance
Provide insights that can help improve academic outcomes

🧠 Model Used
Algorithm: Logistic Regression
Type: Binary Classification
Output: Probability of failure (converted to 0 or 1)

📂 Dataset Features
Feature	Description
sex	Gender of the student (Male/Female)
reason	Reason for choosing the school
guardian	Student's guardian (mother/father/other)
paid	Extra paid classes (yes/no)
internet	Internet access at home (yes/no)

⚙️ Methodology
Data Preprocessing
Handle missing values
Encode categorical variables (e.g., One-Hot Encoding or Label Encoding)
Feature Selection
Selected relevant predictors: sex, reason, guardian, paid, internet
Train-Test Split
Typically 80% training, 20% testing
Model Training
Logistic Regression from sklearn
Evaluation
Accuracy
Confusion Matrix
Precision, Recall, F1-score

📉 Evaluation Metrics
Accuracy Score
Confusion Matrix
Classification Report

🚀 Future Improvements
Include more features (study time, grades, absences)
Try advanced models (Random Forest, XGBoost)
Deploy as a web app (Flask/Streamlit)
Hyperparameter tuning

🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

📜 License
This project is for academic purposes and open for learning and improvement.
