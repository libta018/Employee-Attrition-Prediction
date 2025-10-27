# model_training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Employee.csv')

# Feature Engineering (apna same code)
df_processed = df.copy()
df_processed['YearsInCompany'] = 2024 - df_processed['JoiningYear']
df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                bins=[20, 25, 30, 35, 40], 
                                labels=['20-25', '26-30', '31-35', '36-40'])
df_processed['ExpLevel'] = pd.cut(df_processed['ExperienceInCurrentDomain'],
                                bins=[-1, 1, 3, 10],
                                labels=['Fresher(0-1)', 'Mid(2-3)', 'Senior(4+)'])

# Preprocessing
label_enc = LabelEncoder()
df_processed['Gender'] = label_enc.fit_transform(df_processed['Gender'])
df_processed['EverBenched'] = df_processed['EverBenched'].map({'Yes': 1, 'No': 0})

# One-hot encoding
categorical_cols = ['Education', 'City', 'AgeGroup', 'ExpLevel']
df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

# Drop columns
df_final = df_encoded.drop(['JoiningYear'], axis=1)

# Train-test split
X = df_final.drop('LeaveOrNot', axis=1)
y = df_final['LeaveOrNot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save everything
joblib.dump(model, 'employee_model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

print("Model training completed!")
print(f"Model Accuracy: {model.score(X_test, y_test):.4f}")