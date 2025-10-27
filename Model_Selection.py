# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Load your data
df = pd.read_csv('Employee.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

## 📊 **EXPLORATORY DATA ANALYSIS**

plt.figure(figsize=(15, 10))

# 1. Target Variable Distribution
plt.subplot(2, 3, 1)
df['LeaveOrNot'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Employee Attrition Distribution')
plt.xlabel('Leave (1) vs Stay (0)')
plt.ylabel('Count')

# 2. Education vs Attrition
plt.subplot(2, 3, 2)
pd.crosstab(df['Education'], df['LeaveOrNot']).plot(kind='bar')
plt.title('Education vs Attrition')
plt.xlabel('Education Level')

# 3. Payment Tier vs Attrition
plt.subplot(2, 3, 3)
pd.crosstab(df['PaymentTier'], df['LeaveOrNot']).plot(kind='bar')
plt.title('Payment Tier vs Attrition')
plt.xlabel('Payment Tier')

# 4. City vs Attrition
plt.subplot(2, 3, 4)
pd.crosstab(df['City'], df['LeaveOrNot']).plot(kind='bar')
plt.title('City vs Attrition')
plt.xlabel('City')

# 5. EverBenched vs Attrition
plt.subplot(2, 3, 5)
pd.crosstab(df['EverBenched'], df['LeaveOrNot']).plot(kind='bar')
plt.title('EverBenched vs Attrition')
plt.xlabel('Ever Benched')

plt.tight_layout()
plt.show()

## 🔧 **FEATURE ENGINEERING**

# Create copy for preprocessing
df_processed = df.copy()

# Feature Engineering
df_processed['YearsInCompany'] = 2024 - df_processed['JoiningYear']
df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                bins=[20, 25, 30, 35, 40], 
                                labels=['20-25', '26-30', '31-35', '36-40'])
df_processed['ExpLevel'] = pd.cut(df_processed['ExperienceInCurrentDomain'],
                                bins=[-1, 1, 3, 10],
                                labels=['Fresher(0-1)', 'Mid(2-3)', 'Senior(4+)'])

print("\nNew Features Created:")
print(df_processed[['YearsInCompany', 'AgeGroup', 'ExpLevel']].head())

## ⚙️ **DATA PREPROCESSING**

# Initialize encoders
label_enc = LabelEncoder()

# Encode categorical variables
df_processed['Gender'] = label_enc.fit_transform(df_processed['Gender'])
df_processed['EverBenched'] = df_processed['EverBenched'].map({'Yes': 1, 'No': 0})

# One-hot encoding for categorical features
categorical_cols = ['Education', 'City', 'AgeGroup', 'ExpLevel']
df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

# Drop original columns not needed
columns_to_drop = ['JoiningYear']  # Since we have YearsInCompany
df_final = df_encoded.drop(columns=columns_to_drop, errors='ignore')

print("\nProcessed Dataset Shape:", df_final.shape)
print("\nProcessed Columns:")
print(df_final.columns.tolist())

## 🎯 **TRAIN-TEST SPLIT**

# Separate features and target
X = df_final.drop('LeaveOrNot', axis=1)
y = df_final['LeaveOrNot']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")
print(f"Target distribution in training: {y_train.value_counts().to_dict()}")

## 🔄 **FEATURE SCALING**

# Initialize scaler
scaler = StandardScaler()

# Scale numerical features
numerical_cols = ['Age', 'ExperienceInCurrentDomain', 'YearsInCompany']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("\nFeature scaling completed!")

## 🤖 **MODEL TRAINING**

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Train models and store results
results = {}
cv_results = {}

for name, model in models.items():
    # Use scaled data for Logistic Regression and SVM
    if name in ['Logistic Regression', 'SVM']:
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
    else:
        X_train_used = X_train
        X_test_used = X_test
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_used, y_train, cv=5, scoring='roc_auc')
    cv_results[name] = cv_scores
    
    # Train model
    model.fit(X_train_used, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_used)
    y_pred_proba = model.predict_proba(X_test_used)[:, 1]
    
    # Store results
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'auc_score': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{name} Results:")
    print(f"Cross-Validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Test AUC: {results[name]['auc_score']:.4f}")

## 📈 **MODEL EVALUATION**

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}\nConfusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 2. ROC Curves
plt.figure(figsize=(10, 8))

for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 3. Classification Reports
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORTS")
print("="*50)

for name, result in results.items():
    print(f"\n{name}:")
    print(classification_report(y_test, result['y_pred']))

# 4. Feature Importance (for Random Forest)
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

## 🎯 **MODEL COMPARISON**

# Compare model performance
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test_AUC': [result['auc_score'] for result in results.values()],
    'CV_Mean_AUC': [cv_results[name].mean() for name in results.keys()],
    'CV_Std_AUC': [cv_results[name].std() for name in results.keys()]
}).sort_values('Test_AUC', ascending=False)

print("\n" + "="*50)
print("MODEL PERFORMANCE COMPARISON")
print("="*50)
print(comparison_df)

# Visual comparison
plt.figure(figsize=(12, 6))

# AUC Comparison
plt.subplot(1, 2, 1)
sns.barplot(data=comparison_df, x='Test_AUC', y='Model', palette='viridis')
plt.title('Test AUC Score Comparison')
plt.xlim(0, 1)

# Cross-validation comparison
plt.subplot(1, 2, 2)
sns.boxplot(data=pd.DataFrame(cv_results), palette='Set2')
plt.title('Cross-Validation AUC Scores')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 🔧 **HYPERPARAMETER TUNING (Optional)**

# Tune Random Forest (example)
print("\nPerforming Hyperparameter Tuning for Random Forest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_tuned = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

rf_tuned.fit(X_train, y_train)

print(f"Best parameters: {rf_tuned.best_params_}")
print(f"Best cross-validation score: {rf_tuned.best_score_:.4f}")

# Evaluate tuned model
y_pred_tuned = rf_tuned.predict(X_test)
y_pred_proba_tuned = rf_tuned.predict_proba(X_test)[:, 1]
auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)

print(f"Tuned Random Forest Test AUC: {auc_tuned:.4f}")

## 💾 **FINAL MODEL SELECTION & PREDICTIONS**

# Select best model based on test AUC
best_model_name = max(results.items(), key=lambda x: x[1]['auc_score'])[0]
best_model = results[best_model_name]['model']

print(f"\n🎯 BEST MODEL: {best_model_name}")
print(f"📊 Best Test AUC: {results[best_model_name]['auc_score']:.4f}")

# Make final predictions
final_predictions = best_model.predict(X_test_used if best_model_name in ['Logistic Regression', 'SVM'] else X_test)
final_probabilities = best_model.predict_proba(X_test_used if best_model_name in ['Logistic Regression', 'SVM'] else X_test)[:, 1]

# Create results dataframe
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': final_predictions,
    'Probability_Leave': final_probabilities
})

print("\nFinal Predictions Sample:")
print(results_df.head(10))

# Save predictions
results_df.to_csv('employee_attrition_predictions.csv', index=False)
print("\nPredictions saved to 'employee_attrition_predictions.csv'")

## 📊 **BUSINESS INSIGHTS**

# Key insights from the model
print("\n" + "="*50)
print("KEY BUSINESS INSIGHTS")
print("="*50)

# Attrition rate
attrition_rate = y_test.mean() * 100
print(f"📈 Overall Attrition Rate: {attrition_rate:.2f}%")

# High-risk employees (probability > 0.7)
high_risk_employees = results_df[results_df['Probability_Leave'] > 0.7]
high_risk_rate = len(high_risk_employees) / len(results_df) * 100
print(f"⚠️  High-risk Employees (Prob > 0.7): {high_risk_rate:.2f}%")

# Model accuracy on high-risk group
if len(high_risk_employees) > 0:
    high_risk_accuracy = (high_risk_employees['Actual'] == high_risk_employees['Predicted']).mean() * 100
    print(f"🎯 Model Accuracy on High-risk Group: {high_risk_accuracy:.2f}%")

print("\nModel training and evaluation completed successfully! 🎉")