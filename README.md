👥 Employee Attrition Predictor
Smart Machine Learning app to predict employee attrition risk with a user-friendly interface. Built with Python, scikit-learn, and Streamlit. Perfect for HR analytics, business use, and academic demo.

🚀 Features
Instant Attrition Risk Prediction:
Interactive Streamlit dashboard for HR teams.

ML Pipeline & Evaluation:
Trains Random Forest, SVM, and Logistic Regression.
Visualizes ROC curves, feature importance, and more.

Business Insights:
Highlights risk factors, interprets prediction probability, advises on retention strategies.

Easy Customization:
Modify code and retrain on your own data.

Deployment-ready:
Requirements and deployment script for quick setup.

🗂️ Key Files
File                        |  Purpose                           
----------------------------+------------------------------------
Employee.csv                |  Sample HR dataset                 
Model_Training.py           |  End-to-end model training workflow
Model_Selection.py          |  Model & hyperparameter selection  
app.py / app_deployment.py  |  Streamlit web app                 
employee_model.pkl          |  Trained Random Forest model       
feature_names.pkl           |  Ordered list of feature columns   
requirements.txt            |  Python dependencies               
runtime.txt                 |  Runtime Python version/environment

⚡ Quickstart
1. Clone the Repository
bash
git clone https://github.com/libta018/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
Install Requirements

2. Install Requirements
bash
pip install -r requirements.txt
Train Your Own Model (optional)

3. Train Your Own Model
bash
python Model_Training.py
# or tweak hyperparameters in Model_Selection.py

4. Run the Streamlit App
bash
streamlit run app.py

5.Try Live Demo:
Open App Online

🏗️ Workflow & Implementation
1. Data Preparation
=> Cleaned, checked Employee.csv for missing/invalid values
=> Engineered features:
   => YearsInCompany: Years since joining
   => Age bands, experience bands (e.g. Mid/Senior)
   => Encoded categorical variables

2. Model Building
=> Models: Random Forest, Logistic Regression, SVM
=> Train-test split (80:20), stratified by target
=> Metrics:
   => ROC-AUC, Confusion Matrix
   => Cross-validation for robust scores

3. Evaluation & Insights
=> Visualize ROC curves and feature importances
=> Track test AUC and cross-validation scores for all models
=> Output sample predictions and advice for HR teams

5. Deployment
=> Runs locally or online using Streamlit
=> One-click prediction reports (CSV download)
=> User-friendly interface with clear risk explanations

📊 Dashboard Screenshot
<img width="1205" height="747" alt="Screenshot 2025-10-28 131135" src="https://github.com/user-attachments/assets/839d994d-47d3-4a04-bbb1-890b2c1f7dbc" />
<img width="1117" height="707" alt="Screenshot 2025-10-28 131204" src="https://github.com/user-attachments/assets/2580bd73-734a-4628-8b24-019668dc8d2f" />

💡 Sample Usage
1. Enter employee's details in the app form
2. Click “Predict Attrition Risk”
3. View risk score, tailored advice, and download the report

Feature       |  Value                 
--------------+------------------------
Age           |  28                    
Gender        |  Male                  
City          |  Bangalore             
Education     |  Bachelors             
Payment Tier  |  2                     
Experience    |  2 years               
Ever Benched  |  Yes                   
Prediction    |  Likely to Leave (0.69)
Advice: Moderate risk. Regular engagement and feedback may help retain this employee.

🧠 Business Insights
=> Attrition Rate: Automatically calculated, e.g. ~22% (on sample data)
=> Main Risk Factors: Shown using feature importance (typically: Benched status, Payment tier)
=> Retention Strategies: Tips displayed if risk is moderate/high
=> Model Performance: Test AUC, confusion matrices, cross-validation all included

🛠 Tech Stack
=> Python: pandas, numpy, scikit-learn
=> Visualization: matplotlib, seaborn
=> Web App: Streamlit
=> Serialization: joblib

📝 License
MIT License – Free for any use.

👑 Author
Mohammed Talib
Questions? Issues? Fork, star, or open an issue!

🤝 How to Contribute
=> Fork the project and branch off main
=> Submit PRs for improvements, bugfixes, or new features
=> Share feedback!
