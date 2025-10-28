👥 Employee Attrition Predictor
Empower your HR team with real-time attrition risk insights using this easy-to-use Machine Learning tool.
Built with Python, scikit-learn, and Streamlit. Instantly predicts employee attrition risk—ready for business use, analytics, or academic demos.

🚀 Features
One-click Attrition Risk Prediction: Modern Streamlit dashboard for fast insights.

Robust Machine Learning Pipeline: End-to-end workflow using Random Forest, SVM & Logistic Regression.

Business-friendly Insights: Interprets risk factors and provides actionable retention advice.

Seamless Customization: Easily retrain or tweak model for your company’s data.

Ready for Deployment: All dependencies, scripts, and guides included.

🗂️ Key Files
File                        |  Description                              
----------------------------+-------------------------------------------
Employee.csv                |  Sample employee HR dataset               
Model_Training.py           |  Script for feature engineering & training
Model_Selection.py          |  Hyperparameter/model selection workflow  
app.py / app_deployment.py  |  Streamlit web dashboard                  
employee_model.pkl          |  Trained Random Forest attrition model    
feature_names.pkl           |  Required feature columns for prediction  
requirements.txt            |  Python dependencies                      
runtime.txt                 |  Python runtime version spec              

⚡ Quickstart

Clone the repository
bash
git clone https://github.com/libta018/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

Install required packages
bash
pip install -r requirements.txt

Train your own model 
bash
python Model_Training.py

Launch the Streamlit dashboard
bash
streamlit run app.py

Try the online demo:
Live App Demo


🏗️  How It Works
Data Preparation & Feature Engineering:
Checks and cleans Employee.csv
Extracts:
   Years in company
   Age group, experience level
Encodes important categorical HR variables

Model Training & Evaluation:
Trains Random Forest, Logistic Regression, SVM
Uses stratified train-test splits and cross-validation

Evaluates:
ROC-AUC scores, confusion matrix
Displays feature importance

Streamlit Dashboard:
Takes user input for live predictions
Shows probability gauge, actionable advice, and downloadable report
Visualizes all results—risk, probability, supporting factors

Deployment:
Supports local or online hosting
Export results in CSV format

📊 Dashboard Screenshot
<img width="1205" height="747" alt="Screenshot 2025-10-28 131135" src="https://github.com/user-attachments/assets/839d994d-47d3-4a04-bbb1-890b2c1f7dbc" />
<img width="1117" height="707" alt="Screenshot 2025-10-28 131204" src="https://github.com/user-attachments/assets/2580bd73-734a-4628-8b24-019668dc8d2f" />

💡 Usage Example
Fill in employee details in the app
Hit “Predict Attrition Risk”
Instantly get:
   Risk score
   Retention advice
   Downloadable report

Feature       |  Value                                             
--------------+----------------------------------------------------
Age           |  28                                                
Gender        |  Male                                              
City          |  Bangalore                                         
Education     |  Bachelors                                         
Payment Tier  |  2                                                 
Experience    |  2 years                                           
Ever Benched  |  Yes                                               
Prediction    |  Likely to Leave (0.69)                            
Advice        |  Moderate risk. Engage and give feedback to retain.

🧠 Business Insights Dashboard
Attrition Rate: Auto-calculated (Sample: ~22%)
Key Risk Factors: Easily interpreted, e.g. Benched status, Payment tier, experience
Retention Tips: Appears for moderate/high risk employees
Model Performance: ROC-AUC, confusion matrix, and feature importance visuals

🛠 Tech Stack
Python: pandas, numpy, scikit-learn, joblib
Visualization: matplotlib, seaborn
App interface: Streamlit

📝 License
MIT License — Use freely for education, research, or commercial apps.

👑 Author
Mohammed Talib
Questions? Issues? Fork, star, or open an issue!

🤝 How to Contribute
Fork the project and branch off main
Submit PRs for improvements, bugfixes, or new features
Share feedback!
