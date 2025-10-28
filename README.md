# 👥 Employee Attrition Predictor  

Empower your HR team with real-time **attrition risk insights** using this easy-to-use **Machine Learning web app**.  
Built with **Python, Scikit-learn, and Streamlit**, this tool instantly predicts employee attrition risk — perfect for **business use, analytics, or academic demos**.  

---

## 🚀 Live Demo  

🔗 [Try the App Here](https://employee-attrition-prediction-ml-model.streamlit.app)  

---

## 📊 Features  

- ✔️ One-click **Attrition Risk Prediction** via Streamlit dashboard  
- ✔️ **End-to-end ML pipeline** using Random Forest, SVM & Logistic Regression  
- ✔️ **Business-friendly insights** with actionable retention advice  
- ✔️ **Easily customizable** — retrain with your company’s HR data  
- ✔️ **Ready for deployment** — all scripts, dependencies & runtime included  

---

## 🛠️ Tech Stack  

- **Python 3.9+**  
- **Streamlit** – Web App Framework  
- **Scikit-Learn** – Machine Learning  
- **Pandas & NumPy** – Data Processing  
- **Joblib** – Model Serialization  
- **Matplotlib & Seaborn** – Visualization  

---

## ⚙️ How It Works  

The app predicts **whether an employee is likely to leave** based on various HR parameters such as:  

- Age  
- Gender  
- Education  
- City  
- Payment Tier  
- Experience (Years in Company)  
- Benched Status, etc.  

The trained **Random Forest model** processes these inputs and outputs the **attrition risk score** along with **data-driven retention advice**.  

---

## 📸 Screenshots  

![Dashboard Screenshot 1](<img width="1205" height="747" alt="Screenshot 2025-10-28 131135" src="https://github.com/user-attachments/assets/0846a918-c425-43a2-89ff-bc03ce5f0256" />
)  
![Dashboard Screenshot 2](<img width="1117" height="707" alt="Screenshot 2025-10-28 131204" src="https://github.com/user-attachments/assets/df8d26fc-6429-4578-9670-d2df44d1ae35" />
)  

---

## 📂 Project Structure  

Employee-Attrition-Prediction/
│-- Employee.csv             # Sample HR dataset  
│-- Model_Training.py        # Script for feature engineering & training  
│-- Model_Selection.py       # Model comparison & evaluation workflow  
│-- app.py / app_deployment.py  # Streamlit web app  
│-- employee_model.pkl       # Trained Random Forest model  
│-- feature_names.pkl        # Feature names for prediction  
│-- requirements.txt         # Dependencies  
│-- runtime.txt              # Python runtime version  
│-- README.md                # Documentation  
│-- Screenshots/             # Dashboard preview images  


---

## 📋 Requirements  


streamlit>=1.26.0  
pandas>=2.1.0  
numpy>=1.26.0  
scikit-learn>=1.3.0  
matplotlib>=3.8.0  
seaborn>=0.13.0  
joblib>=1.3.0  


---

## ⚡ Quickstart  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/libta018/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction

2. Install dependencies
pip install -r requirements.txt

3. Train your own model
python Model_Training.py

4. Run the Streamlit dashboard
streamlit run app.py

---

##💡 Example Prediction

| Feature        | Value                                              |
| -------------- | -------------------------------------------------- |
| Age            | 28                                                 |
| Gender         | Male                                               |
| City           | Bangalore                                          |
| Education      | Bachelors                                          |
| Payment Tier   | 2                                                  |
| Experience     | 2 years                                            |
| Ever Benched   | Yes                                                |
| **Prediction** | Likely to Leave (0.69)                             |
| **Advice**     | Moderate risk. Engage and give feedback to retain. |

## 🧠 Business Insights Dashboard  

📈 **Attrition Rate:** Auto-calculated (Sample: ~22%)  

🔍 **Key Risk Factors:** Benched status, Payment tier, Experience level  

💬 **Retention Tips:** Personalized for moderate/high-risk employees  

📊 **Model Metrics:** ROC-AUC, Confusion Matrix, Feature Importance  

---

## 📝 License  

**MIT License** — Free to use for education, research, or commercial purposes.  

---

## 👨‍💻 Author  

**Mohammed Talib**  

📧 Email: [mohammedtalib306@gmail.com](mailto:mohammedtalib306@gmail.com)  
💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/mohammed-talib-analytics-ds)  
🌐 Live Demo: [Try it Here](https://employee-attrition-prediction-ml-model.streamlit.app)  

---

## 🤝 How to Contribute  

- Fork the project & create a feature branch  
- Submit PRs for improvements or bug fixes  
- ⭐ Star the repo if you found it useful  
- Share feedback and suggestions!  


