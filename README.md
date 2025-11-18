# 🧬 Multiple Disease Prediction Using Machine Learning  
A Streamlit-based web application that predicts **Diabetes**, **Heart Disease**, and **Parkinson’s Disease** using trained Machine Learning models.  
This project provides an easy-to-use interface, clean UI navigation, real-time predictions, and integrated preprocessing logic.

---

## 🚀 Features

### ✔ Multiple Disease Prediction  
This application can predict the following diseases:

- **Diabetes Prediction**  
- **Heart Disease Prediction**  
- **Parkinson’s Disease Prediction**

### ✔ Clean & Modern UI  
- Sidebar navigation using `streamlit-option-menu`  
- Multi-page layout inside a single Streamlit application  
- User-friendly input fields and instant results  

### ✔ Machine Learning Models Used  
- Diabetes → SVM / Logistic Regression / Random Forest (depending on your model)  
- Heart Disease → Logistic Regression / SVM / Decision Tree  
- Parkinson’s Disease → XGBoost / SVM / Random Forest  

*(Feel free to modify models based on your training notebook)*

---

## 📁 Project Structure

📦 Multidisease Prediction
┣ 📂 notebook
┃ ┣ diabetes_model.pkl
┃ ┣ heart_model.pkl
┃ ┗ parkinsons_model.pkl
┣ 📂 images (optional)
┣ app.py
┣ requirements.txt
┗ README.md



---

## 🛠 Tech Stack

### **Frontend**
- Streamlit
- streamlit-option-menu

### **Backend**
- Python
- Pickle (for ML model serialization)

### **Machine Learning**
- Scikit-learn
- Pandas
- NumPy

---

## 🧪 How to Run the Project Locally

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/your-username/multi-disease-prediction.git
cd multi-disease-prediction

2️⃣ Create & activate virtual environment
 i. python -m venv ven
For Windows:
venv\Scripts\activate
For Linux/Mac:
source venv/bin/activate

3️⃣ Install required dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
python -m streamlit run app.py


Your app will open automatically in a browser at:

http://localhost:8501/

🧠 How It Works
1. User enters medical data in the input fields
2. Streamlit processes the input
3. Input is converted into a NumPy array
4. Pretrained ML model predicts the disease
5. Result is displayed instantly

📦 Deployment (Optional)

You can deploy this application to:

Streamlit Cloud

Render

Railway

Heroku (Buildpack Python)

Example (Streamlit Cloud):

Push your code to GitHub

Go to https://share.streamlit.io

Select the repo

Deploy


🔧 Requirements

Create a file named requirements.txt:

streamlit
streamlit-option-menu
numpy
pandas
scikit-learn
pickle-mixin


If using joblib:

joblib

🧑‍💻 Author

Vivek Kumar
Machine Learning & Full Stack Developer

🤝 Contributing

Contributions are always welcome!
You can contribute by:

Fixing bugs

Improving UI

Adding new models

Improving documentation

Steps:

Fork the project

Create your feature branch

Commit changes

Push to your branch

Open a Pull Request
