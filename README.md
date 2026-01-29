# 🎓 University Admission Prediction

A simple Machine Learning web application that predicts the **probability of university admission** based on student academic details.

🔗 **Live Demo:**  
[https://university-admission-predictionn.streamlit.app/](https://university-admission-predictionn.streamlit.app/)

---

## 📌 About the Project

This project uses a trained Machine Learning model to estimate a student’s chance of getting admitted to a university.  
Users enter their academic information, and the app returns an admission probability.

---

## 📂 Project Files

- `Admission_Predict.csv` – Dataset used for training  
- `admission prediction.ipynb` – Model training notebook  
- `app.py` – Streamlit web app  
- `case_study_university.pkl` – Trained ML model  
- `requirements.txt` – Required libraries  

---

## 🧠 Model

- Type: Regression Model  
- Output: Admission probability (%)  
- Model saved using Pickle (`.pkl`)

---

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py



