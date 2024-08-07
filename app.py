import streamlit as st
import numpy as np
import pickle

# Load the model
clf = pickle.load(open("case_study_university.pkl", "rb"))

def predict(data):
    # Predict admission chances
    return clf.predict(data)

# Streamlit interface
st.title("Case Study On University Admission Prediction")
st.markdown("Let's Predict Admission Chances")

# Sidebar inputs for user to provide data
st.sidebar.header("Input Parameters")

GRE = st.sidebar.slider("GRE Score", 1.0, 10000.0, 0.5)
TOEFL = st.sidebar.slider("TOEFL Score", 1.0, 10000.0, 0.5)
University_Rating = st.sidebar.slider("University Rating", 1.0, 5.0, 1.0)
SOP = st.sidebar.slider("SOP", 1.0, 5.0, 1.0)
LOR = st.sidebar.slider("LOR", 1.0, 5.0, 1.0)
CGPA = st.sidebar.slider("CGPA", 1.0, 10.0, 0.5)
Research = st.sidebar.slider("Research", 0.0, 1.0, 1.0)

# When the user clicks the button, make prediction
if st.button("Predict Admission Chances"):
    # Format the input data into a 2D array
    input_data = np.array([[GRE, TOEFL, University_Rating, SOP, LOR, CGPA, Research]])
    # Predict
    result = predict(input_data)
    # Display prediction
    st.success(f"Predicted Admission Chance: {result[0]:.2%}")

# Footer note
st.markdown("Developed at Suman")
