import streamlit as st
import pandas as pd
import numpy as np
import pickle

clf = pickle.load(open("case_study_university.pkl","rb"))

def predict(data):
    clf = pickle.load(open("case_study_university.pkl","rb"))
    return clf.predict(data)

st.title("Advertising Spends Prediction Using Machine Learning")
st.markdown("This Model Identify Total Spends On Advertising")

st.header("Advertising Spend On Various Media")
col1,col2 = st.columns(2)

with col1:
    st.text("GRE Score")
    GRE = st.slider("GRE Score", 1.0, 10000.0, 0.5)
    st.text("TOEFL Score")
    TOEFL  = st.slider("TOEFL Score", 1.0, 10000.0, 0.5)
    st.text("University Rating")
    University  = st.slider("University Rating", 1.0, 10000.0, 0.5)
    st.text("SOP")
    SOP = st.slider("SOP", 1.0, 10000.0, 0.5)
    st.text("LOR")
    LOR = st.slider("LOR", 1.0, 10000.0, 0.5)
    st.text("CGPA")
    CGPA = st.slider("CGPA", 1.0, 10000.0, 0.5)
    st.text("Research")
    Research = st.slider("Research", 1.0, 10000.0, 0.5)
                          
st.text('')
if st.button("Sales Prediction"):
    result= clf.predict(np.array([[GRE Score,TOEFL Score,University Rating,SOP,LOR,CGPA,Research]]))
    st.text(result[0])
    
st.markdown("Developed  at Darshit")
                  
