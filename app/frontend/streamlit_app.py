import streamlit as st
import requests

st.title("Language Detector")

text = st.text_area("Enter text:", height=100)

if st.button("Detect"):
    if text:
        try:
            response = requests.post("http://api:80/predict",
                                     json={"text": text})
            result = response.json()
            st.success(f"Detected language: {result['language']}")
        
        except Exception as e:
            st.error(f"Error: {e}")