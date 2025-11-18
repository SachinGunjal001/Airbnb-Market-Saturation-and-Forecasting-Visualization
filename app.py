import streamlit as st
import pandas as pd

st.title("My Streamlit App")
st.write("This is my first Streamlit app!")

# File upload example
uploaded_file = st.file_uploader("Upload a CSV", type="csv")

if uploaded_file:
    df = pd.read_csv('AB_NYC_2019_cleaned.csv')
    st.write("Preview:")
    st.dataframe(df)
