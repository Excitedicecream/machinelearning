import streamlit as st
import panda as pd

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')

st.title('Machine Learning App 🤖')

st.info('This app build a machine learning model')
