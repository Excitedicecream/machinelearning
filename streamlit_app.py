import streamlit as st
import panda as pd

data = pd.readcsc('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')

st.title('Machine Learning App 🤖')

st.info('This app build a machine learning model')
