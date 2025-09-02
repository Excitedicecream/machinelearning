import streamlit as st
import pandas as pd
from nympy.random import default_rng as rng

st.title('Machine Learning App 🤖')

st.info('This app build a machine learning model')

with st.expander('Data'):
  st.write('**Raw Data')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write ('**X**')
  X=df.drop('species', axis = 1)
  X

  st.write('**y**')
  y=df.species
  y

with st.expander('Data Visualisation')
#"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  st.scatter_chart(data=df, x=bill_length_mm, y=bill_mass_g, color=species)
