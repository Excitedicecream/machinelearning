import streamlit as st
import pandas as pd
import plotly.express as px

st.title('Machine Learning App 🤖')

st.info('This app builds a machine learning model')

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')

with st.expander('Data'):
    st.write('**Raw Data**')
    st.dataframe(df)

    st.write('**X**')
    X = df.drop('species', axis=1)
    st.dataframe(X)

    st.write('**y**')
    y = df['species']
    st.dataframe(y)

with st.expander('Data Visualisation'):
    st.write('**Scatter Plot**')

    # Dropdown for X and Y axis
    numeric_columns = df.select_dtypes(['float64', 'int64']).columns.tolist()
    x_axis = st.selectbox('Select X-axis', options=numeric_columns, index=0)
    y_axis = st.selectbox('Select Y-axis', options=numeric_columns, index=1)

    # Plot scatter chart
    st.scatter_chart(data=df, x=x_axis, y=y_axis, color='species')

with st.expander('3D Data Visualisation'):
    st.write('**3D Scatter Plot**')
    col1 = st.selectbox('Select X-axis', options=numeric_columns, index=0, key='x_axis_3d')
    col2 = st.selectbox('Select Y-axis', options=numeric_columns, index=1, key='y_axis_3d')
    col3 = st.selectbox('Select Z-axis', options=numeric_columns, index=2, key='z_axis_3d')

    fig = px.scatter_3d(df, x=col1, y=col2, z=col3, color='species', size_max=10)
    st.plotly_chart(fig, use_container_width=True)
