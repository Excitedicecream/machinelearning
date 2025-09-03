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


# ✅ Detailed Statistics with Dropdown
with st.expander("Show Detailed Statistics"):
    st.write("### Basic Statistics")

    # Select numeric column for statistics
    numeric_columns = df.select_dtypes(['float64', 'int64']).columns.tolist()
    selected_column = st.selectbox("Select a column to view statistics", options=numeric_columns)

    # Calculate stats for selected column
    stats = {
        "Min": df[selected_column].min(),
        "Max": df[selected_column].max(),
        "Mean": df[selected_column].mean(),
        "Median": df[selected_column].median(),
        "Std Dev": df[selected_column].std(),
        "Variance": df[selected_column].var()
    }

    # Display stats
    st.write(f"### Statistics for **{selected_column}**")
    st.write(pd.DataFrame(stats, index=[selected_column]))

    # Optional: Full describe table for all columns
    with st.expander("Show All Columns Summary"):
        st.write(df.describe())

    
with st.expander('2D Data Visualisation'):
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


# Data Preparation
with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.selectbox('Sex', ('female', 'male'))
    bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill Depth (mm)', min_value=13.1, max_value=21.5, value=17.0)
    flipper_length_mm = st.slider('Flipper Length (mm)', min_value=172, max_value=231, value=200)
    body_mass_g = st.slider('Body Mass (g)', min_value=2700, max_value=6300, value=4200)

# ✅ Create a dataframe for the input features
data = {
    'island': [island],
    'sex': [sex],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g]
}

input_df = pd.DataFrame(data)
input_penguins= pd.concat([input_df,X],axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    input_df
    st.write('Combined penguin data**')
    input_penguins

#Encode
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, columns=encode)
df_penguins
