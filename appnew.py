import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('ML App')

# Load the data
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Split the data into features and target
    target_column = st.selectbox('Select the target column', df.columns)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train a model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Create a form for inputting new data
    st.subheader('Predict for new data')
    new_data = {}
    for column in X.columns:
        new_data[column] = st.number_input(column)
    new_data = pd.DataFrame(new_data, index=[0])

    # Predict the target for the new data
    prediction = model.predict(new_data)
    st.write('Prediction:', prediction[0])
