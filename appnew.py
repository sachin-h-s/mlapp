import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('ML App')

# Load the data
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Split the data into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train a model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Create a sidebar with a slider for selecting a sample
    sample_index = st.sidebar.slider('Select a sample', 0, len(df)-1)

    # Display the selected sample
    st.subheader('Selected sample')
    st.write(X.iloc[sample_index, :])

    # Predict the target for the selected sample
    prediction = model.predict(X.iloc[sample_index, :].values.reshape(1, -1))
    st.write('Prediction:', prediction[0])
