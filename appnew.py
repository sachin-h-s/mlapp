# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# st.title('ML App')

# # Load the data
# uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     # Split the data into features and target
#     X = df.drop('target', axis=1)
#     y = df['target']

#     # Train a model
#     model = RandomForestClassifier()
#     model.fit(X, y)

#     # Create a form for inputting new data
#     st.subheader('Predict for new data')
#     new_data = {}
#     for column in X.columns:
#         new_data[column] = st.number_input(column)
#     new_data = pd.DataFrame(new_data, index=[0])

#     # Predict the target for the new data
#     prediction = model.predict(new_data)
#     st.write('Prediction:', prediction[0])
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.title('ML App')

# Load the data
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Split the data into features and target
    target_column = st.selectbox('Select the target column', df.columns)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Perform numerical and categorical transformation on X
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    X_categorical = X.select_dtypes(include=['object'])
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_categorical_encoded = encoder.fit_transform(X_categorical)
    X_transformed = pd.concat([pd.DataFrame(X_numeric_scaled), pd.DataFrame(X_categorical_encoded.toarray())], axis=1)

    # Train a model
    model = LinearRegression()()
    model.fit(X_transformed, y)

    # Create a form for inputting new data
    st.subheader('Predict for new data')
    new_data = {}
    for column in X.columns:
        if column in X_numeric:
            new_data[column] = st.number_input(column)
        else:
            new_data[column] = st.selectbox(column, X_categorical[column].unique())
    new_data = pd.DataFrame(new_data, index=[0])

    # Perform numerical and categorical transformation on new_data
    new_data_numeric = new_data.select_dtypes(include=['int64', 'float64'])
    new_data_categorical = new_data.select_dtypes(include=['object'])
    new_data_numeric_scaled = scaler.transform(new_data_numeric)
    new_data_categorical_encoded = encoder.transform(new_data_categorical)
    new_data_transformed = pd.concat([pd.DataFrame(new_data_numeric_scaled), pd.DataFrame(new_data_categorical_encoded.toarray())], axis=1)

    # Predict the target for the new data
    prediction = model.predict(new_data_transformed)
    st.write('Prediction:', prediction[0])
