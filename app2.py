import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function to load data from a CSV file
@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to select features and apply transformations
def select_and_transform_features(df):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    # Apply transformations
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])
    return preprocessor


def main():
        st.title("End-to-End Machine Learning App")
           
# Load data
file = st.file_uploader("Upload CSV file", type="csv")
if file is not None:
    df = load_data(file)

# Select features and apply transformations
if "df" in locals():
    # Ask the user to specify the target column
    target_col = st.text_input("Enter the name of the target column:")
    if target_col:
        preprocessor = select_and_transform_features(df,target_col)
        X = preprocessor.fit_transform(df.iloc[:, :-1])
        y = df[target_col]
        #fish_frame = fish_frame.iloc[:, :-1]

#     if "df" in locals():
#         preprocessor = select_and_transform_features(df)
#         X = preprocessor.fit_transform(df.drop(columns=["target"]))
#         y = df["target"]

# Split data into training and test sets
if "X" in locals() and "y" in locals():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    
# Choose machine learning model
if "X_train" in locals() and "y_train" in locals():
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select a model", ("Random Forest", "Other model"))
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    else:
        model = OtherModel(param1=value1, param2=value2)

# Train and test the model
if "model" in locals():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_absolute_error(y_test, y_pred)
    st.write(f"Test MAE: {score:.2f}")
    st.write("Test Results:")
    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe(results)

# Show feature importances
if "model_choice" in locals() and model_choice == "Random Forest":
    st.write("Feature importances:")
    importances = pd.Series(model.feature_importances_, index=df.drop(columns=["target"]).columns)
    st.write(importances.sort_values(ascending=False))

# Allow user to make predictions on new data
st.header("Make Predictions")
new_data = st.text_input("Enter new data as comma-separated values:")
if new_data:
    new_data = [float(x) for x in new_data.split(",")]
    prediction = model.predict([new_data])
    st.write(f"Prediction: {prediction[0]:.2f}")
    
    
if __name__ == "__main__":
        main()


