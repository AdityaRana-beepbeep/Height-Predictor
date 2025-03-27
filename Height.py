import streamlit as slt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

slt.set_page_config(page_title="Height Prediction", layout="wide")

uploaded_file = slt.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    slt.write("Dataset Preview", df.head())

    slt.sidebar.title("Father & Son Height Analysis")
    unit = slt.sidebar.radio("Choose the unit of measurement (cm/inches): ", ("cm", "inches"))
    father_height = slt.sidebar.number_input("Enter father's height (inches/cm) for prediction", min_value=60, max_value=270, value=70)

    slt.sidebar.header("Son's Height Prediction Model")
    slt.sidebar.subheader("Select Columns:")
    x_col = slt.sidebar.selectbox("Select Feature (X-axis)", df.columns, index=0)
    y_col = slt.sidebar.selectbox("Select Target (Y-axis)", df.columns, index=1)

    if df[[x_col, y_col]].isnull().any().any():
        slt.error("The selected columns contain missing values. Please clean the data.")
    else:
        X = df[[x_col]]
        y = df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        slt.write("Regression Model Results")
        slt.write(f"**Selected Feature (X-axis):** {x_col}")
        slt.write(f"**Selected Target (Y-axis):** {y_col}")
        slt.write(f"**Intercept:** {model.intercept_}")
        slt.write(f"**Slope (Coefficient):** {model.coef_[0]}")

        if slt.sidebar.button("Predict"):
            predicted_son_height = model.predict([[father_height]])[0]
            slt.sidebar.write(f"Predicted Son's Height: {predicted_son_height:.2f} ({unit})")

        slt.write("### Data Visualization & Regression Line")
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', label="Data Points")
        plt.plot(X, model.predict(X), color='red', label="Regression Line")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Linear Regression: {y_col} vs {x_col}")
        plt.legend()
        slt.pyplot(plt)
else:
    slt.write("Please upload a CSV file to begin.")
