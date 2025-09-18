import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    mean_squared_error, r2_score, classification_report
)

# --------------------------------------
# Streamlit App
# --------------------------------------
st.title("🚗 Used Car Price Prediction App")
st.write("Upload a dataset and try different models for regression/classification.")

# --------------------------------------
# Upload dataset
# --------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    old_car = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(old_car.head())
    st.write(f"Shape: {old_car.shape}")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(old_car.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --------------------------------------
    # Preprocessing
    # --------------------------------------
    categorical_cols = ['transmission', 'brand', 'color', 'fuel_type', 'insurance_valid', 'service_history']
    for col in categorical_cols:
        if col in old_car.columns:
            old_car[col] = old_car[col].astype('category').cat.codes

    # Choose task
    task = st.radio("Choose Task", ["Regression (Predict Price)", "Classification (Cheap vs Expensive)"])

    if task == "Regression (Predict Price)":
        X = old_car.drop(columns=["price_usd"])
        y = old_car["price_usd"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model selection
        model_name = st.selectbox("Choose Model", ["Linear Regression", "KNN Regressor", "Random Forest Regressor"])

        if model_name == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        elif model_name == "KNN Regressor":
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(X_train_scaled, y_train)
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)  # Tree models don't need scaling

        # Predictions
        y_pred = model.predict(X_test_scaled if model_name != "Random Forest Regressor" else X_test)

        # Evaluation
        st.subheader("Model Evaluation")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        # Scatter Plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
        ax.set_xlabel("Actual Price (USD)")
        ax.set_ylabel("Predicted Price (USD)")
        ax.set_title("Actual vs Predicted Prices")
        st.pyplot(fig)

    else:  # Classification
        old_car['price_category'] = (old_car['price_usd'] > old_car['price_usd'].median()).astype(int)
        X = old_car.drop(columns=['price_usd', 'price_category'])
        y = old_car['price_category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Choose classifier
        model_name = st.selectbox("Choose Classifier", ["KNN Classifier", "Logistic Regression"])
        if model_name == "KNN Classifier":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Evaluation
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues")
        st.pyplot(fig)