import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.set_page_config(
    page_title="AI Predictive Analytics Dashboard",
    layout="wide"
)

st.title("📈 AI Predictive Analytics Dashboard")

st.markdown(
    "### Forecast Business Sales Using Historical Data"
)

# Load Data
data = pd.read_csv("data/business_sales_data.csv")

# Sidebar
st.sidebar.header("Dashboard Filters")

region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + list(data["Region"].unique())
)

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(data["Category"].unique())
)

filtered_data = data.copy()

if region != "All":
    filtered_data = filtered_data[
        filtered_data["Region"] == region
    ]

if category != "All":
    filtered_data = filtered_data[
        filtered_data["Category"] == category
    ]

# Display Dataset
st.subheader("📊 Business Dataset")

st.dataframe(filtered_data)

# KPI Metrics
total_sales = filtered_data["Sales"].sum()
total_profit = filtered_data["Profit"].sum()
total_customers = filtered_data["Customers"].sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Customers", f"{total_customers:,}")

# Features and Target
X = filtered_data[
    ["Month", "Advertising", "Customers"]
]

y = filtered_data["Sales"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("🤖 Model Accuracy")

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"{mae:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("R² Score", f"{r2:.2f}")

# Sales Trend Chart
st.subheader("📉 Sales Trend Analysis")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    filtered_data["Month"],
    filtered_data["Sales"],
    marker='o'
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales Trend")

st.pyplot(fig)

# Profit Trend
st.subheader("💰 Profit Analysis")

fig2, ax2 = plt.subplots(figsize=(12,5))

ax2.bar(
    filtered_data["Month"],
    filtered_data["Profit"]
)

ax2.set_xlabel("Month")
ax2.set_ylabel("Profit")
ax2.set_title("Monthly Profit")

st.pyplot(fig2)

# Future Prediction
st.subheader("🔮 Future Sales Prediction")

future_month = st.number_input(
    "Future Month",
    min_value=13,
    max_value=60,
    value=13
)

future_ad = st.number_input(
    "Advertising Budget",
    min_value=1000,
    max_value=50000,
    value=10000
)

future_customers = st.number_input(
    "Expected Customers",
    min_value=100,
    max_value=5000,
    value=1000
)

if st.button("Predict Future Sales"):

    future_data = pd.DataFrame({
        "Month": [future_month],
        "Advertising": [future_ad],
        "Customers": [future_customers]
    })

    prediction = model.predict(future_data)

    st.success(
        f"Predicted Sales: ${prediction[0]:,.2f}"
    )

# Download Report
st.subheader("📥 Download Prediction Report")

report = pd.DataFrame({
    "Predicted Sales": prediction
}) if 'prediction' in locals() else pd.DataFrame()

csv = report.to_csv(index=False)

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name="prediction_report.csv",
    mime="text/csv"
)