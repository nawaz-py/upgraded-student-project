
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="AI Student Performance Analyzer",
    page_icon="📊",
    layout="centered"
)

st.title("AI Student Performance Analyzer")
st.write("Predict student scores using Machine Learning.")

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    2, 4, 6, 8, 3, 5, 7, 9, 1, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                  60, 70, 80, 90, 65, 75, 85, 95, 45, 100],
    "Previous_Score": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
                       42, 58, 68, 78, 48, 62, 72, 88, 35, 90],
    "Final_Score": [35, 40, 48, 52, 60, 68, 72, 82, 88, 95,
                    42, 55, 70, 85, 45, 65, 75, 90, 30, 98]
}

df = pd.DataFrame(data)

# Prepare training data
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Final_Score"]

# Train Machine Learning model
model = LinearRegression()
model.fit(X, y)

# User inputs
st.subheader("Enter Student Details")

study_hours = st.slider(
    "Daily Study Hours", 0, 12, 5
)

attendance = st.slider(
    "Attendance Percentage", 0, 100, 75
)

previous_score = st.slider(
    "Previous Exam Score", 0, 100, 60
)

# Prediction
if st.button("Predict Student Score"):
    input_data = np.array([[
        study_hours,
        attendance,
        previous_score
    ]])

    prediction = model.predict(input_data)[0]

    prediction = np.clip(prediction, 0, 100)

    st.success(
        f"Predicted Final Score: {prediction:.2f}/100"
    )

    if prediction >= 90:
        st.write("Grade: A+")
        st.write("Excellent performance!")
    elif prediction >= 80:
        st.write("Grade: A")
        st.write("Very good performance!")
    elif prediction >= 70:
        st.write("Grade: B")
        st.write("Good performance!")
    elif prediction >= 60:
        st.write("Grade: C")
        st.write("Keep improving!")
    elif prediction >= 35:
        st.write("Grade: D")
        st.write("More practice is needed.")
    else:
        st.write("Grade: F")
        st.write("Additional preparation is recommended.")

# Dataset preview
with st.expander("View Sample Training Dataset"):
    st.dataframe(df)

st.caption("Developed as part of Codomax Module 5")
