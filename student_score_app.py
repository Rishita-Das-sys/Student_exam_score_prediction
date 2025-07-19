
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('StudentPerformanceFactors.csv')
    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df[['Hours_Studied', 'Previous_Scores', 'Attendance']]
    y = df['Exam_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    st.title('Student Exam Score Predictor')
    st.write('Predict exam scores based on study hours, previous marks, and attendance.')

    # Load and train
    df = load_data()
    model = train_model(df)

    # User input
    st.header("Input Student Details")
    study_hours = st.number_input('Study Hours', 0.0, 12.0, 2.0)
    previous_marks = st.number_input('Previous Marks (%)', 0, 100, 50)
    attendance = st.slider('Attendance (%)', 0, 100, 75)

    if st.button('Predict Exam Score'):
        input_data = np.array([[study_hours, previous_marks, attendance]])
        prediction = model.predict(input_data)[0]
        st.success(f'📊 Predicted Exam Score: {prediction:.2f}')

        # Visualization
        st.header("📈 Relationship with Exam Score")
        fig = px.scatter_matrix(df,
                                dimensions=['Hours_Studied', 'Previous_Scores', 'Attendance', 'Exam_Score'],
                                title='Feature Relationships')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
