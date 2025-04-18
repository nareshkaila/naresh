import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import streamlit as st
import plotly.graph_objects as go
from time import sleep

# Load the dataset
dataset = pd.read_csv(r'/Users/kailanaresh/Downloads/A VS CODE/22nd - slr with streamlit/Salary_Data.csv')

# Split the data into independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset into training and testing sets (80-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the model
regressor = LinearRegression()

# Streamlit title and instructions
st.title('Salary Prediction Based on Experience')
st.write("This app predicts salaries based on years of experience using linear regression.")

# Show a loading spinner while training the model
with st.spinner('Training the model...'):
    sleep(2)
    regressor.fit(X_train, y_train)

# After model is trained
st.success('Model Trained Successfully!')

# Predict the test set
y_pred = regressor.predict(X_test)

# Visualizing the training set using Plotly for animation
train_fig = go.Figure()

train_fig.add_trace(go.Scatter(x=X_train.flatten(), y=y_train, mode='markers', name='Training Data', marker=dict(color='red')))
train_fig.add_trace(go.Scatter(x=X_train.flatten(), y=regressor.predict(X_train), mode='lines', name='Regression Line', line=dict(color='blue')))

train_fig.update_layout(
    title='Salary vs Experience (Training Set)',
    xaxis_title='Years of Experience',
    yaxis_title='Salary',
    template='plotly_dark'
)

st.plotly_chart(train_fig)

# Visualizing the test set using Plotly for animation
test_fig = go.Figure()

test_fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers', name='Test Data', marker=dict(color='green')))
test_fig.add_trace(go.Scatter(x=X_train.flatten(), y=regressor.predict(X_train), mode='lines', name='Regression Line', line=dict(color='blue')))

test_fig.update_layout(
    title='Salary vs Experience (Test Set)',
    xaxis_title='Years of Experience',
    yaxis_title='Salary',
    template='plotly_dark'
)

st.plotly_chart(test_fig)

# Predict salary for 12 and 20 years of experience
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])

# Display predictions
st.subheader(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
st.subheader(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model performance
train_score = regressor.score(X_train, y_train)
test_score = regressor.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

st.write(f"Training R²: {train_score:.2f}")
st.write(f"Testing R²: {test_score:.2f}")
st.write(f"Training MSE: {train_mse:.2f}")
st.write(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk (pickle)
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    
st.write("Model has been pickled and saved as `linear_regression_model.pkl`.")
