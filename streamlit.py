import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_path = '/Users/jeffr/Documents/semester 4/MODEL DEPLOY/2602158784/xgb_class.pkl' 
model = joblib.load(model_path)

# Function to make predictions
def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

def main():
    st.title('Churn or Not Churn Prediction')
    
    credit_score = st.number_input('CreditScore', min_value=350.0, max_value=850.0)
    st.write('Credit Score (350-850):', credit_score)
    
    geography_options = ['France', 'Spain', 'Germany']
    geography = st.slider('Geography', min_value= 0, max_value=2)
    st.write('You are from', geography_options[geography])
    
    gender_options = ["Male", "Female"]
    gender = st.radio('Gender', gender_options)
    st.write('You are a', gender)
    
    age = st.number_input('Age', min_value=18.0, max_value=92.0)
    st.write('Age (18.0-92.0):', age)

    tenure = st.slider('Tenure', min_value=0, max_value=10)
    st.write('Tenure (0-10):', tenure)
    
    balance = st.number_input('Balance', min_value=0.0, max_value=238387.56)
    st.write('Balance (0-238387.56):', balance)

    product = st.slider('NumOfProducts', min_value=1, max_value=4)
    st.write('Number of Products (1-4):', product)
    
    crcard = st.checkbox('HasCrCard')
    st.write('You have a credit card' if crcard else 'You do not have a credit card')
    
    active = st.checkbox('IsActiveMember')
    st.write('You have an active membership' if active else 'You do not have an active membership')

    salary = st.number_input('EstimatedSalary', min_value=11.58, max_value=199992.48, value=11.58)
    st.write('Estimated Salary (11.58-199992.48):', salary)

    if st.button('Make Prediction'):
        features = [credit_score, geography, 1 if gender == 'Male' else 0, age, tenure, balance, product, 1 if crcard else 0, 1 if active else 0, salary]
        result = make_prediction(features)
        st.success('The prediction is: {}'.format("Churn customer" if result == 1 else "Not churn customer"))

if __name__ == '__main__':
    main()
