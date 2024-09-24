
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
def load_model():
    with open('model_gb (gb 83).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict using the loaded model
def predict_income(input_data, model):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

    # Create a feature list based on user inputs (Assuming workclass, education, and occupation are already label-encoded)
age_group_mapping = {'Teenager': 0, 'Young Adult': 1, 'adult': 2, 'middle age': 3, 'Senior': 4}
workclass_map = {'Private': 2, 'Self-emp-not-inc': 4, 'Self-emp-inc': 3, 'Federal-gov': 0, 'Local-gov': 1, 'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7}
occupation_mapping = {'Tech-support': 12, 'Craft-repair': 2, 'Other-service': 7, 'Sales': 11, 'Exec-managerial': 3, 'Prof-specialty': 9, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Adm-clerical': 0, 'Farming-fishing': 4, 'Transport-moving': 13, 'Priv-house-serv': 8, 'Protective-serv': 10, 'Armed-Forces': 1}
marital_status_mapping = {'Never-married': 4, 'Married-civ-spouse': 2, 'Divorced': 0, 'Married-spouse-absent': 3, 'Separated': 5, 'Married-AF-spouse': 1, 'Widowed': 6}
relationship_mapping = {'Not-in-family': 1, 'Husband': 0, 'Wife': 5, 'Own-child': 3, 'Unmarried': 4, 'Other-relative': 2}
race_mapping = {'White': 4, 'Black': 2, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 3}
gender_mapping = {'Male': 1, 'Female': 0}
hours_group_mapping = {'Full-Time': 0, 'Over-Time': 1, 'Part-Time': 2, 'Extreme': 3}
   
def main():
    st.title("Income Category Prediction")
    st.write("Enter the following details to predict whether the income is greater than or less than $50K:")

    # User input fields
    workclass = st.selectbox('Workclass', [ 'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
       'Local-gov','Self-emp-inc', 'Without-pay', 'Never-worked'])
    age_group = st.selectbox('Age Group',['Teenager', 'Young Adult', 'adult', 'middle age',
    'Senior'])
    educationNum = st.number_input('Education Number (The number of years of education completed)', min_value=1, max_value=15)
    marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse', 'Divorced',
       'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
       'Widowed'])
    occupation = st.selectbox('Occupation',['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
       'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
       'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
       'Tech-support', 'Protective-serv', 'Armed-Forces',
       'Priv-house-serv'])
    relationship = st.selectbox('Relationship',['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
       'Other-relative'])
    race = st.selectbox('Race',  ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
       'Other'])
    gender = st.radio('Gender',['Male', 'Female'])
    hours_group = st.selectbox('Hours Group', ['Full-Time', 'Over-Time', 'Part-Time', 'Extreme'])


    with st.expander("Your Selected Options"):
        result = {
            'Workclass':workclass,
            'Age':age_group,
            'education':educationNum,
            'Marital Status':marital_status,
            'Occupation':occupation,
            'Relationship':relationship,
            'Race':race,
            'Gender':gender,
            'Hour per Week':hours_group,
    
        }
    st.write(result)

    input_data = [
        age_group_mapping[age_group],
        workclass_map[workclass],
        educationNum,  
        marital_status_mapping[marital_status],
        occupation_mapping[occupation],
        relationship_mapping[relationship],
        race_mapping[race],
        gender_mapping[gender],
        hours_group_mapping[hours_group]
    ]

    
    # Load model
    model = load_model()

    # When user clicks predict
    if st.button("Predict"):
        prediction = predict_income(input_data, model)
        if prediction == 1:
            st.write("Income: Greater than 50K")
        else:
            st.write("Income: Less than or equal to 50K")

if __name__ == '__main__':
    main()

