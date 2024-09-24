
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
    # age_group_mapping = {'Teenager' :0, 'Young Adult':1, 'adult':2, 'middle age':3,
    # 'Senior':4}
    # workclass_mapping = {'Private': 2, 'Self-emp-not-inc': 4, 'Self-emp-inc': 3, 'Federal-gov': 0, 'Local-gov': 1, 'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7}
    # occupation_mapping = {'Tech-support': 12, 'Craft-repair': 2, 'Other-service': 7, 'Sales': 11, 'Exec-managerial': 3, 'Prof-specialty': 9, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Adm-clerical': 0, 'Farming-fishing': 4, 'Transport-moving': 13, 'Priv-house-serv': 8, 'Protective-serv': 10, 'Armed-Forces': 1}
    # marital_status_mapping = {'Never-married':4, 'Married-civ-spouse':2, 'Divorced':0,
    #     'Married-spouse-absent':3, 'Separated':5, 'Married-AF-spouse':1,
    #     'Widowed':6}
    # relationship_mapping = {'Not-in-family':1, 'Husband':0, 'Wife':5, 'Own-child':3, 'Unmarried':4,
    #     'Other-relative':2}
    # race_mapping = {'White':4, 'Black':2, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':0,
    #     'Other':3}
    # gender_mapping = {'Male':1, 'Female':0}
    # hours_group_mapping = {'Full-Time':0, 'Over-Time':1, 'Part-Time':2, 'Extreme':3}
    

    # education_mapping = {'Bachelors': 2, 'HS-grad': 9, '11th': 1, 'Masters': 12, '9th': 5, 'Some-college': 10, 'Assoc-acdm': 7, 'Assoc-voc': 8, '7th-8th': 4, 'Doctorate': 14, 'Prof-school': 15, '5th-6th': 3, '10th': 0, '1st-4th': 13, 'Preschool': 11, '12th': 6}
# Streamlit app
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

    # # One-Hot Encoded Columns
    # age_group_categories = ['Teenager (18<=age)', 'Young Adult (18<age<=30)', 'adult (30<age<=45)', 'middle age (45<age<=60)',
    # 'Senior (60<age<=100)']
    # workclass_categories = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
    # marital_status_categories = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    # occupation_categories = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    # relationship_categories = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    # race_categories = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    # gender_categories = ['Male', 'Female']
   

    # # User input fields
    # age_group = st.selectbox('Age Group', age_group_categories)
    # gender = st.radio('Gender',gender_categories)
    # educationNum = st.number_input('Education Number (The number of years of education completed)', min_value=1, max_value=15)
    # workclass = st.selectbox('Workclass', workclass_categories)
    # marital_status = st.selectbox('Marital Status', marital_status_categories)
    # occupation = st.selectbox('Occupation', occupation_categories)
    # relationship = st.selectbox('Relationship', relationship_categories)
    # race = st.selectbox('Race', race_categories)
    # hours_per_week = st.slider('Hours per Week', 1, 99, 40)
    # final_weight = st.number_input('Final Weight')

    # #Gender Label encode
    # gender_encoded = {'Male':0, 'Female':1}

    # # One-Hot Encode the user inputs
    # workclass_encoded = one_hot_encode(workclass, workclass_categories)
    # marital_status_encoded = one_hot_encode(marital_status, marital_status_categories)
    # occupation_encoded = one_hot_encode(occupation, occupation_categories)
    # relationship_encoded = one_hot_encode(relationship, relationship_categories)
    # race_encoded = one_hot_encode(race, race_categories)
    # age_group_encoded = one_hot_encode(age_group, age_group_categories)

    # # Combine all the one-hot encoded data
    # input_data = age_group_encoded+gender_encoded[gender]+workclass_encoded + marital_status_encoded + occupation_encoded + relationship_encoded + race_encoded + age_group_encoded + [hours_per_week]
 
 
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

