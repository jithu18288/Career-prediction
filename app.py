import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the scaler, label encoder, model, and class names
scaler = pickle.load(open("scaler.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score,average_score):
    
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0
    
    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,total_score,average_score]])
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Predict using the model
    probabilities = model.predict_proba(scaled_features)
    
    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs

# Streamlit app layout
st.title('Career Recommendation System')

# Sidebar inputs
st.sidebar.header('User Input Features')

gender = st.sidebar.radio('Gender', ['Male', 'Female'])
part_time_job = st.sidebar.checkbox('Part-Time Job')
absence_days = st.sidebar.slider('Absence Days', 0, 20, 2)
extracurricular_activities = st.sidebar.checkbox('Extracurricular Activities')
weekly_self_study_hours = st.sidebar.slider('Weekly Self-Study Hours', 0, 20, 7)
math_score = st.sidebar.slider('Math Score', 0, 100, 65)
history_score = st.sidebar.slider('History Score', 0, 100, 60)
physics_score = st.sidebar.slider('Physics Score', 0, 100, 97)
chemistry_score = st.sidebar.slider('Chemistry Score', 0, 100, 94)
biology_score = st.sidebar.slider('Biology Score', 0, 100, 71)
english_score = st.sidebar.slider('English Score', 0, 100, 81)
geography_score = st.sidebar.slider('Geography Score', 0, 100, 66)

total_score = math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score
average_score = total_score / 7

# Prediction
if st.button('Get Career Recommendations'):
    final_recommendations = Recommendations(gender=gender,
                                            part_time_job=part_time_job,
                                            absence_days=absence_days,
                                            extracurricular_activities=extracurricular_activities,
                                            weekly_self_study_hours=weekly_self_study_hours,
                                            math_score=math_score,
                                            history_score=history_score,
                                            physics_score=physics_score,
                                            chemistry_score=chemistry_score,
                                            biology_score=biology_score,
                                            english_score=english_score,
                                            geography_score=geography_score,
                                            total_score=total_score,
                                            average_score=average_score)

    st.subheader('Top recommended studies with probabilities:')
    for class_name, probability in final_recommendations:
        st.write(f"{class_name} with probability {probability}")