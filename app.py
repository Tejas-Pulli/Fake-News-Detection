import pickle
import streamlit as st
# import numpy as np

# loading the saved models
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('fake_news.sav', 'rb') as file:
    news_models = pickle.load(file)


#page title

# getting the input data from the user
Title = st.text_input("Enter Title of News")
Author = st.text_input("Enter Author's Name")
Text = st.text_input("Enter News Text")

input_data = [Title+' '+Author+' '+Text]
# input_data_reshaped = input_data.reshape(1,-1)

input_transformed = vectorizer.transform(input_data)


#code for Prediction
news_pred = ''

#creating a button for prediction
if st.button('Check News'):
    news_prediction = news_models.predict(input_transformed)
    
    if(news_prediction[0]==1):
        news_pred='Yes'
    else:
        news_pred='No'
st.success(news_pred)
