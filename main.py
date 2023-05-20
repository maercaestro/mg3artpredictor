import streamlit as st
import xgboost as xgb
import json
import numpy as np
import pandas as pd
from PIL import Image

v1ss_reg=xgb.XGBRegressor()
v1ss_reg.load_model(r'/Users/abuhuzaifahbidin/Documents/MG3 ART Predictor Project/XGBoost Predictor/100Dxgb.json')

v2ss_reg=xgb.XGBRegressor()
v2ss_reg.load_model(r'/Users/abuhuzaifahbidin/Documents/MG3 ART Predictor Project/XGBoost Predictor/150Dxgb.json')

v3ss_reg=xgb.XGBRegressor()
v3ss_reg.load_model(r'/Users/abuhuzaifahbidin/Documents/MG3 ART Predictor Project/XGBoost Predictor/500Dxgb.json')

image=Image.open('MEGATLogo.png')

# Define the scale factor
scale_factor = 0.25  # Replace with the desired scale factor

# Calculate the new dimensions based on the scale factor
new_width = int(image.width * scale_factor)
new_height = int(image.height * scale_factor)

# Resize the image
resized_image = image.resize((new_width, new_height))

st.image(resized_image)
st.title("MG3 R-1901 ART Predictor")
st.write("""Predicting R-1901 ART using XGBoost ML Model""")


operation_mode=st.sidebar.selectbox("Select your mode of operation",("100D","150D","500D"))
Feed = float(st.text_input('Unit 19 current feed'))
LBO_VI = float(st.text_input('Input your desired VI'))
LBO_KV100 = float(st.text_input('Select your desired KV100'))
ART_R2 = float(st.text_input('Select your current R1902 ART'))
PP = float(st.text_input('Input your desired PP'))

X=np.array([[Feed,LBO_VI,LBO_KV100,ART_R2,PP]])

def predict(X):
    if operation_mode=="100D":
        predictions=v1ss_reg.predict(X)
    elif operation_mode=="150D":
        predictions=v2ss_reg.predict(X)
    elif operation_mode=="500D":
        predictions=v3ss_reg.predict(X)
    
    return predictions

button = st.button('Predict R-1901 ART')

if button:
    st.write('Predicted ART:',predict(X)[0],unsafe_allow_html=True, font={'size': 36})

