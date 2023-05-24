import streamlit as st
import xgboost as xgb
import json
import numpy as np
import pandas as pd
from PIL import Image
import datetime

v1ss_reg=xgb.XGBRegressor()
v1ss_reg.load_model('100Dxgbnew.json')

v2ss_reg=xgb.XGBRegressor()
v2ss_reg.load_model('150Dxgbnew.json')

v3ss_reg=xgb.XGBRegressor()
v3ss_reg.load_model('500Dxgbnew.json')

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
Feed = float(st.number_input('Unit 19 current feed in m3/hr'))
LBO_VI = float(st.number_input('Input your desired VI'))
LBO_KV100 = float(st.number_input('Select your desired KV100'))
LBO_KV40 = float(st.number_input('Select your desired KV40'))
ART_R2 = float(st.number_input('Select your current R1902 ART'))
PP = float(st.number_input('Input your desired PP'))

date=datetime.datetime.now()
Hour=date.hour
Day=date.day
Month=date.month
Year=date.year


X=np.array([[LBO_VI,LBO_KV100,LBO_KV40,ART_R2,PP,Feed,Day,Month,Year,Hour]])

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
    st.write('Predicted ART:',predict(X)[0])
