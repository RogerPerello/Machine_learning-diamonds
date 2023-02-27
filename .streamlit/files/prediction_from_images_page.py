import streamlit as st
from PIL import Image
import numpy as np
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup as bs
import re
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def predict_from_images():
    # Title and subtitle
    st.header('Prediction from images')
    st.write('Use the photo of a diamond and its weight to predict the price.')
    st.write('If you prefer to add its characteristics yourself, click on "Prediction from characteristics" from the sidebar menu.')
    
    # Form
    with st.form('Send your image'):
        st.subheader('First step: how to upload')
        st.write('The image must be a .jpg file.')
        st.write('Put the diamond on a white paper and take the picture as close as you can without losing resolution.')
        st.write('The resulting photo should look as much as possible like this:')
        image = Image.open('.streamlit/images/sample_image.jpg').resize((300, 300))
        image_array = np.array(image)/255
        st.image(image_array)
        image_submit = st.file_uploader('When you are ready, upload the image:', type='jpg')
        submitted = st.form_submit_button('Submit the image')
        st.subheader('Second step: how to add the weight')
        st.write('If you are not sure, use the most precise scale you have at hand.')
        st.write('Select the chosen metric to measure it.')
        st.write('Write down the number you get on the cell below:')
        weight_metric = st.selectbox('How did your measure the weight?', options=['Carat', 'Grams', 'Centigrams', 'Milligrams'])
        input_weight = st.number_input('Weight (carat)', min_value=0.0, max_value=5.0, step=0.01)
        deactivated_button = True
        if submitted and image_submit and weight_metric and input_weight > 0.0:
            deactivated_button = False

    # Prediction preparation
    prediction_button = st.button('Begin prediction', type='primary', disabled=deactivated_button)
    if prediction_button:
        with st.spinner('Loading prediction...'):
            time.sleep(1)

            # Image resizing
            img = Image.open(image_submit)
            img = img.resize((224, 224))
            img_array = img_to_array(img)

            # Inflation webscrapping
            if 'inflation_2022' not in st.session_state:
                current_year = datetime.now().year
                inflation_estimated_2022 = ' using estimated inflation'
                try:
                    url = f'https://www.in2013dollars.com/Jewelry/price-inflation/2022-to-{current_year}'
                    r = requests.get(url)
                    soup = bs(r.text, 'html.parser')
                    info = soup.find_all(class_='highlight')[0].text
                    inflation_2022 = float(re.search('^(.+)%', info)[0][:-1])
                    inflation_estimated_2022 = ''
                except Exception:
                    inflation_2022 = (int(current_year) - 2017) * 1.78
                st.session_state.inflation_2022 = inflation_2022
                st.session_state.inflation_estimated_2022 = inflation_estimated_2022

            # First prediction
            model_cnn = load_model('src/models/predict_from_images/price_prediction_images.h5')
            first_prediction = model_cnn.predict(img_array)

            # Second prediction
            model_knn = joblib.load('src/models/predict_from_images/price_image_prediction.pkl')
            second_prediction = model_knn.predict(first_prediction)

            # Final prediction
            df_images_data = pd.read_csv('src/data/processed/images_data_processed.csv')
            scaler = StandardScaler()
            df_images_data['price'] = scaler.fit_transform(df_images_data[['price']])
            final_prediction = scaler.inverse_transform(second_prediction)
            inflated_prediction = ((final_prediction / 100) * st.session_state.inflation_2022) + final_prediction

            # Prediction display
            st.success(f'Prediction loaded{st.session_state.inflation_estimated_2017}:')
            st.write(f'Your diamond costs {str(inflated_prediction).split(".")[0] + "." + str(inflated_prediction).split(".")[1][:2]} dollars approximately.')
