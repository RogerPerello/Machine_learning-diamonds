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

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


def predict_from_images():

    # Title and subtitle
    st.header('Prediction from Images')
    st.write('Use the photo of a diamond and its weight to predict the price.')
    st.write('If you prefer to add its characteristics yourself, click on "Prediction from Characteristics" from the sidebar menu.')
    st.write('Keep in mind that the result is a generalization, and the price of diamonds with similar characteristics may oscilate a lot.')
    
    # Form
    with st.form('Send your image'):
        st.subheader('First step: add the weight')
        st.write('If you are not sure, use the most precise scale you have at hand.')
        st.write('Remember that 1 gram equals 5 carat.')
        st.write('Write down the number you got from your measurements in the cell below:')
        input_weight = st.number_input('Weight (carat)', min_value=0.01, max_value=1.00, step=0.01)
        st.write('')
        st.subheader('Second step: upload')
        st.write('The image must be a .jpg file.')
        st.write('Place the diamond on a white paper and take the picture as close as you can without losing resolution.')
        st.write('The resulting photo should have a white/grey background and the diamond should be centered and well-lit, with no shadows or reflections obscuring its features:')
        image_sample = Image.open('.streamlit/images/image_sample.jpg').resize((405, 318))
        st.write('')
        st.image(image_sample)
        st.write('')
        st.write('You can try by dragging the above image to the uploader and setting the weight to 0.42 carat. To check the result, compare it with the real price of around 1812 dollars in the year 2023.')
        st.write('This particular image comes from [77diamonds](https://www.77diamonds.com/diamonds/loose-diamonds).')
        st.write('')
        image_submit = st.file_uploader('When you are ready, upload the image:', type='jpg')
        submitted = st.form_submit_button('Submit the image')
        deactivated_button = True
        if submitted and image_submit and input_weight > 0.0:
            deactivated_button = False

    # Loads
    model_cnn = load_model('src/models/predict_from_images/price_prediction_images.h5')
    model_knn = joblib.load('src/models/predict_from_images/price_image_prediction.pkl')
    df_images_data = pd.read_csv('src/data/processed/images_data_processed.csv')
    scaler = StandardScaler()
    df_images_data['price'] = scaler.fit_transform(df_images_data[['price']])


    # Prediction preparation
    prediction_button = st.button('Begin prediction', type='primary', disabled=deactivated_button)
    if prediction_button:
        with st.spinner('Loading prediction...'):
            time.sleep(1)

            # Image resizing
            img = Image.open(image_submit)
            img = img.resize((224, 224))
            img = preprocess_input(img)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

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
            first_prediction = model_cnn.predict(img_array)

            # Second prediction
            df_to_predict = pd.DataFrame(data={'predicted_price': first_prediction[0], 'Weight': input_weight})
            second_prediction = model_knn.predict(df_to_predict)

            # Final prediction
            final_prediction = scaler.inverse_transform(second_prediction.reshape(-1, 1))
            inflated_prediction = ((final_prediction / 100) * st.session_state.inflation_2022) + final_prediction

            # Prediction display
            st.success(f'Prediction loaded{st.session_state.inflation_estimated_2022}:')
            st.write(f'That kind of diamond usually costs {str(inflated_prediction).split(".")[0][2:] + "." + str(inflated_prediction).split(".")[1][:2]} dollars approximately.')
