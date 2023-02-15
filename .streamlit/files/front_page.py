import streamlit as st
import numpy as np
from PIL import Image 
import pandas as pd


def set_front_page():
    # Title and subtitle
    st.header('Diamond APPraiser')
    st.write('by Roger Perelló Gumbau')

    # Main image
    image = Image.open('.streamlit/images/front_picture.png').resize((1022, 400))
    image_array = np.array(image)/255
    st.image(image_array)

    # Introduction
    st.write('')
    st.subheader('The question')
    st.write('''Forming a diamond deep down the Earth's mantle takes at least one billion years.
Under high pressure and temperature, carbon-containing fluids dissolve various minerals, and their atoms are arranged in an extremely rigid way, unyielding.
Defects and impurities try to permeate them, which determines what color will be adopted, if any, as well as their clarity.
The resulting stones, which may come in many weights and shapes, are later carried to the surface in volcanic eruptions and deposited in igneous rocks.
Those deposits are prospected and mined. Then, the diamonds are sliced, studied, splitted, bruted, polished and inspected to meet quality standards.
''')
    st.write('After taking into account this long process, only one thing remains unanswered: how much do they cost?')
    st.subheader('The answer')
    st.write('As a response to that question, the Diamond APPraiser detects the weight and dimensions of a diamond, as well as its clarity, color and cut.')
    st.write('Subsequently, uses those measures to predict an approximate price.')
    st.subheader('How to make it work')
    st.write('If you have a photo of a diamond, click on "Prediction from images" from the sidebar menu.')
    st.write('Otherwise, if you prefer to annotate its characteristics yourself, click on "Prediction from characteristics".')
    st.write('Through the dropdown below you may check some of the inner workings of the app.')

    # Additional information about the data
    with st.expander('A peek into the process'):
        st.header('Image recognition')
        st.write('If an image of a diamond is given, the app uses a deep learning model to obtain its characteristics.')
        st.write('The dataset used to train that first model can be found [here](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images).')
        st.subheader('Images data sample')
        st.write('The column "Id" represents each of the images. The column "price" is set aside before the training.')
        df_images = pd.read_csv('src/data/processed/images_data_processed.csv')
        data_sample_images = df_images.head(10)
        st.write(data_sample_images)
        st.header('Price prediction')
        st.write('Once the characteristics of the given diamond are determined, a supervised machine learning model trained with a larger pool of diamonds decides the appropiate price.')
        st.write('The dataset used to train that second model can be found [here](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond).')
        st.subheader('Prices data sample')
        st.write('To evaluate the prediction, the resulting price is compared to the price of the diamonds reserved in the previous data frame.')
        df_prices = pd.read_csv('src/data/processed/original_processed.csv')
        data_sample_prices = df_prices.head(10)
        st.write(data_sample_prices)
