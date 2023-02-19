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
    st.subheader('How it works')
    st.write('If you have a photo of a diamond, click on "Prediction from images" from the sidebar menu.')
    st.write('Otherwise, if you prefer to point its characteristics yourself, select "Prediction from characteristics".')
    st.write('Through the dropdown below you may check some of the inner workings of the app.')

    # Additional information about the data
    with st.expander('A peek into the process'):
        st.header('General information')
        st.write('Current inflation is taken into account thanks to [this calculator](https://www.in2013dollars.com/Jewelry/price-inflation).')
        st.write('There are some other factors that could increase or decrease the value of a diamond that are not considered when calculating the price:')
        st.write('- Fluorescence: a diamond with strong fluorescence can appear hazy or milky, while a diamond with no fluorescence can appear more transparent and bright.')
        st.write('- Shape: like round, princess or pear. Some are more popular than others and may be more valuable, depending on market trends.')
        st.write('- Symmetry and polish: which are ignored, because they are dependant on cut quality in general.')
        st.write('- Subjective appreciations: for example, colored diamonds tend to be less valuable. However, a diamond with a fancy color might be valued higher simply because it looks good, or due to current trends.')
        st.write('These factors can be usually neglected when evaluating tiny diamonds. Therefore, the smaller the diamond, the better the prediction.')
        st.header('Image recognition')
        st.write('If an image of a diamond is given, the app uses a deep learning model to obtain its characteristics.')
        st.write('The dataset used to train that first model can be found [here](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images).')
        st.subheader('Images data sample')
        st.write('The column "Id" represents each of the images. The rest of the variables are the target')
        df_images = pd.read_csv('src/data/processed/images_data_processed.csv').drop(columns='price')
        data_sample_images = df_images.sample(100)
        st.write(data_sample_images)
        st.header('Price prediction')
        st.write('Once the characteristics of the given diamond are determined, a supervised machine learning model trained with a larger pool of diamonds decides the appropiate price.')
        st.write('The dataset used to train that second model can be found [here](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond).')
        st.subheader('Prices data sample')
        st.write('The column "price" is the target.')
        df_prices = pd.read_csv('src/data/processed/original_processed.csv')
        data_sample_prices = df_prices.sample(100)
        st.write(data_sample_prices)
        st.subheader('Price prediction metrics')
        st.write('When splitted and tested with the y test, the model gives the following results:')
        st.write('- Root mean squared error (rmse): 523.053')
        st.write('- Mean squared error (mse): 273583.983')
        st.write('- Mean absolute error (mae): 258.973')
        st.write('- R2 score: 0.983')
