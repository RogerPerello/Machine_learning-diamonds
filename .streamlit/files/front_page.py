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
Those deposits are prospected and mined. Then, the diamonds are sliced, studied, split, bruted, polished and inspected to meet quality standards.
''')
    st.write('After taking into account this long process, only one thing remains unanswered: how much do they cost?')
    st.subheader('The answer')
    st.write('As a response to that question, the Diamond APPraiser detects the weight and dimensions of a diamond, as well as its clarity, color and cut.')
    st.write('Subsequently, uses those measures to predict an approximate price.')
    st.subheader('How it works')
    st.write('If you have a photo of a diamond and its weight, click on "Prediction from images" from the sidebar menu.')
    st.write('Otherwise, if you prefer to point out all of its characteristics yourself, select "Prediction from characteristics".')
    st.write('Through the dropdown below, you may check some of the inner workings of the app.')

    # Additional information
    with st.expander('A peek into the process'):
        # General information
        st.header('General information')
        st.write('Current inflation is taken into account in real time thanks to [this calculator](https://www.in2013dollars.com/Jewelry/price-inflation).')
        st.write('If the calculator stops working temporarily for some reason, such as website maintenance, inflation is estimated. If that happens, it will be notified upon prediction delivery.')
        st.write('There are some other factors that could increase or decrease the value of a diamond that are not considered when calculating the price:')
        st.write('- Fluorescence: a diamond with strong fluorescence can appear hazy or milky, while a diamond with no fluorescence can appear more transparent and bright.')
        st.write('- Shape: like "round", "princess" or "pear". Some categories are more popular than others and may be more valuable depending on market trends.')
        st.write('- Symmetry and polish: which are ignored, because they are dependent on cut quality in general.')
        st.write('- Subjective appreciations: for example, colored diamonds tend to be less valuable. However, a diamond with a fancy color might be valued higher simply because it looks good, or due to current trends.')
        st.write('These factors can usually be neglected when evaluating tiny diamonds. Therefore, the smaller the diamond, the better the prediction.')

        # Image recognition information
        st.header('Image recognition')
        st.write('If an image of a diamond is given, the app uses a MobilenetV3Large transfer learning model to obtain an approximation its price.')
        st.write('Afterwards, a linear support vector machine regression algorithm uses the weight of the diamond to refine the final result.')
        st.write('The dataset used to train the models can be found [here](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images).')
        st.subheader('Images data sample')
        st.write('The column "id" represents each of the images. "weight" is the variable added to the second model. The variable "price" is the target.')
        df_images = pd.read_csv('src/data/processed/images_data_processed.csv')[['Weight','price']]
        df_images['id'] = df_images.index.apply(lambda x: x + '.jpg')
        df_images = df_images.rename(columns={'Weight': 'weight'})
        df_images = df_images.reset_index(drop=True)
        data_sample_images = df_images.sample(100)
        st.write(data_sample_images)
        st.subheader('Metrics')
        st.write('When split and tested with the "y_test", the model gives the following results:')
        st.write('- Root mean squared error (rmse): 346.892')
        st.write('- Mean squared error (mse): 120334.110')
        st.write('- Mean absolute error (mae): 260.244')
        st.write('- Mean absolute percentage error (mape): 0.175')
        st.write('- R2 score: 0.619')

        # Price prediction through caracteristics information
        st.header('Price prediction')
        st.write('If the characteristics of a diamond are passed, a supervised machine learning model trained with a large pool of diamonds decides the price.')
        st.write('This model was created by stacking a k-nearest neighbors algorithm with XGBoost, using linear regression as the final estimator.')
        st.write('The dataset used to train that last model can be found [here](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond).')
        st.subheader('Prices data sample')
        st.write('The column "price" is the target.')
        df_prices = pd.read_csv('src/data/processed/original_processed.csv')
        data_sample_prices = df_prices.sample(100)
        st.write(data_sample_prices)
        st.subheader('Metrics')
        st.write('When split and tested with the "y_test", the model gives the following results:')
        st.write('- Root mean squared error (rmse): 514.513')
        st.write('- Mean squared error (mse): 264723.399')
        st.write('- Mean absolute error (mae): 256.703')
        st.write('- Mean absolute percentage error (mape):  0.060')
        st.write('- R2 score: 0.983')
