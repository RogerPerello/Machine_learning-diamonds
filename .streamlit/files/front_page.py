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
    st.subheader('The Question')
    st.write('''Forming a diamond deep down in the Earth's mantle takes at least one billion years.
Under high pressure and temperature, carbon-containing fluids dissolve various minerals, and their atoms arrange in an extremely rigid and unyielding way.
Defects and impurities try to permeate them, which determines what color will be adopted, if any, as well as their clarity.
The resulting stones, which may come in many weights and shapes, are later carried to the surface in volcanic eruptions and deposited in igneous rocks.
Those deposits are prospected and mined. Then, the diamonds are sliced, studied, split, bruted, polished, and inspected to meet quality standards.
''')
    st.write('After taking into account this long process, only one question remains unanswered: how much do they cost?')
    st.subheader('The Answer')
    st.write('The Diamond APPraiser detects the weight and dimensions of a diamond, as well as its clarity, color, and cut to predict an approximate price.')
    st.subheader('How It Works')
    st.write('If you have a photo of a diamond and its weight, click on "Prediction from Images" in the sidebar menu.')
    st.write('Otherwise, if you prefer to specify all its characteristics yourself, select "Prediction from Characteristics".')
    st.write('Through the dropdown below, you may check some of the inner workings of the app.')

    # Additional information
    with st.expander('A peek into the process'):
        # General information
        st.header('Prediction from Images')
        st.write('The weight of the diamonds to be predicted is limited to 2 carats to ensure an optimal level of accuracy while still covering the majority of diamonds.')
        st.write('Current inflation is taken into account in real-time using [this calculator](https://www.in2013dollars.com/Jewelry/price-inflation).')
        st.write('If the calculator stops working temporarily due to website maintenance or other reasons, inflation is estimated, and you will be notified upon prediction delivery.')
        st.write('There are other factors that could increase or decrease the value of a diamond that are not considered when calculating the price:')
        st.write('- Fluorescence: a diamond with strong fluorescence can appear hazy or milky, while a diamond with no fluorescence can appear more transparent and bright.')
        st.write('- Shape: categories like "round", "princess", or "pear" may be more valuable depending on market trends.')
        st.write('- Symmetry and polish: these are ignored because they depend on cut quality in general.')
        st.write('- Subjective appreciations: for example, colored diamonds tend to be less valuable. However, a diamond with a fancy color might be valued higher simply because it looks good, or due to current trends.')
        st.write('These factors can usually be neglected when evaluating tiny diamonds. Therefore, the smaller the diamond, the better the prediction.')

        # Image recognition information
        st.header('Image Recognition')
        st.write('If an image of a diamond is provided, the app uses a MobilenetV3Large transfer learning model to approximate its price.')
        st.write('Afterwards, a support vector machine regression algorithm (rbf) uses the weight of the diamond to refine the final result.')
        st.write('The dataset used to train the models can be found [here](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images).')
        st.subheader('Image Data Sample')
        st.write('Here is a sample of the diamond data. The column "id" represents each of the images. "weight" is the variable added to the second model. The variable "price" is the target.')
        df_images = pd.read_csv('src/data/processed/images_data_processed.csv')[['Weight','price']]
        df_images['id'] = df_images.index
        df_images['id'] = df_images['id'].apply(lambda x: str(x) + '.jpg')
        df_images = df_images.rename(columns={'Weight': 'weight'})
        df_images = df_images.reset_index(drop=True)
        data_sample_images = df_images.sample(100)
        st.write(data_sample_images)
        st.subheader('Performance')
        st.write('The model was evaluated on a test dataset, and the following metrics were obtained:')
        st.write('- Root mean squared error (RMSE): 346.892')
        st.write('- Mean squared error (MSE): 120334.110')
        st.write('- Mean absolute error (MAE): 260.244')
        st.write('- Mean absolute percentage error (MAPE): 0.175')
        st.write('- R2 score: 0.619')

        # Price prediction through characteristics information
        st.header('Prediction from Characteristics')
        st.write('Given the characteristics of a diamond, the app uses a machine learning model to predict its price. The model is a supervised learning algorithm created by stacking a k-nearest neighbors algorithm with XGBoost and using linear regression as the final estimator.')
        st.write('The model was trained with a large dataset of diamonds and has been optimized to provide accurate predictions for diamonds up to 2 carats in weight.')
        st.write('The dataset used to train the model can be found on Kaggle at the following link: [https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond).')
        st.subheader('Prices Data Sample')
        st.write('Here is a sample of the diamond data used to train the model. The "price" column is the target variable:')
        df_prices = pd.read_csv('src/data/processed/original_processed.csv')
        data_sample_prices = df_prices.sample(100)
        st.write(data_sample_prices)
        st.subheader('Performance')
        st.write('The model was evaluated on a test dataset, and the following metrics were obtained:')
        st.write('- Root mean squared error (RMSE): 514.513')
        st.write('- Mean squared error (MSE): 264723.399')
        st.write('- Mean absolute error (MAE): 256.703')
        st.write('- Mean absolute percentage error (MAPE):  0.060')
        st.write('- R2 score: 0.983')
