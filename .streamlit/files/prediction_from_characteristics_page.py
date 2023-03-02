import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup as bs
import re
import sklearn
import xgboost


def predict_from_characteristics():
    # Title and subtitle
    st.header('Prediction from characteristics')
    st.write('Use the attributes of a diamond to predict its price.')
    st.write('If you prefer to simply upload a photo and write down the weight, click on "Prediction from image" from the sidebar menu.')

    # Form
    with st.form('Diamond characteristics'):
        st.subheader('Primary values')
        st.write('Try to be precise when assigning primary values.')
        st.write('Remember that the table is the flat facet on its surface, while diameter refers to the entire girdle.')
        input_weight = st.number_input('Weight (carat)', min_value=0.01, max_value=6.0, step=0.01)
        depth_metric = st.selectbox('How did you measure the depth?', options=['Millimeters', 'Percentage'])
        input_depth = st.number_input('Depth', min_value=0.01, max_value=60.0, step=0.01)
        table_metric = st.selectbox('How did you measure the table?', options=['Millimeters', 'Percentage'])
        input_table = st.number_input('Table', min_value=0.01, max_value=60.0, step=0.01)
        st.write('----- Fill only the diameter, if your diamond is rounded, or the length/width if it is squared -----')
        st.write('- For squared diamonds:')
        input_length = st.number_input('Length (millimeters)', min_value=0.0, max_value=60.0, step=0.01)
        input_width = st.number_input('Width (millimeters)', min_value=0.0, max_value=60.0, step=0.01)
        st.write('- For rounded diamonds:')
        input_diameter = st.number_input('Diameter (millimeters)', min_value=0.0, max_value=60.0, step=0.01)
        st.subheader('Secondary values')
        st.write('Try to be precise, but if you do not know some of the secondary values, make a guess.')
        slider_cut = st.select_slider('Cut quality', options=['Fair (F)', 
                                                              'Good (GD)', 
                                                              'Very Good (VG)', 
                                                              'Premium', 
                                                              'Ideal or Excelent (EX)'
                                                              ]
                                        )
        slider_color = st.select_slider('Color', options=['Y-Z (light)', 
                                                            'W-X (light)', 
                                                            'W (light)', 
                                                            'U-V (light)', 
                                                            'S-T (light)', 
                                                            'Q-R (very light)', 
                                                            'O-P (very light)', 
                                                            'O (very light)', 
                                                            'N (very light)', 
                                                            'M (faint)', 
                                                            'L (faint)', 
                                                            'K (faint)', 
                                                            'J (near colorless)', 
                                                            'I (near colorless)', 
                                                            'H (near colorless)', 
                                                            'G (near colorless)', 
                                                            'F (colorless)', 
                                                            'E (colorless)', 
                                                            'D (colorless)'
                                                            ]
                                        )
        slider_clarity = st.select_slider('Clarity (prevalence of inclusions)', options=['Included (I1)', 
                                                                                            'Slightly included 2 (SI2)', 
                                                                                            'Slightly included 1 (SI1)', 
                                                                                            'Very slightly included 2 (VS2)', 
                                                                                            'Very slightly included 1 (VS1)', 
                                                                                            'Very, very slightly included 2 (VVS2)', 
                                                                                            'Very, very slightly included 1 (VVS1)', 
                                                                                            'Internally flawless (IF)', 
                                                                                            'Flawless (FL)'
                                                                                        ]
                                            )
        submitted = st.form_submit_button('Submit')
        deactivated_button = True
        if submitted and (input_diameter == 0 and input_length != 0 and input_width != 0):
            deactivated_button = False
        elif submitted and (input_diameter != 0 and input_length == 0 and input_width == 0):
            deactivated_button = False
        elif submitted and (input_diameter == 0 and input_length == 0 and input_width == 0):
            st.write('You assigned a value of zero to both diameter and length/width. Please, put a correct value on either one to ensure a proper prediction.')
        elif submitted and (input_diameter != 0 and input_length != 0 and input_width != 0):
            st.write('You assigned a value higher than zero to both diameter and length/width. Please, put a zero on either one to ensure a proper prediction.')
        elif submitted and (input_diameter == 0 and input_length == 0 and input_width != 0):
             st.write('You assigned a value of zero to length. Please, put a value higher than zero to ensure a proper prediction.')
        elif submitted and (input_diameter == 0 and input_length != 0 and input_width == 0):
             st.write('You assigned a value of zero to width. Please, put a value higher than zero to ensure a proper prediction.') 
        elif submitted and (input_diameter != 0 and input_length != 0 and input_width == 0):
             st.write('You assigned a value higher than zero to length. Please, put a zero to ensure a proper prediction.')
        elif submitted and (input_diameter != 0 and input_length == 0 and input_width != 0):
             st.write('You assigned a value higher than zero to width. Please, put a zero to ensure a proper prediction.')
        if submitted and input_diameter and not (input_length and input_width):
            st.write(f'Weight: {input_weight}. Depth: {input_depth}. Diameter: {input_diameter}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
        elif submitted and (input_length and input_width) and not input_diameter:
            st.write(f'Weight: {input_weight}. Depth: {input_depth}. Length: {input_length}. Width: {input_width}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
        elif submitted:
            st.write(f'Weight: {input_weight}. Depth: {input_depth}. Table: {input_table}. Diameter: {input_diameter}. Length: {input_length}. Width: {input_width}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')

    # Model load
    model = joblib.load('src/models/predict_from_variables/price_prediction.pkl')

    # Prediction preparation
    prediction_button = st.button('Begin prediction', type='primary', disabled=deactivated_button)
    if prediction_button:
        with st.spinner('Loading prediction...'):
            time.sleep(1)
            if not input_diameter:
                input_diameter = (input_length + input_width) / 2
            else:
                input_length = input_diameter / 2
                input_width = input_length
            for key, value in {'Y-Z (light)': -12, 
                                'W-X (light)': -11, 
                                'W (light)': -10, 
                                'U-V (light)': -9, 
                                'S-T (light)': -8, 
                                'Q-R (very light)': -7, 
                                'O-P (very light)': -6, 
                                'O (very light)': -5, 
                                'N (very light)': -4, 
                                'M (faint)': -3, 
                                'L (faint)': -2, 
                                'K (faint)': -1, 
                                'J (near colorless)': 0, 
                                'I (near colorless)': 1, 
                                'H (near colorless)': 2, 
                                'G (near colorless)': 3, 
                                'F (colorless)': 4, 
                                'E (colorless)': 5, 
                                'D (colorless)': 6
                                }.items():
                if slider_color == key:
                    slider_color = value
            for key, value in {'Fair (F)':0, 
                                'Good (GD)': 1, 
                                'Very Good (VG)': 2,
                                'Premium': 3, 
                                'Ideal or Excelent (EX)': 4
                                }.items():
                if slider_cut == key:
                    slider_cut = value
            for key, value in {'Included (I1)':0, 
                                'Slightly included 2 (SI2)': 1, 
                                'Slightly included 1 (SI1)': 2, 
                                'Very slightly included 2 (VS2)': 3, 
                                'Very slightly included 1 (VS1)': 4,
                                'Very, very slightly included 2 (VVS2)': 5,
                                'Very, very slightly included 1 (VVS1)': 6,
                                'Internally flawless (IF)': 7,
                                'Flawless (FL)': 8
                                }.items():
                if slider_clarity == key:
                    slider_clarity = value
            if input_depth == 'Millimeters':
                depth_percentage = (input_depth / input_diameter) * 100
            if input_depth == 'Percentage':
                depth_percentage = input_depth
                input_depth = (input_depth * input_diameter) / 100
            if table_metric == 'Millimeters':
                input_table = (input_table * 100) / input_diameter
            data_array = np.array([[input_weight, slider_cut, slider_color, slider_clarity, depth_percentage, input_table, input_length, input_width, input_depth]])

            # Inflation webscrapping
            if 'inflation_2017' not in st.session_state:
                current_year = datetime.now().year
                inflation_estimated_2017 = ' using estimated inflation'
                try:
                    url = f'https://www.in2013dollars.com/Jewelry/price-inflation/2017-to-{current_year}'
                    r = requests.get(url)
                    soup = bs(r.text, 'html.parser')
                    info = soup.find_all(class_='highlight')[0].text
                    inflation_2017 = float(re.search('^(.+)%', info)[0][:-1])
                    inflation_estimated_2017 = ''
                except Exception:
                    inflation_2017 = (int(current_year) - 2017) * 1.78
                st.session_state.inflation_2017 = inflation_2017
                st.session_state.inflation_estimated_2017 = inflation_estimated_2017

            # Prediction
            prediction = np.exp(model.predict(data_array)[0])
            inflated_prediction = ((prediction / 100) * st.session_state.inflation_2017) + prediction

        # Prediction display
        st.success(f'Prediction loaded{st.session_state.inflation_estimated_2017}:')
        st.write(f'Your diamond costs {str(inflated_prediction).split(".")[0] + "." + str(inflated_prediction).split(".")[1][:2]} dollars approximately.')
