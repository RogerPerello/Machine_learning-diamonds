import streamlit as st
import pickle
import numpy as np
import sklearn
import xgboost


def predict_from_characteristics():
    # Title and subtitle
    st.header('Prediction from characteristics')
    st.write('Use the attributes of a diamond to predict its price.')
    st.write('If you prefer to simply upload a photo, click on "Prediction from image" from the sidebar menu.')

    # Form
    with st.form('Diamond characteristics'):
        st.subheader('Key values')
        st.write('Try to be precise when assigning key values.')
        input_weight = st.number_input('Weight (carat)', min_value=0.01, max_value=10.0, step=0.01)
        input_depth = st.number_input('Depth (millimeters)', min_value=0.01, max_value=50.0, step=0.01)
        st.write('----- Fill only the diameter, if your diamond is rounded, or the lenght/width if it is squared. -----')
        st.write('- For squared diamonds:')
        input_lenght = st.number_input('Lenght (millimeters)', min_value=0.0, max_value=20.0, step=0.01)
        input_width = st.number_input('Width (millimeters)', min_value=0.0, max_value=80.0, step=0.01)
        st.write('- For rounded diamonds:')
        input_diameter = st.number_input('Diameter (millimeters)', min_value=0.0, max_value=60.0, step=0.01)
        st.subheader('Secondary values')
        st.write('Try to be precise, but if you do not know some of the secondary values, make a guess.')
        slider_cut = st.select_slider('Cut quality', options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
        slider_color = st.select_slider("Color (letter D means colorless)", options=['Y-Z', 
                                                                                        'W-X', 
                                                                                        'W', 
                                                                                        'U-V', 
                                                                                        'S-T', 
                                                                                        'Q-R', 
                                                                                        'O-P', 
                                                                                        'O', 
                                                                                        'N', 
                                                                                        'M', 
                                                                                        'L', 
                                                                                        'K', 
                                                                                        'J', 
                                                                                        'I', 
                                                                                        'H', 
                                                                                        'G', 
                                                                                        'F', 
                                                                                        'E', 
                                                                                        'D'
                                                                                    ]
                                        )
        slider_clarity = st.select_slider("Clarity (number of inclusions)", options=['Included', 
                                                                                        'Slightly included (2)', 
                                                                                        'Slightly included (1)', 
                                                                                        'Very slightly included (2)', 
                                                                                        'Very slightly included (1)', 
                                                                                        'Very, very slightly included (2)', 
                                                                                        'Very, very slightly included (1)', 
                                                                                        'Internally flawless', 
                                                                                        'Flawless'
                                                                                    ]
                                            )
        submitted = st.form_submit_button('Submit')
        deactivated_button = True
        if submitted and (input_diameter == 0 and input_lenght != 0 and input_width != 0):
            deactivated_button = False
        elif submitted and (input_diameter != 0 and input_lenght == 0 and input_width == 0):
            deactivated_button = False
        elif submitted and (input_diameter == 0 and input_lenght == 0 and input_width == 0):
            st.write('You assigned a value of zero to both diameter and lenght/width. Please, put a correct value to either one to ensure a proper prediction.')
        elif submitted and (input_diameter != 0 and input_lenght != 0 and input_width != 0):
            st.write('You assigned a value higher than zero to both diameter and lenght/width. Please, put a zero to either one to ensure a proper prediction.')
        elif submitted and (input_diameter == 0 and input_lenght == 0 and input_width != 0):
             st.write('You assigned a value of zero to lenght. Please, put a value higher than zero to ensure a proper prediction.')
        elif submitted and (input_diameter == 0 and input_lenght != 0 and input_width == 0):
             st.write('You assigned a value of zero to width. Please, put a value higher than zero to ensure a proper prediction.') 
        elif submitted and (input_diameter != 0 and input_lenght != 0 and input_width == 0):
             st.write('You assigned a value higher than zero to lenght. Please, put a zero to ensure a proper prediction.')
        elif submitted and (input_diameter != 0 and input_lenght == 0 and input_width != 0):
             st.write('You assigned a value higher than zero to width. Please, put a zero to ensure a proper prediction.')
        if input_diameter and not (input_lenght and input_width):
            st.write(f'Weight: {input_weight}. Depth: {input_depth}. Diameter: {input_diameter}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
        elif (input_lenght and input_width) and not input_diameter:
            st.write(f'Weight: {input_weight}. Depth: {input_depth}. Lenght: {input_lenght}. Width: {input_width}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
        else:
            st.write(f'Weight: {input_weight}. Depth: {input_depth}. Diameter: {input_diameter}. Lenght: {input_lenght}. Width: {input_width}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')

    # Prediction preparation
    prediction_button = st.button('Begin prediction', type='primary', disabled=deactivated_button)
    if prediction_button:
        data_load_state = st.text('Loading prediction...')
        if not input_diameter:
            input_diameter = (input_lenght + input_width) / 2
        else:
            input_lenght = input_diameter / 2
            input_width = input_lenght
        for key, value in {'Y-Z': -12, 
                        'W-X': -11, 
                        'W': -10, 
                        'U-V': -9, 
                        'S-T': -8, 
                        'Q-R': -7, 
                        'O-P': -6, 
                        'O': -5, 
                        'N': -4, 
                        'M': -3, 
                        'L': -2, 
                        'K': -1, 
                        'J': 0, 
                        'I': 1, 
                        'H': 2, 
                        'G': 3, 
                        'F': 4, 
                        'E': 5, 
                        'D': 6
                        }.items():
            if slider_color == key:
                slider_color = value
        for key, value in {'Fair':0, 
                            'Good': 1, 
                            'Very Good': 2, 
                            'Premium': 3, 
                            'Ideal': 4
                            }.items():
            if slider_cut == key:
                slider_cut = value
        for key, value in {'Included':0, 
                            'Slightly included (2)': 1, 
                            'Slightly included (1)': 2, 
                            'Very slightly included (2)': 3, 
                            'Very slightly included (1)': 4,
                            'Very, very slightly included (2)': 5,
                            'Very, very slightly included (1)': 6,
                            'Internally flawless': 7,
                            'Flawless': 8
                            }.items():
            if slider_clarity == key:
                slider_clarity = value      
        depth_percentage = (input_depth / ((input_lenght + input_width) / 2)) * 100
        data_array = np.array([[input_weight, slider_cut, slider_color, slider_clarity, depth_percentage, input_lenght, input_width, input_depth]])

        # Prediction
        model = pickle.load(open('src/models/new_model_price_prediction.pkl', 'rb'))
        prediction = np.exp(model.predict(data_array)[0])
        data_load_state.text('Prediction loaded:')
        st.write(f'Your diamond costs {str(prediction).split(".")[0] + "." + str(prediction).split(".")[1][:2]} dollars approximately.')
