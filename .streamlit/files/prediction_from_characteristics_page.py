import streamlit as st


def predict_from_characteristics():
    # Title and subtitle
    st.header('Prediction from characteristics')
    st.write('Use the attributes of a diamond to predict its price.')
    st.write('If you prefer to simply upload a photo, click on "Prediction from image" from the sidebar menu.')


    # Form
    with st.form('Diamond characteristics'):
        st.subheader('Key values')
        st.write('Try to be precise when assigning key values.')
        slider_weight = st.slider('Weight (carat)', min_value=0.01, max_value=10.0, step=0.01)
        slider_depth = st.slider('Depth (millimeters)', min_value=0.01, max_value=50.0, step=0.01)
        st.write('----- Fill only the diameter, if your diamond is rounded, or the lenght/width if it is squared. -----')
        st.write('- For squared diamonds:')
        slider_lenght = st.slider('Lenght (millimeters)', min_value=0.0, max_value=20.0, step=0.01)
        slider_width = st.slider('Width (millimeters)', min_value=0.0, max_value=80.0, step=0.01)
        st.write('- For rounded diamonds:')
        slider_diameter = st.slider('Diameter (millimeters)', min_value=0.0, max_value=60.0, step=0.01)
        st.subheader('Secondary values')
        st.write('If you do not know some of the secondary values, make a guess.')
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
        if submitted and (slider_diameter == 0 and slider_lenght != 0 and slider_width != 0):
            st.write(f'Weight: {slider_weight}. Depth: {slider_depth}. Lenght: {slider_lenght}. Width: {slider_width}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
        elif submitted and (slider_diameter != 0 and slider_lenght == 0 and slider_width == 0):
            st.write(f'Weight: {slider_weight}. Depth: {slider_depth}. Diameter: {slider_diameter}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
        elif submitted and (slider_diameter == 0 and slider_lenght == 0 and slider_width == 0):
            st.write('You assigned a value of 0 to both diameter and lenght/width. Please add a correct value to one or another to ensure a proper prediction.')
            st.write(f'Weight: {slider_weight}. Depth: {slider_depth}. Diameter: {slider_diameter}. Lenght: {slider_lenght}. Width: {slider_width}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
        elif submitted and (slider_diameter != 0 and slider_lenght != 0 and slider_width != 0):
            st.write('You assigned a value above 0 to both diameter and lenght/width. The diamond will be assumed to be squared and the diameter will be ignored.')
            st.write(f'Weight: {slider_weight}. Depth: {slider_depth}. Lenght: {slider_lenght}. Width: {slider_width}. Cut: {slider_cut.lower()}. Color: {slider_color.lower()}. Clarity: {slider_clarity.lower()}.')
            slider_diameter = 0
