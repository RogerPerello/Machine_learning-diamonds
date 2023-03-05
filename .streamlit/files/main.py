import streamlit as st

from front_page import set_front_page
from prediction_from_images_page import predict_from_images
from prediction_from_characteristics_page import predict_from_characteristics


# Configuration
st.set_page_config(page_title='Diamond APPraiser', layout='wide', page_icon='💠')

# Page selection and execution
page_names_to_funcs = {'Introduction': set_front_page,
                       'Prediction from Images': predict_from_images,
                       'Prediction from Characteristics': predict_from_characteristics
                        }

selected_page = st.sidebar.selectbox('Sidebar menu', page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
