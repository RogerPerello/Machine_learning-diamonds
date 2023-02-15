import os
import streamlit as st

from front_page import *
from prediction_from_images_page import *
from prediction_from_characteristics_page import *


os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Configuration
st.set_page_config(page_title='Diamond APPraiser', layout='wide', page_icon='💠')


# Page selection and execution
page_names_to_funcs = {'Introduction': set_front_page,
                       'Prediction from images': predict_from_images,
                       'Prediction from characteristics': predict_from_characteristics
                        }

selected_page = st.sidebar.selectbox('Sidebar menu', page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
