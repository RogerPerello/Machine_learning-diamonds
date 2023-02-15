import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from pages.front_page import *


# CONFIGURACIÓN --------------------------------------------------------------------
# ----------------------------------------------------------------------------------
st.set_page_config(page_title='Diamonds price predictor', layout='wide', page_icon='🔹') # menu_items (about & report a bug)


# CARGAR y ENTRENAR DF --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


page_names_to_funcs = {"Main page: historical data": front_page,
                        }
selected_page = st.sidebar.selectbox("Possible predictions", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()