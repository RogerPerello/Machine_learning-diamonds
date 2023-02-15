import os
import streamlit as st
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import pandas as pd


def front_page():
    '''Title and subtitle'''
    st.header('Diamond price predictor')
    st.write('by Roger Perelló Gumbau')

    '''Main image'''
    image = Image.open(r'.streamlit\images\front_picture.png').resize((1022, 400))
    image_array = np.array(image)/255
    st.image(image_array)

    '''Some info'''
    with st.expander('Learn more'):
        st.write('This radar data was collected by a system in Goose Bay, Labrador. This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts. See the paper for more details. The targets were free electrons in the ionosphere. "Good" radar returns are those showing evidence of some type of structure in the ionosphere. "Bad" returns are those that do not; their signals pass through the ionosphere.')
        st.write("")
        st.write('Received signals were processed using an autocorrelation function whose arguments are the time of a pulse and the pulse number. There were 17 pulse numbers for the Goose Bay system. Instances in this databse are described by 2 attributes per pulse number, corresponding to the complex values returned by the function resulting from the complex electromagnetic signal.')


    data = pd.read_csv(r'src\data\processed\original_processed.csv')
    st.title('About the data')

    data_load_state = st.text('Loading data...')

    data_sample = data.head(100)

    data_load_state.text("Done! (using st.cache_data)")

    st.subheader('Data sample')
    st.write(data_sample)