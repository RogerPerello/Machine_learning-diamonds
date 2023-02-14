import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import joblib

os.chdir(os.path.dirname(os.getcwdb()))


# CONFIGURACIÓN --------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
st.set_page_config(page_title='Diamonds price predictor', layout='wide', page_icon='🔹') # menu_items (about & report a bug)


# CARGAR y ENTRENAR DF -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

data = pd.read_csv(r'src\data\processed\original_processed.csv')



def main_page():


    # PÁGINA PRINCIPAL ------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    st.header('Ionosphere App')
    st.write('Prediction: is that signal good or bad?')

    st.image('https://svs.gsfc.nasa.gov/vis/a010000/a012900/a012960/Airglow_Layers_print.jpg', caption='This is inspired by this Ionosphere Data Set (https://archive.ics.uci.edu/ml/datasets/ionosphere).')

    with st.expander('Data information'):
        st.write('This radar data was collected by a system in Goose Bay, Labrador. This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts. See the paper for more details. The targets were free electrons in the ionosphere. "Good" radar returns are those showing evidence of some type of structure in the ionosphere. "Bad" returns are those that do not; their signals pass through the ionosphere.')
        st.write("")
        st.write('Received signals were processed using an autocorrelation function whose arguments are the time of a pulse and the pulse number. There were 17 pulse numbers for the Goose Bay system. Instances in this databse are described by 2 attributes per pulse number, corresponding to the complex values returned by the function resulting from the complex electromagnetic signal.')


    # MOSTRAR EJEMPLO -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    st.title('About the data')

    data_load_state = st.text('Loading data...')

    data_sample = data.head(100)

    data_load_state.text("Done! (using st.cache_data)")

    st.subheader('Data sample')
    st.write(data_sample)


    # PREDICCIÓN -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------


    st.title('Prediction')

    st.subheader('Accuracy')

    y_pred = knn.predict(X_test)

    st.write(accuracy_score(y_test, y_pred))


    st.subheader('Recall')

    y_pred = knn.predict(X_test)

    st.write(recall_score(y_test, y_pred))


    st.subheader('Precision')

    y_pred = knn.predict(X_test)

    st.write(precision_score(y_test, y_pred))


    st.subheader('F1 score')

    y_pred = knn.predict(X_test)

    st.write(f1_score(y_test, y_pred))


def page2():
    # PÁGINA SECUNDARIA ------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    st.header('Ionosphere data submission')
    st.write('Please, submit your dataframe')

    with st.form("Submission_form"):
       csv_submit = st.file_uploader('Upload your dataframe as csv', type='csv')
       submitted = st.form_submit_button("Submit")
       if submitted:
            st.write('Done! Here is your prediction:')
            y_pred_submit = knn.predict(pd.read_csv(csv_submit))
            st.write(y_pred_submit)



page_names_to_funcs = {
    "Main page: historical data": main_page,
    "Submit your own data": page2,


}

selected_page = st.sidebar.selectbox("Possible predictions", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()