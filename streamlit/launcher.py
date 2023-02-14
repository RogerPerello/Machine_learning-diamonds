import os
from subprocess import run


# Run this file to launch app.py
os.chdir(os.path.dirname(__file__))

run('streamlit run app.py')
