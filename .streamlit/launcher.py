import os
from subprocess import run

os.chdir('..')

run('streamlit run .streamlit\\files\\main.py')
