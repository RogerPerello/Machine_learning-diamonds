import os
from subprocess import run


start = False
while start == False:
    try:
        run('streamlit run .streamlit\\files\\main.py')
        start = True
    except Exception:
        os.chdir('..')
