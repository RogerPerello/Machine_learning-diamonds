import numpy as np
import joblib
import time
from sklearn.neighbors import KNeighborsRegressor

from utils.classes import *

# The dataframe is obtained using pickle
# If the image generator were to be used directly, the result would oscillate slightly
df_images_weight = joblib.load(r'src\fixed_images_dataframe.pkl')

training = Regression(df_images_weight, 'original_price')
X, y = training.split_dataframe(return_entire_Xy=True)

# Training
print('--- Training started ---')

start_time = time.time()

model = SVR(C=3.02,
            epsilon=0.516,
            kernel='linear',
            shrinking=True,
            tol=0.04
            )

model.fit(X, y)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

# Serialization
training.send_pickle(model, open('src/models/predict_from_images/price_image_prediction.pkl', 'wb'))

print('--- Serialization done ---')
