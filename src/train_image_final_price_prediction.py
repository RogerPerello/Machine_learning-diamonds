import numpy as np
import joblib
import time
from sklearn.neighbors import KNeighborsRegressor


# The dataframes are obtained using pickle
# If the image generator were to be used directly, the result would oscillate slightly
X_train, X_test, y_train, y_test = joblib.load(r'src\fixed_images_dataframe.pkl')

X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test[:, 0]])

# Training
print('--- Training started ---')

start_time = time.time()

model = KNeighborsRegressor(n_neighbors=6, 
                            weights='distance',
                            p=1,
                            algorithm='brute', 
                            metric='chebyshev', 
                            leaf_size=1
                            )

model.fit(X, y)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

# Serialization
joblib.dump(model, open('src/models/predict_from_images/price_image_prediction.pkl', 'wb'))

print('--- Serialization done ---')
