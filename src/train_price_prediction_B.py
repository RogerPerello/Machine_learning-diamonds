from utils.classes import *
from utils.functions import *


df_diamonds = pd.read_csv('src/data/processed/original_processed.csv')

df_diamonds = remove_all(df_diamonds, zeros_only=True)
df_diamonds = assign_values(df_diamonds, outlier=False)
df_diamonds['price'] = np.log(df_diamonds['price'])

# Training
print('--- Training started ---')

start_time = time.time()

training = Regression(df_diamonds, 'price')
X_train, X_test, y_train, y_test = training.split_dataframe()

model = KNeighborsRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

print(f'Predicted rmse: {mean_squared_error(y_test, y_pred, squared=False)}\n')

# Serialization
training.send_pickle(model, open('src/models/price_prediction_B.pkl', 'wb'))

print('--- Serialization done ---')
