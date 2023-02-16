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
X, y = training.split_dataframe(return_entire_Xy=True)

model = KNeighborsRegressor(algorithm='ball_tree', 
                            leaf_size=2,
                            metric='cityblock', 
                            n_neighbors=12, 
                            weights='distance'
                            )

model.fit(X, y)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

# Serialization
training.send_pickle(model, open('src/models/price_prediction_B.pkl', 'wb'))

print('--- Serialization done ---')
