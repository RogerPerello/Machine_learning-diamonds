from sklearn.ensemble import StackingRegressor

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

model_A = XGBRegressor(n_estimators=1065, 
                        eta=0.15,
                        monotone_constraints={"weight (carat)": 1}, 
                        subsample=1.0, 
                        colsample_bytree=0.6, 
                        max_depth=6, 
                        min_child_weight=8, 
                        max_delta_step=5,
                        gamma=0,
                        reg_lambda=1.0,
                        reg_alpha=1.0,
                        num_parallel_tree=9
                        )

model_B = KNeighborsRegressor(algorithm='brute', 
                                leaf_size=1,
                                metric='cityblock', 
                                n_neighbors=16, 
                                weights='distance'
                                )

estimators = [('xgb', model_A), ('knn', model_B)]

stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

stacking.fit(X, y)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

# Serialization
training.send_pickle(stacking, open('src/models/predict_from_variables/price_prediction.pkl', 'wb'))

print('--- Serialization done ---')
