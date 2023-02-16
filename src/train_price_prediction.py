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

model_A = XGBRegressor(n_estimators=788, 
                        eta=0.15,
                        monotone_constraints={"weight (carat)": 1}, 
                        subsample=0.8, 
                        colsample_bytree=0.8, 
                        max_depth=6, 
                        min_child_weight=4, 
                        max_delta_step=5,
                        gamma=0,
                        reg_lambda=0.8,
                        reg_alpha=0.6,
                        num_parallel_tree=10
                    )

model_B = KNeighborsRegressor(algorithm='ball_tree', 
                            leaf_size=2,
                            metric='cityblock', 
                            n_neighbors=12, 
                            weights='distance'
                            )

estimators = [('xgb', model_A), ('knn', model_B)]

stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

stacking.fit(X, y)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

# Serialization
training.send_pickle(stacking, open('src/models/price_prediction.pkl', 'wb'))

print('--- Serialization done ---')
