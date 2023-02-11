from utils.classes import *


df_diamonds = pd.read_csv(r'src\data\processed\diamonds_training.csv')

# Entrenamiento
print('--- Training started ---')

start_time = time.time()

training = Regression(df_diamonds, 'price')
X, y = training.split_dataframe(return_entire_Xy=True)

model = XGBRegressor(colsample_bytree=0.9, 
                     eta=0.15,
                     reg_lambda=0.6,
                     max_delta_step=3,
                     max_depth=5,
                     min_child_weight=1,
                     monotone_constraints={'weight (carat)': 1},
                     n_estimators=746,
                     num_parallel_tree=7,
                     reg_alpha=0.2,
                     subsample=0.8,
                    )
model.fit(X, y)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---')

# Serialización
training.send_pickle(model, r'src\model\new_model.pkl')

print('--- Serialization done ---')
