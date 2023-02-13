from utils.classes import *
from utils.functions import *


df_diamonds = pd.read_csv(r'src\data\processed\diamonds_training.csv')

df_diamonds = remove_all(df_diamonds, zeros_only=True)
df_diamonds = assign_values(df_diamonds, outlier=False)
df_diamonds = df_diamonds.drop(columns='table (percentage)')

# Entrenamiento
print('--- Training started ---')

start_time = time.time()

training = Regression(df_diamonds, 'price')
X_train, X_test, y_train, y_test = training.split_dataframe()

model = XGBRegressor(n_estimators=825, 
                        eta=0.15,
                        monotone_constraints={"weight (carat)": 1}, 
                        subsample=0.9, 
                        colsample_bytree=0.7, 
                        max_depth=6, 
                        min_child_weight=2, 
                        max_delta_step=0,
                        gamma=0,
                        reg_lambda=0.8,
                        reg_alpha=1,
                        num_parallel_tree=10,
                    )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

print(f'Predicted rmse: {mean_squared_error(y_test, y_pred, squared=False)}\n')

# Serialización
training.send_pickle(model, r'src\model\new_model_price_prediction.pkl')

print('--- Serialization done ---')
