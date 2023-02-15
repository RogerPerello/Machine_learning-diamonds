from utils.classes import *
from utils.functions import *


df_diamonds = pd.read_csv(r'src\data\processed\competition\train_processed.csv')

df_diamonds = remove_all(df_diamonds, zeros_only=True)
df_diamonds = assign_values(df_diamonds, outlier=False)

# Entrenamiento
print('--- Training started ---')

start_time = time.time()

training = Regression(df_diamonds, 'price')
X_train, X_test, y_train, y_test = training.split_dataframe()

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

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

print(f'Predicted rmse: {mean_squared_error(y_test, y_pred, squared=False)}\n')

# Serialización
training.send_pickle(model, r'src\models\new_model_competition_only.pkl')

print('--- Serialization done ---')
