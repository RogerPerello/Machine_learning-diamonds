import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path_df = r'src\data\processed\processed_diamonds_1.csv'
df_diamonds = pd.read_csv(os.path.abspath(path_df))

X = df_diamonds.drop(columns='price (dollars)')
y = df_diamonds['price (dollars)']

regression = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=43)
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

print(r2_score(y_test, y_pred))
