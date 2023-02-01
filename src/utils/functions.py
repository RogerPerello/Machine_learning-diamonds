import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def remove_all(df, zeros_only=False):
    '''Removes specific values from the diamonds dataframe'''
    df = df.drop(df[(df['width (millimeters)'] == 0)
                    & (df['lenght (millimeters)'] == 0)
                    & (df['depth (millimeters)'] == 0)
                    ].index
                )
    if not zeros_only:
        df = df.drop(df[(df['depth (percentage)'] > 75) | (df['depth (percentage)'] < 45)].index)
        df = df.drop(df[(df['table (percentage)'] > 90) | (df['table (percentage)'] < 45)].index)
        df = df.drop(df[(df['width (millimeters)'] > 30)].index)
        df = df.drop(df[df['depth (millimeters)'] > 30].index)
        df = df.drop(df[(df['table (percentage)'] > 75) | (df['depth (percentage)'] < 52.3)].index)
    return df


def assign_values(df):
    '''Assigns especific values for the diamonds dataframe'''
    df.loc[df.index == 34815, 'lenght (millimeters)'] = 6.62
    df.loc[df['lenght (millimeters)'] > 10.7, 'lenght (millimeters)'] = 10.54
    df.loc[df['depth (millimeters)'] == 0, 'depth (millimeters)'] = df['depth (percentage)'] / 100 * (df['lenght (millimeters)']+df['width (millimeters)']) / 2
    return df


def impute_boxplot_min_max(df, list_of_columns, min=True, max=True):
    '''Imputes the values above the min or the max of the boxplot to the min or the max for the selected columns'''
    for column in list_of_columns:
        q3, q1 = np.percentile(df[column], [75, 25])
        iqr = q3 - q1
        if min:
            df.loc[df[column] < q1 - 1.5*iqr, column] = q1 - 1.5*iqr
        if max:
            df.loc[df[column] > q3 + 1.5*iqr, column] = q3 + 1.5*iqr
        return df


def apply_ridge(df):
    '''Uses ridge to impute a few outliers from the depth (millimeters) column of the diamonds dataframe'''
    q3, q1 = np.percentile(df['depth (millimeters)'], [75, 25])
    iqr = q3 - q1
    y_test = df[(df['depth (millimeters)'] > q3 + 1.5*iqr) | (df['depth (millimeters)'] < q1 - 1.5*iqr)]['depth (millimeters)']
    y_train = df.drop(y_test.index)['depth (millimeters)']
    X_train = df.drop(y_test.index)[['weight (carat)', 'lenght (millimeters)', 'width (millimeters)']]
    X_test = df[(df['depth (millimeters)'] > q3 + 1.5*iqr) | (df['depth (millimeters)'] < q1 - 1.5*iqr)][['weight (carat)', 'lenght (millimeters)', 'width (millimeters)']]
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    df_depth = pd.DataFrame(data={'Original depth': y_test, 'Predicted depth': y_pred})
    for index in df_depth.index:
        df.loc[index, 'depth (millimeters)'] = df_depth.loc[index, 'Predicted depth']
    return df


def impute_next_higher(df, log=True):
    '''Imputes the outliers of the column weight (carat) of the diamonds dataframe to the next higher number'''
    q3, q1 = np.percentile(df['weight (carat)'], [75, 25])
    iqr = q3 - q1
    if log:
        df.loc[df['weight (carat)'] > q3 + 1.5*iqr, 'weight (carat)'] = 1.3862943611198906
    else:
        df.loc[df['weight (carat)'] > q3 + 1.5*iqr, 'weight (carat)'] = np.exp(1.3862943611198906)
    return df
