import numpy as np
from sklearn.preprocessing import StandardScaler


def apply_scalar(list_of_df, list_of_columns, method, model=None):
    '''Requires numpy as np and from sklearn.preprocessing import StandardScaler'''
    '''Requires a list of dataframes, the chosen method (string: "log" or "standard") and accepts a a list of column names (list of strings), if requried'''
    '''Applies the selected scalar method to a list of train and test dataframes for the chosen columns'''
def apply_scalar(list_of_df, method, list_of_columns=None):
    if method == 'log':
        for df in list_of_df:
            for column in list_of_columns:
                df[column] = np.log(df[column])
    elif method == 'standard':
        scaler = StandardScaler().fit(list_of_df[0].values)
        for df in list_of_df:
            df.loc[:, :] = scaler.transform(df.values)


def impute_boxplot_min_max(df, list_of_columns, min=True, max=True):
    '''Requires numpy as np'''
    '''Requires a dataframe, a list of column names (list of strings) and accepts a boolean for the key arguments "min" and "max"'''
    '''Imputes the outliers of a boxplot for the chosen columns to its min and max values'''
    for column in list_of_columns:
        q3, q1 = np.percentile(df[column], [75, 25])
        iqr = q3 - q1
        if min:
            df.loc[df[column] < q1 - 1.5*iqr, column] = q1 - 1.5*iqr
        if max:
            df.loc[df[column] > q3 + 1.5*iqr, column] = q3 + 1.5*iqr


def remove_elements(df, conditioned_columns, condition, number):
    '''Requires a dataframe, a list of column names (list of strings), a condition (string) and a number'''
    '''Removes the rows of a dataframe based on a condition'''
    for column in conditioned_columns:
        if condition == 'equal':
            df.drop(df[(df[column] == number)].index, inplace=True)
        elif condition == 'bigger':
            df.drop(df[(df[column] > number)].index, inplace=True)
        elif condition == 'bigger_or_equal':
            df.drop(df[(df[column] >= number)].index, inplace=True)   
        elif condition == 'smaller':
            df.drop(df[(df[column] < number)].index, inplace=True)   
        elif condition == 'smaller_or_equal':
            df.drop(df[(df[column] <= number)].index, inplace=True)
